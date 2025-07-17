#include <iostream>
#include <vector>
#include <future>
#include <thread>
#include <mutex>
#include <chrono>
#include <random>
#include <string>
#include <cstring>  // memcpy
#include <cstdint>  // size_t
#include <fcntl.h>  // open, O_RDONLY
#include <unistd.h> // read, lseek, close
#include <sys/stat.h> // fstat
#include <sys/types.h> // off_t
#include <errno.h>  // errno
#include <cstdlib>  // calloc, free

const int NUM_LAYERS = 36; // 총 레이어 수 업데이트
const int LOOK_AHEAD = 10; // 미리 로딩 버퍼 수 (look_ahead)
const double COMPUTE_TIME = 0.0023; // 2.3ms (sleep 시뮬)
const size_t LAYER_SIZE = (65LL * 1024 * 1024) / sizeof(float); // 65MB / float (≈17,039,360 elements)
const size_t NUM_CHUNKS = 4; // 청크 수
const std::string WEIGHTS_FILE = "./weights.bin"; // 단일 파일
const size_t LAYER_BYTES = LAYER_SIZE * sizeof(float); // ~65MB
const size_t TOTAL_WEIGHTS_BYTES = LAYER_BYTES * NUM_LAYERS; // 총 ≈2.34GB

std::mutex pool_mutex; // 풀 보호

// 메모 풀: vector<float*> 로 raw 포인터 재활용
std::vector<float*> memory_pool;
std::vector<size_t> layer_offsets; // 오프셋 캐싱
int fd = -1; // 공유 fd

double total_load_time = 0.0; // 총 로딩 시간 (ms)
double total_compute_time = 0.0; // 총 계산 시간 (ms)

// 메모리 미리 할당 함수 (Extract Method 리팩토링)
void preallocate_mem_pool() {
    memory_pool.resize(LOOK_AHEAD + 1);
    for (auto& ptr : memory_pool) {
        ptr = static_cast<float*>(calloc(LAYER_SIZE, sizeof(float))); // 0 초기화 미리 할당
        if (!ptr) {
            std::cerr << "Calloc failed!" << std::endl;
            exit(1);
        }
    }
}

// 로드 완료 검사 함수 (Extract Method 리팩토링)
bool is_layer_loaded(int slot) {
    // calloc으로 0 초기화되었으므로, 첫 요소가 0이 아닌지 간단 체크 (실제 데이터 로드 확인)
    if (memory_pool[slot][0] == 0.0f) {
        std::cerr << "Layer not loaded for slot " << slot << std::endl;
        return false;
    }
    return true;
}

// 로드 함수 (fd 기반 청크 병렬 읽기 + memcpy)
void load_layer(int layer_id, int slot) {
    auto start = std::chrono::high_resolution_clock::now();

    // 청크 병렬 읽기
    size_t chunk_size = LAYER_SIZE / NUM_CHUNKS;
    std::vector<std::future<void>> chunk_futures;

    size_t offset = layer_offsets[layer_id - 1];

    for (size_t idx = 0; idx < NUM_CHUNKS; ++idx) {
        chunk_futures.push_back(std::async(std::launch::async, [&](size_t i) {
            int chunk_fd = dup(fd); // FD 복제
            if (chunk_fd == -1) return;
            off_t chunk_start = static_cast<off_t>(offset + i * chunk_size * sizeof(float));
            if (lseek(chunk_fd, chunk_start, SEEK_SET) == -1) {
                close(chunk_fd);
                return;
            }
            std::vector<float> buffer(chunk_size);
            ssize_t bytes_read = read(chunk_fd, buffer.data(), chunk_size * sizeof(float));
            if (bytes_read != static_cast<ssize_t>(chunk_size * sizeof(float))) {
                std::cerr << "Read error: " << strerror(errno) << std::endl;
                close(chunk_fd);
                return;
            }
            {
                std::lock_guard<std::mutex> chunk_lock(pool_mutex);
                std::memcpy(memory_pool[slot] + i * chunk_size, buffer.data(), chunk_size * sizeof(float));
            }
            close(chunk_fd);
        }, idx));
    }
    for (auto& f : chunk_futures) f.get();

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Loaded layer " << layer_id << " time: " << duration << " ms" << std::endl;
    total_load_time += duration;
}

// 계산 함수 (곱셈 시뮬)
void compute_layer(int layer_id, int slot) {
    auto start = std::chrono::high_resolution_clock::now();

    std::this_thread::sleep_for(std::chrono::duration<double>(COMPUTE_TIME));

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Computed layer " << layer_id << " time: " << duration << " ms" << std::endl;
    total_compute_time += duration;
}

int main(int argc, char* argv[]) {
    auto program_start = std::chrono::high_resolution_clock::now();

    fd = open(WEIGHTS_FILE.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "Error opening file: " << strerror(errno) << std::endl;
        return 1;
    }

    struct stat st;

    // 오프셋 캐싱 초기화
    layer_offsets.resize(NUM_LAYERS);
    for (int i = 0; i < NUM_LAYERS; ++i) {
        layer_offsets[i] = static_cast<size_t>(i) * LAYER_BYTES;
    }

    preallocate_mem_pool(); // 미리 calloc 할당

    // 초기 버퍼 로드 (비동기 병렬)
    std::vector<std::future<void>> futures;
    for (int i = 0; i < LOOK_AHEAD; ++i) {
        futures.push_back(std::async(std::launch::async, load_layer, i + 1, i));
    }
    for (auto& f : futures) f.get();

    int current_layer = 1;
    int current_slot = 1;

    while (current_layer <= NUM_LAYERS) {
        // 계산 전에 로드 완료 검사
        if (!is_layer_loaded(current_slot)) {
            std::cerr << "Error: Layer " << current_layer << " not loaded!" << std::endl;
            break;
        }

        auto compute_future = std::async(std::launch::async, compute_layer, current_layer, current_slot);

        std::future<void> load_future;
        int next_load_id = current_layer + LOOK_AHEAD;
        if (next_load_id <= NUM_LAYERS) {
            int load_slot = LOOK_AHEAD;
            load_future = std::async(std::launch::async, load_layer, next_load_id, load_slot);
        }

        compute_future.get();

        if (next_load_id <= NUM_LAYERS) {
            load_future.get();
            std::lock_guard<std::mutex> lock(pool_mutex);
            std::swap(memory_pool[current_slot], memory_pool[LOOK_AHEAD]);
        }

        current_layer++;
        current_slot = (current_slot + 1) % LOOK_AHEAD;
    }

    auto program_end = std::chrono::high_resolution_clock::now();
    double program_duration = std::chrono::duration<double, std::milli>(program_end - program_start).count();

    std::cout << "Total loading time: " << total_load_time << " ms" << std::endl;
    std::cout << "Total compute time: " << total_compute_time << " ms" << std::endl;
    std::cout << "Total program execution time: " << program_duration << " ms" << std::endl;

    for (auto ptr : memory_pool) {
        if (ptr) free(ptr);
    }

    close(fd);

    return 0;
}
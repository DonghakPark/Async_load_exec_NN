#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <bs_thread_pool_manager.hpp>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

constexpr int NUM_LAYERS = 34;
constexpr int LOOK_AHEAD = 8;
constexpr double COMPUTE_TIME = 0.0023;
constexpr size_t LAYER_SIZE = (((3072 * 3072 * 2) + (3072 * 256 * 2) +
                                (3072 * 8192 * 2) + (8192 * 8192)) *
                               4 / 8);
constexpr size_t NUM_THREAD = 64;
const std::string WEIGHTS_FILE = "./weights.bin";
std::vector<std::future<void>> load_futures;
auto &bs_thread_pool = nntrainer::ThreadPoolManager::getInstance();

std::vector<void *> memory_pool;
std::vector<size_t> layer_offsets;
int fd = -1;

double total_load_time = 0.0;
double total_compute_time = 0.0;

void preallocate_mem_pool() {
  for (unsigned int i = 0; i < NUM_LAYERS; i++) {
    void *ptr = nullptr;
    posix_memalign(&ptr, 4096, LAYER_SIZE);
    memory_pool.emplace_back(ptr);
  }
}

void mmap_worker(void *to, void *from, size_t size) { memcpy(to, from, size); }

void load_layer(int layer_id) {
  if (layer_id >= NUM_LAYERS) return;

  size_t chunk_size = LAYER_SIZE / NUM_THREAD;
  size_t offset = layer_offsets[layer_id];
  auto start = std::chrono::high_resolution_clock::now();

  char *mapped_ptr =
      static_cast<char *>(mmap(nullptr, LAYER_SIZE, PROT_READ | O_DIRECT,
                               MAP_PRIVATE | MAP_POPULATE, fd, offset));
  madvise(mapped_ptr, LAYER_SIZE, MADV_WILLNEED);

  for (size_t i = 0; i < NUM_THREAD; ++i) {
    bs_thread_pool.detach_task([=] {
      memcpy(memory_pool[layer_id] + i * chunk_size,
             mapped_ptr + i * chunk_size, chunk_size);
    });
  }
  bs_thread_pool.wait();

  auto end = std::chrono::high_resolution_clock::now();
  double duration =
      std::chrono::duration<double, std::milli>(end - start).count();
  printf("Loaded Layer[%d] : %f ms (chunk size : %d)\n", layer_id, duration,
         chunk_size);

  total_load_time += duration;
  munmap(mapped_ptr, LAYER_SIZE);
}

void compute_layer(int layer_id) {
  auto start = std::chrono::high_resolution_clock::now();

  std::this_thread::sleep_for(std::chrono::duration<double>(COMPUTE_TIME));

  auto end = std::chrono::high_resolution_clock::now();
  double duration =
      std::chrono::duration<double, std::milli>(end - start).count();

  printf("Computed Layer[%d] : %f ms \n", layer_id, duration);
  total_compute_time += duration;
}

int main(int argc, char *argv[]) {
  fd = open(WEIGHTS_FILE.c_str(), O_RDONLY | O_DIRECT);
  for (int i = 0; i < NUM_LAYERS; ++i) {
    layer_offsets.emplace_back(static_cast<size_t>(i) * LAYER_SIZE);
  }

  preallocate_mem_pool();

  auto program_start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < LOOK_AHEAD; ++i) {
    load_futures.push_back(std::async(std::launch::async, load_layer, i));
  }

  for (unsigned int order = 0; order < NUM_LAYERS; ++order) {
    compute_layer(order);
    load_futures.push_back(
        std::async(std::launch::async, load_layer, order + LOOK_AHEAD));
  }

  auto program_end = std::chrono::high_resolution_clock::now();
  double program_duration =
      std::chrono::duration<double, std::milli>(program_end - program_start)
          .count();

  std::cout << "Total loading time: " << total_load_time << " ms" << std::endl;
  std::cout << "Total compute time: " << total_compute_time << " ms"
            << std::endl;
  std::cout << "Total Forwarding execution time: " << program_duration << " ms"
            << std::endl;

  for (auto ptr : memory_pool) {
    if (ptr) free(ptr);
  }
  close(fd);
  return 0;
}
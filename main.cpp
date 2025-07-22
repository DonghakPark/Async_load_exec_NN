#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

const int NUM_LAYERS = 34;
const int LOOK_AHEAD = 10;
const double COMPUTE_TIME = 0.0023;
const size_t LAYER_SIZE = (((3072 * 3072 * 2) + (3072 * 256 * 2) +
                            (3072 * 8192 * 2) + (8192 * 8192)) *
                           4);
const size_t NUM_CHUNKS = 4;
const std::string WEIGHTS_FILE = "./weights.bin";
const size_t LAYER_BYTES = LAYER_SIZE;
const size_t TOTAL_WEIGHTS_BYTES = LAYER_BYTES * NUM_LAYERS;
std::vector<std::future<void>> load_futures;
std::unordered_map<int, bool> load_status;

std::mutex pool_mutex;

std::vector<void *> memory_pool;
std::vector<size_t> layer_offsets;
int fd = -1;

double total_load_time = 0.0;
double total_compute_time = 0.0;

void preallocate_mem_pool() {
  for (unsigned int i = 0; i < NUM_LAYERS; i++) {
    void *ptr = aligned_alloc(4096, LAYER_SIZE);
    memory_pool.emplace_back(ptr);
    std::cout << "Allocate Mem for [" << i << "] " << ptr
              << " - size : " << LAYER_SIZE << std::endl;
  }
}

bool is_layer_loaded(int layer_order) {
  if (load_status[layer_order] == false) {
    load_futures[layer_order].get();
    return true;
  }
  return true;
}

void load_layer(int layer_id, int slot) {
  auto start = std::chrono::high_resolution_clock::now();

  size_t chunk_size = LAYER_SIZE / NUM_CHUNKS;

  std::vector<std::future<void>> chunk_futures;
  size_t offset = layer_offsets[layer_id - 1];

  for (size_t idx = 0; idx < NUM_CHUNKS; ++idx) {
    chunk_futures.push_back(std::async(
        std::launch::async,
        [&](size_t i) {
          off_t chunk_start = static_cast<off_t>(offset + i * chunk_size);
          std::vector<float> buffer(chunk_size);
          ssize_t bytes_read = read(fd, buffer.data(), chunk_size);
          {
            std::lock_guard<std::mutex> chunk_lock(pool_mutex);
            std::memcpy(memory_pool[slot] + i * chunk_size, buffer.data(),
                        chunk_size);
          }
        },
        idx));
  }
  for (auto &f : chunk_futures) f.get();

  load_status[layer_id] = true;

  auto end = std::chrono::high_resolution_clock::now();
  double duration =

      std::chrono::duration<double, std::milli>(end - start).count();
  printf("Loaded Layer : %d, Time : %f ms\n", layer_id, duration);

  total_load_time += duration;
}

void compute_layer(int layer_id) {
  auto start = std::chrono::high_resolution_clock::now();
  std::this_thread::sleep_for(std::chrono::duration<double>(COMPUTE_TIME));

  auto end = std::chrono::high_resolution_clock::now();
  double duration =
      std::chrono::duration<double, std::milli>(end - start).count();

  printf("Computed Layer : %d, Time : %f ms\n", layer_id, duration);
  total_compute_time += duration;
}

int main(int argc, char *argv[]) {
  auto program_start = std::chrono::high_resolution_clock::now();

  fd = open(WEIGHTS_FILE.c_str(), O_RDONLY);
  for (int i = 0; i < NUM_LAYERS; ++i) {
    layer_offsets.emplace_back(static_cast<size_t>(i) * LAYER_BYTES);
  }

  preallocate_mem_pool();
  std::cout << "Allocate memmory Done " << std::endl;

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// Forwarding Logic
  ///

  for (int i = 0; i < LOOK_AHEAD; ++i) {
    load_futures.push_back(std::async(std::launch::async, load_layer, i, i));
  }  // Load Weight (0 ~ Look Ahead)
  std::cout << "Pre Load Done " << std::endl;

  for (unsigned int order = 0; order < NUM_LAYERS; ++order) {
    is_layer_loaded(order);
    compute_layer(order);
    load_futures.push_back(
        std::async(std::launch::async, load_layer, order + LOOK_AHEAD, order));
  }

  auto program_end = std::chrono::high_resolution_clock::now();
  double program_duration =
      std::chrono::duration<double, std::milli>(program_end - program_start)
          .count();

  std::cout << "Total loading time: " << total_load_time << " ms" << std::endl;
  std::cout << "Total compute time: " << total_compute_time << " ms"
            << std::endl;
  std::cout << "Total program execution time: " << program_duration << " ms"
            << std::endl;

  for (auto ptr : memory_pool) {
    if (ptr) free(ptr);
  }

  close(fd);

  return 0;
}
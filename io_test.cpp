#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

constexpr size_t LAYER_SIZE = (((3072 * 3072 * 2) + (3072 * 256 * 2) +
                               (3072 * 8192 * 2) + (8192 * 8192)) *
                              4 / 8);
std::atomic<size_t> total_bytes{0};
// std::vector<char> tmp(2097152);

void mmap_worker(char* base, size_t chunk_size, size_t offset) {
  std::vector<char> tmp(chunk_size);
  memcpy(tmp.data(), base + offset, chunk_size);
  total_bytes += chunk_size;
}

void benchmark_mmap(const char* path, size_t threads, size_t chunk) {
  auto t0 = std::chrono::high_resolution_clock::now();
  total_bytes = 0;

  int fd = open(path, O_RDONLY);
  char* base = (char*)mmap(NULL, LAYER_SIZE, PROT_READ, MAP_PRIVATE, fd, 0);
  madvise(base, LAYER_SIZE, MADV_WILLNEED);

  std::vector<std::thread> ths;
  for (size_t i = 0; i < threads; ++i)
    ths.emplace_back(mmap_worker, base, chunk, i * chunk);
  for (auto& t : ths) t.join();

  munmap(base, LAYER_SIZE);
  close(fd);

  auto t1 = std::chrono::high_resolution_clock::now();
  double sec = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double mbps = (double(total_bytes) / (1024 * 1024)) / sec * 1000;

  std::cout << "[mmap"
            << "] threads=" << threads << ", chunk=" << chunk
            << ", time=" << sec << "ms, speed=" << mbps << " MB/s\n";
}

int main(int argc, char** argv) {
  const char* path = "./weights.bin";

  for (size_t c : {4096, 4096 * 2, 2096 * 3, 4096 * 4, 4096 * 16, 4096 * 32,
                   4096 * 64, 4096 * 128, 4096 * 256, 4096 * 512}) {
    for (size_t threads : {1, 2, 4, 8, 16, 32}) {
      benchmark_mmap(path, threads, c);
      std::cout << "---------------------\n";
    }
  }

  return 0;
}

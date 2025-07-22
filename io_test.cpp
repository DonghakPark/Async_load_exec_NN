// random_read_crossplatform.cpp
#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <atomic>
#include <chrono>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

// constexpr size_t FILE_SIZE = 2200ULL * 1024 * 1024;
constexpr size_t FILE_SIZE = ((( 3072*3072 * 2) + ( 3072*256 * 2)  + (3072*8192*2) + (8192*8192)) * 4 );
std::atomic<size_t> total_bytes{0};
constexpr size_t ALIGNMENT = 4096;

void* aligned_alloc(size_t alignment, size_t size) {
#ifdef _WIN32
    return VirtualAlloc(NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* ptr = nullptr;
    posix_memalign(&ptr, alignment, size);
    return ptr;
#endif
}

void aligned_free(void* ptr) {
#ifdef _WIN32
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    free(ptr);
#endif
}

//
// mmap worker (shared base ptr)
//
void mmap_worker(char* base, size_t chunk_size, size_t ops, size_t fsize) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, fsize - chunk_size);
    std::vector<char> tmp(chunk_size);

    for (size_t i = 0; i < ops; ++i) {
        size_t offset = dist(rng);
        memcpy(tmp.data(), base + offset, chunk_size);
        total_bytes += chunk_size;
    }
}

void benchmark_mmap(const char* path, size_t threads, size_t chunk, size_t ops, bool use_madvise) {
    total_bytes = 0;

#ifdef _WIN32
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    HANDLE map = CreateFileMapping(file, NULL, PAGE_READONLY, 0, 0, NULL);
    char* base = (char*)MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0);
#else
    int fd = open(path, O_RDONLY);
    char* base = (char*)mmap(NULL, FILE_SIZE, PROT_READ, MAP_SHARED, fd, 0);
    if (use_madvise) madvise(base, FILE_SIZE, MADV_WILLNEED);
#endif

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> ths;
    for (size_t i = 0; i < threads; ++i)
        ths.emplace_back(mmap_worker, base, chunk, ops, FILE_SIZE);
    for (auto& t : ths) t.join();

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    double mbps = (double(total_bytes) / (1024 * 1024)) / sec;

    std::cout << "[mmap" << (use_madvise ? "+madvise" : "") << "] threads=" << threads
              << ", chunk=" << chunk << ", time=" << sec << "s, speed=" << mbps << " MB/s\n";

#ifdef _WIN32
    UnmapViewOfFile(base);
    CloseHandle(map);
    CloseHandle(file);
#else
    munmap(base, FILE_SIZE);
    close(fd);
#endif
}

//
// pread worker
//
void pread_worker(const char* path, size_t chunk, size_t ops, size_t fsize) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, fsize - chunk);
    char* buffer = (char*)aligned_alloc(ALIGNMENT, chunk);

#ifdef _WIN32
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
#else
    int fd = open(path, O_RDONLY);
#endif

    for (size_t i = 0; i < ops; ++i) {
        size_t offset = dist(rng);

#ifdef _WIN32
        LARGE_INTEGER li; li.QuadPart = offset;
        SetFilePointerEx(file, li, NULL, FILE_BEGIN);
        DWORD br;
        ReadFile(file, buffer, DWORD(chunk), &br, NULL);
#else
        pread(fd, buffer, chunk, offset);
#endif
        total_bytes += chunk;
    }

#ifdef _WIN32
    CloseHandle(file);
#else
    close(fd);
#endif
    aligned_free(buffer);
}

void benchmark_pread(const char* path, size_t threads, size_t chunk, size_t ops) {
    total_bytes = 0;
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> ths;
    for (size_t i = 0; i < threads; ++i)
        ths.emplace_back(pread_worker, path, chunk, ops, FILE_SIZE);
    for (auto& t : ths) t.join();

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    double mbps = (double(total_bytes) / (1024 * 1024)) / sec;

    std::cout << "[pread] threads=" << threads << ", chunk=" << chunk
              << ", time=" << sec << "s, speed=" << mbps << " MB/s\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <testfile> <op> \n";
        return 1;
    }

    const char* path = argv[1];
    size_t chunk = 4096;
    size_t ops = std::stoi(argv[2]);

    for(size_t c : {4096, 4096*2, 2096*3, 4096*4, 4096 * 16, 4096 * 32, 4096 * 64, 4096*128}){
      for (size_t threads : {1, 2, 4, 8, 16}) {
        benchmark_mmap(path, threads, c, ops, false);
        benchmark_mmap(path, threads, c, ops, true);
        benchmark_pread(path, threads, c, ops);
        std::cout << "---------------------\n";
      }
    }

    return 0;
}

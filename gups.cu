// -*- c++ -*-
#include <sys/mman.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <chrono>
#include <cstring>
#include <cassert>
#include <fstream>

struct node {
  node *next;
};

template <typename T>
void swap(T &x, T &y) {
  T t = x;
  x = y; y = t;
}

template <typename T>
void shuffle(std::vector<T> &vec) {
  const size_t len = vec.size();
  for(size_t i = 0; i < len; i++) {
    size_t j = i + (rand() % (len - i));
    swap(vec[i], vec[j]);
  }
}

template <typename T>
size_t partition(T *arr, size_t n) {
  size_t d=0;
  size_t r = rand() % n;
  T p = arr[r];
  arr[r] = arr[n-1];
  arr[n-1] = p;
  
  for(size_t i=0;i<(n-1);i++) {
    if(arr[i] < p) {
      swap(arr[i], arr[d]);
      d++;
    }
  }
  arr[n-1] = arr[d];
  arr[d] = p;
  return d;
}

template <typename T>
void sort(T *arr, size_t len) {
  size_t d;
  if(len <= 16) {
    for(size_t i=1;i<len;i++) {
      size_t j=i;
      while((j > 0) && (arr[j-1] > arr[j])) {
	swap(arr[j-1], arr[j]);
	j--;
      }
    }    
    return;
  }
  d = partition(arr, len);
  sort(arr, d);
  sort(arr+d+1, len-d-1);
}


__global__ void gups(uint64_t *mem, uint32_t *xx, uint32_t lg_n, int64_t iters) {
  int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
  uint32_t x = xx[idx];
  const uint32_t m = (1U<<lg_n)-1;
  
  while(iters >= 0) {
    mem[x&m] ^= x;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    mem[x&m] ^= x;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    
    iters -= 2;
  }
}

int main(int argc, char *argv[]) {
  static const int warp_size = 32;  
  static const int nthr = warp_size;
  static const uint32_t max_n = 1U<<28;
  static const int64_t iters = 1L<<20;
  
  
  uint64_t *buf = nullptr;
  uint32_t *xx = nullptr;

  node *nodes = nullptr, **nodes_out = nullptr;
  int64_t *cycles = nullptr;
  
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if(deviceProp.kernelExecTimeoutEnabled) {
    std::cout << "Warning : kernel timeout enabled (long runs will fail)\n";
  }
  double freq = deviceProp.clockRate * 1000.0;

  auto ce = cudaMallocManaged((void**)&xx, sizeof(uint32_t)*nthr);
  if(ce != cudaSuccess) {
    std::cout << cudaGetErrorString(ce) << "\n";
    exit(-1);
  }

  
  uint32_t n_keys = 1U<<24;
  std::vector<uint32_t> keys(nthr);
  for(uint32_t i = 0; i < n_keys; i++) {
    keys[i] = i+1;
  }
  shuffle(keys);
  //copy to shared buffer
  for(uint32_t i = 0; i < nthr; i++) {
    xx[i] = keys[i];
  }
  keys.clear();
  assert(cudaMallocManaged((void**)&buf, sizeof(uint64_t)*max_n) == cudaSuccess);

  for(uint32_t lg_n = 10; (1U<<lg_n) <= max_n; ++lg_n) {

    gups<<<nthr/warp_size, warp_size>>>(buf, xx, lg_n, iters);
    cudaDeviceSynchronize();
    auto ce = cudaGetLastError();
    if(ce != cudaSuccess) {
      std::cerr << "iters = " << iters << " : "
		<< cudaGetErrorString(ce) << "\n";
    }
  }
  
#if 0
  assert(cudaMallocManaged((void**)&nodes, sizeof(node)*max_keys) == cudaSuccess);
  assert(cudaMallocManaged((void**)&nodes_out, sizeof(node*)*nthr) == cudaSuccess);  
  assert(cudaMallocManaged((void**)&cycles, sizeof(int64_t)*nthr) == cudaSuccess);

  uint64_t max_iter_step = 1UL<<22;
  
  for(uint64_t n_keys = 1UL<<8; n_keys <= max_keys; n_keys *= 2) {
    node *h = &nodes[keys[0]];
    node *c = h;  
    h->next = h;
    for(uint64_t i = 1; i < n_keys; i++) {
      node *n = &nodes[keys[i]];
      node *t = c->next;
      c->next = n;
      n->next = t;
      c = n;
    }
    uint64_t iters = n_keys*16;
    
    for(int i = 0; i < nthr; i++) {
      //int r = rand() % n_keys;
      node *n = &nodes[keys[11]];
      nodes_out[i] = n;
      cycles[i] = 0;
    }
    
    if(iters < (1UL<<20)) {
      iters = 1UL<<20;
    }

    
    if(iters <= max_iter_step) {
      traverse<<<nthr/warp_size, warp_size>>>(nodes_out, cycles, iters);
      cudaDeviceSynchronize();
      auto ce = cudaGetLastError();
      if(ce != cudaSuccess) {
	std::cerr << "iters = " << iters << " : "
		  << cudaGetErrorString(ce) << "\n";
      }
    }
    else {
      for(uint64_t itrs = 0; itrs < iters; itrs += max_iter_step) {
	traverse<<<nthr/warp_size, warp_size>>>(nodes_out, cycles, max_iter_step);
	cudaDeviceSynchronize();
	assert(cudaGetLastError() == cudaSuccess);
      }
      max_iter_step /= 2;
    }
    sort(cycles, nthr);
    double cpl = static_cast<double>(cycles[nthr/2]) / iters;
    double nspl = (cpl/freq) / (1e-9);
    std::cout << sizeof(node)*n_keys << " bytes, GPU cycles per load "
	      << cpl << ", nanosec per load " << nspl << " \n";
    
  }
#endif
  cudaFree(nodes);
  cudaFree(nodes_out);
  cudaFree(cycles);
  return 0;
}

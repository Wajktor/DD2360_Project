#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

class Managed {
public:
  void *operator new(size_t len) {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
    return ptr;
  }

  void operator delete(void *ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }
};

struct particle2 : public Managed{
	int id;
	int len;
	char *name;
};

struct particle{
	int id;
	int len;
	float *nums;
};


void allocate(struct particle **par){

	// **par = &oldpar
	cudaMallocManaged( &(*par), sizeof(particle));

	cudaMallocManaged( &((*par)->nums), sizeof(float) * 3 );
}

__global__ void change(struct particle *par){

	par->nums[0] = 1337.0;

}

__global__ void blesd(){

	float weight[2000][2000][2000];
	weight[0][0][0] = 99;

	for(int i = 0; i < 1; ++i){
		printf("weight: %f \n", weight[0][0][0]);
	}

}


int main(){


	blesd<<<1,1>>>();

	cudaDeviceSynchronize();

	return 0;
}
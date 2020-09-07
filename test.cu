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

struct particle : public Managed{
	int id;
	int len;
	char *name;
};

struct par{
	int id;
	int len;
	char *name;
};

__global__ void testKern(struct particle *part){
	printf("id: %d \nlen: %d \nname: %s\n", part->id, part->len, part->name);
}

__global__ void testKern2(struct par *part){
	printf("Inom GPU \n");
	part->name[2] = 'k';
	printf("id: %d \nlen: %d \nname: %s\n", part->id, part->len, part->name);
}

__global__ void testkern3(struct par *part){

}


int main(){

	par *part = new par();

	par *part2;

	cudaMallocManaged(&part2, sizeof(par));
	cudaMallocManaged(&(part2->name), 8*sizeof(char));
	cudaMallocManaged(&(part->name), part->len);

	part2->id = 33;
	part2->len = 8;

	part2->name[0] = 'a';
	part2->name[1] = 'b';
	part2->name[2] = 'c';

	strncpy();

	std::cout << "r48" << std::endl;


	part->id = 3;
	part->len = 7;
	part->name ="wakanda";

	std::cout << "r57" << std::endl;




	cudaDeviceSynchronize();


	std::cout << "CPU: " << std::endl;
	printf("id: %d \nlen: %d \nname: %s\n", part2->id, part2->len, part2->name);

	std::cout << "GPU: " << std::endl;
	testKern2<<<1,1>>>(part2);
	cudaDeviceSynchronize();
	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

	printf("id: %d \nlen: %d \nname: %s\n", part2->id, part2->len, part2->name);


	return 0;
}
#include "Compare.h"
#include <cstdlib>

void compare(struct particles *part_cpu, struct particles *part_gpu){

	for(int i = 0; i < part_cpu->nop; i++){
		//Compare if error is large enough
		if( abs( part_cpu->x[i] - part_gpu->x[i] ) > 0.001f ) {
			std::cout << "CPU: " << part_cpu->x[i] << ", GPU: " << part_gpu->x[i] << std::endl;
		}
	}
}

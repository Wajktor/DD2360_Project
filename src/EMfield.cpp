#include "EMfield.h"
#include <cuda.h>
#include <cuda_runtime.h>


/** allocate electric and magnetic field */
void field_allocate(struct grid* grd, struct EMfield* field)
{

    // E on nodes
    field->Ex  = newArr3<FPfield>(&field->Ex_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Ey  = newArr3<FPfield>(&field->Ey_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Ez  = newArr3<FPfield>(&field->Ez_flat, grd->nxn, grd->nyn, grd->nzn);
    // B on nodes
    field->Bxn = newArr3<FPfield>(&field->Bxn_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Byn = newArr3<FPfield>(&field->Byn_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Bzn = newArr3<FPfield>(&field->Bzn_flat, grd->nxn, grd->nyn, grd->nzn);
}

void field_allocate_gpu(struct grid* grd, struct EMfield **field){


    cudaMallocManaged( &(*field), sizeof(EMfield) );

/*
    cudaMallocManaged(&((*field)->Ex_flat), sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    cudaMallocManaged(&((*field)->Ey_flat), sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    cudaMallocManaged(&((*field)->Ez_flat), sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);

    cudaMallocManaged(&((*field)->Bxn_flat), sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    cudaMallocManaged(&((*field)->Byn_flat), sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    cudaMallocManaged(&((*field)->Bzn_flat), sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
*/
    

    (*field)->Ex  = newArr3<FPfield>(&(*field)->Ex_flat, grd->nxn, grd->nyn, grd->nzn);
    (*field)->Ey  = newArr3<FPfield>(&(*field)->Ey_flat, grd->nxn, grd->nyn, grd->nzn);
    (*field)->Ez  = newArr3<FPfield>(&(*field)->Ez_flat, grd->nxn, grd->nyn, grd->nzn);
    // B on nodes
    (*field)->Bxn = newArr3<FPfield>(&(*field)->Bxn_flat, grd->nxn, grd->nyn, grd->nzn);
    (*field)->Byn = newArr3<FPfield>(&(*field)->Byn_flat, grd->nxn, grd->nyn, grd->nzn);
    (*field)->Bzn = newArr3<FPfield>(&(*field)->Bzn_flat, grd->nxn, grd->nyn, grd->nzn);

    
    cudaDeviceSynchronize();

    std::cout << "FIELD ALLOC: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    std::cout << "tot size: " << grd->nxn * grd->nyn * grd->nzn << std::endl;

}

/** deallocate electric and magnetic field */
void field_deallocate(struct grid* grd, struct EMfield* field)
{
    // E deallocate 3D arrays
    delArr3(field->Ex, grd->nxn, grd->nyn);
    delArr3(field->Ey, grd->nxn, grd->nyn);
    delArr3(field->Ez, grd->nxn, grd->nyn);

    // B deallocate 3D arrays
    delArr3(field->Bxn, grd->nxn, grd->nyn);
    delArr3(field->Byn, grd->nxn, grd->nyn);
    delArr3(field->Bzn, grd->nxn, grd->nyn);
}

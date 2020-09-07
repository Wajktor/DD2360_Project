/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"

// Field structure
#include "EMfield.h" // Just E and Bn
#include "EMfield_aux.h" // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include "RW_IO.h"

// Compare function, NEW CODE
#include "Compare.h"


int main(int argc, char **argv){
    
    // Read the inputfile and fill the param structure
    parameters param;
    parameters *param_gpu;
    cudaMallocManaged(&param_gpu, sizeof(parameters));
    // Read the input file name from command line
    readInputFile(&param,argc,argv);
    printParameters(&param);
    saveParameters(&param);

    *param_gpu = param;
    
    // Timing variables
    double iStart = cpuSecond();
    double iMover, iInterp, eMover = 0.0, eInterp = 0.0;
    
    // Set-up the grid information
    grid grd;
    grid *grd_gpu;

    setGrid(&param, &grd);
    setGrid_gpu(&param, &grd_gpu);

    std::cout << "grd: " <<  grd_gpu->XN_flat[99] << "\n";
    
    // Allocate Fields
    EMfield field;
    EMfield *field_gpu;

    field_allocate(&grd,&field);
    field_allocate_gpu(grd_gpu, &field_gpu);

    field_gpu->Ex_flat[0] = 929;

    cudaDeviceSynchronize();
    printf("field allocated gpu\n");
    

    EMfield_aux field_aux;
    EMfield_aux field_aux_gpu;
    field_aux_allocate(&grd,&field_aux);
    field_aux_allocate(grd_gpu, &field_aux_gpu);
    
    
    // Allocate Interpolated Quantities
    // per species
    interpDensSpecies *ids     = new interpDensSpecies[param.ns];
    interpDensSpecies *ids_gpu = new interpDensSpecies[param.ns];

    for (int is=0; is < param.ns; is++){
        interp_dens_species_allocate(&(*grd_gpu), &ids_gpu[is], is);
        interp_dens_species_allocate(&grd,     &ids[is],     is);
    }
    // Net densities

    interpDensNet idn;
    interpDensNet idn_gpu;
    interp_dens_net_allocate(&grd,&idn);
    interp_dens_net_allocate(grd_gpu, &idn_gpu);
    
    // Allocate Particles
    particles *part     = new particles[param.ns];
    //particles *part_gpu = new particles[param.ns]; //for comparison
    particles *part_gpu;
    cudaMallocManaged(&part_gpu, sizeof(particles) * param.ns);
    // allocation
    for (int is=0; is < param.ns; is++){
        particle_allocate(&param,&part[is],    is);
        particle_allocate_device(&param,&part_gpu[is],is); // for comparison
    }


    
    // Initialization
    //initGEM(&param,&grd,    &field,    &field_aux,    part,    ids);
    printf("INIT GPU vars \n");
    initGEM(&param, grd_gpu, field_gpu,&field_aux_gpu,part_gpu,ids);


    std::cout << "after INIT\n" << cudaGetErrorString(cudaGetLastError()) << std::endl;

    cudaDeviceSynchronize();


    //Test device mover
    mover_PC_device(&part_gpu[0], field_gpu, grd_gpu, param_gpu);
    cudaDeviceSynchronize();
    return 0;

    // 

    for (int is=0; is < param.ns; is++){
        mover_PC(&part[is]    ,&field,     &grd,     &param);
        mover_PC_device(&part_gpu[is],field_gpu, grd_gpu, param_gpu);
    }

    cudaDeviceSynchronize();

    std::cout << "partX: " << field_gpu->Bxn_flat[0] << std::endl;

    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

    return 0;

    // NEW CODE ************************
    //**********************************
    //**********************************
    // Compare GPU and CPU versions
    std::cout << "Compare CPU and GPU" << std::endl;
    for(int is = 0; is < param.ns; is++){
        std::cout << "Comparing species: " << is << std::endl; 
        compare(&part[is], &part_gpu[is]);
    }
    return 0;
    // END NEW CODE ********************
    
    
    // **********************************************************//
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {
        
        std::cout << std::endl;
        std::cout << "***********************" << std::endl;
        std::cout << "   cycle = " << cycle << std::endl;
        std::cout << "***********************" << std::endl;
    
        // set to zero the densities - needed for interpolation
        setZeroDensities(&idn,ids,&grd,param.ns);
        
        
        
        // implicit mover
        iMover = cpuSecond(); // start timer for mover
        for (int is=0; is < param.ns; is++)
            mover_PC(&part[is],&field,&grd,&param);
        eMover += (cpuSecond() - iMover); // stop timer for mover
        
        
        
        
        // interpolation particle to grid
        iInterp = cpuSecond(); // start timer for the interpolation step
        // interpolate species
        for (int is=0; is < param.ns; is++)
            interpP2G(&part[is],&ids[is],&grd);
        // apply BC to interpolated densities
        for (int is=0; is < param.ns; is++)
            applyBCids(&ids[is],&grd,&param);
        // sum over species
        sumOverSpecies(&idn,ids,&grd,param.ns);
        // interpolate charge density from center to node
        applyBCscalarDensN(idn.rhon,&grd,&param);
        
        
        
        // write E, B, rho to disk
        if (cycle%param.FieldOutputCycle==0){
            VTK_Write_Vectors(cycle, &grd,&field);
            VTK_Write_Scalars(cycle, &grd,ids,&idn);
        }
        
        eInterp += (cpuSecond() - iInterp); // stop timer for interpolation
        
        
    
    }  // end of one PIC cycle
    
    /// Release the resources
    // deallocate field
    grid_deallocate(&grd);
    field_deallocate(&grd,&field);
    // interp
    interp_dens_net_deallocate(&grd,&idn);
    
    // Deallocate interpolated densities and particles
    for (int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd,&ids[is]);
        particle_deallocate(&part[is]);
    }
    
    
    // stop timer
    double iElaps = cpuSecond() - iStart;
    
    // Print timing of simulation
    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    std::cout << "   Mover Time / Cycle   (s) = " << eMover/param.ncycles << std::endl;
    std::cout << "   Interp. Time / Cycle (s) = " << eInterp/param.ncycles  << std::endl;
    std::cout << "**************************************" << std::endl;
    
    // exit
    return 0;
}



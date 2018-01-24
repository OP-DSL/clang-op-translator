//
// Skeleton for direct kernels using OpenMP
//

// user function
void skeleton(double *a) {}

// CUDA kernel function
__global__ void op_cuda_skeleton(double *arg0, int set_size) {

  double arg4_l[1];

  // process set elements
  for (int n = threadIdx.x + blockIdx.x * blockDim.x; n < set_size;
       n += blockDim.x * gridDim.x) {

    // user-supplied kernel call
    skeleton(arg0);
  }

  //global reductions
  for ( int d=0; d<1; d++ ){
    op_reduction<OP_INC>(&arg4[d+blockIdx.x*1],arg4_l[d]);
  }
}

// host stub function
void op_par_loop_skeleton(char const *name, op_set set, op_arg arg0) {

  int nargs = 1;
  op_arg args[1];

  args[0] = arg0;

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(0);
  op_timers_core(&cpu_t1, &wall_t1);


  if (OP_diags > 2) {
    printf("");
  }

  op_mpi_halo_exchanges(set, nargs, args);
  if (set->size > 0) {

    // set CUDA execution parameters
    int nthread = OP_block_size;

    int nblocks = 200;

    op_cuda_skeleton<<<nblocks, nthread>>>((double *)arg0.data_d);
    // combine reduction data
    op_mpi_reduce(&arg0, (double *)arg0.data);
  }

  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());

  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[0].name = name;
  OP_kernels[0].count += 1;
  OP_kernels[0].time += wall_t2 - wall_t1;
  OP_kernels[0].transfer += 0;
}

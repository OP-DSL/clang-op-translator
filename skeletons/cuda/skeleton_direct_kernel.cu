//
// Skeleton for direct kernels using CUDA
//

int direct_skeleton_stride_OP2HOST = -1;

// user function
__device__ void skeleton(double *a) {}

// CUDA kernel function
__global__ void op_cuda_skeleton(double *arg0, int set_size) {
  double arg0_l[1];

  for (int d = 0; d < 1; ++d) {
    arg0_l[d] = ZERO_double;
  }

  for (int n = threadIdx.x + blockIdx.x * blockDim.x; n < set_size;
       n += blockDim.x * gridDim.x) {
    // user-supplied kernel call
    skeleton(arg0);
  }

  for (int d = 0; d < 1; d++) {
    op_reduction<OP_INC>(&arg0[d + blockIdx.x * 1], arg0_l[d]);
  }
}

// host stub function
void op_par_loop_skeleton(char const *name, op_set set, op_arg arg0) {

  int nargs = 1;
  op_arg args[1];

  args[0] = arg0;

  double *arg0h = (double *)arg0.data;

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(0);
  op_timers_core(&cpu_t1, &wall_t1);

  if (OP_diags > 2) {
    printf("");
  }

  op_mpi_halo_exchanges_cuda(set, nargs, args);
  if (set->size > 0) {

    int const_bytes = 0;
    op_setup_constants(const_bytes, args, nargs);
    setConstantArrToArg<double>(args[0], arg0h);
    mvConstArraysToDevice(const_bytes);

    if (OP_kernels[0].count == 0) {
      direct_skeleton_stride_OP2HOST = getSetSizeFromOpArg(&arg0);
    }

    // set CUDA execution parameters
    int nthread = OP_block_size;

    int nblocks = 200;

    int maxblocks = nblocks;
    reduct_supp_data_t reduct;
    op_setup_reductions(reduct, args, nargs, maxblocks);
    setRedArrToArg<double, OP_INC>(args[0], maxblocks, arg0h);
    mvReductArraysToDevice(reduct.reduct_bytes);
    int nshared = reduct.reduct_size * nthread;
    op_cuda_skeleton<<<nblocks, nthread, nshared>>>((double *)arg0.data_d,
                                                    set->size);

    mvReductArraysToHost(reduct.reduct_bytes);
    updateRedArrToArg<double, OP_INC>(args[0], maxblocks, arg0h);
  }

  op_mpi_set_dirtybit_cuda(nargs, args);

  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[0].name = name;
  OP_kernels[0].count += 1;
  OP_kernels[0].time += wall_t2 - wall_t1;
  OP_kernels[0].transfer += 0;
}

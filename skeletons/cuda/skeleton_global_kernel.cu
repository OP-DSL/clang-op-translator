//
// Skeleton for direct kernels using OpenMP
//

int direct_skeleton_stride_OP2HOST = -1;

// user function
__device__ void skeleton(double *a) {}

// CUDA kernel function
__global__ void op_cuda_skeleton(double *arg0, int start, int end,
                                 int *col_reord, int set_size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid + start >= end)
    return;

  double arg0_l[1];

  for (int d = 0; d < 1; ++d) {
    arg0_l[d] = ZERO_double;
  }
  int n = col_reord[tid + start];

  int map1idx;
  map1idx = 0;

  // user-supplied kernel call
  skeleton(arg0);

  for (int d = 0; d < 1; d++) {
    op_reduction<OP_INC>(&arg0[d + blockIdx.x * 1], arg0_l[d]);
  }
}

// host stub function
void op_par_loop_skeleton(char const *name, op_set set, op_arg arg0) {

  double *arg0h = (double *)arg0.data;
  int nargs = 1;
  op_arg args[1];

  args[0] = arg0;

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(0);
  op_timers_core(&cpu_t1, &wall_t1);

  int ninds = 1;
  int inds[1] = {0};

  if (OP_diags > 2) {
    printf("");
  }

  // get plan
  int part_size = OP_part_size;

  int set_size = op_mpi_halo_exchanges_cuda(set, nargs, args);
  if (set->size > 0) {

    op_plan *Plan = op_plan_get_stage(name, set, part_size, nargs, args, ninds,
                                      inds, OP_COLOR2);

    int const_bytes = 0;
    op_setup_constants(const_bytes, args, nargs);
    setConstantArrToArg<double>(args[0], arg0h);
    mvConstArraysToDevice(const_bytes);

    if (OP_kernels[0].count == 0) {
      direct_skeleton_stride_OP2HOST = getSetSizeFromOpArg(&arg0);
    }

    int nthread = OP_block_size;
    int maxblocks = 0;
    for (int col = 0; col < Plan->ncolors; col++) {
      int start = Plan->col_offsets[0][col];
      int end = Plan->col_offsets[0][col + 1];
      int nblocks = (end - start - 1) / nthread + 1;
      maxblocks = MAX(maxblocks, nblocks);
    }
    reduct_supp_data_t reduct;
    op_setup_reductions(reduct, args, nargs, maxblocks);
    setRedArrToArg<double, OP_INC>(args[0], maxblocks, arg0h);
    mvReductArraysToDevice(reduct.reduct_bytes);

    // execute plan
    for (int col = 0; col < Plan->ncolors; col++) {
      if (col == Plan->ncolors_core) {
        op_mpi_wait_all_cuda(nargs, args);
      }

      int start = Plan->col_offsets[0][col];
      int end = Plan->col_offsets[0][col + 1];
      int nblocks = (end - start - 1) / nthread + 1;
      int nshared = reduct.reduct_size * nthread;

      op_cuda_skeleton<<<nblocks, nthread, nshared>>>(
          (double *)arg0.data_d, start, end, Plan->col_reord, set->size);
    }
    mvReductArraysToHost(reduct.reduct_bytes);
    updateRedArrToArg<double, OP_INC>(args[0], maxblocks, arg0h);
    // update kernel record
    OP_kernels[0].transfer += Plan->transfer;
    OP_kernels[0].transfer2 += Plan->transfer2;
  }
  op_mpi_set_dirtybit_cuda(nargs, args);

  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[0].name = name;
  OP_kernels[0].count += 1;
  OP_kernels[0].time += wall_t2 - wall_t1;
}

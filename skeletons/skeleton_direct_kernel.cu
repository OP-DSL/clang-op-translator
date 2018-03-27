//
// Skeleton for direct kernels using CUDA
//

// user function
void skeleton(double *a) {}

// CUDA kernel function
void op_cuda_skeleton(double *arg0, int set_size) {
  int n = 0;
  if (n < set_size) {
    // user-supplied kernel call
    skeleton(arg0);
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

    // set CUDA execution parameters
    int nthread = OP_block_size;

    int nblocks = (set->size - 1) / nthread + 1;

    int reduct_bytes = 0;

    op_cuda_skeleton((double *)arg0.data_d, set->size);

    mvReductArraysToHost(reduct_bytes);
  }

  op_mpi_set_dirtybit_cuda(nargs, args);

  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[0].name = name;
  OP_kernels[0].count += 1;
  OP_kernels[0].time += wall_t2 - wall_t1;
  OP_kernels[0].transfer += 0;
}

//
// Skeleton for direct kernels with vectorization
//

// user function
void skeleton(double *a) {}
void skeleton_vec(double *a) {}

// host stub function
void op_par_loop_skeleton(char const *name, op_set set, op_arg arg0) {

  int nargs = 1;
  op_arg args[1];

  args[0] = arg0;

  // create aligned pointers for dats
  double *__restrict__ ptr0 = (double *)arg0.data;

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(0);
  op_timers_core(&cpu_t1, &wall_t1);

  if (OP_diags > 2) {
    printf("");
  }

  int exec_size = op_mpi_halo_exchanges(set, nargs, args);

  if (exec_size > 0) {

#ifdef VECTORIZE
#pragma novector
    for (int n = 0; n < (exec_size / SIMD_VEC) * SIMD_VEC; n += SIMD_VEC) {
      double dat[SIMD_VEC] = {0.0};
#pragma simd
      for (int i = 0; i < SIMD_VEC; i++) {
        skeleton_vec(&((double *)arg0.data)[4 * n]);
      }
      for (int i = 0; i < SIMD_VEC; i++) {
        dat[i] = 0;
      }
    }
    // remainder
    for (int n = (exec_size / SIMD_VEC) * SIMD_VEC; n < exec_size; n++) {
#else
    for (int n = 0; n < exec_size; n++) {
#endif
      skeleton(&((double *)arg0.data)[4 * n]);
    }
  }

  // combine reduction data
  op_mpi_reduce(&arg0, (double *)arg0.data);
  op_mpi_set_dirtybit(nargs, args);

  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[0].name = name;
  OP_kernels[0].count += 1;
  OP_kernels[0].time += wall_t2 - wall_t1;
  OP_kernels[0].transfer += 0;
}

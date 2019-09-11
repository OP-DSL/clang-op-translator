//
// Skeleton for direct kernels using OpenMP
//

// user function
void skeleton(double *a) {}

// host stub function
void op_par_loop_skeleton(char const *name, op_set set, op_arg arg0) {

  int nargs = 1;
  op_arg args[1];

  args[0] = arg0;

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(0);
  op_timers_core(&cpu_t1, &wall_t1);

  // local variables for reduction
  double arg0_l = *(double *)arg0.data;

  if (OP_diags > 2) {
    printf("");
  }

  op_mpi_halo_exchanges(set, nargs, args);

  if (set->size > 0) {
    for (int n = 0; n < set->size; n++) {
      int map0idx = arg0.map_data[n * arg0.map->dim + 0];

      skeleton(&((double *)arg0.data)[4 * n]);
    }
  }
  *((double *)arg0.data) = arg0_l;

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

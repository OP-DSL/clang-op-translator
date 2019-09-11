//
// Skeleton for direct kernels using OpenMP
//

// user function
void skeleton_OMP4(double *a) {}

// host stub function
void op_par_loop_skeleton(char const *name, op_set set, op_arg arg0) {

  int nargs = 1;
  op_arg args[1];

  args[0] = arg0;

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(0);
  op_timers_core(&cpu_t1, &wall_t1);


  double arg0_l = *(double *)arg0.data;

  if (OP_diags > 2) {
    printf("");
  }

  op_mpi_halo_exchanges_cuda(set, nargs, args);

  #ifdef OP_PART_SIZE_1
    int part_size = OP_PART_SIZE_1;
  #else
    int part_size = OP_part_size;
  #endif
  #ifdef OP_BLOCK_SIZE_1
    int nthread = OP_BLOCK_SIZE_1;
  #else
    int nthread = OP_block_size;
  #endif

  if (set->size > 0) {

    //Set up typed device pointers for OpenMP

    int *mapStart;
    skeleton_OMP4(&((double *)arg0.data)[4 * 0]);
  }
  *((double *)arg0.data) = arg0_l;

  // combine reduction data
  op_mpi_reduce(&arg0, (double *)arg0.data);
  op_mpi_set_dirtybit_cuda(nargs, args);


  if (OP_diags>1) deviceSync();
  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[0].name = name;
  OP_kernels[0].count += 1;
  OP_kernels[0].time += wall_t2 - wall_t1;
  OP_kernels[0].transfer += 0;
}

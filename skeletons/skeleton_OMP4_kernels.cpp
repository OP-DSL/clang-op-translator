//
// Skeleton for indirect kernels using OpenMP
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

  int ninds = 1;
  int inds[1] = {0};

  // local variables for reduction
  double arg0_l = *(double *)arg0.data;

  if (OP_diags > 2) {
    printf("");
  }

  // get plan
  int part_size = OP_part_size;

  int set_size = op_mpi_halo_exchanges_cuda(set, nargs, args);


  #ifdef OP_BLOCK_SIZE_1
    int nthread = OP_BLOCK_SIZE_1;
  #else
    int nthread = OP_block_size;
  #endif

  int ncolors = 0;
  int set_size1 = set->size + set->exec_size;

  if (set->size > 0) {
    //Set up typed device pointers for OpenMP
    int *map_ = arg0.map_data_d;

    op_plan *Plan = op_plan_get_stage(name,set,part_size,nargs,args,ninds,inds,OP_COLOR2);
    ncolors = Plan->ncolors;
    int *col_reord = Plan->col_reord;

    // execute plan
    for ( int col=0; col<Plan->ncolors; col++ ){
      if (col==1) {
        op_mpi_wait_all_cuda(nargs, args);
      }
      int start = Plan->col_offsets[0][col];
      int end = Plan->col_offsets[0][col+1];

      skeleton_OMP4(&((double *)arg0.data)[0]);
    }
    OP_kernels[0].transfer  += Plan->transfer;
    OP_kernels[0].transfer2 += Plan->transfer2;
  }
  *((double *)arg0.data) = arg0_l;

  if (set_size == 0 || set_size == set->core_size || ncolors == 1) {
    op_mpi_wait_all_cuda(nargs, args);
  }
  
  // combine reduction data
  op_mpi_reduce(&arg0, (double *)arg0.data);
  op_mpi_set_dirtybit_cuda(nargs, args);


  if (OP_diags>1) deviceSync();
  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[0].name = name;
  OP_kernels[0].count += 1;
  OP_kernels[0].time += wall_t2 - wall_t1;
}

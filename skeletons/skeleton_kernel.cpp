//
// Skeleton for indirect kernels using OpenMP
//

// user function
void skeleton(double* a){}

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

  int set_size = op_mpi_halo_exchanges(set, nargs, args);

  if (set->size > 0) {

    op_plan *Plan = op_plan_get(name, set, part_size, nargs, args, ninds, inds);

    // execute plan
    int block_offset = 0;
    for (int col = 0; col < Plan->ncolors; col++) {
      if (col == Plan->ncolors_core) {
        op_mpi_wait_all(nargs, args);
      }
      int nblocks = Plan->ncolblk[col];

#pragma omp parallel for reduction(+:arg0_l)
      for (int blockIdx = 0; blockIdx < nblocks; blockIdx++) {
        int blockId = Plan->blkmap[blockIdx + block_offset];
        int nelem = Plan->nelems[blockId];
        int offset_b = Plan->offset[blockId];
        for (int n = offset_b; n < offset_b + nelem; n++) {
          int map0idx = arg0.map_data[n * arg0.map->dim + 0];

          skeleton(&((double *)arg0.data)[4 * n]);
        }
      }

      block_offset += nblocks;
    }
    OP_kernels[0].transfer += Plan->transfer;
    OP_kernels[0].transfer2 += Plan->transfer2;
  }
  *((double *)arg0.data) = arg0_l;

  if (set_size == 0 || set_size == set->core_size) {
    op_mpi_wait_all(nargs, args);
  }
  // combine reduction data
  op_mpi_reduce(&arg0, (double *)arg0.data);
  op_mpi_set_dirtybit(nargs, args);

  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[0].name = name;
  OP_kernels[0].count += 1;
  OP_kernels[0].time += wall_t2 - wall_t1;
}

//
// Skeleton for direct kernels using OpenMP
//

// user function
__device__ void skeleton(double *a) {}

// CUDA kernel function
__global__ void op_cuda_skeleton(double *arg0, int block_offset, int *blkmap,
                                 int *offset, int *nelems, int *ncolors,
                                 int *colors, int nblocks, int set_size) {
  double arg0_l[1];

  for (int d = 0; d < 1; ++d) {
    arg0_l[d] = ZERO_double;
  }

  __shared__ int nelem, offset_b;
  extern __shared__ char shared[];
  if (blockIdx.x + blockIdx.y * gridDim.x >= nblocks) {
    return;
  }

  if (threadIdx.x == 0) {
    // get sizes and shift pointers and direct-mapped data
    int blockId = blkmap[blockIdx.x + blockIdx.y * gridDim.x + block_offset];
    nelem = nelems[blockId];
    offset_b = offset[blockId];
  }
  __syncthreads(); // make sure all of above completed

  for (int n = threadIdx.x; n < nelem; n += blockDim.x) {
    int map1idx;
    // user-supplied kernel call
    skeleton(arg0);
  }

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

    op_plan *Plan = op_plan_get(name, set, part_size, nargs, args, ninds, inds);

    int const_bytes = 0;
    op_setup_constants(const_bytes, args, nargs);
    setConstantArrToArg<double>(args[0], arg0h);
    mvConstArraysToDevice(const_bytes);

    // execute plan
    int block_offset = 0;
    for (int col = 0; col < Plan->ncolors; col++) {
      if (col == Plan->ncolors_core) {
        op_mpi_wait_all_cuda(nargs, args);
      }

      int nthread = OP_block_size;

      dim3 nblocks = dim3(
          Plan->ncolblk[col] >= (1 << 16) ? 65535 : Plan->ncolblk[col],
          Plan->ncolblk[col] >= (1 << 16) ? (Plan->ncolblk[col] - 1) / 65535 + 1
                                          : 1,
          1);
      if (Plan->ncolblk[col] > 0) {
        op_cuda_skeleton<<<nblocks, nthread>>>(
            (double *)arg0.data_d, block_offset, Plan->blkmap, Plan->offset,
            Plan->nelems, Plan->nthrcol, Plan->thrcol, Plan->ncolblk[col],
            set->size);
      }
      block_offset += Plan->ncolblk[col];
    }
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

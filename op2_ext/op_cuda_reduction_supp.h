#ifndef OP_CUDA_REDUCTION_SUPP_H_INCLUDED
#define OP_CUDA_REDUCTION_SUPP_H_INCLUDED

struct reduct_supp_data_t {
  int reduct_bytes = 0;
  int reduct_size = 0;
};

void op_setup_reductions(reduct_supp_data_t &reduct, op_arg *args, int nargs,
                         int maxblocks) {
  for (int i = 0; i < nargs; ++i) {
    if (args[i].argtype == OP_GBL && args[i].acc != OP_READ &&
        args[i].acc != OP_WRITE) {
      args[i].data = OP_reduct_h + reduct.reduct_bytes;
      args[i].data_d = OP_reduct_d + reduct.reduct_bytes;
      reduct.reduct_bytes += ROUND_UP(maxblocks * args[i].dim * args[i].size);
      reduct.reduct_size = MAX(reduct.reduct_size, args[i].size);
    }
  }
  reallocReductArrays(reduct.reduct_bytes);
}
template <typename T, op_arg_type argtype>
void setRedArrToArg(op_arg &arg, int maxblocks, T *argh) {
  for (int b = 0; b < maxblocks; ++b) {
    for (int d = 0; d < arg.dim; ++d) {
      ((T *)arg.data)[d + b * arg.dim] = argh[d];
    }
  }
}

template <typename T, op_arg_type argtype>
void updateRedArrToArg(op_arg &arg, int maxblocks, T *argh) {
  for (int b = 0; b < maxblocks; ++b) {
    for (int d = 0; d < arg.dim; ++d) {
      argh[d] += ((T *)arg.data)[d + b * arg.dim];
    }
  }
  arg.data = (char *)argh;
  op_mpi_reduce(&arg,argh);
}

void op_setup_constants(int &const_bytes, op_arg *args, int nargs) {
  for (int i = 0; i < nargs; ++i) {
    if (args[i].argtype == OP_GBL && args[i].acc == OP_READ) {
      args[i].data = OP_consts_h + const_bytes;
      args[i].data_d = OP_consts_d + const_bytes;
      const_bytes += ROUND_UP(args[i].dim * args[i].size);
    }
  }
  reallocConstArrays(const_bytes);
}

template <typename T>
void setConstantArrToArg(op_arg &arg, T*argh) {
  for (int d = 0; d < arg.dim; ++d) {
    ((T *)arg.data)[d] = argh[d];
  }
}

#endif /* ifndef OP_CUDA_REDUCTION_SUPP_H_INCLUDED */

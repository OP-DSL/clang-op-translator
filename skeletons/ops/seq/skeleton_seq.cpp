#include "ops_lib_cpp.h"

void kernel() {}

void par_loop_skeleton(char const *name, ops_block block,
                                          int dim, int *range, ops_arg arg0) {
  const unsigned num_args = 1;

  // Timing
  double __t1, __t2, __c1, __c2;
  ops_arg args[1] = {arg0};


  kernel();
}

#include "ops_globals.h"

// user specified operation
void kernel( double *) {}

// host stub function
void op_par_loop_skeleton(char const *name, op_set set, op_arg arg0) {


  kernel((double*)arg0.data);

}

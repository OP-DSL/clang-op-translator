# Op2 Clang Translator

## Build Instructions
You will need a checkout of the llvm, clang and clang-tools-extra source code first (see http://clang.llvm.org/get_started.html for instructions)
Check out this repository and set OP2_CLANG_INSTALL_PATH enviroment variable (e.g. with `git clone https://github.com/bgd54/OP2-Clang.git;
 export OP2_CLANG_INSTALL_PATH=/path/to/current/dir/OP2-Clang`) then in the OP2-Clang directory:
> cmake .

## Usage Instructions
Run as follows:
> op2-clang /path/to/op2/apps/c/airfoil/airfoil_plain/dp/airfoil.cpp -- clang++ -I/path/to/op2/include/

If clang doesn't find omp.h:
> op2-clang /path/to/op2/apps/c/airfoil/airfoil_plain/dp/airfoil.cpp -- clang++ -I/path/to/op2/include/ -I/path/to/gcc/lib/include/ -fopenmp


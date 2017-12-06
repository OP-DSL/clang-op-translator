# Op2 Clang Translator

## Build Instructions
You will need a checkout of the llvm, clang and clang-tools-extra source code first (see http://clang.llvm.org/get_started.html for instructions)
Check out this repository (e.g. with `git clone https://github.com/bgd54/OP2-Clang.git) then in the OP2-Clang directory:

> mkdir build  
> cd build  
> cmake ..  
> make  

## Usage Instructions
Run as follows:
> ./op2-clang /path/to/op2/apps/c/airfoil/airfoil_plain/dp/airfoil.cpp -- clang++

If clang doesn't find omp.h:
> ./op2-clang /path/to/op2/apps/c/airfoil/airfoil_plain/dp/airfoil.cpp -- clang++ -I/path/to/gcc/lib/include/

If you don't want to generate all versions you can specify wich version you want to generate with -optarget flag as follows:

> ./op2-clang -optarget=seq /path/to/op2/apps/c/airfoil/airfoil_plain/dp/airfoil.cpp -- clang++

Check available options with:

> ./op2-clang -help

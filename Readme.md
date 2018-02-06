# OP2 Clang Translator
OP2 is an API with associated libraries and preprocessors to generate
parallel executables for applications on unstructured grids.

This repository contains the implementation of a clang based preprocessor,
which performs source-to-source translation of OP2 applications.

## Build Instructions
You will need a checkout of the llvm, clang and clang-tools-extra source code first (see http://clang.llvm.org/get_started.html for instructions).
Check out this repository and set OP2_INSTALL_PATH (e.g. with `git clone https://github.com/bgd54/OP2-Clang.git; export OP2_INSTALL_PATH="/path/to/op2/"`) then in the OP2-Clang directory:

> mkdir build  
> cd build  
> cmake ..  
> make  

## Usage Instructions
Run as follows:
> ./op2-clang /path/to/op2/apps/c/airfoil/airfoil_plain/dp/airfoil.cpp -- clang++

This will generate `airfoil_op.cpp` with the master kernel files `airfoil_kernels.cpp`, `airfoil_seqkernels.cpp` and `airfoil_veckernels.cpp` these files including the generated user kernel files (`<loop_name>_xxxkernel.cpp`).

If clang doesn't find omp.h:
> ./op2-clang /path/to/op2/apps/c/airfoil/airfoil_plain/dp/airfoil.cpp -- clang++ -I/path/to/gcc/lib/include/

If you don't want to generate all versions you can specify wich version you want to generate with -optarget flag as follows:

> ./op2-clang -optarget=seq /path/to/op2/apps/c/airfoil/airfoil_plain/dp/airfoil.cpp -- clang++

Check available options with:

> ./op2-clang -help

## Directory Structure
This repository is structured as follows:  
OP2-Clang  
|  
\`- doc: documentation  
|  
\`- skeletons: kernel and masterkernel skeletons used by the tool  
|  
\`- src: source of the tool  
|  
\`- CMakeLists.txt: cmake file for build  
|  
\`- Readme.md: this file  

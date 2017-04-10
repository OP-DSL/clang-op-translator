# Op2 Clang Translator

## Build Instructions
You will need a checkout of the llvm, clang and clang-tools-extra source code first (see http://clang.llvm.org/get_started.html for instructions)
Check out this repository into `[llvm_src_dir]/tools/clang/tools/extra/op2` (e.g. with git clone `https://github.com/DavidTruby/OP2-Clang op2` run from that directory) and then in that directory run the following command:
> echo 'add_subdirectory(op2)' >> CMakeLists.txt

## Usage Instructions
Run as follows:
> bin/op2 /path/to/op2/apps/c/airfoil/airfoil_plain/dp/airfoil.cpp -- clang++ -I/path/to/op2/include/ 

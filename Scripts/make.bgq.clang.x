# RZ test case
export CXX=/usr/apps/gnu/clang/2017.06.06/bin/mpiclang++11-fastmpi;
export CC=/usr/apps/gnu/clang/2017.06.06/bin/mpiclang-fastmpi;
export CXXFLAGS="-std=c++11 -g -O3 -Wno-write-strings -ffp-contract=off -mcpu=a2 -mtune=a2";
export CFLAGS="  -g -O3 -ffp-contract=off -mcpu=a2 -mtune=a2";

gmake clean; \
gmake \
PREFIX=`pwd`/test_install/bgq/clang \
install;

export -n CXX;
export -n CC;
export -n CXXFLAGS;
export -n CFLAGS;

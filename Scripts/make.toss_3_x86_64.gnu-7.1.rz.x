export CXX=/usr/tce/packages/gcc/gcc-7.1.0/bin/g++;
export CC=/usr/tce/packages/gcc/gcc-7.1.0/bin/gcc;
export CXXFLAGS="-g -O2 -std=c++11 -ffp-contract=off";
export CFLAGS="  -g -O2 -std=gnu11 -ffp-contract=off";
gmake clean; \
gmake \
PREFIX=`pwd`/test_install/toss_3_x86_64/gnu-7.1 \
NUCLEAR_PATH=/usr/gapps/bdiv/toss_3_x86_64/gnu-7.1-mvapich2-2.2/nuclear/r184/lib \
install;
export -n CXX;
export -n CC;
export -n CXXFLAGS;
export -n CFLAGS;

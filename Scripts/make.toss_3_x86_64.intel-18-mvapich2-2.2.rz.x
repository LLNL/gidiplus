export CXX=/usr/tce/packages/intel/intel-18.0.1/bin/icpc;
export CC=/usr/tce/packages/intel/intel-18.0.1/bin/icc;
export CXXFLAGS="-gxx-name=/usr/tce/packages/gcc/gcc-7.1.0/bin/g++ -std=c++11 -g -O2 -fp-model precise -fp-model source -axCORE-AVX2 -xAVX -no-fma -ip -no-ftz -prec-div -prec-sqrt -diag-disable cpu-dispatch";
export CFLAGS="  -gcc-name=/usr/tce/packages/gcc/gcc-7.1.0/bin/gcc -std=gnu11 -g -O2 -fp-model precise -fp-model source -axCORE-AVX2 -xAVX -no-fma -ip -no-ftz -prec-div -prec-sqrt -diag-disable cpu-dispatch";
gmake clean; \
gmake \
NUCLEAR_PATH=/usr/gapps/bdiv/toss_3_x86_64_ib/intel-18-mvapich2-2.2/nuclear/r184/lib \
PREFIX=`pwd`/test_install/toss_3_x86_64/intel-18-mvapich2-2.2 \
install; 
export -n CXX;
export -n CC;
export -n CXXFLAGS;
export -n CCFLAGS;

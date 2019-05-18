export CXX=/usr/tce/packages/intel/intel-17.0.0/bin/icpc;
export CC=/usr/tce/packages/intel/intel-17.0.0/bin/icc;
export CXXFLAGS="-g -O2 -std=c++11 -no-ftz -fp-model precise -fp-model source -finline-functions -nolib-inline -prec-div -prec-sqrt -gxx-name=/usr/tce/packages/gcc/gcc-6.1.0/bin/g++"; \
export CFLAGS="  -g -O2 -std=gnu11 -no-ftz -fp-model precise -fp-model source -finline-functions -nolib-inline -prec-div -prec-sqrt -gcc-name=/usr/tce/packages/gcc/gcc-6.1.0/bin/gcc -std=gnu11"; \
gmake clean; \
gmake \
PREFIX=`pwd`/test_install/toss_3_x86_64/intel-17-mvapich2-2.2 \
NUCLEAR_PATH=/usr/gapps/bdiv/toss_3_x86_64_ib/intel-17-mvapich2-2.2/nuclear/r184/lib \
install; 
export -n CXX;
export -n CC;
export -n CXXFLAGS;
export -n CCFLAGS;

export CXX=/usr/tce/packages/mvapich2/mvapich2-2.3.6-intel-2022.1.0-magic/bin/mpicxx
export CC=/usr/tce/packages/mvapich2/mvapich2-2.3.6-intel-2022.1.0-magic/bin/mpicc
export CXXFLAGS="-g -O2 -std=c++14 -ffp-contract=off -fno-fast-math -ffp-model=precise "
export CFLAGS="  -g -O2 -std=gnu11 -ffp-contract=off -fno-fast-math -ffp-model=precise -D__STDC_LIMIT_MACROS"
#export CXXFLAGS="-gxx-name=/usr/tce/packages/gcc/gcc-7.1.0/bin/g++ -std=c++11 -g -O2 -fp-model precise -fp-model source -axCORE-AVX2 -xAVX -no-fma -ip -no-ftz -prec-div -prec-sqrt -diag-disable cpu-dispatch";
#export CFLAGS="  -gcc-name=/usr/tce/packages/gcc/gcc-7.1.0/bin/gcc -std=gnu11 -g -O2 -fp-model precise -fp-model source -axCORE-AVX2 -xAVX -no-fma -ip -no-ftz -prec-div -prec-sqrt -diag-disable cpu-dispatch";
gmake clean -s; \
gmake -s -j16 \
PREFIX=`pwd`/test_install/toss_4_x86_64/intel-2022-mvapich2-2.3 \
install;
export -n CXX;
export -n CC;
export -n CXXFLAGS;
export -n CCFLAGS;

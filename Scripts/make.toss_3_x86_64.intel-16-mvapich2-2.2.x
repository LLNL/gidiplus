export -n CXX;
export -n CC;
export -n CXXFLAGS;
export -n CFLAGS;

gmake clean; \
gmake \
CXX=/usr/tce/packages/mvapich2/mvapich2-2.2-intel-16.0.3/bin/mpic++ \
CC=/usr/tce/packages/mvapich2/mvapich2-2.2-intel-16.0.3/bin/mpicc \
CXXFLAGS="-g -O2 -D__STDC_LIMIT_MACROS -no-ftz  -fp-model precise -fp-model source -finline-functions -nolib-inline -prec-div -prec-sqrt -diag-disable cpu-dispatch -gxx-name=/usr/tce/packages/gcc/gcc-4.9.3/bin/g++ -std=c++11" \
CFLAGS="   -g -O2 -D__STDC_LIMIT_MACROS -no-ftz  -fp-model precise -fp-model source -finline-functions -nolib-inline -prec-div -prec-sqrt -diag-disable cpu-dispatch -gcc-name=/usr/tce/packages/gcc/gcc-4.9.3/bin/gcc -std=gnu11" \
PREFIX=`pwd`/test_install/toss_3_x86_64/intel-16-mvapich2-2.2 \
NUCLEAR_PATH=/usr/gapps/nuclear/toss_3_x86_64_ib/versions/nuclear.svn183/Src/lib \
install;

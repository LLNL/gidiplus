export CXX=clang++
export CC=clang
#export CXXFLAGS="-std=c++11 -g -O0 -Wstrict-aliasing -Wno-write-strings"
#export CFLAGS="  -std=gnu11 -g -O0 -Wstrict-aliasing "
export CXXFLAGS="-std=c++11 -g -O3 -mavx -ffp-contract=off -funroll-loops -finline-functions -Wstrict-aliasing -Wno-write-strings"
export CFLAGS="  -std=gnu11 -g -O2 -mavx -ffp-contract=off -funroll-loops -finline-functions -Wstrict-aliasing "
#export GPERF_PATH=/usr/gapps/bdiv/toss_3_x86_64_ib/clang-11-mvapich2-2.3/gperftools/2.4
#export GPERF_LIB=$GPERF_PATH/lib
#export GPERF_INCLUDE=$GPERF_PATH/include
gmake -s clean; \
gmake -s -j16 \
PREFIX=`pwd`/test_install/toss_4_x86_64/clang \
install;
export -n CXX;
export -n CC;
export -n CXXFLAGS;
export -n CFLAGS;

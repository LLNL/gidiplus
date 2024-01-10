export CXX=/usr/tce/packages/gcc/gcc-7.3.1/bin/g++
export CC=/usr/tce/packages/gcc/gcc-7.3.1/bin/gcc
export CXXFLAGS="-g -O2 -std=c++11 -D__STDC_LIMIT_MACROS "
export CFLAGS="  -g -O2 -std=gnu11 -D__STDC_LIMIT_MACROS "

gmake realclean -s
gmake -s -j16 \
PREFIX=`pwd`/test_install/blueos_3_ppc64le_ib_p9/gnu \
install;
export -n CXX;
export -n CC;
export -n CXXFLAGS;
export -n CFLAGS;


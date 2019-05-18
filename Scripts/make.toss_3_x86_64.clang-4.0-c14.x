export CXX=/usr/tce/packages/clang/clang-4.0.0/bin/clang++
export CC=/usr/tce/packages/clang/clang-4.0.0/bin/clang
export CXXFLAGS="-g -O2 -D__STDC_LIMIT_MACROS -std=c++14 -x c++ -stdlib=libc++ -fstandalone-debug";
export CFLAGS="-g -O2"
gmake clean; \
gmake \
PREFIX=`pwd`/test_install/toss_3_x86_64/clang-4.0.c14 \
NUCLEAR_PATH=/usr/gapps/nuclear/toss_3_x86_64_ib/versions/nuclear.svn183/gnu/lib \
install;
export -n CXX;
export -n CC;
export -n CXXFLAGS;
export -n CFLAGS;

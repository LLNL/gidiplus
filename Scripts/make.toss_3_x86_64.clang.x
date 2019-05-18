export CXX=/usr/tce/packages/clang/clang-6.0.0/bin/clang++
export CC=/usr/tce/packages/clang/clang-6.0.0/bin/clang
export CXXFLAGS="-std=c++11 -g -O3 -mavx -ffp-contract=off -funroll-loops -finline-functions -Wstrict-aliasing -Wno-write-strings"
export CFLAGS="  -std=gnu11 -g -O2 -mavx -ffp-contract=off -funroll-loops -finline-functions -Wstrict-aliasing "
gmake -s clean; \
gmake -s -j \
PREFIX=`pwd`/test_install/toss_3_x86_64/clang \
NUCLEAR_PATH=/usr/gapps/nuclear/toss_3_x86_64_ib/versions/nuclear.svn183/gnu/lib \
install;
export -n CXX;
export -n CC;
export -n CXXFLAGS;
export -n CFLAGS;

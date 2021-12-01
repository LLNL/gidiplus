export CXX=c++
export CC=cc
export CXXFLAGS="-std=c++11 -g -O3 -mavx -ffp-contract=off -funroll-loops -finline-functions -Wstrict-aliasing -Wno-write-strings -Wsign-compare -Wall"
export CFLAGS="  -std=gnu11 -g -O2 -mavx -ffp-contract=off -funroll-loops -finline-functions -Wstrict-aliasing -Wsign-compare -Wall"
make -s clean; \
make -s -j \
PREFIX=`pwd`/test_install/toss_3_x86_64/clang \
install;
export -n CXX;
export -n CC;
export -n CXXFLAGS;
export -n CFLAGS;

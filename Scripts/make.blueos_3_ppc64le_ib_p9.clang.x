export CXX=/usr/tce/bin/mpiclang++
export CC=/usr/tce/bin/mpiclang
export CXXFLAGS="-g -O2 -std=c++11 -ffp-contract=off"
export CFLAGS="  -g -O2 -std=gnu11 -ffp-contract=off"

gmake realclean -s
gmake -s -j16 \
PREFIX=`pwd`/test_install/blueos_3_ppc64le_ib_p9/clang \
install

export -n CXX CC CXXFLAGS CLAGS


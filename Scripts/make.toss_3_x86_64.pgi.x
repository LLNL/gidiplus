export CXX=/usr/tce/packages/pgi/pgi-16.9/bin/pgc++
export CC=/usr/tce/packages/pgi/pgi-16.9/bin/pgcc
export CXXFLAGS="-g -O2 --c++11 --display_error_number --diag_suppress177 --diag_suppress111"
export CFLAGS="  -g -O2 "
gmake clean; \
gmake \
PREFIX=`pwd`/test_install/toss_3_x86_64/pgi \
NUCLEAR_PATH=/usr/gapps/nuclear/toss_3_x86_64_ib/versions/nuclear.svn183/gnu/lib \
install;
export -n CXX;
export -n CC;
export -n CXXFLAGS;
export -n CFLAGS;

# Note, this compile complains about the use of INFINITY -- floating point overflow occurs in nf library

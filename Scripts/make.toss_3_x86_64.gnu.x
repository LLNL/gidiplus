export CXX=g++
export CXXFLAGS="-g -O2"
export CC=gcc
export CFLAGS=" -g -O2"
gmake clean; \
gmake \
PREFIX=`pwd`/test_install/toss_3_x86_64/gnu \
NUCLEAR_PATH=/usr/gapps/nuclear/toss_3_x86_64_ib/versions/nuclear.svn183/gnu/lib \
install;
export -n CXX;
export -n CC;
export -n CXXFLAGS;
export -n CFLAGS;

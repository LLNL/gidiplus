export CXX=/usr/local/apps/gnu/4.9.3/bin/g++
export CXXFLAGS="-g -Wall -Wshadow  -Woverloaded-virtual -Wno-long-long -Wno-strict-aliasing -g -O2 -std=c++11"
export CC=/usr/local/apps/gnu/4.9.3/bin/gcc
export CFLAGS=" -g -Wall -Wshadow  -Wno-long-long -Wno-strict-aliasing -g -O2"
gmake clean; \
gmake \
PREFIX=`pwd`/test_install/chaos/gnu-4.9 \
NUCLEAR_PATH=/usr/gapps/nuclear/chaos_5_x86_64/versions/nuclear.svn183/gnu/lib \
install;
export -n CXX;
export -n CC;
export -n CXXFLAGS;
export -n CFLAGS;

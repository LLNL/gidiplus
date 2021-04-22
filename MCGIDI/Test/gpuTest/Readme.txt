This tests MCGIDI on the GPU. To compile this, first build GIDI with nvcc. 


Then make this test with the following command after changing the directory to MCGIDI/Test/gpuTest:

For RZAnsel Cuda10 opt:
gmake CXX=/usr/tce/packages/cuda/cuda-10.1.243/bin/nvcc CXXFLAGS='-x cu --relocatable-device-code=true -lineinfo -g -O2 -std=c++11 -gencode=arch=compute_70,code=sm_70 -I$(CUDA_PATH)/include'
For RZAnsel Cuda11 opt (doesn't work):
gmake CXX=/usr/tce/packages/cuda/cuda-11.2.0-beta/bin/nvcc CXXFLAGS='-x cu --relocatable-device-code=true -lineinfo -g -O2 -std=c++11 -gencode=arch=compute_70,code=sm_70 -I$(CUDA_PATH)/include'


To run it, grab a process like
lalloc 1

And then executa it like
gputest <doPrint> <numCollisions> <numIsotopes> <doCompare>

default is
gpuTest 0 100000 1 0

doPrint - If nonzero, print out the list of reactions on the original host, unpacked host, and gpu for the last isotope
numCollisions - Sample these number of collisions on the CPU and GPU
numIsotopes - Number of isotopes to load in. Collisions is only on last isotope. Up to 100.
doCompare - If 0, do nothing. If 1, copy the protare back from the GPU and write it to disk. If 2, copy the protare back from the GPU and compare to the one on disk.

For timing, use nvprof like

/usr/tce/packages/cuda/cuda-11.2.0-beta/bin/nvprof gpuTest 0 0 100 0

/usr/tce/packages/cuda/cuda-11.2.0-beta/bin/nvprof gpuTest 0 1e7 1 0

For cpu timing, try:
/usr/tce/packages/cuda/cuda-11.2.0-beta/bin/nvprof --cpu-profiling on gpuTest 0 1e7 1 0

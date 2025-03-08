This tests MCGIDI on the GPU. To compile this, first build GIDI with nvcc. 


Then make this test with the following command after changing the directory to MCGIDI/Test/gpuTest:

For RZAnsel Cuda10 opt:
gmake CXX=/usr/tce/packages/cuda/cuda-10.1.243/bin/nvcc CXXFLAGS='-x cu --relocatable-device-code=true -lineinfo -g -O2 -std=c++11 -gencode=arch=compute_70,code=sm_70 -I$(CUDA_PATH)/include'
For RZAnsel Cuda11 opt:
gmake CXX=/usr/tce/packages/cuda/cuda-11.7.0/bin/nvcc CXXFLAGS='-x cu --relocatable-device-code=true -lineinfo -g -O2 -std=c++11 -gencode=arch=compute_70,code=sm_70 -I$(CUDA_PATH)/include'
For RZvernal/HIP Rocm 5.7.1:
gmake CXX=/usr/tce/packages/cray-mpich/cray-mpich-8.1.27-rocmcc-5.7.1-cce-16.0.1d-magic/bin/mpihipcc CXXFLAGS='-g -ggdb -O0 -std=c++14 -finline-functions -ffp-contract=off -x hip -fgpu-rdc --offload-arch=gfx90a,gfx940,gfx942 -w -Wno-unused-command-line-argument -D __HIP__ -I/usr/tce/packages/cray-mpich/cray-mpich-8.1.27-rocmcc-5.7.1-cce-16.0.1d-magic/include' HDF5_LIBS='/usr/gapps/bdivport/toss_4_x86_64_ib_cray/rocm-5.7-mpich-8.1.27/hdf5/1.14.3/lib/libhdf5.a'
For RZvernal/HIP Rocm 6.0.3:
gmake CXX=/usr/tce/packages/cray-mpich/cray-mpich-8.1.29-rocmcc-6.0.3-magic/bin/mpiamdclang++ CXXFLAGS='-g -ggdb -O0 -std=c++14 -finline-functions -ffp-contract=off  --hip-link  -x hip -fgpu-rdc --offload-arch=gfx90a,gfx940,gfx942 -w -Wno-unused-command-line-argument -D __HIP__ -I/usr/tce/packages/cray-mpich/cray-mpich-8.1.29-rocmcc-6.0.3-magic/include' HDF5_LIBS='/usr/gapps/bdiv/toss_4_x86_64_ib_cray/rocmcc-6.0.3-cce-17/hdf5/1.14.3/lib/libhdf5.a'

To run it, grab a process like
RZansel:
lalloc 1
RZwhamo:
salloc -N1 -n64  -p mi100 --exclusive

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

/usr/tce/packages/cuda/cuda-11.2.0-beta/bin/nvprof gpuTest 0 10000000 1 0

For cpu timing, try:
/usr/tce/packages/cuda/cuda-11.2.0-beta/bin/nvprof --cpu-profiling on gpuTest 0 10000000 1 0

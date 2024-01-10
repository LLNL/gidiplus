/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#ifndef LUPI_declare_macro_hpp_included
#define LUPI_declare_macro_hpp_included

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    #define LUPI_ON_GPU 1
#endif

#ifdef __CUDACC__
    #define LUPI_HOST __host__
    #define LUPI_DEVICE __device__
    #define LUPI_HOST_DEVICE __host__ __device__
    #define LUPI_THROW(arg) printf("%s", arg)
    #define LUPI_WARP_SIZE 32
    #define LUPI_THREADID threadIdx.x
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUASSERT: %s File: %s line: %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#elif HAVE_OPENMP_TARGET
//    #define LUPI_HOST_DEVICE _Pragma( "omp declare target" )
//    #define LUPI_HOST_DEVICE_END _Pragma("omp end declare target")
    //#define LUPI_HOST_DEVICE #pragma omp declare target
    //#define LUPI_HOST_DEVICE_END #pragma omp end declare target
    //#define LUPI_DEVICE #pragma omp declare target 
    //#define LUPI_DEVICE_END #pragma omp end declare target
    #define LUPI_HOST 
    #define LUPI_DEVICE 
    #define LUPI_HOST_DEVICE 
    #define LUPI_THROW(arg) printf("%s", arg)
    #define LUPI_WARP_SIZE 1
    #define LUPI_THREADID 
inline void gpuAssert(int code, const char *file, int line, bool abort=true) {}
#elif defined(__HIP__)
    #include <hip/hip_version.h>
    #include <hip/hip_runtime.h>
    #include <hip/hip_runtime_api.h>
    #include <hip/hip_common.h>

    #define LUPI_HOST __host__
    #define LUPI_DEVICE __device__
    #define LUPI_HOST_DEVICE __host__ __device__
    //#define LUPI_THROW(arg) printf("%s", arg)
    #define LUPI_THROW(arg) 
    #define LUPI_WARP_SIZE 1
    #define LUPI_THREADID hipThreadIdx_x
inline void gpuAssert(hipError_t code, const char *file, int line, bool do_abort=true)
{
    if (code == hipSuccess) { return; }
    printf("GPUassert code %d: %s %s %d\n", code, hipGetErrorString(code), file, line);
    if (do_abort) { abort(); }
}

#else
    #define LUPI_HOST
    #define LUPI_DEVICE 
    #define LUPI_HOST_DEVICE
    #define LUPI_THROW(arg) throw arg
    #define LUPI_WARP_SIZE 1
    #define LUPI_THREADID 
inline void gpuAssert(int code, const char *file, int line, bool abort=true) {}
#endif

#endif      // End of LUPI_declare_macro_hpp_included

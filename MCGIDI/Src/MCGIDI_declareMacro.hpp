/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#ifndef MCGIDI_DECLARE_MACRO_HH
#define MCGIDI_DECLARE_MACRO_HH

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#ifdef __CUDACC__
    #define HOST __host__
    #define DEVICE __device__
    #define HOST_DEVICE __host__ __device__
    #define THROW(arg) printf(arg)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUASSERT: %s File: %s line: %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#elif HAVE_OPENMP_TARGET
//    #define HOST_DEVICE _Pragma( "omp declare target" )
//    #define HOST_DEVICE_END _Pragma("omp end declare target")
    //#define HOST_DEVICE #pragma omp declare target
    //#define HOST_DEVICE_END #pragma omp end declare target
    //#define DEVICE #pragma omp declare target 
    //#define DEVICE_END #pragma omp end declare target
    #define HOST 
    #define DEVICE 
    #define HOST_DEVICE 
    #define THROW(arg) printf(arg)
inline void gpuAssert(int code, const char *file, int line, bool abort=true) {}
#else
    #define HOST
    #define DEVICE 
    #define HOST_DEVICE
    #define THROW(arg) throw arg
inline void gpuAssert(int code, const char *file, int line, bool abort=true) {}
#endif

#endif

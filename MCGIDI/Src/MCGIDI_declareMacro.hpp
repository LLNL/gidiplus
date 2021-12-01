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
    #define MCGIDI_HOST __host__
    #define MCGIDI_DEVICE __device__
    #define MCGIDI_HOST_DEVICE __host__ __device__
    #define MCGIDI_THROW(arg) printf("%s", arg)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUASSERT: %s File: %s line: %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#elif HAVE_OPENMP_TARGET
//    #define MCGIDI_HOST_DEVICE _Pragma( "omp declare target" )
//    #define MCGIDI_HOST_DEVICE_END _Pragma("omp end declare target")
    //#define MCGIDI_HOST_DEVICE #pragma omp declare target
    //#define MCGIDI_HOST_DEVICE_END #pragma omp end declare target
    //#define MCGIDI_DEVICE #pragma omp declare target 
    //#define MCGIDI_DEVICE_END #pragma omp end declare target
    #define MCGIDI_HOST 
    #define MCGIDI_DEVICE 
    #define MCGIDI_HOST_DEVICE 
    #define MCGIDI_THROW(arg) printf("%s", arg)
inline void gpuAssert(int code, const char *file, int line, bool abort=true) {}
#else
    #define MCGIDI_HOST
    #define MCGIDI_DEVICE 
    #define MCGIDI_HOST_DEVICE
    #define MCGIDI_THROW(arg) throw arg
inline void gpuAssert(int code, const char *file, int line, bool abort=true) {}
#endif

#endif

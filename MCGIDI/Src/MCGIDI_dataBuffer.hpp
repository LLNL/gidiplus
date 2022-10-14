/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#ifndef MCGIDI_data_buffer_hpp_included
#define MCGIDI_data_buffer_hpp_included 1

#include "MCGIDI_declareMacro.hpp"

namespace MCGIDI {

/*
============================================================
========================= DataBuffer =======================
============================================================
*/
class DataBuffer {

    public:
        size_t m_intIndex;
        size_t m_floatIndex;
        size_t m_charIndex;
        size_t m_longIndex;

        int *m_intData;
        double *m_floatData;
        char *m_charData;
        uint64_t *m_longData;

        // For unpacking into pre-allocated memory
        char *m_placementStart;
        char *m_placement;
        size_t m_maxPlacementSize;

        // If m_sharedPlacementStart is not a nullPtr, place int and double vector information here
        // m_sharedMaxPlacementSize is how much shared memory will be used.
        char *m_sharedPlacementStart;
        char *m_sharedPlacement;
        size_t m_sharedMaxPlacementSize;

        enum class Mode { Count, Pack, Unpack, Reset, Memory };

        MCGIDI_HOST_DEVICE DataBuffer( void ) :
                m_intIndex( 0 ),
                m_floatIndex( 0 ),
                m_charIndex( 0 ),
                m_longIndex( 0 ),
                m_intData( nullptr ),
                m_floatData( nullptr ),
                m_charData( nullptr ),
                m_longData( nullptr ),
                m_placementStart( nullptr ),
                m_placement( nullptr ),
                m_maxPlacementSize( 0 ),
                m_sharedPlacementStart( nullptr ),
                m_sharedPlacement( nullptr ),
                m_sharedMaxPlacementSize( 0 ) {
        }

        MCGIDI_HOST_DEVICE DataBuffer( DataBuffer const &rhs ) :
                m_intIndex( 0 ),
                m_floatIndex( 0 ),
                m_charIndex( 0 ),
                m_longIndex( 0 ),
                m_intData( nullptr ),
                m_floatData( nullptr ),
                m_charData( nullptr ),
                m_longData( nullptr ),
                m_placementStart( nullptr ),
                m_placement( nullptr ),
                m_maxPlacementSize( 0 ),
                m_sharedPlacementStart( nullptr ),
                m_sharedPlacement( nullptr ),
                m_sharedMaxPlacementSize( 0 ) {

        }

        MCGIDI_HOST_DEVICE ~DataBuffer( ) {

            delete [] m_intData;
            delete [] m_floatData;
            delete [] m_charData;
            delete [] m_longData;
        }

        MCGIDI_HOST_DEVICE void zeroIndexes( void ) { m_intIndex = m_floatIndex = m_charIndex = m_longIndex = 0; }

        MCGIDI_HOST_DEVICE void copyIndexes( DataBuffer const &a_input ) {

            m_intIndex   = a_input.m_intIndex;
            m_floatIndex = a_input.m_floatIndex;
            m_charIndex  = a_input.m_charIndex;
            m_longIndex  = a_input.m_longIndex;
        }

        MCGIDI_HOST_DEVICE void simpleCopy( DataBuffer const &a_input ) {

            m_intIndex               = a_input.m_intIndex;
            m_floatIndex             = a_input.m_floatIndex;
            m_charIndex              = a_input.m_charIndex;
            m_longIndex              = a_input.m_longIndex;
            m_intData                = a_input.m_intData;
            m_floatData              = a_input.m_floatData;
            m_charData               = a_input.m_charData;
            m_longData               = a_input.m_longData;
            m_placementStart         = a_input.m_placementStart;
            m_placement              = a_input.m_placement;
            m_maxPlacementSize       = a_input.m_maxPlacementSize;
            m_sharedPlacementStart   = a_input.m_sharedPlacementStart;
            m_sharedPlacement        = a_input.m_sharedPlacement;
            m_sharedMaxPlacementSize = a_input.m_sharedMaxPlacementSize;
        }

        // Useful for temporary buffers that we don't want destroying the data in the destructor
        MCGIDI_HOST_DEVICE void nullOutPointers( void ) {

            m_intData = nullptr;
            m_floatData = nullptr;
            m_charData = nullptr;
            m_longData = nullptr;
        }

        MCGIDI_HOST_DEVICE void allocateBuffers( void ) {

            m_intData = new int[m_intIndex];
            m_floatData = new double[m_floatIndex];
            m_charData = new char[m_charIndex];
            m_longData = new uint64_t[m_longIndex];
        }

        MCGIDI_HOST_DEVICE void freeMemory( void ) {

            delete [] m_intData;
            delete [] m_floatData;
            delete [] m_charData;
            delete [] m_longData;
            zeroIndexes( );
            nullOutPointers( );
        }

        MCGIDI_HOST_DEVICE bool compareIndexes( char const *a_file, int a_line, DataBuffer const &a_input ) {

            return( ( a_input.m_intIndex  == m_intIndex  ) && ( a_input.m_floatIndex == m_floatIndex ) &&
                    ( a_input.m_charIndex == m_charIndex ) && ( a_input.m_longIndex  == m_longIndex  ) );
        }

        MCGIDI_HOST_DEVICE void incrementPlacement(size_t a_delta) {

            size_t sub = a_delta % 8;
            if (sub != 0) a_delta += (8-sub);
            m_placement += a_delta;
        }

        MCGIDI_HOST_DEVICE void incrementSharedPlacement(size_t a_delta) {

            size_t sub = a_delta % 8;
            if (sub != 0) a_delta += (8-sub);
            m_sharedPlacement += a_delta;
        }

        // Returns true if data buffer has not gone over any memory limits
        MCGIDI_HOST_DEVICE bool validate() {

            if (m_placementStart == 0 && m_sharedPlacementStart == 0) return true;
            if (m_placement > m_maxPlacementSize + m_placementStart) return false;
            if (m_sharedPlacement > m_sharedMaxPlacementSize + m_sharedPlacementStart) return false;
            return true;
        }

#if defined(__CUDACC__) || defined (__HIP__)
    #ifdef __CUDACC__
        #define MCGIDI_GPU_MALLOC cudaMalloc
        #define MCGIDI_GPU_MEMCPY cudaMemcpy
        #define MCGIDI_GPU_HTOD   cudaMemcpyHostToDevice
    #else
        #define MCGIDI_GPU_MALLOC hipMalloc
        #define MCGIDI_GPU_MEMCPY hipMemcpy
        #define MCGIDI_GPU_HTOD   hipMemcpyHostToDevice
    #endif
        // Copy this host object to the device and return its pointer
        MCGIDI_HOST DataBuffer *copyToDevice(size_t a_cpuSize, char *&a_protarePtr) {

            DataBuffer *devicePtr = nullptr;
            DataBuffer buf_tmp;

            buf_tmp.copyIndexes(*this);
            buf_tmp.m_maxPlacementSize = a_cpuSize;

            gpuErrchk(MCGIDI_GPU_MALLOC((void **) &buf_tmp.m_intData, sizeof(int) * m_intIndex));
            gpuErrchk(MCGIDI_GPU_MEMCPY(buf_tmp.m_intData, m_intData, sizeof(int) * m_intIndex, MCGIDI_GPU_HTOD));
            gpuErrchk(MCGIDI_GPU_MALLOC((void **) &buf_tmp.m_floatData, sizeof(double) * m_floatIndex));
            gpuErrchk(MCGIDI_GPU_MEMCPY(buf_tmp.m_floatData, m_floatData, sizeof(double) * m_floatIndex, MCGIDI_GPU_HTOD));
            gpuErrchk(MCGIDI_GPU_MALLOC((void **) &buf_tmp.m_charData, sizeof(char) * m_charIndex));
            gpuErrchk(MCGIDI_GPU_MEMCPY(buf_tmp.m_charData, m_charData, sizeof(char) * m_charIndex, MCGIDI_GPU_HTOD));
            gpuErrchk(MCGIDI_GPU_MALLOC((void **) &buf_tmp.m_longData, sizeof(uint64_t) * m_longIndex));
            gpuErrchk(MCGIDI_GPU_MEMCPY(buf_tmp.m_longData, m_longData, sizeof(uint64_t) * m_longIndex, MCGIDI_GPU_HTOD));

            gpuErrchk(MCGIDI_GPU_MALLOC((void **) &buf_tmp.m_placementStart, buf_tmp.m_maxPlacementSize));
            // Set to 0 for easier byte comparisons. This may be removed after testing is done
            //gpuErrchk(cudaMemset((void *) buf_tmp.m_placementStart, 0, buf_tmp.m_maxPlacementSize));
            buf_tmp.m_placement = buf_tmp.m_placementStart;

            a_protarePtr = buf_tmp.m_placementStart;

            gpuErrchk(MCGIDI_GPU_MALLOC((void **) &devicePtr, sizeof(DataBuffer)));
            gpuErrchk(MCGIDI_GPU_MEMCPY(devicePtr, &buf_tmp, sizeof(DataBuffer), MCGIDI_GPU_HTOD));

            // Don't need destructor trying to free the device memory.
            buf_tmp.nullOutPointers();

            return devicePtr;
        }
    #undef MCGIDI_GPU_MALLOC
    #undef MCGIDI_GPU_MEMCPY
    #undef MCGIDI_GPU_HTOD
#endif

    private:
        DataBuffer &operator=( DataBuffer const &tmp );     // disable assignment operator

};

}       // End of namespace MCGIDI.

#define DATA_MEMBER_SIMPLE(member, buffer, index, mode) \
    {if (     mode == DataBuffer::Mode::Count )  {(index)++; } \
     else if ( mode == DataBuffer::Mode::Pack   ) {(buffer)[ (index)++ ] = (member); } \
     else if ( mode == DataBuffer::Mode::Unpack ) {member = (buffer)[ (index)++ ]; }   \
     else if ( mode == DataBuffer::Mode::Reset )  {(index)++; member = 0; }}

#define DATA_MEMBER_CAST(member, buf, mode, someType) \
    {if (     mode == DataBuffer::Mode::Count )  {((buf).m_intIndex)++; } \
     else if ( mode == DataBuffer::Mode::Pack   ) {(buf).m_intData[ ((buf).m_intIndex)++ ] = (int)(member); } \
     else if ( mode == DataBuffer::Mode::Unpack ) {member = (someType) (buf).m_intData[ ((buf).m_intIndex)++ ]; } \
     else if ( mode == DataBuffer::Mode::Reset )  {((buf).m_intIndex)++; member = (someType) 0; }}

#define DATA_MEMBER_CHAR( member, buf, mode) DATA_MEMBER_SIMPLE(member, (buf).m_charData,  (buf).m_charIndex,  mode)
#define DATA_MEMBER_INT(  member, buf, mode) DATA_MEMBER_SIMPLE(member, (buf).m_intData,   (buf).m_intIndex,   mode)
#define DATA_MEMBER_FLOAT(member, buf, mode) DATA_MEMBER_SIMPLE(member, (buf).m_floatData, (buf).m_floatIndex, mode)

#define DATA_MEMBER_STRING(member, buf, mode) \
    {if (     mode == DataBuffer::Mode::Count ) {((buf).m_charIndex) += member.size(); ((buf).m_intIndex)++; } \
     else if ( mode == DataBuffer::Mode::Pack   ) {size_t array_size = member.size(); \
             (buf).m_intData[((buf).m_intIndex)++] = array_size; \
             for (size_t size_index = 0; size_index < array_size; size_index++)\
                 {(buf).m_charData[ ((buf).m_charIndex)++ ] = (member[size_index]); }} \
     else if ( mode == DataBuffer::Mode::Unpack ) {size_t array_size = (buf).m_intData[((buf).m_intIndex)++]; \
         member.resize(array_size, &(buf).m_placement); \
         for (size_t size_index = 0; size_index < array_size; size_index++) \
             {member[size_index] = (buf).m_charData[ ((buf).m_charIndex)++ ]; }} \
     else if ( mode == DataBuffer::Mode::Reset ) {size_t array_size = member.size(); \
         for (size_t size_index = 0; size_index < array_size; size_index++) \
            {((buf).m_charIndex)++; member[size_index] = '\0'; }} \
     else if ( mode == DataBuffer::Mode::Memory ) { (buf).incrementPlacement(sizeof(char) * (member.size()+1)); } }

#if MCGIDI_WARP_SIZE > 1 and defined(MC_ON_GPU)
#define DATA_MEMBER_VECTOR_DOUBLE(member, buf, mode) \
    { \
        size_t vector_size = member.size(); \
        DATA_MEMBER_INT(vector_size, (buf), mode); \
        if ( mode == DataBuffer::Mode::Unpack ) member.resize(vector_size, &(buf).m_placement); \
        size_t bufferIndex = (buf).m_floatIndex; \
        for ( size_t member_index = 0; member_index < vector_size; member_index += MCGIDI_WARP_SIZE, bufferIndex += MCGIDI_WARP_SIZE ) \
        { \
            size_t thrMemberId = member_index+MCGIDI_THREADID; \
            if (thrMemberId >= vector_size) continue; \
            member[thrMemberId] = (buf).m_floatData[bufferIndex + MCGIDI_THREADID]; \
        } \
        (buf).m_floatIndex += vector_size; \
    }
#else
#define DATA_MEMBER_VECTOR_DOUBLE(member, buf, mode) \
    { \
        size_t vector_size = member.size(); \
        DATA_MEMBER_INT(vector_size, (buf), mode); \
        if ( mode == DataBuffer::Mode::Unpack ) { \
            if ((buf).m_sharedPlacement == nullptr) { \
                member.resize(vector_size, &(buf).m_placement); \
            } else { \
                member.resize(vector_size, &(buf).m_sharedPlacement); \
            } \
        }\
        if ( mode == DataBuffer::Mode::Memory ) { \
            (buf).incrementSharedPlacement(sizeof(double) * member.capacity()); \
        } \
        for ( size_t member_index = 0; member_index < vector_size; member_index++ ) \
        { \
            DATA_MEMBER_FLOAT(member[member_index], (buf), mode); \
        } \
    }
#endif

#if MCGIDI_WARP_SIZE > 1 and defined(MC_ON_GPU)
#define DATA_MEMBER_VECTOR_INT(member, buf, mode) \
    { \
        size_t vector_size = member.size(); \
        DATA_MEMBER_INT(vector_size, (buf), mode); \
        if ( mode == DataBuffer::Mode::Unpack ) member.resize(vector_size, &(buf).m_placement); \
        size_t bufferIndex = (buf).m_intIndex; \
        for ( size_t member_index = 0; member_index < vector_size; member_index += MCGIDI_WARP_SIZE, bufferIndex += MCGIDI_WARP_SIZE ) \
        { \
            size_t thrMemberId = member_index+MCGIDI_THREADID; \
            if (thrMemberId >= vector_size) continue; \
            member[thrMemberId] = (buf).m_intData[bufferIndex + MCGIDI_THREADID]; \
        } \
        (buf).m_intIndex += vector_size; \
    }
#else
#define DATA_MEMBER_VECTOR_INT(member, buf, mode) \
    { \
        size_t vector_size = member.size(); \
        DATA_MEMBER_INT(vector_size, (buf), mode); \
        if ( mode == DataBuffer::Mode::Unpack ) { \
            if ((buf).m_sharedPlacement == nullptr) { \
                member.resize(vector_size, &(buf).m_placement); \
            } else { \
                member.resize(vector_size, &(buf).m_sharedPlacement); \
            } \
        }\
        if ( mode == DataBuffer::Mode::Memory ) { \
            (buf).incrementSharedPlacement(sizeof(int) * member.capacity()); \
        } \
        for ( size_t member_index = 0; member_index < vector_size; member_index++ ) \
        { \
            DATA_MEMBER_INT(member[member_index], (buf), mode); \
        } \
    }
#endif

#if MCGIDI_WARP_SIZE > 1 and defined(MC_ON_GPU)
#define DATA_MEMBER_CHAR_ARRAY( member, buf, mode ) { \
        size_t array_size = sizeof( member ); \
        size_t bufferIndex = (buf).m_charIndex; \
        for ( size_t member_index = 0; member_index < array_size; member_index += MCGIDI_WARP_SIZE, bufferIndex += MCGIDI_WARP_SIZE ) { \
            size_t thrMemberId = member_index+MCGIDI_THREADID; \
            if( thrMemberId >= array_size ) continue; \
            member[thrMemberId] = (buf).m_charData[bufferIndex + MCGIDI_THREADID]; \
        } \
        (buf).m_charIndex += array_size; \
    }
#else
#define DATA_MEMBER_CHAR_ARRAY( member, buf, mode ) { \
        size_t array_size = sizeof( member ); \
        for ( size_t member_index = 0; member_index < array_size; member_index++ ) DATA_MEMBER_CHAR( member[member_index], (buf), mode ); \
    }
#endif

#endif      // End of MCGIDI_data_buffer_hpp_included

/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#ifndef LUPI_data_buffer_hpp_included
#define LUPI_data_buffer_hpp_included 1

#include <cstdint>

#include <LUPI_declareMacro.hpp>

namespace LUPI {

/*
============================================================
========================= DataBuffer =======================
============================================================
*/
class DataBuffer {

    public:
        std::size_t m_intIndex;
        std::size_t m_floatIndex;
        std::size_t m_charIndex;
        std::size_t m_longIndex;

        int *m_intData;
        double *m_floatData;
        char *m_charData;
        std::uint64_t *m_longData;

        // For unpacking into pre-allocated memory
        char *m_placementStart;
        char *m_placement;
        std::size_t m_maxPlacementSize;

        // If m_sharedPlacementStart is not a nullPtr, place int and double vector information here
        // m_sharedMaxPlacementSize is how much shared memory will be used.
        char *m_sharedPlacementStart;
        char *m_sharedPlacement;
        std::size_t m_sharedMaxPlacementSize;

        enum class Mode { Count, Pack, Unpack, Reset, Memory };

        LUPI_HOST_DEVICE DataBuffer( void ) :
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

        LUPI_HOST_DEVICE DataBuffer( DataBuffer const &rhs ) :
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

        LUPI_HOST_DEVICE ~DataBuffer( ) {

            delete [] m_intData;
            delete [] m_floatData;
            delete [] m_charData;
            delete [] m_longData;
        }

        LUPI_HOST_DEVICE void zeroIndexes( void ) {

            m_intIndex = m_floatIndex = m_charIndex = m_longIndex = 0;
        }

        LUPI_HOST_DEVICE void copyIndexes( DataBuffer const &a_input ) {

            m_intIndex   = a_input.m_intIndex;
            m_floatIndex = a_input.m_floatIndex;
            m_charIndex  = a_input.m_charIndex;
            m_longIndex  = a_input.m_longIndex;
        }

        LUPI_HOST_DEVICE void simpleCopy( DataBuffer const &a_input ) {

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
        LUPI_HOST_DEVICE void nullOutPointers( void ) {

            m_intData = nullptr;
            m_floatData = nullptr;
            m_charData = nullptr;
            m_longData = nullptr;
        }

        LUPI_HOST_DEVICE void allocateBuffers( void ) {

            m_intData = new int[m_intIndex];
            m_floatData = new double[m_floatIndex];
            m_charData = new char[m_charIndex];
            m_longData = new std::uint64_t[m_longIndex];
        }

        LUPI_HOST_DEVICE void freeMemory( void ) {

            delete [] m_intData;
            delete [] m_floatData;
            delete [] m_charData;
            delete [] m_longData;
            zeroIndexes( );
            nullOutPointers( );
        }

        LUPI_HOST_DEVICE bool compareIndexes( char const *a_file, int a_line, DataBuffer const &a_input ) {

            return( ( a_input.m_intIndex  == m_intIndex  ) && ( a_input.m_floatIndex == m_floatIndex ) &&
                    ( a_input.m_charIndex == m_charIndex ) && ( a_input.m_longIndex  == m_longIndex  ) );
        }

        LUPI_HOST_DEVICE void incrementPlacement(std::size_t a_delta) {

            std::size_t sub = a_delta % 8;
            if (sub != 0) a_delta += (8-sub);
            m_placement += a_delta;
        }

        LUPI_HOST_DEVICE void incrementSharedPlacement(std::size_t a_delta) {

            std::size_t sub = a_delta % 8;
            if (sub != 0) a_delta += (8-sub);
            m_sharedPlacement += a_delta;
        }

        // Returns true if data buffer has not gone over any memory limits
        LUPI_HOST_DEVICE bool validate() {

            if (m_placementStart == 0 && m_sharedPlacementStart == 0) return true;
            if (m_placement > m_maxPlacementSize + m_placementStart) return false;
            if (m_sharedPlacement > m_sharedMaxPlacementSize + m_sharedPlacementStart) return false;
            return true;
        }

#if defined(__CUDACC__) || defined (__HIP__)
    #ifdef __CUDACC__
        #define LUPI_GPU_MALLOC cudaMalloc
        #define LUPI_GPU_MEMCPY cudaMemcpy
        #define LUPI_GPU_HTOD   cudaMemcpyHostToDevice
    #else
        #define LUPI_GPU_MALLOC hipMalloc
        #define LUPI_GPU_MEMCPY hipMemcpy
        #define LUPI_GPU_HTOD   hipMemcpyHostToDevice
    #endif
        // Copy this host object to the device and return its pointer
        LUPI_HOST DataBuffer *copyToDevice(std::size_t a_cpuSize, char *&a_protarePtr) {

            DataBuffer *devicePtr = nullptr;
            DataBuffer buf_tmp;

            buf_tmp.copyIndexes(*this);
            buf_tmp.m_maxPlacementSize = a_cpuSize;

            gpuErrorCheck( LUPI_GPU_MALLOC( (void **) &buf_tmp.m_intData, sizeof(int) * m_intIndex) );
            gpuErrorCheck( LUPI_GPU_MEMCPY( buf_tmp.m_intData, m_intData, sizeof(int) * m_intIndex, LUPI_GPU_HTOD ) );
            gpuErrorCheck( LUPI_GPU_MALLOC( (void **) &buf_tmp.m_floatData, sizeof(double) * m_floatIndex ) );
            gpuErrorCheck( LUPI_GPU_MEMCPY( buf_tmp.m_floatData, m_floatData, sizeof(double) * m_floatIndex, LUPI_GPU_HTOD ) );
            gpuErrorCheck( LUPI_GPU_MALLOC( (void **) &buf_tmp.m_charData, sizeof(char) * m_charIndex ) );
            gpuErrorCheck( LUPI_GPU_MEMCPY( buf_tmp.m_charData, m_charData, sizeof(char) * m_charIndex, LUPI_GPU_HTOD ) );
            gpuErrorCheck( LUPI_GPU_MALLOC( (void **) &buf_tmp.m_longData, sizeof(std::uint64_t) * m_longIndex ) );
            gpuErrorCheck( LUPI_GPU_MEMCPY( buf_tmp.m_longData, m_longData, sizeof(std::uint64_t) * m_longIndex, LUPI_GPU_HTOD ) );

            gpuErrorCheck( LUPI_GPU_MALLOC( (void **) &buf_tmp.m_placementStart, buf_tmp.m_maxPlacementSize ) );
            // Set to 0 for easier byte comparisons. This may be removed after testing is done
            //gpuErrorCheck( cudaMemset( (void *) buf_tmp.m_placementStart, 0, buf_tmp.m_maxPlacementSize ) );
            buf_tmp.m_placement = buf_tmp.m_placementStart;

            a_protarePtr = buf_tmp.m_placementStart;

            gpuErrorCheck( LUPI_GPU_MALLOC( (void **) &devicePtr, sizeof(DataBuffer) ) );
            gpuErrorCheck( LUPI_GPU_MEMCPY( devicePtr, &buf_tmp, sizeof(DataBuffer), LUPI_GPU_HTOD ) );

            // Don't need destructor trying to free the device memory.
            buf_tmp.nullOutPointers( );

            return devicePtr;
        }
    #undef LUPI_GPU_MALLOC
    #undef LUPI_GPU_MEMCPY
    #undef LUPI_GPU_HTOD
#endif

    private:
        DataBuffer &operator=( DataBuffer const &tmp );     // disable assignment operator

};

}       // End of namespace LUPI.

#define DATA_MEMBER_SIMPLE(member, buffer, index, mode) \
    {if (     mode == LUPI::DataBuffer::Mode::Count )  {(index)++; } \
     else if ( mode == LUPI::DataBuffer::Mode::Pack   ) {(buffer)[ (index)++ ] = (member); } \
     else if ( mode == LUPI::DataBuffer::Mode::Unpack ) {member = (buffer)[ (index)++ ]; }   \
     else if ( mode == LUPI::DataBuffer::Mode::Reset )  {(index)++; member = 0; }}

#define DATA_MEMBER_CAST(member, buf, mode, someType) \
    {if (     mode == LUPI::DataBuffer::Mode::Count )  {((buf).m_intIndex)++; } \
     else if ( mode == LUPI::DataBuffer::Mode::Pack   ) {(buf).m_intData[ ((buf).m_intIndex)++ ] = (int)(member); } \
     else if ( mode == LUPI::DataBuffer::Mode::Unpack ) {member = (someType) (buf).m_intData[ ((buf).m_intIndex)++ ]; } \
     else if ( mode == LUPI::DataBuffer::Mode::Reset )  {((buf).m_intIndex)++; member = (someType) 0; }}

#define DATA_MEMBER_CHAR( member, buf, mode) DATA_MEMBER_SIMPLE(member, (buf).m_charData,  (buf).m_charIndex,  mode)
#define DATA_MEMBER_INT(  member, buf, mode) DATA_MEMBER_SIMPLE(member, (buf).m_intData,   (buf).m_intIndex,   mode)
#define DATA_MEMBER_FLOAT(member, buf, mode) DATA_MEMBER_SIMPLE(member, (buf).m_floatData, (buf).m_floatIndex, mode)

#define DATA_MEMBER_STRING(member, buf, mode) \
    {if (     mode == LUPI::DataBuffer::Mode::Count ) {((buf).m_charIndex) += member.size(); ((buf).m_intIndex)++; } \
     else if ( mode == LUPI::DataBuffer::Mode::Pack   ) {std::size_t array_size = member.size(); \
             (buf).m_intData[((buf).m_intIndex)++] = array_size; \
             for (std::size_t size_index = 0; size_index < array_size; size_index++)\
                 {(buf).m_charData[ ((buf).m_charIndex)++ ] = (member[size_index]); }} \
     else if ( mode == LUPI::DataBuffer::Mode::Unpack ) {std::size_t array_size = (buf).m_intData[((buf).m_intIndex)++]; \
         member.resize(array_size, &(buf).m_placement); \
         for (std::size_t size_index = 0; size_index < array_size; size_index++) \
             {member[size_index] = (buf).m_charData[ ((buf).m_charIndex)++ ]; }} \
     else if ( mode == LUPI::DataBuffer::Mode::Reset ) {std::size_t array_size = member.size(); \
         for (std::size_t size_index = 0; size_index < array_size; size_index++) \
            {((buf).m_charIndex)++; member[size_index] = '\0'; }} \
     else if ( mode == LUPI::DataBuffer::Mode::Memory ) { (buf).incrementPlacement(sizeof(char) * (member.size()+1)); } }

#define DATA_MEMBER_STD_STRING(member, buf, mode) { \
    if (     mode == LUPI::DataBuffer::Mode::Count ) \
        {((buf).m_charIndex) += member.size(); ((buf).m_intIndex)++; } \
    else if ( mode == LUPI::DataBuffer::Mode::Pack   ) {std::size_t array_size = member.size(); \
             (buf).m_intData[((buf).m_intIndex)++] = array_size; \
             for (std::size_t size_index = 0; size_index < array_size; size_index++)\
                 {(buf).m_charData[((buf).m_charIndex)++] = (member[size_index]); }} \
    else if ( mode == LUPI::DataBuffer::Mode::Unpack ) {std::size_t array_size = (buf).m_intData[((buf).m_intIndex)++]; \
         member.resize(array_size); \
         for (std::size_t size_index = 0; size_index < array_size; size_index++) \
             {member[size_index] = (buf).m_charData[ ((buf).m_charIndex)++ ]; }} }

#if LUPI_WARP_SIZE > 1 and defined(LUPI_ON_GPU)
#define DATA_MEMBER_VECTOR_DOUBLE(member, buf, mode) \
    { \
        std::size_t vector_size = member.size(); \
        DATA_MEMBER_INT(vector_size, (buf), mode); \
        if ( mode == LUPI::DataBuffer::Mode::Unpack ) member.resize(vector_size, &(buf).m_placement); \
        std::size_t bufferIndex = (buf).m_floatIndex; \
        for ( std::size_t member_index = 0; member_index < vector_size; member_index += LUPI_WARP_SIZE, bufferIndex += LUPI_WARP_SIZE ) \
        { \
            std::size_t thrMemberId = member_index + LUPI_THREADID; \
            if (thrMemberId >= vector_size) continue; \
            member[thrMemberId] = (buf).m_floatData[bufferIndex + LUPI_THREADID]; \
        } \
        (buf).m_floatIndex += vector_size; \
    }
#else
#define DATA_MEMBER_VECTOR_DOUBLE(member, buf, mode) \
    { \
        std::size_t vector_size = member.size(); \
        DATA_MEMBER_INT(vector_size, (buf), mode); \
        if ( mode == LUPI::DataBuffer::Mode::Unpack ) { \
            if ((buf).m_sharedPlacement == nullptr) { \
                member.resize(vector_size, &(buf).m_placement); \
            } else { \
                member.resize(vector_size, &(buf).m_sharedPlacement); \
            } \
        }\
        if ( mode == LUPI::DataBuffer::Mode::Memory ) { \
            (buf).incrementSharedPlacement(sizeof(double) * member.capacity()); \
        } \
        for ( std::size_t member_index = 0; member_index < vector_size; member_index++ ) \
        { \
            DATA_MEMBER_FLOAT(member[member_index], (buf), mode); \
        } \
    }
#endif

#if LUPI_WARP_SIZE > 1 and defined(LUPI_ON_GPU)
#define DATA_MEMBER_VECTOR_INT(member, buf, mode) \
    { \
        std::size_t vector_size = member.size(); \
        DATA_MEMBER_INT(vector_size, (buf), mode); \
        if ( mode == LUPI::DataBuffer::Mode::Unpack ) member.resize(vector_size, &(buf).m_placement); \
        std::size_t bufferIndex = (buf).m_intIndex; \
        for ( std::size_t member_index = 0; member_index < vector_size; member_index += LUPI_WARP_SIZE, bufferIndex += LUPI_WARP_SIZE ) \
        { \
            std::size_t thrMemberId = member_index + LUPI_THREADID; \
            if (thrMemberId >= vector_size) continue; \
            member[thrMemberId] = (buf).m_intData[bufferIndex + LUPI_THREADID]; \
        } \
        (buf).m_intIndex += vector_size; \
    }
#else
#define DATA_MEMBER_VECTOR_INT(member, buf, mode) \
    { \
        std::size_t vector_size = member.size(); \
        DATA_MEMBER_INT(vector_size, (buf), mode); \
        if ( mode == LUPI::DataBuffer::Mode::Unpack ) { \
            if ((buf).m_sharedPlacement == nullptr) { \
                member.resize(vector_size, &(buf).m_placement); \
            } else { \
                member.resize(vector_size, &(buf).m_sharedPlacement); \
            } \
        }\
        if ( mode == LUPI::DataBuffer::Mode::Memory ) { \
            (buf).incrementSharedPlacement(sizeof(int) * member.capacity()); \
        } \
        for ( std::size_t member_index = 0; member_index < vector_size; member_index++ ) \
        { \
            DATA_MEMBER_INT(member[member_index], (buf), mode); \
        } \
    }
#endif

#if LUPI_WARP_SIZE > 1 and defined(LUPI_ON_GPU)
#define DATA_MEMBER_CHAR_ARRAY( member, buf, mode ) { \
        std::size_t array_size = sizeof( member ); \
        std::size_t bufferIndex = (buf).m_charIndex; \
        for ( std::size_t member_index = 0; member_index < array_size; member_index += LUPI_WARP_SIZE, bufferIndex += LUPI_WARP_SIZE ) { \
            std::size_t thrMemberId = member_index + LUPI_THREADID; \
            if( thrMemberId >= array_size ) continue; \
            member[thrMemberId] = (buf).m_charData[bufferIndex + LUPI_THREADID]; \
        } \
        (buf).m_charIndex += array_size; \
    }
#else
#define DATA_MEMBER_CHAR_ARRAY( member, buf, mode ) { \
        std::size_t array_size = sizeof( member ); \
        for ( std::size_t member_index = 0; member_index < array_size; member_index++ ) DATA_MEMBER_CHAR( member[member_index], (buf), mode ); \
    }
#endif

#endif      // End of LUPI_data_buffer_hpp_included

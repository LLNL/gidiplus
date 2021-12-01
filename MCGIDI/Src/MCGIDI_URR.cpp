/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "MCGIDI.hpp"

namespace MCGIDI {

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE void URR_protareInfo::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    DATA_MEMBER_CAST( m_inURR, a_buffer, a_mode, bool );
    DATA_MEMBER_FLOAT( m_rng_Value, a_buffer, a_mode );
}

/* *********************************************************************************************************//**
 * URR_protareInfos constructor.
 *
 * @param a_protares            [in]    The list of protares to be check for URR data. Each protare with URR data add to *a_URR_protareInfos*.
 ***********************************************************************************************************/

MCGIDI_HOST URR_protareInfos::URR_protareInfos( Vector<Protare *> &a_protares ) {

    setup( a_protares );
}

/* *********************************************************************************************************//**
 * URR_protareInfos setup.
 *
 * @param a_protares            [in]    The list of protares to be check for URR data. Each protare with URR data add to *a_URR_protareInfos*.
 ***********************************************************************************************************/

MCGIDI_HOST void URR_protareInfos::setup( Vector<Protare *> &a_protares ) {

    std::vector<URR_protareInfo> URR_protareInfo_1;

    for( MCGIDI_VectorSizeType i1 = 0; i1 < a_protares.size( ); ++i1 ) {
        Protare *protare = a_protares[i1];

        for( MCGIDI_VectorSizeType i2 = 0; i2 < protare->numberOfProtares( ); ++i2 ) {
            ProtareSingle *protareSingle = const_cast<ProtareSingle *>( protare->protare( i2 ) );

            if( protareSingle->hasURR_probabilityTables( ) ) {
                protareSingle->URR_index( URR_protareInfo_1.size( ) );
                URR_protareInfo_1.push_back( URR_protareInfo( ) );
            }
        }
    }

    m_URR_protareInfos.reserve( URR_protareInfo_1.size( ) );
    m_URR_protareInfos.clear( );
    for( std::size_t i1 = 0; i1 < URR_protareInfo_1.size( ); ++i1 ) m_URR_protareInfos.push_back( URR_protareInfo_1[i1] );
}

/* *********************************************************************************************************//**
 * Updates *this* if *a_protare* has a non-negative *URR_index*.
 *
 * @param a_protare             [in]    The protare whose *URR_index* is used to see if *this* needs updating.
 * @param a_energy              [in]    The energy of the projectile.
 * @param a_userrng             [in]    The random number generator function the uses *a_rngState* to generator a double in the range [0, 1.0).
 * @param a_rngState            [in]    The random number generator state.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE void URR_protareInfos::updateProtare( MCGIDI::Protare const *a_protare, double a_energy, double (*a_userrng)( void * ), void *a_rngState ) {

    for( MCGIDI_VectorSizeType i1 = 0; i1 < a_protare->numberOfProtares( ); ++i1 ) {
        ProtareSingle *protareSingle = const_cast<ProtareSingle *>( a_protare->protare( i1 ) );

        if( protareSingle->URR_index( ) >= 0 ) {
            URR_protareInfo &URR_protare_info = m_URR_protareInfos[protareSingle->URR_index( )];

            URR_protare_info.m_inURR = protareSingle->inURR( a_energy );
            if( URR_protare_info.inURR( ) ) URR_protare_info.m_rng_Value = a_userrng( a_rngState );
        }
    }
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE void URR_protareInfos::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    std::size_t vectorSize = m_URR_protareInfos.size( );
    int vectorSizeInt = (int) vectorSize;
    DATA_MEMBER_INT( vectorSizeInt, a_buffer, a_mode );
    vectorSize = (std::size_t) vectorSizeInt;
    if( a_mode == DataBuffer::Mode::Unpack ) m_URR_protareInfos.resize( vectorSize, &a_buffer.m_placement );
    if( a_mode == DataBuffer::Mode::Memory ) a_buffer.m_placement += m_URR_protareInfos.internalSize();

    for( std::size_t vectorIndex = 0; vectorIndex < vectorSize; ++vectorIndex ) {
        m_URR_protareInfos[vectorIndex].serialize( a_buffer, a_mode );
    }
}

}       // End namespace MCGIDI.

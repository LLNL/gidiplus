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

/*! \class DelayedNeutron
 * This class represents a **GNDS** <**DelayedNeutron**> node.
 */

/* *********************************************************************************************************//**
 * Default constructor used when broadcasting a Protare as needed by MPI or GPUs.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE DelayedNeutron::DelayedNeutron( ) :
        m_delayedNeutronIndex( -1 ),
        m_rate( 0.0 ),
        m_product( ) {

}

/* *********************************************************************************************************//**
 * @param a_index               [in]    Fix me.
 * @param a_delayedNeutron      [in]    The GIDI::DelayedNeutron whose data is to be used to construct *this*.
 * @param a_setupInfo           [in]    Used internally when constructing a Protare to pass information to other constructors.
 * @param a_settings            [in]    Used to pass user options to the *this* to instruct it which data are desired.
 * @param a_particles           [in]    List of transporting particles and their information (e.g., multi-group boundaries and fluxes).
 ***********************************************************************************************************/

MCGIDI_HOST DelayedNeutron::DelayedNeutron( int a_index, GIDI::DelayedNeutron const *a_delayedNeutron, SetupInfo &a_setupInfo, Transporting::MC const &a_settings, GIDI::Transporting::Particles const &a_particles ) :
        m_delayedNeutronIndex( a_index ),
        m_rate( 0.0 ),
        m_product( &a_delayedNeutron->product( ), a_setupInfo, a_settings, a_particles, false ) {

    GIDI::PhysicalQuantity const *rate = a_delayedNeutron->rate( ).get<GIDI::PhysicalQuantity>( 0 );
    m_rate = rate->value( );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE DelayedNeutron::~DelayedNeutron( ) {

}

/* *********************************************************************************************************//**
 * Updates the m_userParticleIndex to *a_userParticleIndex* for all particles with PoPs index *a_particleIndex*.
 *  
 * @param a_particleIndex       [in]    The PoPs id of the particle whose userPid is to be set.
 * @param a_userParticleIndex   [in]    The particle id specified by the user.
 ***********************************************************************************************************/

MCGIDI_HOST void DelayedNeutron::setUserParticleIndex( int a_particleIndex, int a_userParticleIndex ) {
    
    m_product.setUserParticleIndex( a_particleIndex, a_userParticleIndex );
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE void DelayedNeutron::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    DATA_MEMBER_INT( m_delayedNeutronIndex, a_buffer, a_mode );
    DATA_MEMBER_FLOAT( m_rate, a_buffer, a_mode );
    m_product.serialize( a_buffer, a_mode );
}

}

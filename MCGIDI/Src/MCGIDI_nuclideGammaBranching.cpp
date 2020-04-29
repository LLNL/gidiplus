/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <string.h>

#include "MCGIDI.hpp"

namespace MCGIDI {

/*
============================================================
================== NuclideGammaBranchInfo ==================
============================================================
*/

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

NuclideGammaBranchInfo::NuclideGammaBranchInfo( ) :
        m_probability( 0.0 ),
        m_photonEmissionProbability( 0.0 ),
        m_gammaEnergy( 0.0 ),
        m_residualStateIndex( -1 ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

NuclideGammaBranchInfo::NuclideGammaBranchInfo( PoPI::NuclideGammaBranchInfo const &a_nuclideGammaBranchInfo, std::map<std::string, int> &a_stateNamesToIndices ) :
        m_probability( a_nuclideGammaBranchInfo.probability( ) ),
        m_photonEmissionProbability( a_nuclideGammaBranchInfo.photonEmissionProbability( ) ),
        m_gammaEnergy( a_nuclideGammaBranchInfo.gammaEnergy( ) ),
        m_residualStateIndex( -1 ) {

        std::map<std::string,int>::iterator iter = a_stateNamesToIndices.find( a_nuclideGammaBranchInfo.residualState( ) );
        if( iter != a_stateNamesToIndices.end( ) ) m_residualStateIndex = iter->second;
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

HOST_DEVICE void NuclideGammaBranchInfo::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    DataBuffer *workingBuffer = &a_buffer;

    DATA_MEMBER_FLOAT( m_probability, *workingBuffer, a_mode );
    DATA_MEMBER_FLOAT( m_photonEmissionProbability, *workingBuffer, a_mode );
    DATA_MEMBER_FLOAT( m_gammaEnergy, *workingBuffer, a_mode );
    DATA_MEMBER_INT( m_residualStateIndex, *workingBuffer, a_mode );
}

/*
============================================================
============== NuclideGammaBranchStateInfo =================
============================================================
*/

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

NuclideGammaBranchStateInfo::NuclideGammaBranchStateInfo( ) :
        m_multiplicity( 0.0 ),
        m_averageGammaEnergy( 0.0 ) {

        m_state[0] = 0;
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

NuclideGammaBranchStateInfo::NuclideGammaBranchStateInfo( PoPI::NuclideGammaBranchStateInfo const &a_nuclideGammaBranchingInfo,
                std::vector<NuclideGammaBranchInfo *> &a_nuclideGammaBranchInfos, std::map<std::string, int> &a_stateNamesToIndices ) :
        m_multiplicity( a_nuclideGammaBranchingInfo.multiplicity( ) ),
        m_averageGammaEnergy( a_nuclideGammaBranchingInfo.averageGammaEnergy( ) ) {

    strncpy( m_state, a_nuclideGammaBranchingInfo.state( ).c_str( ), sizeof( m_state ) );
    m_state[sizeof( m_state )-1] = 0;

    std::vector<PoPI::NuclideGammaBranchInfo> const &branches = a_nuclideGammaBranchingInfo.branches( );
    m_branches.reserve( branches.size( ) );

    for( std::size_t i1 = 0; i1 < branches.size( ); ++i1 ) {
        m_branches.push_back( a_nuclideGammaBranchInfos.size( ) );
        a_nuclideGammaBranchInfos.push_back( new NuclideGammaBranchInfo( branches[i1], a_stateNamesToIndices ) );
    }
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

HOST_DEVICE void NuclideGammaBranchStateInfo::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    DataBuffer *workingBuffer = &a_buffer;

    DATA_MEMBER_CHAR_ARRAY( m_state, *workingBuffer, a_mode );
    DATA_MEMBER_FLOAT( m_multiplicity, *workingBuffer, a_mode );
    DATA_MEMBER_FLOAT( m_averageGammaEnergy, *workingBuffer, a_mode );
    DATA_MEMBER_VECTOR_INT( m_branches, a_buffer, a_mode );
}

}

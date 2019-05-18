/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "PoPs.hpp"

namespace PoPs {

/*
============================================================
================== NuclideGammaBranchInfo ==================
============================================================
*/
NuclideGammaBranchInfo::NuclideGammaBranchInfo( double a_probability, double a_photonEmissionProbability, double a_gammaEnergy, std::string const &a_residualState ) :
        m_probability( a_probability ),
        m_photonEmissionProbability( a_photonEmissionProbability ),
        m_gammaEnergy( a_gammaEnergy ),
        m_residualState( a_residualState ) {

}
/*
=========================================================
*/
NuclideGammaBranchInfo::NuclideGammaBranchInfo( NuclideGammaBranchInfo const &a_nuclideGammaBranchInfo ) :
        m_probability( a_nuclideGammaBranchInfo.probability( ) ),
        m_photonEmissionProbability( a_nuclideGammaBranchInfo.photonEmissionProbability( ) ),
        m_gammaEnergy( a_nuclideGammaBranchInfo.gammaEnergy( ) ),
        m_residualState( a_nuclideGammaBranchInfo.residualState( ) ) {

}

/*
============================================================
================= NuclideGammaBranchStateInfo ================
============================================================
*/
NuclideGammaBranchStateInfo::NuclideGammaBranchStateInfo( std::string a_state ) :
        m_state( a_state ),
        m_derivedCalculated( false ),
        m_multiplicity( 0.0 ),
        m_averageGammaEnergy( 0.0 ) {

}
/*
=========================================================
*/
void NuclideGammaBranchStateInfo::add( NuclideGammaBranchInfo const &a_nuclideGammaBranchInfo ) {

    m_branches.push_back( a_nuclideGammaBranchInfo );
}
/*
=========================================================
*/
void NuclideGammaBranchStateInfo::calculateDerivedData( NuclideGammaBranchStateInfos &a_nuclideGammaBranchStateInfos ) {

    if( m_derivedCalculated ) return;

    for( std::size_t i1 = 0; i1 < m_branches.size( ); ++i1 ) {
        NuclideGammaBranchInfo &nuclideGammaBranchInfo = m_branches[i1];

        std::string const &residualState = nuclideGammaBranchInfo.residualState( );
        NuclideGammaBranchStateInfo *nuclideGammaBranchStateInfo = a_nuclideGammaBranchStateInfos.find( residualState );

        double chainedMultiplicity = 0.0;
        double chainedAverageGammaEnergy = 0.0;
        if( nuclideGammaBranchStateInfo != NULL ) {
            nuclideGammaBranchStateInfo->calculateDerivedData( a_nuclideGammaBranchStateInfos );
            chainedMultiplicity = nuclideGammaBranchStateInfo->multiplicity( );
            chainedAverageGammaEnergy = nuclideGammaBranchStateInfo->averageGammaEnergy( );
        }

        m_multiplicity += nuclideGammaBranchInfo.probability( ) * ( nuclideGammaBranchInfo.photonEmissionProbability( ) + chainedMultiplicity );
        m_averageGammaEnergy += nuclideGammaBranchInfo.probability( ) * 
                ( nuclideGammaBranchInfo.photonEmissionProbability( ) * nuclideGammaBranchInfo.gammaEnergy( ) + chainedAverageGammaEnergy );
    }

    m_derivedCalculated = true;
}

/*
============================================================
================ NuclideGammaBranchStateInfos ================
============================================================
*/
NuclideGammaBranchStateInfos::NuclideGammaBranchStateInfos( ) {

}
/*
=========================================================
*/
NuclideGammaBranchStateInfos::~NuclideGammaBranchStateInfos( ) {

    for( std::size_t i1 = 0; i1 < m_nuclideGammaBranchStateInfos.size( ); ++i1 ) delete m_nuclideGammaBranchStateInfos[i1];
}
/*
=========================================================
*/
void NuclideGammaBranchStateInfos::add( NuclideGammaBranchStateInfo *a_nuclideGammaBranchStateInfo ) {

    m_nuclideGammaBranchStateInfos.push_back( a_nuclideGammaBranchStateInfo );
}
/*
=========================================================
*/
NuclideGammaBranchStateInfo *NuclideGammaBranchStateInfos::find( std::string const &a_state ) {

    for( std::size_t i1 = 0; i1 < m_nuclideGammaBranchStateInfos.size( ); ++i1 ) {
        NuclideGammaBranchStateInfo *nuclideGammaBranchStateInfo = m_nuclideGammaBranchStateInfos[i1];

        if( nuclideGammaBranchStateInfo->state( ) == a_state ) return( nuclideGammaBranchStateInfo );
    }

    return( NULL );
}

}

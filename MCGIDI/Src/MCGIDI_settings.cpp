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

namespace Transporting {

/*
============================================================
============================ MC ============================
============================================================
*/
MCGIDI_HOST MC::MC( PoPI::Database const &a_pops, std::string const &a_projectileID, GIDI::Styles::Suite const *a_styles, std::string const &a_label, GIDI::Transporting::DelayedNeutrons a_delayedNeutrons, double a_energyDomainMax ) :
        GIDI::Transporting::Settings( a_projectileID, a_delayedNeutrons ),
        m_pops( a_pops ),
        m_neutronIndex( a_pops[PoPI::IDs::neutron] ),
        m_photonIndex( a_pops[PoPI::IDs::photon] ),
        m_styles( a_styles ),
        m_label( a_label ),
        m_energyDomainMax( a_energyDomainMax ),
        m_ignoreENDF_MT5( false ),
        m_sampleNonTransportingParticles( false ),
        m_crossSectionLookupMode( LookupMode::Data1d::continuousEnergy ),
        m_other1dDataLookupMode( LookupMode::Data1d::continuousEnergy ),
        m_distributionLookupMode( LookupMode::Distribution::pdf_cdf ),
        m_upscatterModel( Sampling::Upscatter::Model::none ),
        m_upscatterModelALabel( "" ),
        m_want_URR_probabilityTables( false ),
        m_wantTerrellPromptNeutronDistribution( false ) {

}
/*
=========================================================
*/
MCGIDI_HOST void MC::crossSectionLookupMode( LookupMode::Data1d a_crossSectionLookupMode ) {

    if( ( a_crossSectionLookupMode != LookupMode::Data1d::continuousEnergy ) && 
        ( a_crossSectionLookupMode != LookupMode::Data1d::multiGroup ) ) {
        throw( "Invalided cross section mode request." );
    }
    m_crossSectionLookupMode = a_crossSectionLookupMode;
}
/*
=========================================================
*/
MCGIDI_HOST void MC::other1dDataLookupMode( LookupMode::Data1d a_other1dDataLookupMode ) {

    if( a_other1dDataLookupMode != LookupMode::Data1d::continuousEnergy ) throw( "Invalided other mode request." );
    m_other1dDataLookupMode = a_other1dDataLookupMode;
}
/*
=========================================================
*/
MCGIDI_HOST void MC::distributionLookupMode( LookupMode::Distribution a_distributionLookupMode ) {

    if( a_distributionLookupMode != LookupMode::Distribution::pdf_cdf ) throw( "Invalided distribution mode request." );
    m_distributionLookupMode = a_distributionLookupMode;
}
/*
=========================================================
*/
MCGIDI_HOST void MC::setUpscatterModelA( std::string const &a_upscatterModelALabel ) {

    m_upscatterModel = Sampling::Upscatter::Model::A;
    m_upscatterModelALabel = a_upscatterModelALabel;
}

}

}

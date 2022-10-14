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

/*! \class MC
 * Class to store user defined preferences for creating an MCGIDI::Protare instance.
 */

/* *********************************************************************************************************//**
 * Class to store user defined preferences for creating an MCGIDI::Protare instance.
 *
 * @param a_pops                        [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_projectileID                [in]    The PoPs id for the projectile.
 * @param a_styles                      [in]    The styles child node of the GIDI::Protare.
 * @param a_label                       [in]    
 * @param a_delayedNeutrons             [in]    Sets whether delayed neutron data will be load or not if available.
 * @param a_energyDomainMax             [in]    The maximum projectile energy for which data should be loaded.
 ***********************************************************************************************************/

MCGIDI_HOST MC::MC( PoPI::Database const &a_pops, std::string const &a_projectileID, GIDI::Styles::Suite const *a_styles, std::string const &a_label, 
                GIDI::Transporting::DelayedNeutrons a_delayedNeutrons, double a_energyDomainMax ) :
        GIDI::Transporting::Settings( a_projectileID, a_delayedNeutrons ),
        m_pops( a_pops ),
        m_neutronIndex( a_pops[PoPI::IDs::neutron] ),
        m_photonIndex( a_pops[PoPI::IDs::photon] ),
        m_electronIndex( a_pops[PoPI::IDs::electron] ),
        m_styles( a_styles ),
        m_label( a_label ),
        m_energyDomainMax( a_energyDomainMax ),
        m_ignoreENDF_MT5( false ),
        m_sampleNonTransportingParticles( false ),
        m_useSlowerContinuousEnergyConversion( false ),
        m_crossSectionLookupMode( LookupMode::Data1d::continuousEnergy ),
        m_other1dDataLookupMode( LookupMode::Data1d::continuousEnergy ),
        m_distributionLookupMode( LookupMode::Distribution::pdf_cdf ),
        m_upscatterModel( Sampling::Upscatter::Model::none ),
        m_upscatterModelALabel( "" ),
        m_URR_mode( URR_mode::none ),
        m_wantTerrellPromptNeutronDistribution( false ),
        m_wantRawTNSL_distributionSampling( true ) {

}

/* *********************************************************************************************************//**
 * Class to store user defined preferences for creating an MCGIDI::Protare instance.
 *
 * @param a_pops                        [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_protare                     [in]    GIDI::Protare whose information is used to fill *this*.
 * @param a_label                       [in]    
 * @param a_delayedNeutrons             [in]    Sets whether delayed neutron data will be load or not if available.
 * @param a_energyDomainMax             [in]    The maximum projectile energy for which data should be loaded.
 ***********************************************************************************************************/

MCGIDI_HOST MC::MC( PoPI::Database const &a_pops, GIDI::Protare const &a_protare, std::string const &a_label, 
                GIDI::Transporting::DelayedNeutrons a_delayedNeutrons, double a_energyDomainMax ) :
        GIDI::Transporting::Settings( a_protare.projectile( ).ID( ), a_delayedNeutrons ),
        m_pops( a_pops ),
        m_neutronIndex( a_pops[PoPI::IDs::neutron] ),
        m_photonIndex( a_pops[PoPI::IDs::photon] ),
        m_electronIndex( a_pops[PoPI::IDs::electron] ),
        m_styles( &a_protare.styles( ) ),
        m_label( a_label ),
        m_energyDomainMax( a_energyDomainMax ),
        m_ignoreENDF_MT5( false ),
        m_sampleNonTransportingParticles( false ),
        m_useSlowerContinuousEnergyConversion( false ),
        m_crossSectionLookupMode( LookupMode::Data1d::continuousEnergy ),
        m_other1dDataLookupMode( LookupMode::Data1d::continuousEnergy ),
        m_distributionLookupMode( LookupMode::Distribution::pdf_cdf ),
        m_upscatterModel( Sampling::Upscatter::Model::none ),
        m_upscatterModelALabel( "" ),
        m_URR_mode( URR_mode::none ),
        m_wantTerrellPromptNeutronDistribution( false ),
        m_wantRawTNSL_distributionSampling( true ) {

}

/* *********************************************************************************************************//**
 * Sets the *m_crossSectionLookupMode* member of *this* to *a_crossSectionLookupMode*.
 *
 * @param a_crossSectionLookupMode      [in]    The *LookupMode::Data1d* data mode.
 ***********************************************************************************************************/

MCGIDI_HOST void MC::setCrossSectionLookupMode( LookupMode::Data1d a_crossSectionLookupMode ) {

    if( ( a_crossSectionLookupMode != LookupMode::Data1d::continuousEnergy ) && 
        ( a_crossSectionLookupMode != LookupMode::Data1d::multiGroup ) ) {
        throw( "Invalided cross section mode request." );
    }
    m_crossSectionLookupMode = a_crossSectionLookupMode;
}

/* *********************************************************************************************************//**
 * Sets the *m_other1dDataLookupMode* member of *this* to *a_other1dDataLookupMode*.
 *
 * @param a_other1dDataLookupMode       [in]    The *LookupMode::Data1d* data mode.
 ***********************************************************************************************************/

MCGIDI_HOST void MC::setOther1dDataLookupMode( LookupMode::Data1d a_other1dDataLookupMode ) {

    if( a_other1dDataLookupMode != LookupMode::Data1d::continuousEnergy ) throw( "Invalided other mode request." );
    m_other1dDataLookupMode = a_other1dDataLookupMode;
}

/* *********************************************************************************************************//**
 * Sets the *m_distributionLookupMode* member of *this* to *a_distributionLookupMode*.
 *
 * @param a_distributionLookupMode      [in]    The *LookupMode::Data1d* data mode.
 ***********************************************************************************************************/

MCGIDI_HOST void MC::setDistributionLookupMode( LookupMode::Distribution a_distributionLookupMode ) {

    if( a_distributionLookupMode != LookupMode::Distribution::pdf_cdf ) throw( "Invalided distribution mode request." );
    m_distributionLookupMode = a_distributionLookupMode;
}

/* *********************************************************************************************************//**
 * Sets the *m_upscatterModel* member of *this* to **Sampling::Upscatter::Model::A** and the *m_upscatterModelALabel* member
 * to *a_upscatterModelALabel*.
 *
 * @param a_upscatterModelALabel        [in]    The *LookupMode::Data1d* data mode.
 ***********************************************************************************************************/

MCGIDI_HOST void MC::set_upscatterModelA( std::string const &a_upscatterModelALabel ) {

    m_upscatterModel = Sampling::Upscatter::Model::A;
    m_upscatterModelALabel = a_upscatterModelALabel;
}

}

}

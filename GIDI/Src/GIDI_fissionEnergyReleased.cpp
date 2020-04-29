/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "GIDI.hpp"

namespace GIDI {

namespace Functions {

/*! \class FissionEnergyRelease
 * Class to store GNDS <**fissionEnergyRelease**> node.
 */

/* *********************************************************************************************************//**
 * @param a_construction        [in]    Used to pass user options for parsing.
 * @param a_node                [in]    The pugi::xml_node to be parsed to construct the instance.
 * @param a_parent              [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

FissionEnergyRelease::FissionEnergyRelease( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Function1dForm( a_construction, a_node, FormType::fissionEnergyRelease1d, a_parent ) {

    m_promptProductKE = data1dParse( a_construction, a_node.child( "promptProductKE" ).first_child( ), NULL );
    m_promptNeutronKE = data1dParse( a_construction, a_node.child( "promptNeutronKE" ).first_child( ), NULL );
    m_delayedNeutronKE = data1dParse( a_construction, a_node.child( "delayedNeutronKE" ).first_child( ), NULL );
    m_promptGammaEnergy = data1dParse( a_construction, a_node.child( "promptGammaEnergy" ).first_child( ), NULL );
    m_delayedGammaEnergy = data1dParse( a_construction, a_node.child( "delayedGammaEnergy" ).first_child( ), NULL );
    m_delayedBetaEnergy = data1dParse( a_construction, a_node.child( "delayedBetaEnergy" ).first_child( ), NULL );
    m_neutrinoEnergy = data1dParse( a_construction, a_node.child( "neutrinoEnergy" ).first_child( ), NULL );
    m_nonNeutrinoEnergy = data1dParse( a_construction, a_node.child( "nonNeutrinoEnergy" ).first_child( ), NULL );
    m_totalEnergy = data1dParse( a_construction, a_node.child( "totalEnergy" ).first_child( ), NULL );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

FissionEnergyRelease::~FissionEnergyRelease( ) {

    if( m_promptProductKE != NULL ) delete m_promptProductKE;
    if( m_promptNeutronKE != NULL ) delete m_promptNeutronKE;
    if( m_delayedNeutronKE != NULL ) delete m_delayedNeutronKE;
    if( m_promptGammaEnergy != NULL ) delete m_promptGammaEnergy;
    if( m_delayedGammaEnergy != NULL ) delete m_delayedGammaEnergy;
    if( m_delayedBetaEnergy != NULL ) delete m_delayedBetaEnergy;
    if( m_neutrinoEnergy != NULL ) delete m_neutrinoEnergy;
    if( m_nonNeutrinoEnergy != NULL ) delete m_nonNeutrinoEnergy;
    if( m_totalEnergy != NULL ) delete m_totalEnergy;
}

/* *********************************************************************************************************//**
 * Returns the multi-group Q-value.
 *
 * @param a_settings            [in]    Specifies the requested label.
 * @param a_temperatureInfo     [in]    Specifies the temperature and labels use to lookup the requested data.
 *
 * @return                              Multi-group Q-value.
 ***********************************************************************************************************/

Vector FissionEnergyRelease::multiGroupQ( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const {

    Vector vector( 0 );

    if( a_settings.delayedNeutrons( ) == Transporting::DelayedNeutrons::on ) {
        Gridded1d const *gridded1d = dynamic_cast<Gridded1d const *>( m_delayedNeutronKE );

        vector += gridded1d->data( );
        gridded1d = dynamic_cast<Gridded1d const *>( m_delayedGammaEnergy );
        vector += gridded1d->data( );
        gridded1d = dynamic_cast<Gridded1d const *>( m_delayedBetaEnergy );
        vector += gridded1d->data( );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void FissionEnergyRelease::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );

    a_writeInfo.addNodeStarter( a_indent, moniker( ), a_writeInfo.addAttribute( "label", label( ) ) );

    energyReleaseToXMLList( a_writeInfo, "promptProductKE", indent2, m_promptProductKE );
    energyReleaseToXMLList( a_writeInfo, "promptNeutronKE", indent2, m_promptNeutronKE );
    energyReleaseToXMLList( a_writeInfo, "delayedNeutronKE", indent2, m_delayedNeutronKE );
    energyReleaseToXMLList( a_writeInfo, "promptGammaEnergy", indent2, m_promptGammaEnergy );
    energyReleaseToXMLList( a_writeInfo, "delayedGammaEnergy", indent2, m_delayedGammaEnergy );
    energyReleaseToXMLList( a_writeInfo, "delayedBetaEnergy", indent2, m_delayedBetaEnergy );
    energyReleaseToXMLList( a_writeInfo, "neutrinoEnergy", indent2, m_neutrinoEnergy );
    energyReleaseToXMLList( a_writeInfo, "nonNeutrinoEnergy", indent2, m_nonNeutrinoEnergy );
    energyReleaseToXMLList( a_writeInfo, "totalEnergy", indent2, m_totalEnergy );

    a_writeInfo.addNodeEnder( moniker( ) );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_moniker           [in]        The moniker for the energy type.
 * @param       a_indent            [in]        The amount to indent *this* node.
 * @param       a_function          [in]        The component of the energy released in fission.
 ***********************************************************************************************************/

void FissionEnergyRelease::energyReleaseToXMLList( WriteInfo &a_writeInfo, std::string const &a_moniker, std::string const &a_indent, Function1dForm *a_function ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );

    if( a_function == NULL ) return;

    a_writeInfo.addNodeStarter( a_indent, a_moniker, "" );
    a_function->toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( a_moniker );
}

}               // End namespace Functions.

}               // End namespace GIDI.

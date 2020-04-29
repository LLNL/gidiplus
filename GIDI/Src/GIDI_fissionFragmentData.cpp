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

/*! \class FissionFragmentData
 * This class represents a **GNDS** fissionFragmentData.
*/

/* *********************************************************************************************************//**
 * Default constructor for FissionFragmentData.
 ***********************************************************************************************************/

FissionFragmentData::FissionFragmentData( ) :
        Ancestry( fissionFragmentDataMoniker ),
        m_delayedNeutrons( delayedNeutronsMoniker ),
        m_fissionEnergyReleases( fissionEnergyReleasesMoniker ) {

}

/* *********************************************************************************************************//**
 * Constructed from data in a <**outputChannel**> node.
 *
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The reaction pugi::xml_node to be parsed and used to construct the reaction.
 * @param a_pops            [in]    The *external* PoPI::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs    [in]    The *internal* PoPI::Database instance used to get particle indices and possibly other particle information.
 *                                  This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_styles          [in]    The <**styles**> node under the <**reactionSuite**> node.
 ***********************************************************************************************************/

FissionFragmentData::FissionFragmentData( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPI::Database const &a_pops, 
                PoPI::Database const &a_internalPoPs, Styles::Suite const *a_styles ) :
        Ancestry( a_node.name( ) ),
        m_delayedNeutrons( a_construction, delayedNeutronsMoniker, a_node, a_pops, a_internalPoPs, parseDelayedNeutronsSuite, a_styles ),
        m_fissionEnergyReleases( a_construction, fissionEnergyReleasesMoniker, a_node, a_pops, a_internalPoPs, parseFissionEnergyReleasesSuite, a_styles ) {

    m_delayedNeutrons.setAncestor( this );
    m_fissionEnergyReleases.setAncestor( this );

    for( std::size_t i1 = 0; i1 < m_delayedNeutrons.size( ); ++i1 ) {
        DelayedNeutron *delayedNeutron = m_delayedNeutrons.get<DelayedNeutron>( i1 );

        delayedNeutron->setDelayedNeutronIndex( i1 );
    }
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

FissionFragmentData::~FissionFragmentData( ) {

}

/* *********************************************************************************************************//**
 * Used by Ancestry to tranverse GNDS nodes. This method returns a pointer to a derived class' a_item member or NULL if none exists.
 *
 * @param a_item    [in]    The name of the class member whose pointer is to be return.
 * @return                  The pointer to the class member or NULL if class does not have a member named a_item.
 ***********************************************************************************************************/

Ancestry *FissionFragmentData::findInAncestry3( std::string const &a_item ) {

    if( a_item == delayedNeutronsMoniker ) return( &m_delayedNeutrons );
    if( a_item == fissionEnergyReleasesMoniker ) return( &m_fissionEnergyReleases );

    return( NULL );
}

/* *********************************************************************************************************//**
 * Used by Ancestry to tranverse GNDS nodes. This method returns a pointer to a derived class' a_item member or NULL if none exists.
 *
 * @param a_item    [in]    The name of the class member whose pointer is to be return.
 * @return                  The pointer to the class member or NULL if class does not have a member named a_item.
 ***********************************************************************************************************/

Ancestry const *FissionFragmentData::findInAncestry3( std::string const &a_item ) const {

    if( a_item == delayedNeutronsMoniker ) return( &m_delayedNeutrons );
    if( a_item == fissionEnergyReleasesMoniker ) return( &m_fissionEnergyReleases );

    return( NULL );
}

/* *********************************************************************************************************//**
 * Insert a std::set with the products id and any product in in its output channel.
 * If a_transportablesOnly is true, only transportable product indices are return.
 *
 * @param a_indices                 [out]   The unique list of product indices.
 * @param a_particles               [in]    The list of particles to be transported.
 * @param a_transportablesOnly      [in]    If true, only transportable product indices are added in the list.
 ***********************************************************************************************************/

void FissionFragmentData::productIDs( std::set<std::string> &a_indices, Transporting::Particles const &a_particles, bool a_transportablesOnly ) const {

    for( std::size_t index = 0; index < m_delayedNeutrons.size( ); ++index ) {
        DelayedNeutron const &delayedNeutrons1 = *m_delayedNeutrons.get<DelayedNeutron>( index );

        delayedNeutrons1.productIDs( a_indices, a_particles, a_transportablesOnly );
    }
}

/* *********************************************************************************************************//**
 * Returns the product multiplicity (e.g., 0, 1, 2, ...) or -1 if energy dependent or not an integer for particle with id *a_id*.
 * 
 * @param a_id;                     [in]    The id of the requested particle.
 *
 * @return                                  The multiplicity for the requested particle.
 ***********************************************************************************************************/
 
int FissionFragmentData::productMultiplicity( std::string const &a_id ) const {

    int total_multiplicity = 0;

    for( std::size_t index = 0; index < m_delayedNeutrons.size( ); ++index ) {
        DelayedNeutron const &delayedNeutrons1 = *m_delayedNeutrons.get<DelayedNeutron>( index );

        int multiplicity = delayedNeutrons1.productMultiplicity( a_id );
        if( multiplicity < 0 ) return( -1 );
        total_multiplicity += multiplicity;
    }

    return( total_multiplicity );
}

/* *********************************************************************************************************//**
 * Determines the maximum Legredre order present in the multi-group transfer matrix for the specified products of this output channel.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_temperatureInfo [in]    Specifies the temperature and labels use to lookup the requested data.
 * @param a_productID       [in]    Particle id of the requested product.
 *
 * @return                          The maximum Legredre order. If no transfer matrix data are present for the requested product, -1 is returned.
 ***********************************************************************************************************/

int FissionFragmentData::maximumLegendreOrder( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const {

    int _maximumLegendreOrder = -1;

    for( std::size_t index = 0; index < m_delayedNeutrons.size( ); ++index ) {
        DelayedNeutron const &delayedNeutrons1 = *m_delayedNeutrons.get<DelayedNeutron>( index );
        int r_maximumLegendreOrder = delayedNeutrons1.maximumLegendreOrder( a_settings, a_temperatureInfo, a_productID );

        if( r_maximumLegendreOrder > _maximumLegendreOrder ) _maximumLegendreOrder = r_maximumLegendreOrder;
    }

    return( _maximumLegendreOrder );
}

/* *********************************************************************************************************//**
 * Returns the sum of the multi-group multiplicity for the requested label for the request product of this output channel. 
 * This is a cross section weighted multiplicity.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_temperatureInfo [in]    Specifies the temperature and labels use to lookup the requested data.
 * @param a_productID       [in]    Particle id for the requested product.
 *
 * @return                          The requested multi-group multiplicity as a GIDI::Vector.
 ***********************************************************************************************************/

Vector FissionFragmentData::multiGroupMultiplicity( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const {

    Vector vector( 0 );

    if( a_settings.delayedNeutrons( ) != Transporting::DelayedNeutrons::on ) return( vector );

    for( std::size_t index = 0; index < m_delayedNeutrons.size( ); ++index ) {
        DelayedNeutron const &delayedNeutrons1 = *m_delayedNeutrons.get<DelayedNeutron>( index );

        vector += delayedNeutrons1.multiGroupMultiplicity( a_settings, a_temperatureInfo, a_productID );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the sum of the multi-group, Q for the requested label for the this output channel. This is a cross section weighted Q.
 * If a_final is false, only the Q for the output channels directly under each reaction is summed. Otherwise, the Q for all output channels
 * summed, including output channels for each products.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_temperatureInfo [in]    Specifies the temperature and labels use to lookup the requested data.
 * @param a_final           [in]    If true, the Q is calculated for all output channels, including those for products.
 *
 * @return                          The requested multi-group Q as a GIDI::Vector.
 ***********************************************************************************************************/

Vector FissionFragmentData::multiGroupQ( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, bool a_final ) const {

    Vector vector( 0 );

    if( a_settings.delayedNeutrons( ) != Transporting::DelayedNeutrons::on ) return( vector );

    if( m_fissionEnergyReleases.size( ) == 0 ) return( vector );

    Functions::FissionEnergyRelease const *form = dynamic_cast<Functions::FissionEnergyRelease const *>( a_settings.form( m_fissionEnergyReleases, a_temperatureInfo ) );

    vector += form->multiGroupQ( a_settings, a_temperatureInfo );

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, product matrix for the requested label for the requested product index for the requested Legendre order.
 * If no data are found, an empty GIDI::Matrix is returned.
 *
 * @param a_settings        [in]    Specifies the requested label and if delayed neutrons should be included.
 * @param a_temperatureInfo [in]    Specifies the temperature and labels use to lookup the requested data.
 * @param a_particles       [in]    The list of particles to be transported.
 * @param a_productID       [in]    Particle id for the requested product.
 * @param a_order           [in]    Requested product matrix, Legendre order.
 *
 * @return                          The requested multi-group product matrix as a GIDI::Matrix.
 ***********************************************************************************************************/

Matrix FissionFragmentData::multiGroupProductMatrix( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, std::string const &a_productID, int a_order ) const {

    Matrix matrix( 0, 0 );

    if( a_settings.delayedNeutrons( ) != Transporting::DelayedNeutrons::on ) return( matrix );

    for( std::size_t index = 0; index < m_delayedNeutrons.size( ); ++index ) {
        DelayedNeutron const &delayedNeutrons1 = *m_delayedNeutrons.get<DelayedNeutron>( index );

        matrix += delayedNeutrons1.multiGroupProductMatrix( a_settings, a_temperatureInfo, a_particles, a_productID, a_order );
    }

    return( matrix );
}

/* *********************************************************************************************************//**
 * Returns the sum of the multi-group, average energy for the requested label for the requested product. This is a cross section weighted average energy.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_temperatureInfo [in]    Specifies the temperature and labels use to lookup the requested data.
 * @param a_productID       [in]    Particle id for the requested product.
 *
 * @return                          The requested multi-group average energy as a GIDI::Vector.
 ***********************************************************************************************************/

Vector FissionFragmentData::multiGroupAverageEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const {

    Vector vector( 0 );

    if( a_settings.delayedNeutrons( ) != Transporting::DelayedNeutrons::on ) return( vector );

    for( std::size_t index = 0; index < m_delayedNeutrons.size( ); ++index ) {
        DelayedNeutron const &delayedNeutrons1 = *m_delayedNeutrons.get<DelayedNeutron>( index );

        vector += delayedNeutrons1.multiGroupAverageEnergy( a_settings, a_temperatureInfo, a_productID );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the sum of the multi-group, average momentum for the requested label for the requested product. This is a cross section weighted average momentum.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_temperatureInfo [in]    Specifies the temperature and labels use to lookup the requested data.
 * @param a_productID       [in]    Particle id for the requested product.
 *
 * @return                          The requested multi-group average momentum as a GIDI::Vector.
 ***********************************************************************************************************/

Vector FissionFragmentData::multiGroupAverageMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const {

    Vector vector( 0 );

    if( a_settings.delayedNeutrons( ) != Transporting::DelayedNeutrons::on ) return( vector );

    for( std::size_t index = 0; index < m_delayedNeutrons.size( ); ++index ) {
        DelayedNeutron const &delayedNeutrons1 = *m_delayedNeutrons.get<DelayedNeutron>( index );

        vector += delayedNeutrons1.multiGroupAverageMomentum( a_settings, a_temperatureInfo, a_productID );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void FissionFragmentData::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );

    if( moniker( ) == "" ) return;
    if( ( m_delayedNeutrons.size( ) == 0 ) && ( m_fissionEnergyReleases.size( ) == 0 ) ) return;
    a_writeInfo.addNodeStarter( a_indent, moniker( ), "" );
    m_delayedNeutrons.toXMLList( a_writeInfo, indent2 );
    m_fissionEnergyReleases.toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( moniker( ) );
}

}

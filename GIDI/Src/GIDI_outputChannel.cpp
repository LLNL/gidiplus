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

/*! \class OutputChannel
 * This class represents a **GNDS** outputChannel.
*/

OutputChannel::OutputChannel( bool a_twoBody, bool a_fissions, std::string a_process ) :
        Ancestry( outputChannelMoniker ),
        m_twoBody( a_twoBody ),
        m_fissions( a_fissions ),
        m_process( a_process ),
        m_Q( QMoniker ),
        m_products( productsMoniker ),
        m_fissionFragmentData( ) {

    m_Q.setAncestor( this );
    m_products.setAncestor( this );
    m_fissionFragmentData.setAncestor( this );
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
 * @param a_isFission       [in]    Boolean indicating if output channel is a fission channel (true) or not (false).
 ***********************************************************************************************************/

OutputChannel::OutputChannel( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPI::Database const &a_pops, 
                PoPI::Database const &a_internalPoPs, Styles::Suite const *a_styles, bool a_isFission ) :
        Ancestry( a_node.name( ) ),
        m_twoBody( std::string( a_node.attribute( "genre" ).value( ) ) == "twoBody" ),
        m_fissions( a_isFission ),
        m_process( std::string( a_node.attribute( "process" ).value( ) ) ),
        m_Q( a_construction, QMoniker, a_node, a_pops, a_internalPoPs, parseQSuite, a_styles ),
        m_products( a_construction, productsMoniker, a_node, a_pops, a_internalPoPs, parseProductSuite, a_styles ),
        m_fissionFragmentData( a_construction, a_node.child( fissionFragmentDataMoniker ), a_pops, a_internalPoPs, a_styles ) {

    m_Q.setAncestor( this );
    m_products.setAncestor( this );
    m_fissionFragmentData.setAncestor( this );

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

OutputChannel::~OutputChannel( ) {

}

/* *********************************************************************************************************//**
 * Returns the maximum product depth for this output channel.
 *
 * @return The maximum product depth.
 ***********************************************************************************************************/

int OutputChannel::depth( ) const {

    int _depth = 0;
    std::size_t size = m_products.size( );

    for( std::size_t index = 0; index < size; ++index ) {
        Product const &_product = *m_products.get<Product>( index );

        int productDepth = _product.depth( );
        if( productDepth > _depth ) _depth = productDepth;
    }
    return( _depth + 1 );
}

/* *********************************************************************************************************//**
 * Used by Ancestry to tranverse GNDS nodes. This method returns a pointer to a derived class' a_item member or NULL if none exists.
 *
 * @param a_item    [in]    The name of the class member whose pointer is to be return.
 * @return                  The pointer to the class member or NULL if class does not have a member named a_item.
 ***********************************************************************************************************/

Ancestry *OutputChannel::findInAncestry3( std::string const &a_item ) {

    if( a_item == QMoniker ) return( &m_Q );
    if( a_item == productsMoniker ) return( &m_products );
    if( a_item == fissionFragmentDataMoniker ) return( &m_fissionFragmentData );

    return( NULL );
}

/* *********************************************************************************************************//**
 * Used by Ancestry to tranverse GNDS nodes. This method returns a pointer to a derived class' a_item member or NULL if none exists.
 *
 * @param a_item    [in]    The name of the class member whose pointer is to be return.
 * @return                  The pointer to the class member or NULL if class does not have a member named a_item.
 ***********************************************************************************************************/

Ancestry const *OutputChannel::findInAncestry3( std::string const &a_item ) const {

    if( a_item == QMoniker ) return( &m_Q );
    if( a_item == productsMoniker ) return( &m_products );
    if( a_item == fissionFragmentDataMoniker ) return( &m_fissionFragmentData );

    return( NULL );
}

/* *********************************************************************************************************//**
 * Returns true if the product has an output channel and its output channel hasFission returns true, and false otherwise.
 *
 * @return  true if at least one output channel is a fission channel.
 ***********************************************************************************************************/

bool OutputChannel::hasFission( ) const {

    if( m_fissions ) return( true );

    std::size_t size = m_products.size( );
    for( std::size_t index = 0; index < size; ++index ) {
        Product const &_product = *m_products.get<Product>( index );

        if( _product.hasFission( ) ) return( true );
    }
    return( false );
}

/* *********************************************************************************************************//**
 * Insert a std::set with the products id and any product in in its output channel.
 * If a_transportablesOnly is true, only transportable product indices are return.
 *
 * @param a_indices                 [out]   The unique list of product indices.
 * @param a_particles               [in]    The list of particles to be transported.
 * @param a_transportablesOnly      [in]    If true, only transportable product indices are added in the list.
 ***********************************************************************************************************/

void OutputChannel::productIDs( std::set<std::string> &a_indices, Transporting::Particles const &a_particles, bool a_transportablesOnly ) const {

    std::size_t size = m_products.size( );

    for( std::size_t index = 0; index < size; ++index ) {
        Product const &_product = *m_products.get<Product>( index );

        _product.productIDs( a_indices, a_particles, a_transportablesOnly );
    }

    m_fissionFragmentData.productIDs( a_indices, a_particles, a_transportablesOnly );
}

/* *********************************************************************************************************//**
 * Returns the product multiplicity (e.g., 0, 1, 2, ...) or -1 if energy dependent or not an integer for particle with id *a_id*.
 *
 * @param a_id;                     [in]    The id of the requested particle.
 *
 * @return                                  The multiplicity for the requested particle.
 ***********************************************************************************************************/

int OutputChannel::productMultiplicity( std::string const &a_id ) const {

    int total_multiplicity = 0;
    std::size_t size = m_products.size( );

    for( std::size_t index = 0; index < size; ++index ) {
        Product const &product = *m_products.get<Product>( index );
        int multiplicity = product.productMultiplicity( a_id );

        if( multiplicity < 0 ) return( -1 );
        total_multiplicity += multiplicity;
    }

    int multiplicity = m_fissionFragmentData.productMultiplicity( a_id );
    if( multiplicity < 0 ) return( -1 );

    return( total_multiplicity + multiplicity );
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

int OutputChannel::maximumLegendreOrder( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const {

    std::size_t size = m_products.size( );
    int _maximumLegendreOrder = -1;

    for( std::size_t index = 0; index < size; ++index ) {
        Product const &_product = *m_products.get<Product>( index );
        int r_maximumLegendreOrder = _product.maximumLegendreOrder( a_settings, a_temperatureInfo, a_productID );

        if( r_maximumLegendreOrder > _maximumLegendreOrder ) _maximumLegendreOrder = r_maximumLegendreOrder;
    }

    int r_maximumLegendreOrder = m_fissionFragmentData.maximumLegendreOrder( a_settings, a_temperatureInfo, a_productID );
    if( r_maximumLegendreOrder > _maximumLegendreOrder ) _maximumLegendreOrder = r_maximumLegendreOrder;

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

Vector OutputChannel::multiGroupMultiplicity( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const {

    Vector vector( 0 );

    for( std::size_t index = 0; index < m_products.size( ); ++index ) {
        Product const &_product = *m_products.get<Product>( index );

        vector += _product.multiGroupMultiplicity( a_settings, a_temperatureInfo, a_productID );
    }

    vector += m_fissionFragmentData.multiGroupMultiplicity( a_settings, a_temperatureInfo, a_productID );

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

Vector OutputChannel::multiGroupQ( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, bool a_final ) const {

    Vector vector( 0 );

    Functions::Gridded1d const *form = dynamic_cast<Functions::Gridded1d const*>( a_settings.form( m_Q, a_temperatureInfo ) );

    vector += form->data( );

    if( a_final ) {
        for( std::size_t index = 0; index < m_products.size( ); ++index ) {
            Product const &product1 = *m_products.get<Product>( index );

            vector += product1.multiGroupQ( a_settings, a_temperatureInfo, a_final );
        }
    }

    vector += m_fissionFragmentData.multiGroupQ( a_settings, a_temperatureInfo, a_final );

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

Matrix OutputChannel::multiGroupProductMatrix( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, std::string const &a_productID, int a_order ) const {

    Matrix matrix( 0, 0 );

    for( std::size_t index = 0; index < m_products.size( ); ++index ) {
        Product const &_product = *m_products.get<Product>( index );

        matrix += _product.multiGroupProductMatrix( a_settings, a_temperatureInfo, a_particles, a_productID, a_order );
    }

    matrix += m_fissionFragmentData.multiGroupProductMatrix( a_settings, a_temperatureInfo, a_particles, a_productID, a_order ); 

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

Vector OutputChannel::multiGroupAverageEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const {

    Vector vector( 0 );

    for( std::size_t index = 0; index < m_products.size( ); ++index ) {
        Product const &_product = *m_products.get<Product>( index );

        vector += _product.multiGroupAverageEnergy( a_settings, a_temperatureInfo, a_productID );
    }

    vector += m_fissionFragmentData.multiGroupAverageEnergy( a_settings, a_temperatureInfo, a_productID ); 

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

Vector OutputChannel::multiGroupAverageMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const {

    Vector vector( 0 );

    for( std::size_t index = 0; index < m_products.size( ); ++index ) {
        Product const &_product = *m_products.get<Product>( index );

        vector += _product.multiGroupAverageMomentum( a_settings, a_temperatureInfo, a_productID );
    }

    vector += m_fissionFragmentData.multiGroupAverageMomentum( a_settings, a_temperatureInfo, a_productID ); 

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns, via arguments, the average energy and momentum, and gain for product with particle id *a_particleID*.
 *
 * @param a_particleID          [in]    The particle id of the product.
 * @param a_energy              [in]    The energy of the projectile.
 * @param a_productEnergy       [in]    The average energy of the product.
 * @param a_productMomentum     [in]    The average momentum of the product.
 * @param a_productGain         [in]    The gain of the product.
 ***********************************************************************************************************/

void OutputChannel::continuousEnergyProductData( std::string const &a_particleID, double a_energy, double &a_productEnergy, double &a_productMomentum, double &a_productGain ) const {

    for( std::size_t index = 0; index < m_products.size( ); ++index ) {
        Product const &product = *m_products.get<Product>( index );

        product.continuousEnergyProductData( a_particleID, a_energy, a_productEnergy, a_productMomentum, a_productGain );
    }
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *  
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/
 
void OutputChannel::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {
    
    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string attributes;

    if( m_twoBody ) {
        attributes = a_writeInfo.addAttribute( "genre", "twoBody" ); }
    else {
        attributes = a_writeInfo.addAttribute( "genre", "NBody" );
    }

    if( m_process != "" ) attributes += a_writeInfo.addAttribute( "process", m_process );

    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );

    m_Q.toXMLList( a_writeInfo, indent2 ); 
    m_products.toXMLList( a_writeInfo, indent2 );
    m_fissionFragmentData.toXMLList( a_writeInfo, indent2 );

    a_writeInfo.addNodeEnder( moniker( ) );
}

}

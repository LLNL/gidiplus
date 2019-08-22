/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>
#include "GIDI.hpp"

namespace GIDI {

/*! \class Product
 * Class to store a GNDS <**product**> node.
 */

/* *********************************************************************************************************//**
 * Constructed 
 ***********************************************************************************************************/

Product::Product( PoPs::Database const &a_pops, std::string const &a_productID, std::string const &a_label ) :
        Form( f_product ),
        m_particle( ParticleInfo( a_productID, a_pops, a_pops, true ) ),
        m_multiplicity( multiplicityMoniker ),
        m_distribution( distributionMoniker ),
        m_averageEnergy( averageEnergyMoniker ),
        m_averageMomentum( averageMomentumMoniker ),
        m_outputChannel( NULL ) {

    moniker( productMoniker );
    label( a_label );

    m_multiplicity.setAncestor( this );
    m_distribution.setAncestor( this );
    m_averageEnergy.setAncestor( this );
    m_averageMomentum.setAncestor( this );
}

/* *********************************************************************************************************//**
 * Constructed from data in a <**product**> node.
 *
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The pugi::xml_node to be parsed and used to construct the Product.
 * @param a_pops            [in]    The *external* PoPs::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs    [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                  This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_parent          [in]    The **m_products** member of GIDI::OutputChannel this product belongs to.
 * @param a_styles          [in]    The <**styles**> node under the <**reactionSuite**> node.
 ***********************************************************************************************************/

Product::Product( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, 
                PoPs::Database const &a_internalPoPs, Suite *a_parent, Styles::Suite const *a_styles ) :
        Form( a_node, f_product, a_parent ),
        m_particle( a_node.attribute( "pid" ).value( ), a_pops, a_internalPoPs, false ),
        m_productMultiplicity( 0 ),
        m_multiplicity( a_construction, multiplicityMoniker, a_node, a_pops, a_internalPoPs, parseMultiplicitySuite, a_styles ),
        m_distribution( a_construction, distributionMoniker, a_node, a_pops, a_internalPoPs, parseDistributionSuite, a_styles ),
        m_averageEnergy( a_construction, averageEnergyMoniker, a_node, a_pops, a_internalPoPs, parseAverageEnergySuite, a_styles ),
        m_averageMomentum( a_construction, averageMomentumMoniker, a_node, a_pops, a_internalPoPs, parseAverageMomentumSuite, a_styles ),
        m_outputChannel( NULL ) {

    m_multiplicity.setAncestor( this );
    m_distribution.setAncestor( this );
    m_averageEnergy.setAncestor( this );
    m_averageMomentum.setAncestor( this );

    pugi::xml_node const _outputChannel = a_node.child( "outputChannel" );
    if( _outputChannel.type( ) != pugi::node_null ) m_outputChannel = new OutputChannel( a_construction, _outputChannel, a_pops, a_internalPoPs, a_styles, false );

    if( m_outputChannel == NULL ) {
        if( m_multiplicity.size( ) > 0 ) {
            GIDI::Function1dForm const *function1d = m_multiplicity.get<GIDI::Function1dForm>( 0 );

            if( function1d->type( ) == f_constant1d ) {
                m_productMultiplicity = function1d->evaluate( 0.0 ); }
            else if( function1d->type( ) != f_unspecified1d ) {
                m_productMultiplicity = -1;
            }
        } }
    else {
        m_outputChannel->setAncestor( this );
    }
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Product::~Product( ) {

    if( m_outputChannel != NULL ) delete m_outputChannel;
}

/* *********************************************************************************************************//**
 * Returns the maximum product depth for this product.
 *
 * @return The maximum product depth.
 ***********************************************************************************************************/

int Product::depth( ) const {

    if( m_outputChannel == NULL ) return( 0 );
    return( m_outputChannel->depth( ) );
}

/* *********************************************************************************************************//**
 * Returns true if the product has an output channel and its output channel hasFission returns true, and false otherwise.
 *
 * @return  true if at least one output channel is a fission channel.
 ***********************************************************************************************************/

bool Product::hasFission( ) const {

    if( m_outputChannel != NULL ) return( m_outputChannel->hasFission( ) );
    return( false );
}

/* *********************************************************************************************************//**
 * Used by Ancestry to tranverse GNDS nodes. This method returns a pointer to a derived class' a_item member or NULL if none exists.
 *
 * @param a_item    [in]    The name of the class member whose pointer is to be return.
 * @return                  The pointer to the class member or NULL if class does not have a member named a_item.
 ***********************************************************************************************************/

Ancestry const *Product::findInAncestry3( std::string const &a_item ) const {

    if( a_item == multiplicityMoniker ) return( &m_multiplicity );
    if( a_item == distributionMoniker ) return( &m_distribution );
    if( a_item == averageEnergyMoniker ) return( &m_averageEnergy );
    if( a_item == averageMomentumMoniker ) return( &m_averageMomentum );
    if( a_item == outputChannelMoniker ) return( m_outputChannel );

    return( NULL );
}

/* *********************************************************************************************************//**
 * Insert a std::set with the products index and any product in in its output channel.
 * If a_transportablesOnly is true, only transportable product indices are return.
 *
 * @param a_indices                 [out]   The unique list of product indices.
 * @param a_particles               [in]    The list of particles to be transported.
 * @param a_transportablesOnly      [in]    If true, only transportable product indices are added in the list.
 ***********************************************************************************************************/

void Product::productIDs( std::set<std::string> &a_indices, Settings::Particles const &a_particles, bool a_transportablesOnly ) const {

    if( m_outputChannel == NULL ) {
        if( a_transportablesOnly && !a_particles.hasParticle( m_particle.ID( ) ) ) return;
        if( m_particle.ID( ) != "" ) a_indices.insert( m_particle.ID( ) ); }
    else {
        m_outputChannel->productIDs( a_indices, a_particles, a_transportablesOnly );
    }
}

/* *********************************************************************************************************//**
 * Returns the product multiplicity (e.g., 0, 1, 2, ...) or -1 if energy dependent or not an integer for particle with id *a_id*.
 *
 * @param a_id;                     [in]    The id of the requested particle.
 *
 * @return                                  The multiplicity for the requested particle.
 ***********************************************************************************************************/

int Product::productMultiplicity( std::string const &a_id ) const {

    if( m_outputChannel != NULL ) return( m_outputChannel->productMultiplicity( a_id ) );

    if( a_id == m_particle.ID( ) ) return( m_productMultiplicity );
    return( 0 );
}

/* *********************************************************************************************************//**
 * Determines the maximum Legredre order present in the multi-group transfer matrix for a this product and any sub-products for a give label.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_productID       [in]    Particle id for the requested product.
 * @return                          The maximum Legredre order. If no transfer matrix data are present for the requested product, -1 is returned.
 ***********************************************************************************************************/

int Product::maximumLegendreOrder( Settings::MG const &a_settings, std::string const &a_productID ) const {

    int _maximumLegendreOrder = -1;

    if( m_outputChannel == NULL ) {
        if( m_particle.ID( ) == a_productID ) {
            MultiGroup3d const &form = *m_distribution.getViaLineage<MultiGroup3d>( a_settings.label( ) );
            Gridded3d const &gdata = form.data( );
            Array3d const &data = gdata.data( );
            _maximumLegendreOrder = data.size( ) - 1;
        } }
    else {
        int __maximumLegendreOrder = m_outputChannel->maximumLegendreOrder( a_settings, a_productID );
        if( __maximumLegendreOrder > _maximumLegendreOrder ) _maximumLegendreOrder = __maximumLegendreOrder;
    }

    return( _maximumLegendreOrder );
}

/* *********************************************************************************************************//**
 * Returns the sum of the multi-group, multiplicity for the requested label for the this product and any sub-product. 
 * This is a cross section weighted multiplicity.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_particles       [in]    The list of particles to be transported.
 * @param a_productID       [in]    Particle id for the requested product.
 * @return                          The requested multi-group multiplicity as a GIDI::Vector.
 ***********************************************************************************************************/

Vector Product::multiGroupMultiplicity( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const {

    Vector _vector( 0 );

    if( m_outputChannel == NULL ) {
        if( m_particle.ID( ) == a_productID ) {
            Gridded1d const &form = *m_multiplicity.getViaLineage<Gridded1d>( a_settings.label( ) );
            _vector += form.data( );
        } }
    else {
        _vector += m_outputChannel->multiGroupMultiplicity( a_settings, a_particles, a_productID );
    }

    return( _vector );
}

/* *********************************************************************************************************//**
 * Returns the sum of the multi-group, Q for the requested label for the this product and any sub-product . This is a cross section weighted Q.
 * If a_final is false, only the Q for the products output channel is returned, otherwise, the Q for all output channels
 * summed, including output channels for each products.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_particles       [in]    The list of particles to be transported.
 * @param a_final           [in]    If true, the Q is calculated for all output channels, including those for products.
 * @return                          The requested multi-group Q as a GIDI::Vector.
 ***********************************************************************************************************/

Vector Product::multiGroupQ( Settings::MG const &a_settings, Settings::Particles const &a_particles, bool a_final ) const {
    
    Vector _vector( 0 );

    if( m_outputChannel != NULL ) _vector += m_outputChannel->multiGroupQ( a_settings, a_particles, a_final );

    return( _vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, product matrix for the requested label for the requested product index for the requested Legendre order.
 * If no data are found, an empty GIDI::Matrix is returned.
 *
 * @param a_settings        [in]    Specifies the requested label and if delayed neutrons should be included.
 * @param a_particles       [in]    The list of particles to be transported.
 * @param a_productID       [in]    Particle id for the requested product.
 * @param a_order           [in]    Requested product matrix, Legendre order.
 * @return                          The requested multi-group product matrix as a GIDI::Matrix.
 ***********************************************************************************************************/

Matrix Product::multiGroupProductMatrix( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID, int a_order ) const {

    Matrix matrix( 0, 0 );

    if( m_outputChannel == NULL ) {
        if( m_particle.ID( ) == a_productID ) {
            MultiGroup3d const &form = *m_distribution.getViaLineage<MultiGroup3d>( a_settings.label( ) );
            Gridded3d const &gdata = form.data( );
            Array3d const &data = gdata.data( );
            matrix = data.matrix( a_order );
        } }
    else {
        matrix += m_outputChannel->multiGroupProductMatrix( a_settings, a_particles, a_productID, a_order );
    }

    return( matrix );
}

/* *********************************************************************************************************//**
 * Returns the sum of the multi-group, average energy for the requested label for the requested product. This is a cross section weighted average energy
 * summed over this and all sub-products.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_particles       [in]    The list of particles to be transported.
 * @param a_productID       [in]    Particle id for the requested product.
 * @return                          The requested multi-group average energy as a GIDI::Vector.
 ***********************************************************************************************************/

Vector Product::multiGroupAverageEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const {

    Vector vector( 0 );

    if( m_outputChannel == NULL ) {
        if( m_particle.ID( ) == a_productID ) {
            Gridded1d const &form = *m_averageEnergy.getViaLineage<Gridded1d>( a_settings.label( ) );
            vector += form.data( );
        } }
    else {
        vector += m_outputChannel->multiGroupAverageEnergy( a_settings, a_particles, a_productID );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the sum of the multi-group, average momentum for the requested label for the requested product. This is a cross section weighted average momentum 
 * summed over this and all sub-products.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_particles       [in]    The list of particles to be transported.
 * @param a_productID       [in]    Particle id for the requested product.
 * @return                          The requested multi-group average momentum as a GIDI::Vector.
 ***********************************************************************************************************/

Vector Product::multiGroupAverageMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const {

    Vector _vector( 0 );

    if( m_outputChannel == NULL ) {
        if( m_particle.ID( ) == a_productID ) {
            Gridded1d const &form = *m_averageMomentum.getViaLineage<Gridded1d>( a_settings.label( ) );
            _vector += form.data( );
        } }
    else {
        _vector += m_outputChannel->multiGroupAverageMomentum( a_settings, a_particles, a_productID );
    }
    return( _vector );
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

void Product::continuousEnergyProductData( std::string const &a_particleID, double a_energy, double &a_productEnergy, double &a_productMomentum, double &a_productGain ) const {

    if( m_outputChannel == NULL ) {
        if( a_particleID == particle( ).ID( ) ) {
            a_productEnergy += averageEnergy( ).get<GIDI::Function1dForm>( 0 )->evaluate( a_energy );
            a_productMomentum += averageMomentum( ).get<GIDI::Function1dForm>( 0 )->evaluate( a_energy );
            a_productGain += multiplicity( ).get<GIDI::Function1dForm>( 0 )->evaluate( a_energy ); } }
    else {
        m_outputChannel->continuousEnergyProductData( a_particleID, a_energy, a_productEnergy, a_productMomentum, a_productGain );
    }
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void Product::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string attributes;

    attributes += a_writeInfo.addAttribute( "pid", m_particle.ID( ) );
    attributes += a_writeInfo.addAttribute( "label", label( ) );
    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );

    m_multiplicity.toXMLList( a_writeInfo, indent2 );
    m_distribution.toXMLList( a_writeInfo, indent2 );
    m_averageEnergy.toXMLList( a_writeInfo, indent2 );
    m_averageMomentum.toXMLList( a_writeInfo, indent2 );
    if( m_outputChannel != NULL ) m_outputChannel->toXMLList( a_writeInfo, indent2 );

    a_writeInfo.addNodeEnder(  moniker( ) );
}

}

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

/*! \class DelayedNeutron
 * This class represents a **GNDS** delayedNeutron.
*/

/* *********************************************************************************************************//**
 * Constructed from data in a <**outputChannel**> node.
 *
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The reaction pugi::xml_node to be parsed and used to construct the reaction.
 * @param a_pops            [in]    The *external* PoPs::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs    [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                  This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_parent          [in]    The parent GIDI::Suite.
 * @param a_styles          [in]    The <**styles**> node under the <**reactionSuite**> node.
 ***********************************************************************************************************/

DelayedNeutron::DelayedNeutron( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, 
                PoPs::Database const &a_internalPoPs, Suite *a_parent, Styles::Suite const *a_styles ) :
        Form( a_node, f_delayedNeutron, a_parent ),
        m_delayedNeutronIndex( 0 ),
        m_rate( a_construction, rateMoniker, a_node, a_pops, a_internalPoPs, parseRateSuite, a_styles ),
        m_product( a_construction, a_node.child( productMoniker ), a_pops, a_internalPoPs, NULL, a_styles ) {

    m_rate.setAncestor( this );
    m_product.setAncestor( this );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

DelayedNeutron::~DelayedNeutron( ) {

}

/* *********************************************************************************************************//**
 * Used by Ancestry to tranverse GNDS nodes. This method returns a pointer to a derived class' a_item member or NULL if none exists.
 *
 * @param a_item    [in]    The name of the class member whose pointer is to be return.
 * @return                  The pointer to the class member or NULL if class does not have a member named a_item.
 ***********************************************************************************************************/

Ancestry const *DelayedNeutron::findInAncestry3( std::string const &a_item ) const {

    if( a_item == rateMoniker ) return( &m_rate );
    if( a_item == productMoniker ) return( &m_product );

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

void DelayedNeutron::productIDs( std::set<std::string> &a_indices, Settings::Particles const &a_particles, bool a_transportablesOnly ) const {

    m_product.productIDs( a_indices, a_particles, a_transportablesOnly );
}

/* *********************************************************************************************************//**
 * Returns the product multiplicity (e.g., 0, 1, 2, ...) or -1 if energy dependent or not an integer for particle with id *a_id*.
 *
 * @param a_id;                     [in]    The id of the requested particle.
 *
 * @return                                  The multiplicity for the requested particle.
 ***********************************************************************************************************/

int DelayedNeutron::productMultiplicity( std::string const &a_id ) const {

    return( m_product.productMultiplicity( a_id ) );
}

/* *********************************************************************************************************//**
 * Determines the maximum Legredre order present in the multi-group transfer matrix for the specified products of this output channel.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_productID  x    [in]    Particle id of the requested product.
 * @return                          The maximum Legredre order. If no transfer matrix data are present for the requested product, -1 is returned.
 ***********************************************************************************************************/

int DelayedNeutron::maximumLegendreOrder( Settings::MG const &a_settings, std::string const &a_productID ) const {

    return( m_product.maximumLegendreOrder( a_settings, a_productID ) );
}

/* *********************************************************************************************************//**
 * Returns the sum of the multi-group multiplicity for the requested label for the request product of this output channel. 
 * This is a cross section weighted multiplicity.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_productID       [in]    Particle id for the requested product.
 * @param a_particles       [in]    The list of particles to be transported.
 * @return                          The requested multi-group multiplicity as a GIDI::Vector.
 ***********************************************************************************************************/

Vector DelayedNeutron::multiGroupMultiplicity( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const {

    return( m_product.multiGroupMultiplicity( a_settings, a_particles, a_productID ) );
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

Matrix DelayedNeutron::multiGroupProductMatrix( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID, int a_order ) const {

    return( m_product.multiGroupProductMatrix( a_settings, a_particles, a_productID, a_order ) );
}

/* *********************************************************************************************************//**
 * Returns the sum of the multi-group, average energy for the requested label for the requested product. This is a cross section weighted average energy.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_particles       [in]    The list of particles to be transported.
 * @param a_productID       [in]    Particle id for the requested product.
 * @return                          The requested multi-group average energy as a GIDI::Vector.
 ***********************************************************************************************************/

Vector DelayedNeutron::multiGroupAverageEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const {

    return( m_product.multiGroupAverageEnergy( a_settings, a_particles, a_productID ) );
}

/* *********************************************************************************************************//**
 * Returns the sum of the multi-group, average momentum for the requested label for the requested product. This is a cross section weighted average momentum.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_particles       [in]    The list of particles to be transported.
 * @param a_productID       [in]    Particle id for the requested product.
 * @return                          The requested multi-group average momentum as a GIDI::Vector.
 ***********************************************************************************************************/

Vector DelayedNeutron::multiGroupAverageMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const {

    return( m_product.multiGroupAverageMomentum( a_settings, a_particles, a_productID ) );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void DelayedNeutron::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {
    
    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    
    a_writeInfo.addNodeStarter( a_indent, moniker( ), a_writeInfo.addAttribute( "label", label( ) ) );
    m_rate.toXMLList( a_writeInfo, indent2 );
    m_product.toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( moniker( ) );
}

}

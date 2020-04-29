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

#define dateAttribute "date"
#define derivedFromAttribute "derivedFrom"
#define libraryAttribute "library"
#define versionAttribute "version"
#define temperatureAttribute "temperature"
#define projectileEnergyDomainNode "projectileEnergyDomain"
#define muCutoffAttribute "muCutoff"
#define lMaxAttribute "lMax"
#define parametersAttribute "parameters"
#define upperCalculatedGroupAttribute "upperCalculatedGroup"
#define fluxNode "flux"
#define inverseSpeedNode "inverseSpeed"
#define gridded1dNode "gridded1d"
#define gridNode "grid"

namespace Styles {

/*! \class Suite
 * This is essentially the GIDI::Suite class with the addition of the **findLabelInLineage** method.
 */

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Suite::Suite( ) :
        GIDI::Suite( stylesMoniker ) {

}

/* *********************************************************************************************************//**
 * Searches the Suite *a_suite* for a form with label *a_label* or, if not found, recursively ascends the **derivedFrom** until
 * a derived form is found. The *this* instance must be an <**styles**> node so that the **derivedFrom**s can be ascended. 
 * If no form is found, an empty string is returned.
 *
 * @param a_suite       [in]    The Suite, typically a component, whose forms are searched for a form with label *a_label* or one of its **derivedFrom**.
 * @param a_label       [in]    The label of the form to start the search.
 * @return                      The label of the form found or an empty string if none is found.
 ***********************************************************************************************************/

std::string const *Suite::findLabelInLineage( GIDI::Suite const &a_suite, std::string const &a_label ) const {

    std::string const *label = &a_label;
    Suite::const_iterator iter = a_suite.find( a_label );

    while( true ) {
        if( iter != a_suite.end( ) ) return( label );

        Base const *form = get<Base>( *label );
        form = form->getDerivedStyle( );
        label = &form->label( );
        if( *label == "" ) break;
        iter = a_suite.find( *label );        
    }

    return( label );
}

/* *********************************************************************************************************//**
 * Searches the styles in a GIDI::Styles::Suite instance for a **multiGroup** style. This method starts at the form with label *a_label* 
 * and ascends until a **multiGroup** style is found or no **multiGroup** style exists.
 *
 * @param a_label       [in]    The label for the style to start the search.
 * @return                      Pointer to the **multiGroup** style or NULL if one is not found.
 ***********************************************************************************************************/

MultiGroup const *Suite::multiGroup( std::string const &a_label ) const {

    std::string const *label( &a_label );

    while( true ) {
        Suite::const_iterator iter = this->find( *label );
        if( iter == this->end( ) ) return( NULL );
        Base const *base = static_cast<Base const *>( *iter );
        if( base->moniker( ) == multiGroupStyleMoniker ) return( static_cast<MultiGroup const *>( base ) );
        if( base->moniker( ) == SnElasticUpScatterStyleMoniker ) {
            label = &(base->derivedStyle( )); }
        else if( base->moniker( ) == heatedMultiGroupStyleMoniker ) {
            HeatedMultiGroup const *heatedMultiGroup = static_cast<HeatedMultiGroup const *>( base );
            label = &(heatedMultiGroup->parameters( )); }
        else {
            return( NULL );
        }
    }
}

/*! \class Base
 * This is the virtual base class inherited by all **style** classes. It handles the *date* and **derivedFrom** members.
 */

/* *********************************************************************************************************//**
 *
 * @param a_node        [in]    The **pugi::xml_node** to be parsed.
 * @param a_parent      [in]    The parent GIDI::Suite.
 * @return
 ***********************************************************************************************************/

Base::Base( pugi::xml_node const &a_node, GIDI::Suite *a_parent ) : 
        Form( a_node, FormType::style, a_parent ),
        m_date( a_node.attribute( dateAttribute ).value( ) ),
        m_derivedStyle( a_node.attribute( derivedFromAttribute ).value( ) ) {
}

/* *********************************************************************************************************//**
 * Returns a pointer to the **derivedFrom** style of *this*.
 *
 * @return          Pointer to the **derivedFrom** style of *this*.
 ***********************************************************************************************************/

Base const *Base::getDerivedStyle( ) const {

    Form const *_form( sibling( m_derivedStyle ) );

    return( dynamic_cast<Base const *>( _form ) );
}

/* *********************************************************************************************************//**
 * Starting at *this*'s **derivedFrom** style, and ascending as needed, returns the **derivedFrom** style whose moniker is *a_moniker*.
 *
 * @param a_moniker         [in]    The moniker to search for.
 * @return                          The style whose moniker is *a_moniker*.
 ***********************************************************************************************************/

Base const *Base::getDerivedStyle( std::string const &a_moniker ) const {

    Form const *_form( sibling( m_derivedStyle ) );
    Base const *_style = dynamic_cast<Base const *>( _form );

    if( _style == NULL ) return( _style );
    if( _style->moniker( ) != a_moniker ) _style = _style->getDerivedStyle( a_moniker );
    return( _style );
}

/* *********************************************************************************************************//**
 * Returns the base attributes for *this* as a *std::string* instance.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 *
 * @return                                      The base attributes as a XML attribute string.
 ***********************************************************************************************************/

std::string Base::baseXMLAttributes( WriteInfo &a_writeInfo ) const {

    std::string attributes( a_writeInfo.addAttribute( "label", label( ) ) );

    if( m_derivedStyle != "" ) attributes += a_writeInfo.addAttribute( "derivedFrom", m_derivedStyle );
    attributes += a_writeInfo.addAttribute( "date", m_date );

    return( attributes );
}

/*! \class Evaluated
 * This is the **evaluated** style class.
 */

/* *********************************************************************************************************//**
 *
 * @param a_node        [in]    The **pugi::xml_node** to be parsed.
 * @param a_parent      [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

Evaluated::Evaluated( pugi::xml_node const &a_node, GIDI::Suite *a_parent ) :
        Base( a_node, a_parent ),
        m_library( a_node.attribute( libraryAttribute ).value( ) ),
        m_version( a_node.attribute( versionAttribute ).value( ) ),
        m_temperature( a_node.child( temperatureAttribute ) ),
        m_projectileEnergyDomain( a_node.child( projectileEnergyDomainNode ) ) {

}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void Evaluated::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string attributes = baseXMLAttributes( a_writeInfo );

    attributes += a_writeInfo.addAttribute( "library", m_library );
    attributes += a_writeInfo.addAttribute( "version", m_version );
    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );
    
    m_temperature.toXMLList( a_writeInfo, indent2 );
    m_projectileEnergyDomain.toXMLList( a_writeInfo, indent2 );
    
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class CrossSectionReconstructed
 * This is the **crossSectionReconstructed** style class.
 */

/* *********************************************************************************************************//**
 *
 * @param a_node        [in]    The **pugi::xml_node** to be parsed.
 * @param a_parent      [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

CrossSectionReconstructed::CrossSectionReconstructed( pugi::xml_node const &a_node, GIDI::Suite *a_parent ) :
        Base( a_node, a_parent ) {
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void CrossSectionReconstructed::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    a_writeInfo.addNodeStarter( a_indent, moniker( ), baseXMLAttributes( a_writeInfo ) );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/* *********************************************************************************************************//**
 * Ascends the **derivedFrom** styles until a temperature is found.
 *
 * @return          Returns the temperature associated with this style.
 ***********************************************************************************************************/

PhysicalQuantity const &CrossSectionReconstructed::temperature( ) const {

    Base const *style = getDerivedStyle( );

    if( style == NULL ) throw Exception( "No style with temperature." );
    return( style->temperature( ) );
}

/*! \class CoulombPlusNuclearElasticMuCutoff
 * This is the **CoulombPlusNuclearElasticMuCutoff** style class.
 */

/* *********************************************************************************************************//**
 *
 * @param a_node        [in]    The **pugi::xml_node** to be parsed.
 * @param a_parent      [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

CoulombPlusNuclearElasticMuCutoff::CoulombPlusNuclearElasticMuCutoff( pugi::xml_node const &a_node, GIDI::Suite *a_parent ) :
        Base( a_node, a_parent ),
        m_muCutoff( a_node.attribute( muCutoffAttribute ).as_double( ) ) {

}

/* *********************************************************************************************************//**
 * Ascends the **derivedFrom** styles until a temperature is found.
 *
 * @return          Returns the temperature associated with this style.
 ***********************************************************************************************************/

PhysicalQuantity const &CoulombPlusNuclearElasticMuCutoff::temperature( ) const {

    Base const *style = getDerivedStyle( );

    if( style == NULL ) throw Exception( "No style with temperature." );
    return( style->temperature( ) );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void CoulombPlusNuclearElasticMuCutoff::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string attributes = baseXMLAttributes( a_writeInfo );

    attributes += a_writeInfo.addAttribute( "muCutoff", doubleToShortestString( m_muCutoff ) );
    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class AverageProductData
 * This is the **averageProductData** style class.
 */

/* *********************************************************************************************************//**
 *
 * @param a_node        [in]    The **pugi::xml_node** to be parsed.
 * @param a_parent      [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

AverageProductData::AverageProductData( pugi::xml_node const &a_node, GIDI::Suite *a_parent ) :
        Base( a_node, a_parent ) {

}

/* *********************************************************************************************************//**
 * Ascends the **derivedFrom** styles until a temperature is found.
 *
 * @return          Returns the temperature associated with this style.
 ***********************************************************************************************************/

PhysicalQuantity const &AverageProductData::temperature( ) const {

    Base const *style = getDerivedStyle( );

    if( style == NULL ) throw Exception( "No style with temperature." );
    return( style->temperature( ) );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void AverageProductData::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {
    
    a_writeInfo.addNodeStarter( a_indent, moniker( ), baseXMLAttributes( a_writeInfo ) );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class TNSL
 * This is the **heated** style class.
 */

/* *********************************************************************************************************//**
 *
 * @param a_node        [in]    The **pugi::xml_node** to be parsed.
 * @param a_parent      [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

TNSL::TNSL( pugi::xml_node const &a_node, GIDI::Suite *a_parent ) :
        Base( a_node, a_parent ),
        m_temperature( a_node.child( temperatureAttribute ) ) {
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void TNSL::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );

    a_writeInfo.addNodeStarter( a_indent, moniker( ), baseXMLAttributes( a_writeInfo ) );
    m_temperature.toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class Heated
 * This is the **heated** style class.
 */

/* *********************************************************************************************************//**
 *
 * @param a_node        [in]    The **pugi::xml_node** to be parsed.
 * @param a_parent      [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

Heated::Heated( pugi::xml_node const &a_node, GIDI::Suite *a_parent ) :
        Base( a_node, a_parent ),
        m_temperature( a_node.child( temperatureAttribute ) ) {
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void Heated::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );

    a_writeInfo.addNodeStarter( a_indent, moniker( ), baseXMLAttributes( a_writeInfo ) );
    m_temperature.toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class MonteCarlo_cdf
 * This is the **MonteCarlo_cdf** style class.
 */

/* *********************************************************************************************************//**
 *
 * @param a_node        [in]    The **pugi::xml_node** to be parsed.
 * @param a_parent      [in]    The parent GIDI::Suite.
 * @return
 ***********************************************************************************************************/

MonteCarlo_cdf::MonteCarlo_cdf( pugi::xml_node const &a_node, GIDI::Suite *a_parent ) :
        Base( a_node, a_parent ) {

}

/* *********************************************************************************************************//**
 * Ascends the **derivedFrom** styles until a temperature is found.
 *
 * @return          Returns the temperature associated with this style.
 ***********************************************************************************************************/

PhysicalQuantity const &MonteCarlo_cdf::temperature( ) const {

    Base const *style = getDerivedStyle( );

    if( style == NULL ) throw Exception( "No style with temperature." );
    return( style->temperature( ) );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void MonteCarlo_cdf::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {
    
    a_writeInfo.addNodeStarter( a_indent, moniker( ), baseXMLAttributes( a_writeInfo ) );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class MultiGroup
 * This is the **multiGroup** style class.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops            [in]    A PoPI::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs    [in]    The *internal* PoPI::Database instance used to get particle indices and possibly other particle information.
 *                                  This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

MultiGroup::MultiGroup( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs, GIDI::Suite *a_parent ) : 
        Base( a_node, a_parent ),
        m_maximumLegendreOrder( a_node.attribute( lMaxAttribute ).as_int( ) ),
        m_transportables( a_construction, transportablesMoniker, a_node, a_pops, a_internalPoPs, parseTransportablesSuite, NULL ) {

    m_transportables.setAncestor( this );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

MultiGroup::~MultiGroup( ) {

}

/* *********************************************************************************************************//**
 * Ascends the **derivedFrom** styles until a temperature is found.
 *
 * @return          Returns the temperature associated with this style.
 ***********************************************************************************************************/
 
PhysicalQuantity const &MultiGroup::temperature( ) const {

    Base const *style = getDerivedStyle( );

    if( style == NULL ) throw Exception( "No style with temperature." );
    return( style->temperature( ) );
}

/* *********************************************************************************************************//**
 * Returns the multi-group boundaries for the product with index *a_productID*.
 *
 * @param a_productID           [in]    Particle id for the requested product.
 * @return                              The multi-group boundaries.
 ***********************************************************************************************************/

std::vector<double> const &MultiGroup::groupBoundaries( std::string const &a_productID ) const {

    for( std::size_t index = 0; index < m_transportables.size( ); ++index ) {
        Transportable const &_transportable = *m_transportables.get<Transportable>( index );

        if( _transportable.pid( ) == a_productID ) {
            return( _transportable.groupBoundaries( ) );
        }
    }
    throw Exception( "MultiGroup::groupBoundaries: product index not found" );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void MultiGroup::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string attributes = baseXMLAttributes( a_writeInfo );

    attributes += a_writeInfo.addAttribute( "lMax", intToString( m_maximumLegendreOrder ) );
    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );
    m_transportables.toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class HeatedMultiGroup
 * This is the **neatedMultiGroup** style class.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops            [in]    A PoPI::Database instance used to get particle indices and possibly other particle information.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

HeatedMultiGroup::HeatedMultiGroup( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPI::Database const &a_pops, GIDI::Suite *a_parent ) : 
        Base( a_node, a_parent ),
        m_parameters( a_node.attribute( parametersAttribute ).value( ) ),
        m_flux( a_construction, a_node.child( fluxNode ) ),
        m_inverseSpeed( a_construction, a_node.child( inverseSpeedNode ).child( gridded1dNode ), NULL ) {

        m_flux.setAncestor( this );
        m_inverseSpeed.setAncestor( this );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

HeatedMultiGroup::~HeatedMultiGroup( ) {

}

/* *********************************************************************************************************//**
 * Returns the **multiGroup** style for *this* **heatedMultiGroup** style.
 *
 * @return          The **multiGroup** style.
 ***********************************************************************************************************/

MultiGroup const &HeatedMultiGroup::multiGroup( ) const {

    Form const *form( sibling( m_parameters ) );
    while( form == NULL ) throw Exception( "parameters style not found." );

    MultiGroup const *_multiGroup = dynamic_cast<MultiGroup const *>( form );
    if( _multiGroup == NULL ) throw Exception( "parameter not a multiGroup style not found." );

    return( *_multiGroup );
}

/* *********************************************************************************************************//**
 * Returns the maximum Legendre order used to process *this* **heatedMultiGroup** style.
 *
 * @return          The maximum Legendre order used to process.
 ***********************************************************************************************************/

int HeatedMultiGroup::maximumLegendreOrder( ) const {

    return( multiGroup( ).maximumLegendreOrder( ) );
}

/* *********************************************************************************************************//**
 * Ascends the **derivedFrom** styles until a temperature is found.
 *
 * @return          Returns the temperature associated with this style.
 ***********************************************************************************************************/

PhysicalQuantity const &HeatedMultiGroup::temperature( ) const {

    Base const *style = getDerivedStyle( );

    if( style == NULL ) throw Exception( "No style with temperature." );
    return( style->temperature( ) );
}

/* *********************************************************************************************************//**
 * Returns the multi-group boundaries for the product with index *a_productID* used for processing this **heatedMultiGroup**.
 *
 * @param a_productID           [in]    Particle id for the requested product.
 * @return                              The multi-group boundaries.
 ***********************************************************************************************************/

std::vector<double> const &HeatedMultiGroup::groupBoundaries( std::string const &a_productID ) const {

    return( multiGroup( ).groupBoundaries( a_productID ) );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void HeatedMultiGroup::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {
    
    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string indent3 = a_writeInfo.incrementalIndent( indent2 );
    std::string attributes = baseXMLAttributes( a_writeInfo );
    
    attributes += a_writeInfo.addAttribute( "parameters", m_parameters );
    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );
    m_flux.toXMLList( a_writeInfo, indent2 );

    a_writeInfo.addNodeStarter( indent2, "inverseSpeed", "" );
    m_inverseSpeed.toXMLList_func( a_writeInfo, indent3, false, false );
    a_writeInfo.addNodeEnder( "inverseSpeed" );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class SnElasticUpScatter
 * This is the **SnElasticUpScatter** style class.
 */

/* *********************************************************************************************************//**
 *
 * @param a_node        [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops        [in]    A PoPI::Database instance used to get particle indices and possibly other particle information.
 * @param a_parent      [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

SnElasticUpScatter::SnElasticUpScatter( pugi::xml_node const &a_node, PoPI::Database const &a_pops, GIDI::Suite *a_parent ) :
        Base( a_node, a_parent ),
        m_upperCalculatedGroup( a_node.attribute( upperCalculatedGroupAttribute ).as_int( ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

SnElasticUpScatter::~SnElasticUpScatter( ) {

}

/* *********************************************************************************************************//**
 * Ascends the **derivedFrom** styles until a temperature is found.
 *
 * @return          Returns the temperature associated with this style.
 ***********************************************************************************************************/

PhysicalQuantity const &SnElasticUpScatter::temperature( ) const {

    Base const *style = getDerivedStyle( );

    if( style == NULL ) throw Exception( "No style with temperature." );
    return( style->temperature( ) );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void SnElasticUpScatter::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {
    
    std::string attributes = baseXMLAttributes( a_writeInfo );
    
    attributes += a_writeInfo.addAttribute( "upperCalculatedGroup", intToString( m_upperCalculatedGroup ) );
    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class GriddedCrossSection
 * This is the **griddedCrossSection** style class.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops            [in]    A PoPI::Database instance used to get particle indices and possibly other particle information.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

GriddedCrossSection::GriddedCrossSection( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPI::Database const &a_pops, GIDI::Suite *a_parent ) :
        Base( a_node, a_parent ),
        m_grid( a_node.child( gridNode ), a_construction.useSystem_strtod( ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

GriddedCrossSection::~GriddedCrossSection( ) {

}

/* *********************************************************************************************************//**
 * Ascends the **derivedFrom** styles until a temperature is found.
 *
 * @return          Returns the temperature associated with this style.
 ***********************************************************************************************************/

PhysicalQuantity const &GriddedCrossSection::temperature( ) const {

    Base const *style = getDerivedStyle( );

    if( style == NULL ) throw Exception( "No style with temperature." );
    return( style->temperature( ) );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void GriddedCrossSection::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {
    
    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    
    a_writeInfo.addNodeStarter( a_indent, moniker( ), baseXMLAttributes( a_writeInfo ) );
    m_grid.toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class URR_probabilityTables
 * This is the **URR_probabilityTables** style class.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops            [in]    A PoPI::Database instance used to get particle indices and possibly other particle information.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

URR_probabilityTables::URR_probabilityTables( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPI::Database const &a_pops, GIDI::Suite *a_parent ) :
        Base( a_node, a_parent ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

URR_probabilityTables::~URR_probabilityTables( ) {

}

/* *********************************************************************************************************//**
 * Ascends the **derivedFrom** styles until a temperature is found.
 *
 * @return          Returns the temperature associated with this style.
 ***********************************************************************************************************/

PhysicalQuantity const &URR_probabilityTables::temperature( ) const {

    Base const *style = getDerivedStyle( );

    if( style == NULL ) throw Exception( "No style with temperature." );
    return( style->temperature( ) );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void URR_probabilityTables::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    a_writeInfo.addNodeStarterEnder( a_indent, moniker( ), baseXMLAttributes( a_writeInfo ) );
}

/*! \class TemperatureInfo
 * This class stores the labels for a given temperature for the **heatedCrossSection**, **griddedCrossSection**, **heatedMultiGroup** and
 * **SnElasticUpScatter** styles. If no style of a given process (e.g., **heatedCrossSection**) type exists, its label is an empty string.
 */

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

TemperatureInfo::TemperatureInfo( ) :
        m_temperature( -1.0, "K" ),
        m_heatedCrossSection( "" ),
        m_griddedCrossSection( "" ),
        m_URR_probabilityTables( "" ),
        m_heatedMultiGroup( "" ),
        m_SnElasticUpScatter( "" ) {

}

/* *********************************************************************************************************//**
 *
 * @param a_temperature             [in]    The temperature.
 * @param a_heatedCrossSection      [in]    The label for the **heatedCrossSection** style.
 * @param a_griddedCrossSection     [in]    The label for the **griddedCrossSection** style.
 * @param a_heatedMultiGroup        [in]    The label for the **heatedMultiGroup** style.
 * @param a_URR_probabilityTables   [in]    The label for the **URR_probabilityTables** style.
 * @param a_SnElasticUpScatter      [in]    The label for the **SnElasticUpScatter** style.
 ***********************************************************************************************************/

TemperatureInfo::TemperatureInfo( PhysicalQuantity const &a_temperature, std::string const &a_heatedCrossSection, std::string const &a_griddedCrossSection,
                std::string const &a_URR_probabilityTables, std::string const &a_heatedMultiGroup, std::string const &a_SnElasticUpScatter ) :
        m_temperature( a_temperature ),
        m_heatedCrossSection( a_heatedCrossSection ),
        m_griddedCrossSection( a_griddedCrossSection ),
        m_URR_probabilityTables( a_URR_probabilityTables ),
        m_heatedMultiGroup( a_heatedMultiGroup ),
        m_SnElasticUpScatter( a_SnElasticUpScatter ) {

}

/* *********************************************************************************************************//**
 * Prints information about *this* to std::cout.
 ***********************************************************************************************************/

void TemperatureInfo::print( ) const {

    std::cout << "temperature = " << m_temperature.value( ) << " " << m_temperature.unit( ) << " heatedCrossSection = '" << m_heatedCrossSection
            << "' griddedCrossSection = '" << m_griddedCrossSection << "' URR_probabilityTables = '" << m_URR_probabilityTables 
            << "' heatedMultiGroup = '" << m_heatedMultiGroup << "' SnElasticUpScatter = '" << m_SnElasticUpScatter << std::endl;
}

}

}

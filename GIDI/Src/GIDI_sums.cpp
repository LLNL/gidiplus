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

namespace Sums {

/*! \class Sums
 * This class represents the **GNDS** <**sums**> node.
 */

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Sums::Sums( ) :
        Ancestry( sumsMoniker ),
        m_crossSections( sumsCrossSectionsMoniker ),
        m_multiplicities( sumsMultiplicitiesMoniker ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Sums::~Sums( ) {

}

/* *********************************************************************************************************//**
 * The Sums method to parse its sub-nodes.
 *
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops            [in]    A PoPI::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs    [in]    The *internal* PoPI::Database instance used to get particle indices and possibly other particle information.
 *                                  This is the <**PoPs**> node under the <**reactionSuite**> node.
 ***********************************************************************************************************/

void Sums::parse( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs ) {

    m_crossSections.parse( a_construction, a_node.child( sumsCrossSectionsMoniker ), a_pops, a_internalPoPs, parseSumsCrossSectionsSuite, NULL );
    m_multiplicities.parse( a_construction, a_node.child( sumsMultiplicitiesMoniker ), a_pops, a_internalPoPs, parseSumsMultiplicitiesSuite, NULL );

    m_crossSections.setAncestor( this );
    m_multiplicities.setAncestor( this );
}

/* *********************************************************************************************************//**
 * Returns a pointer to the member whose moniker is *a_item*.
 *
 * @param a_item            [in]    The moniker of the member to return.
 * @return                          Returns the pointer to the member of NULL if it does not exists.
 ***********************************************************************************************************/

Ancestry *Sums::findInAncestry3( std::string const &a_item ) {

    if( a_item == sumsCrossSectionsMoniker ) return( &m_crossSections );
    if( a_item == sumsMultiplicitiesMoniker ) return( &m_multiplicities );

    return( NULL );
}

/* *********************************************************************************************************//**
 * Returns a pointer to the member whose moniker is *a_item*.
 *
 * @param a_item            [in]    The moniker of the member to return.
 * @return                          Returns the pointer to the member of NULL if it does not exists.
 ***********************************************************************************************************/

Ancestry const *Sums::findInAncestry3( std::string const &a_item ) const {

    if( a_item == sumsCrossSectionsMoniker ) return( &m_crossSections );
    if( a_item == sumsMultiplicitiesMoniker ) return( &m_multiplicities );

    return( NULL );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void Sums::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );

    a_writeInfo.addNodeStarter( a_indent, moniker( ), "" );
    m_crossSections.toXMLList( a_writeInfo, indent2 );
    m_multiplicities.toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class Base
 * Base class for Sums sub-node.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops            [in]    A PoPI::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs    [in]    The *internal* PoPI::Database instance used to get particle indices and possibly other particle information.
 *                                  This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_type            [in]    Type for the node.
 ***********************************************************************************************************/

Base::Base( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs,
                FormType a_type ) :
        Form( a_node, a_type ),
        m_ENDF_MT( a_node.attribute( "ENDF_MT" ).as_int( ) ),
        m_summands( a_construction, a_node.child( sumsSummandsMoniker ) ) {

        m_summands.setAncestor( this );
}

/*! \class CrossSectionSum
 * This class represents the **GNDS** <**crossSectionSum**> node.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops            [in]    A PoPI::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs    [in]    The *internal* PoPI::Database instance used to get particle indices and possibly other particle information.
 *                                  This is the <**PoPs**> node under the <**reactionSuite**> node.
 ***********************************************************************************************************/

CrossSectionSum::CrossSectionSum( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs ) :
        Base( a_construction, a_node, a_pops, a_internalPoPs, FormType::crossSectionSum ),
        m_Q( a_construction, QMoniker, a_node, a_pops, a_internalPoPs, parseQSuite, NULL ),
        m_crossSection( a_construction, crossSectionMoniker, a_node, a_pops, a_internalPoPs, parseCrossSectionSuite, NULL ) {

    m_Q.setAncestor( this );
    m_crossSection.setAncestor( this );
}

/* *********************************************************************************************************//**
 * Returns a pointer to the member whose moniker is *a_item*.
 *
 * @param a_item            [in]    The moniker of the member to return.
 * @return                          Returns the pointer to the member of NULL if it does not exists.
 ***********************************************************************************************************/

Ancestry *CrossSectionSum::findInAncestry3( std::string const &a_item ) {

    if( a_item == QMoniker ) return( &m_Q );
    if( a_item == crossSectionMoniker ) return( &m_crossSection );

    return( NULL );
}

/* *********************************************************************************************************//**
 * Returns a pointer to the member whose moniker is *a_item*.
 *
 * @param a_item            [in]    The moniker of the member to return.
 * @return                          Returns the pointer to the member of NULL if it does not exists.
 ***********************************************************************************************************/

Ancestry const *CrossSectionSum::findInAncestry3( std::string const &a_item ) const {

    if( a_item == QMoniker ) return( &m_Q );
    if( a_item == crossSectionMoniker ) return( &m_crossSection );

    return( NULL );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void CrossSectionSum::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string attributes;

    attributes  = a_writeInfo.addAttribute( "label", label( ) );
    attributes += a_writeInfo.addAttribute( "ENDF_MT", intToString( ENDF_MT( ) ) );
    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );

    summands( ).toXMLList( a_writeInfo, indent2 );

    m_Q.toXMLList( a_writeInfo, indent2 );
    m_crossSection.toXMLList( a_writeInfo, indent2 );

    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class MultiplicitySum
 * This class represents the **GNDS** <**multiplicitySum**> node.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops            [in]    A PoPI::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs    [in]    The *internal* PoPI::Database instance used to get particle indices and possibly other particle information.
 *                                  This is the <**PoPs**> node under the <**reactionSuite**> node.
 ***********************************************************************************************************/

MultiplicitySum::MultiplicitySum( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs ) :
        Base( a_construction, a_node, a_pops, a_internalPoPs, FormType::multiplicitySum ),
        m_multiplicity( a_construction, multiplicityMoniker, a_node, a_pops, a_internalPoPs, parseMultiplicitySuite, NULL ) {

    m_multiplicity.setAncestor( this );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void MultiplicitySum::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {
 
    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string attributes;         
 
    attributes  = a_writeInfo.addAttribute( "label", label( ) );
    attributes += a_writeInfo.addAttribute( "ENDF_MT", intToString( ENDF_MT( ) ) );
    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );

    summands( ).toXMLList( a_writeInfo, indent2 );

    m_multiplicity.toXMLList( a_writeInfo, indent2 );

    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class Summands
 * This class represents the **GNDS** <**summands**> node.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 ***********************************************************************************************************/

Summands::Summands( Construction::Settings const &a_construction, pugi::xml_node const &a_node ) :
        Form( a_node, FormType::summands ) {

    for( pugi::xml_node child = a_node.first_child( ); child; child = child.next_sibling( ) ) {
        std::string name( child.name( ) );

        if( name == sumsAddMoniker ) {
            Summand::Add *add = new Summand::Add( a_construction, child );

            add->setAncestor( this );
            m_summands.push_back( add ); }
        else {
            std::cout << "Sums::Summand::Base: Ignoring unsupported Form '" << name << "'." << std::endl;
        }
    }
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Summands::~Summands( ) {

    for( std::vector<Summand::Base *>::iterator iter = m_summands.begin( ); iter < m_summands.end( ); ++iter ) delete *iter;
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void Summands::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );

    a_writeInfo.addNodeStarter( a_indent, moniker( ), "" );
    for( std::vector<Summand::Base *>::const_iterator iter = m_summands.begin( ); iter != m_summands.end( ); ++iter ) (*iter)->toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( moniker( ) );
}

namespace Summand {

/*! \class Base
 * Base class inherited by sub-nodes of Summands.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 ***********************************************************************************************************/

Base::Base( Construction::Settings const &a_construction, pugi::xml_node const &a_node ) :
        Ancestry( a_node.name( ) ),
        m_href( a_node.attribute( "href" ).value( ) ) {

}
/*
=========================================================
 *
 * @return
 */
Base::~Base( ) {

}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void Base::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string attributes = a_writeInfo.addAttribute( "href", href( ) );

    a_writeInfo.addNodeStarterEnder( a_indent, moniker( ), attributes );
}


/*! \class Add
 * This class represents the **GNDS** <**add**> node.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 ***********************************************************************************************************/

Add::Add( Construction::Settings const &a_construction, pugi::xml_node const &a_node ) :
        Base( a_construction, a_node ) {

}

}               // End of namespace Summand.

}               // End of namespace Sums.

}               // End of namespace GIDI.

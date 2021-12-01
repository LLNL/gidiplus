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
#include <HAPI.hpp>

namespace GIDI {

namespace Sums {

/*! \class Sums
 * This class represents the **GNDS** <**sums**> node.
 */

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Sums::Sums( ) :
        Ancestry( GIDI_sumsChars ),
        m_crossSectionSums( GIDI_sumsCrossSectionsChars ),
        m_multiplicitySums( GIDI_sumsMultiplicitiesChars ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Sums::~Sums( ) {

}

/* *********************************************************************************************************//**
 * The Sums method to parse its sub-nodes.
 *
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **HAPI::Node** to be parsed.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 * @param a_pops            [in]    A PoPI::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs    [in]    The *internal* PoPI::Database instance used to get particle indices and possibly other particle information.
 *                                  This is the <**PoPs**> node under the <**reactionSuite**> node.
 ***********************************************************************************************************/

void Sums::parse( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo,
		PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs ) {

    if( a_node.child( GIDI_crossSectionSumsChars ).empty( ) ) {
        m_crossSectionSums.parse( a_construction, a_node.child( GIDI_sumsCrossSectionsChars ), a_setupInfo, a_pops, a_internalPoPs, parseSumsCrossSectionsSuite, nullptr ); }
    else {
        m_crossSectionSums.parse( a_construction, a_node.child( GIDI_crossSectionSumsChars ), a_setupInfo, a_pops, a_internalPoPs, parseSumsCrossSectionsSuite, nullptr );
    }

    if( a_node.child( GIDI_multiplicitySumsChars ).empty( ) ) {
        m_multiplicitySums.parse( a_construction, a_node.child( GIDI_sumsMultiplicitiesChars ), a_setupInfo, a_pops, a_internalPoPs, parseSumsMultiplicitiesSuite, nullptr ); }
    else {
        m_multiplicitySums.parse( a_construction, a_node.child( GIDI_multiplicitySumsChars ), a_setupInfo, a_pops, a_internalPoPs, parseSumsMultiplicitiesSuite, nullptr );
    }

    m_crossSectionSums.setAncestor( this );
    m_multiplicitySums.setAncestor( this );
}

/* *********************************************************************************************************//**
 * Returns a pointer to the member whose moniker is *a_item*.
 *
 * @param a_item            [in]    The moniker of the member to return.
 * @return                          Returns the pointer to the member of nullptr if it does not exists.
 ***********************************************************************************************************/

Ancestry *Sums::findInAncestry3( std::string const &a_item ) {

    if( a_item == GIDI_crossSectionSumsChars ) return( &m_crossSectionSums );
    if( a_item == GIDI_multiplicitySumsChars ) return( &m_multiplicitySums );

    if( a_item == GIDI_sumsCrossSectionsChars ) return( &m_crossSectionSums );      // GNDS 1.10.
    if( a_item == GIDI_sumsMultiplicitiesChars ) return( &m_multiplicitySums );      // GNDS 1.10.

    return( nullptr );
}

/* *********************************************************************************************************//**
 * Returns a pointer to the member whose moniker is *a_item*.
 *
 * @param a_item            [in]    The moniker of the member to return.
 * @return                          Returns the pointer to the member of nullptr if it does not exists.
 ***********************************************************************************************************/

Ancestry const *Sums::findInAncestry3( std::string const &a_item ) const {

    if( a_item == GIDI_crossSectionSumsChars ) return( &m_crossSectionSums );
    if( a_item == GIDI_multiplicitySumsChars ) return( &m_multiplicitySums );

    if( a_item == GIDI_sumsCrossSectionsChars ) return( &m_crossSectionSums );      // GNDS 1.10.
    if( a_item == GIDI_sumsMultiplicitiesChars ) return( &m_multiplicitySums );      // GNDS 1.10.

    return( nullptr );
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
    m_crossSectionSums.toXMLList( a_writeInfo, indent2 );
    m_multiplicitySums.toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class Base
 * Base class for Sums sub-node.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **HAPI::Node** to be parsed.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 * @param a_pops            [in]    A PoPI::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs    [in]    The *internal* PoPI::Database instance used to get particle indices and possibly other particle information.
 *                                  This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_type            [in]    Type for the node.
 ***********************************************************************************************************/

Base::Base( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops,
		PoPI::Database const &a_internalPoPs, FormType a_type ) :
        Form( a_node, a_setupInfo, a_type ),
        m_ENDF_MT( a_node.attribute_as_int( GIDI_ENDF_MT_Chars ) ),
        m_summands( a_construction, a_node.child( GIDI_sumsSummandsChars ), a_setupInfo ) {

        m_summands.setAncestor( this );
}

/*! \class CrossSectionSum
 * This class represents the **GNDS** <**crossSectionSum**> node.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **HAPI::Node** to be parsed.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 * @param a_pops            [in]    A PoPI::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs    [in]    The *internal* PoPI::Database instance used to get particle indices and possibly other particle information.
 *                                  This is the <**PoPs**> node under the <**reactionSuite**> node.
 ***********************************************************************************************************/

CrossSectionSum::CrossSectionSum( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo,
		PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs ) :
        Base( a_construction, a_node, a_setupInfo, a_pops, a_internalPoPs, FormType::crossSectionSum ),
        m_Q( a_construction, GIDI_QChars, a_node, a_setupInfo, a_pops, a_internalPoPs, parseQSuite, nullptr ),
        m_crossSection( a_construction, GIDI_crossSectionChars, a_node, a_setupInfo, a_pops, a_internalPoPs, parseCrossSectionSuite, nullptr ) {

    m_Q.setAncestor( this );
    m_crossSection.setAncestor( this );
}

/* *********************************************************************************************************//**
 * Returns a pointer to the member whose moniker is *a_item*.
 *
 * @param a_item            [in]    The moniker of the member to return.
 * @return                          Returns the pointer to the member of nullptr if it does not exists.
 ***********************************************************************************************************/

Ancestry *CrossSectionSum::findInAncestry3( std::string const &a_item ) {

    if( a_item == GIDI_QChars ) return( &m_Q );
    if( a_item == GIDI_crossSectionChars ) return( &m_crossSection );

    return( nullptr );
}

/* *********************************************************************************************************//**
 * Returns a pointer to the member whose moniker is *a_item*.
 *
 * @param a_item            [in]    The moniker of the member to return.
 * @return                          Returns the pointer to the member of nullptr if it does not exists.
 ***********************************************************************************************************/

Ancestry const *CrossSectionSum::findInAncestry3( std::string const &a_item ) const {

    if( a_item == GIDI_QChars ) return( &m_Q );
    if( a_item == GIDI_crossSectionChars ) return( &m_crossSection );

    return( nullptr );
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

    attributes  = a_writeInfo.addAttribute( GIDI_labelChars, label( ) );
    attributes += a_writeInfo.addAttribute( GIDI_ENDF_MT_Chars, intToString( ENDF_MT( ) ) );
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
 * @param a_node            [in]    The **HAPI::Node** to be parsed.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 * @param a_pops            [in]    A PoPI::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs    [in]    The *internal* PoPI::Database instance used to get particle indices and possibly other particle information.
 *                                  This is the <**PoPs**> node under the <**reactionSuite**> node.
 ***********************************************************************************************************/

MultiplicitySum::MultiplicitySum( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo,
		PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs ) :
        Base( a_construction, a_node, a_setupInfo, a_pops, a_internalPoPs, FormType::multiplicitySum ),
        m_multiplicity( a_construction, GIDI_multiplicityChars, a_node, a_setupInfo, a_pops, a_internalPoPs, parseMultiplicitySuite, nullptr ) {

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
 
    attributes  = a_writeInfo.addAttribute( GIDI_labelChars, label( ) );
    attributes += a_writeInfo.addAttribute( GIDI_ENDF_MT_Chars, intToString( ENDF_MT( ) ) );
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
 * @param a_node            [in]    The **HAPI::Node** to be parsed.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 ***********************************************************************************************************/

Summands::Summands( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo ) :
        Form( a_node, a_setupInfo, FormType::summands ) {

    for( HAPI::Node child = a_node.first_child( ); !child.empty( ); child.to_next_sibling( ) ) {
        std::string name( child.name( ) );

        if( name == GIDI_sumsAddChars ) {
            Summand::Add *add = new Summand::Add( a_construction, child, a_setupInfo );

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
 * @param a_node            [in]    The **HAPI::Node** to be parsed.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 ***********************************************************************************************************/

Base::Base( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo ) :
        Ancestry( a_node.name( ) ),
        m_href( a_node.attribute_as_string( GIDI_hrefChars ) ) {

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

    std::string attributes = a_writeInfo.addAttribute( GIDI_hrefChars, href( ) );

    a_writeInfo.addNodeStarterEnder( a_indent, moniker( ), attributes );
}


/*! \class Add
 * This class represents the **GNDS** <**add**> node.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **HAPI::Node** to be parsed.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 ***********************************************************************************************************/

Add::Add( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo ) :
        Base( a_construction, a_node, a_setupInfo ) {

}

}               // End of namespace Summand.

}               // End of namespace Sums.

}               // End of namespace GIDI.

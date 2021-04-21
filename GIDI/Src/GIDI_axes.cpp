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

/*! \class Axes
 * Represents a **GNDS axes** node. An axes is a list of Axis and/or Grid nodes. An axes contains a list of *N* independent axis and/or grid nodes,
 * and a dependent axis node. The dimension of an axes is the number of independent nodes.
 */

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Axes::Axes( ) :
        Form( FormType::axes ) {

    setMoniker( GIDI_axesChars );
}

/* *********************************************************************************************************//**
 *
 * @param a_node                [in]    The **pugi::xml_node** to be parsed and used to construct the Axes.
 * @param a_setupInfo           [in]    Information create my the Protare constructor to help in parsing.
 * @param a_useSystem_strtod    [in]    Flag passed to the function nfu_stringToListOfDoubles.
 ***********************************************************************************************************/

Axes::Axes( pugi::xml_node const &a_node, SetupInfo &a_setupInfo, int a_useSystem_strtod ) :
        Form( a_node, a_setupInfo, FormType::axes ) {

    for( pugi::xml_node child = a_node.first_child( ); child; child = child.next_sibling( ) ) {
        std::string name( child.name( ) );

        if(      name == GIDI_axisChars ) {
            m_axes.push_back( new Axis( child, a_setupInfo ) ); }
        else if( name == GIDI_gridChars ) {
            m_axes.push_back( new Grid( child, a_setupInfo, a_useSystem_strtod ) ); }
        else {
            throw Exception( "unknown axes sub-element" );
        }
    }
}

/* *********************************************************************************************************//**
 * Copy constructor for the Axes class.
 *
 * @param a_axes                [in]    The Axes instance to copy.
 ***********************************************************************************************************/

Axes::Axes( Axes const  &a_axes ) :
        Form( a_axes ) {

    for( std::size_t i1 = 0; i1 < a_axes.size( ); ++i1 ) {
        Axis const *axis = a_axes[i1];

        if( axis->type( ) == FormType::axis ) {
            m_axes.push_back( new Axis( *axis ) ); }
        else {
            m_axes.push_back( new Grid( static_cast<Grid const &>( *axis ) ) );
        }
    }
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Axes::~Axes( ) {

    for( std::size_t i1 = 0; i1 < m_axes.size( ); ++i1 ) delete m_axes[i1];
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void Axes::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    if( m_axes.size( ) == 0 ) return;

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );

    a_writeInfo.addNodeStarter( a_indent, moniker( ), "" );
    for( std::vector<Axis *>::const_iterator iter = m_axes.begin( ); iter != m_axes.end( ); ++iter ) (*iter)->toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( moniker( ) );
}

}

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

#define gridKeyName "index"

/*! \class Grid
 * Class to store a **GNDS grid** node. 
 */

/* *********************************************************************************************************//**
 *
 * @param a_node                [in]    The **pugi::xml_node** to be parsed and used to construct the Grid.
 * @param a_useSystem_strtod    [in]    Flag passed to the function nfu_stringToListOfDoubles.
 ***********************************************************************************************************/

Grid::Grid( pugi::xml_node const &a_node, int a_useSystem_strtod ) :
        Axis::Axis( a_node, f_grid ),
        m_style( a_node.attribute( "style" ).value( ) ),
        m_keyName( gridKeyName ),
        m_keyValue( a_node.attribute( gridKeyName ).value( ) ) {

    if( href( ) == "" ) {
        pugi::xml_node values = a_node.first_child( );
        if( values.name( ) != std::string( "values" ) ) throw std::runtime_error( "grid's first child not values" );

        m_valueType = values.attribute( "valueType" ).value( );

        parseValuesOfDoubles( values, m_values, a_useSystem_strtod );
    }
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void Grid::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string attributes = a_writeInfo.addAttribute( "index", intToString( index( ) ) );

    if( href( ) == "" ) {
        attributes += a_writeInfo.addAttribute( "label", label( ) );
        attributes += a_writeInfo.addAttribute( "unit", unit( ) );
        attributes += a_writeInfo.addAttribute( "style", style( ) );
        a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );
        doublesToXMLList( a_writeInfo, indent2, m_values, 0, true, m_valueType );
        a_writeInfo.addNodeEnder( moniker( ) ); }
    else {
        attributes += a_writeInfo.addAttribute( "href", href( ) );
        a_writeInfo.addNodeStarterEnder( a_indent, moniker( ), attributes );
    }
}

}

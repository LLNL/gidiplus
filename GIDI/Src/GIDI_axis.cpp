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

/*! \class Axis
 * Class to store an axis or the base class for a Grid instance. The **type** is *"axis"* if the instance is an Axis or
 * is *"grid"* if the instance is a Grid.
 */

/* *********************************************************************************************************//**
 *
 * @param a_index           [in]     The index for the axis.
 * @param a_label           [in]     The label for the axes.
 * @param a_unit            [in]     The unit for the axis.
 * @param a_type            [in]     The **type** is either *"axis"* or *"grid"*.
 ***********************************************************************************************************/

Axis::Axis( int a_index, std::string a_label, std::string a_unit, FormType a_type ) :
        Form( axisMoniker, a_type, a_label ),
        m_index( a_index ),
        m_unit( a_unit ) {

}

/* *********************************************************************************************************//**
 *
 * @param a_node            [in]     The **pugi::xml_node** to be parsed and used to construct the Axis.
 * @param a_type            [in]     The **type** is either *"axis"* or *"grid"*.
 ***********************************************************************************************************/

Axis::Axis( pugi::xml_node const &a_node, FormType a_type ) :
        Form( a_node, a_type ),
        m_index( a_node.attribute( "index" ).as_int( ) ),
        m_unit( a_node.attribute( "unit" ).value( ) ),
        m_href( a_node.attribute( "href" ).value( ) ) {

}

/* *********************************************************************************************************//**
 * Copy constructor for the Axis class.
 *
 * @param a_axis            [in]     The Axis instance to copy.
 ***********************************************************************************************************/

Axis::Axis( Axis const &a_axis ) :
        Form( a_axis ),
        m_index( a_axis.index( ) ),
        m_unit( a_axis.unit( ) ),
        m_href( a_axis.href( ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Axis::~Axis( ) {

}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void Axis::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string attributes = a_writeInfo.addAttribute( "index", intToString( index( ) ) );

    if( m_href == "" ) {
        attributes += a_writeInfo.addAttribute( "label", label( ) );
        attributes += a_writeInfo.addAttribute( "unit", unit( ) );
        a_writeInfo.addNodeStarterEnder( a_indent, moniker( ), attributes ); }
    else {
        attributes += a_writeInfo.addAttribute( "href", m_href );
        a_writeInfo.addNodeStarterEnder( a_indent, moniker( ), attributes );
    }
}

}

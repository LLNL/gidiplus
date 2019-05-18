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

/*! \class PhysicalQuantity
 * Class to store a physical quantity. A physical quantity is a value (e.g., 13.2) with a unit (e.g., 'cm'). The physical quantity
 * can be unitless (i.e., the unit can be an empty string). Examples a physical quantities are '13.2 cm', '0.132 m', '4.5 kg'.
 */

/* *********************************************************************************************************//**
 *
 * @param a_node            [in]     The **pugi::xml_node** to be parsed and used to construct the PhysicalQuantity.
 ***********************************************************************************************************/

PhysicalQuantity::PhysicalQuantity( pugi::xml_node const &a_node ) :
        Form( a_node, f_physicalQuantity ),
        m_value( a_node.attribute( "value" ).as_double( ) ),
        m_unit( a_node.attribute( "unit" ).value( ) ) {

}

/* *********************************************************************************************************//**
 *
 * @param a_value           [in]     The physical quantity's value.
 * @param a_unit            [in]     The physical quantity's unit.
 ***********************************************************************************************************/

PhysicalQuantity::PhysicalQuantity( double a_value, std::string a_unit ) :
        Form( f_physicalQuantity ),
        m_value( a_value ),
        m_unit( a_unit ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

PhysicalQuantity::~PhysicalQuantity( ) {

}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void PhysicalQuantity::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string attributes = a_writeInfo.addAttribute( "value", doubleToShortestString( value( ) ) ) + a_writeInfo.addAttribute( "unit", unit( ) );

    a_writeInfo.addNodeStarterEnder( a_indent, moniker( ), attributes );
}

}
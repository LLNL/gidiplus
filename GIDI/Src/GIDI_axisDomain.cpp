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

/*! \class AxisDomain
 * Class to store a the minimum and maximum limits for a domain (i.e., a section of an axis) and its unit. 
 */

/* *********************************************************************************************************//**
 *
 * @param a_node            [in]     The **pugi::xml_node** to be parsed and used to construct the AxisDomain.
 ***********************************************************************************************************/

AxisDomain::AxisDomain( pugi::xml_node const &a_node ) :
        Form( a_node, FormType::axisDomain ),
        m_minimum( a_node.attribute( "min" ).as_double( ) ),
        m_maximum( a_node.attribute( "max" ).as_double( ) ),
        m_unit( a_node.attribute( "unit" ).value( ) ) {

}

/* *********************************************************************************************************//**
 *
 * @param a_minimum         [in]     The domain's minimum value.
 * @param a_maximum         [in]     The domain's maximum.
 * @param a_unit            [in]     The domain's unit.
 ***********************************************************************************************************/

AxisDomain::AxisDomain( double a_minimum, double a_maximum, std::string const &a_unit ) :
        Form( FormType::axisDomain ),
        m_minimum( a_minimum ),
        m_maximum( a_maximum ),
        m_unit( a_unit ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

AxisDomain::~AxisDomain( ) {

}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void AxisDomain::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string attributes = a_writeInfo.addAttribute( "min", doubleToShortestString( minimum( ) ) ) + 
                             a_writeInfo.addAttribute( "max", doubleToShortestString( maximum( ) ) ) +
                             a_writeInfo.addAttribute( "unit", unit( ) );

    a_writeInfo.addNodeStarterEnder( a_indent, moniker( ), attributes );
}


}

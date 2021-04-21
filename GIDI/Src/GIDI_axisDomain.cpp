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
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 ***********************************************************************************************************/

AxisDomain::AxisDomain( pugi::xml_node const &a_node, SetupInfo &a_setupInfo ) :
        Form( a_node, a_setupInfo, FormType::axisDomain ),
        m_minimum( a_node.attribute( GIDI_minChars ).as_double( ) ),
        m_maximum( a_node.attribute( GIDI_maxChars ).as_double( ) ),
        m_unit( a_node.attribute( GIDI_unitChars ).value( ) ) {

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

    std::string attributes = a_writeInfo.addAttribute( GIDI_minChars, doubleToShortestString( minimum( ) ) ) + 
                             a_writeInfo.addAttribute( GIDI_minChars, doubleToShortestString( maximum( ) ) ) +
                             a_writeInfo.addAttribute( GIDI_unitChars, unit( ) );

    a_writeInfo.addNodeStarterEnder( a_indent, moniker( ), attributes );
}


}

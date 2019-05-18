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

/*! \class DiscreteGamma2d
 * Class for the GNDS <**discreteGamma**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction    [in]    Used to pass user options for parsing.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed and used to construct the XYs2d.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

DiscreteGamma2d::DiscreteGamma2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Function2dForm( a_construction, a_node, f_discreteGamma2d, a_parent ),
        m_domainMin( a_node.attribute( "domainMin" ).as_double( ) ),
        m_domainMax( a_node.attribute( "domainMax" ).as_double( ) ),
        m_value( a_node.attribute( "value" ).as_double( ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

DiscreteGamma2d::~DiscreteGamma2d( ) {

}

/* *********************************************************************************************************//**
 * Returns the energy of the discrete gamma.
 *
 * @param a_x2              [in]    The is ignored.
 * @param a_x1              [in]    The is ignored.
 * @return                          The energy of the discrete gamma.
 ***********************************************************************************************************/

double DiscreteGamma2d::evaluate( double a_x2, double a_x1 ) const {

// FIXME - Do we need to check domain?
    return( m_value );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 * @param       a_embedded          [in]        If *true*, *this* function is embedded in a higher dimensional function.
 * @param       a_inRegions         [in]        This is not used in this method.
 ***********************************************************************************************************/

void DiscreteGamma2d::toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string attributes = a_writeInfo.addAttribute( "value", doubleToShortestString( value( ) ) );

    attributes += a_writeInfo.addAttribute( "domainMin", doubleToShortestString( domainMin( ) ) );
    attributes += a_writeInfo.addAttribute( "domainMax", doubleToShortestString( domainMax( ) ) );
    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );

    axes( ).toXMLList( a_writeInfo, indent2 );

    a_writeInfo.addNodeEnder( moniker( ) );
}

}

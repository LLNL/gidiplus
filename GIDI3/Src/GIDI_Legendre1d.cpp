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

/*! \class Legendre1d
 * Class for the GNDS <**Legendre**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction    [in]    Used to pass user options for parsing.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed and used to construct the XYs2d.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

Legendre1d::Legendre1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Function1dForm( a_construction, a_node, f_Legendre1d, a_parent ) {

    parseValuesOfDoubles( a_construction, a_node.child( "values" ), m_coefficients );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Legendre1d::~Legendre1d( ) {

}

/* *********************************************************************************************************//**
 * Returns the value of the function evaluated at the specified projectile's energy.
 * Currently not implemented.
 *
 * @param a_x1              [in]    The projectile's energy.
 * @return                          The value of the function evaluated at *a_x1*.
 ***********************************************************************************************************/

double Legendre1d::evaluate( double a_x1 ) const {

    throw std::runtime_error( "Legendre1d::evaluate: not implemented." );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 * @param       a_embedded          [in]        If *true*, *this* function is embedded in a higher dimensional function.
 * @param       a_inRegions         [in]        If *true*, *this* is in a Regions1d container.
 ***********************************************************************************************************/

void Legendre1d::toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string attributes;

    if( a_embedded ) {
        attributes += a_writeInfo.addAttribute( "outerDomainValue", doubleToShortestString( outerDomainValue( ) ) ); }
    else {
        if( a_inRegions ) {
            attributes = a_writeInfo.addAttribute( "index", intToString( index( ) ) ); }
        else {
            if( label( ) != "" ) attributes = a_writeInfo.addAttribute( "label", label( ) );
        }
    }

    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );

    axes( ).toXMLList( a_writeInfo, indent2 );
    doublesToXMLList( a_writeInfo, indent2, m_coefficients, 0, false );
    a_writeInfo.addNodeEnder( moniker( ) );
}

}

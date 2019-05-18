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

/*! \class Constant1d
 * Class for the GNDS <**constant1d**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_value           [in]     The GNDS **value** for *this*.
 * @param a_domainMin       [in]     The minimum value for the domain.
 * @param a_domainMax       [in]     The maximum value for the domain.
 * @param a_domainUnit      [in]     The unit of the domain.
 * @param a_rangeUnit       [in]     The unit of the range.
 ***********************************************************************************************************/

Constant1d::Constant1d( double a_value, double a_domainMin, double a_domainMax, std::string const &a_domainUnit, std::string const &a_rangeUnit ) :
        Function1dForm( f_constant1d, a_domainUnit, a_rangeUnit, ptwXY_interpolationLinLin, 0, 0 ),
        m_value( a_value ),
        m_domainMin( a_domainMin ),
        m_domainMax( a_domainMax ) {

    moniker( constant1dMoniker );
}

/* *********************************************************************************************************//**
 *
 * @param a_construction    [in]    Used to pass user options for parsing.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed and used to construct the XYs2d.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

Constant1d::Constant1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Function1dForm( a_construction, a_node, f_constant1d, a_parent ),
        m_value( a_node.attribute( "value" ).as_double( ) ),
        m_domainMin( a_node.attribute( "domainMin" ).as_double( ) ),
        m_domainMax( a_node.attribute( "domainMax" ).as_double( ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Constant1d::~Constant1d( ) {

}

/* *********************************************************************************************************//**
 * Returns the value of the constant function.
 *
 * @param a_x1              [in]    This is ignored a the function is a constant.
 * @return                          The value of the constant.
 ***********************************************************************************************************/

double Constant1d::evaluate( double a_x1 ) const {

// FIXME - Do we need to check domain?
    return( m_value );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 * @param       a_embedded          [in]        If *true*, *this* function is embedded in a higher dimensional function.
 * @param       a_inRegions         [in]        If *true*, *this* is in a Regions1d container.
 ***********************************************************************************************************/

void Constant1d::toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string attributes;
    
    if( a_embedded ) {
        attributes += a_writeInfo.addAttribute( "outerDomainValue", doubleToShortestString( outerDomainValue( ) ) ); }
    else {
        if( a_inRegions ) {
            attributes = a_writeInfo.addAttribute( "index", intToString( index( ) ) ); }
        else {
            attributes = a_writeInfo.addAttribute( "label", label( ) );
        }
    }

    attributes += a_writeInfo.addAttribute( "value", doubleToShortestString( m_value ) );
    attributes += a_writeInfo.addAttribute( "domainMin", doubleToShortestString( m_domainMin ) );
    attributes += a_writeInfo.addAttribute( "domainMax", doubleToShortestString( m_domainMax ) );

    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );
    axes( ).toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( moniker( ) );
}

}

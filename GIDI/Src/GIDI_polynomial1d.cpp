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

namespace Functions {

/*! \class Polynomial1d
 * Class for the GNDS <**polynomial1d**> node.
 */

/* *********************************************************************************************************//**
 * @param a_axes            [in]    The axes to copy for *this*. 
 * @param a_domainMin       [in]    The minimum value for the domain.
 * @param a_domainMax       [in]    The maximum value for the domain.
 * @param a_coefficients    [in]    The coefficients representing the polynomial.
 * @param a_index               [in]    Currently not used.
 * @param a_outerDomainValue    [in]    If embedded in a higher dimensional function, the value of the domain of the next higher dimension.
 ***********************************************************************************************************/

Polynomial1d::Polynomial1d( Axes const &a_axes, double a_domainMin, double a_domainMax, std::vector<double> const &a_coefficients, int a_index, double a_outerDomainValue ) :
        Function1dForm( polynomial1dMoniker, FormType::polynomial1d, a_axes, ptwXY_interpolationLinLin, a_index, a_outerDomainValue ),
        m_domainMin( a_domainMin ),
        m_domainMax( a_domainMax ),
        m_coefficients( a_coefficients ) {

}

/* *********************************************************************************************************//**
 * @param a_construction    [in]     Used to pass user options to the constructor.
 * @param a_node            [in]     The **pugi::xml_node** to be parsed and used to construct the XYs2d.
 * @param a_parent          [in]     The parent GIDI::Suite.
 ***********************************************************************************************************/

Polynomial1d::Polynomial1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Function1dForm( a_construction, a_node, FormType::polynomial1d, a_parent ),
        m_domainMin( a_node.attribute( "domainMin" ).as_double( ) ),
        m_domainMax( a_node.attribute( "domainMax" ).as_double( ) ) {

    parseValuesOfDoubles( a_construction, a_node.child( "values" ), m_coefficients );
}

/* *********************************************************************************************************//**
 * The Polynomial1d copy constructor.
 *
 * @param a_polynomial1d        [in]    The Polynomial1d instance to copy.
 ***********************************************************************************************************/

Polynomial1d::Polynomial1d( Polynomial1d const &a_polynomial1d ) :
        Function1dForm( a_polynomial1d ),
        m_domainMin( a_polynomial1d.domainMin( ) ),
        m_domainMax( a_polynomial1d.domainMax( ) ),
        m_coefficients( a_polynomial1d.coefficients( ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Polynomial1d::~Polynomial1d( ) {

}

/* *********************************************************************************************************//**
 * The value of the polynomial at the point *a_x1*.
 *
 * @param a_x1          [in]    Domain value to evaluate this at.
 * @return                      The value of the polynomial at the point **a_x1**.
 ***********************************************************************************************************/

double Polynomial1d::evaluate( double a_x1 ) const {

    double _value = 0;

    if( a_x1 < m_domainMin ) return( 0.0 );
    if( a_x1 > m_domainMax ) return( 0.0 );

    for( std::vector<double>::const_reverse_iterator riter = m_coefficients.rbegin( ); riter != m_coefficients.rend( ); ++riter ) {
        _value = *riter + _value * a_x1;
    }
    return( _value );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 * @param       a_embedded          [in]        If *true*, *this* function is embedded in a higher dimensional function.
 * @param       a_inRegions         [in]        If *true*, *this* is in a Regions1d container.
 ***********************************************************************************************************/

void Polynomial1d::toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const {

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

    attributes = a_writeInfo.addAttribute( "domainMin", doubleToShortestString( domainMin( ) ) );
    attributes += a_writeInfo.addAttribute( "domainMax", doubleToShortestString( domainMax( ) ) );
    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );

    axes( ).toXMLList( a_writeInfo, indent2 );
    doublesToXMLList( a_writeInfo, indent2, m_coefficients );
    a_writeInfo.addNodeEnder( moniker( ) );
}

}               // End namespace Functions.

}               // End namespace GIDI.
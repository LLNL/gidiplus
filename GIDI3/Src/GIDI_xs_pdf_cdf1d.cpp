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

/*! \class Xs_pdf_cdf1d
 * Class for the GNDS <**xs_pdf_cdf1d**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]     The **pugi::xml_node** to be parsed and used to construct the XYs2d.
 * @param a_parent          [in]     The parent GIDI::Suite.
 ***********************************************************************************************************/

Xs_pdf_cdf1d::Xs_pdf_cdf1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Function1dForm( a_construction, a_node, f_xs_pdf_cdf1d, a_parent ) {

        parseValuesOfDoubles( a_construction, a_node.child( "xs" ).child( "values" ), m_xs );
        parseValuesOfDoubles( a_construction, a_node.child( "pdf" ).child( "values" ), m_pdf );
        parseValuesOfDoubles( a_construction, a_node.child( "cdf" ).child( "values" ), m_cdf );
}

/* *********************************************************************************************************//**
 *
 * @param a_domainUnit          [in]    The domain unit for the instance.
 * @param a_rangeUnit           [in]    The range unit for the instance.
 * @param a_interpolation       [in]    The interpolation flag.
 * @param a_index               [in]    If imbedded in a two dimensional function, the index of this instance.
 * @param a_outerDomainValue    [in]    If imbedded in a two dimensional function, the domain value for *x2*.
 * @param a_Xs                  [in]    List of x1 values.
 * @param a_pdf                 [in]    The pdf evaluated at the x1 values.
 * @param a_cdf                 [in]    The pdf evaluated at the x1 values.
 ***********************************************************************************************************/

Xs_pdf_cdf1d::Xs_pdf_cdf1d( std::string const &a_domainUnit, std::string const &a_rangeUnit, ptwXY_interpolation a_interpolation, 
            int a_index, double a_outerDomainValue, std::vector<double> const &a_Xs, std::vector<double> const &a_pdf, std::vector<double> const &a_cdf ) :
        Function1dForm( f_xs_pdf_cdf1d, a_domainUnit, a_rangeUnit, a_interpolation, a_index, a_outerDomainValue),
        m_xs( a_Xs ),
        m_pdf( a_pdf ),
        m_cdf( a_cdf ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Xs_pdf_cdf1d::~Xs_pdf_cdf1d( ) {

}

/* *********************************************************************************************************//**
 * The value of *pdf* at the point *a_x1*.
 * Currently not implemented.
 *
 * @param a_x1          [in]    The point for the *x1* axis.
 * @return                      The value of the function at the point *a_x1*.
 ***********************************************************************************************************/

double Xs_pdf_cdf1d::evaluate( double a_x1 ) const {

    throw std::runtime_error( "Xs_pdf_cdf1d::evaluate: not implemented." );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 * @param       a_embedded          [in]        If *true*, *this* function is embedded in a higher dimensional function.
 * @param       a_inRegions         [in]        If *true*, *this* is in a Regions1d container.
 ***********************************************************************************************************/

void Xs_pdf_cdf1d::toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const {

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

    if( interpolation( ) != ptwXY_interpolationLinLin ) attributes += a_writeInfo.addAttribute( "interpolation", interpolationString( ) );

    std::string xml = a_writeInfo.nodeStarter( a_indent, moniker( ), attributes );
    xml += nodeWithValuesToDoubles( a_writeInfo,  "xs", m_xs );
    xml += nodeWithValuesToDoubles( a_writeInfo, "pdf", m_pdf );
    xml += nodeWithValuesToDoubles( a_writeInfo, "cdf", m_cdf );
    xml += a_writeInfo.nodeEnder( moniker( ) );

    a_writeInfo.push_back( xml );
}

}

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

#define GIDI_xsChars "xs"
#define GIDI_pdfChars "pdf"
#define GIDI_cdfChars "cdf"

/*! \class Xs_pdf_cdf1d
 * Class for the GNDS <**xs_pdf_cdf1d**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_axes                [in]    The axes to copy for *this*.
 * @param a_interpolation       [in]    The interpolation flag.
 * @param a_index               [in]    If imbedded in a two dimensional function, the index of this instance.
 * @param a_outerDomainValue    [in]    If imbedded in a two dimensional function, the domain value for *x2*.
 * @param a_Xs                  [in]    List of x1 values.
 * @param a_pdf                 [in]    The pdf evaluated at the x1 values.
 * @param a_cdf                 [in]    The pdf evaluated at the x1 values.
 ***********************************************************************************************************/

Xs_pdf_cdf1d::Xs_pdf_cdf1d( Axes const &a_axes, ptwXY_interpolation a_interpolation, std::vector<double> const &a_Xs, 
                std::vector<double> const &a_pdf, std::vector<double> const &a_cdf, int a_index, double a_outerDomainValue ) :
        Function1dForm( GIDI_xs_pdf_cdf1dChars, FormType::xs_pdf_cdf1d, a_axes, a_interpolation, a_index, a_outerDomainValue ),
        m_xs( a_Xs ),
        m_pdf( a_pdf ),
        m_cdf( a_cdf ) {

}

/* *********************************************************************************************************//**
 *
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed and used to construct the XYs2d.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

Xs_pdf_cdf1d::Xs_pdf_cdf1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent ) :
        Function1dForm( a_construction, a_node, a_setupInfo, FormType::xs_pdf_cdf1d, a_parent ) {

        parseValuesOfDoubles( a_construction, a_node.child( GIDI_xsChars ).child( GIDI_valuesChars ), a_setupInfo, m_xs );
        parseValuesOfDoubles( a_construction, a_node.child( GIDI_pdfChars ).child( GIDI_valuesChars ), a_setupInfo, m_pdf );
        parseValuesOfDoubles( a_construction, a_node.child( GIDI_cdfChars ).child( GIDI_valuesChars ), a_setupInfo, m_cdf );
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

    throw Exception( "Xs_pdf_cdf1d::evaluate: not implemented." );
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
        attributes += a_writeInfo.addAttribute( GIDI_outerDomainValueChars, doubleToShortestString( outerDomainValue( ) ) ); }
    else {
        if( a_inRegions ) {
            attributes = a_writeInfo.addAttribute( GIDI_indexChars, intToString( index( ) ) ); }
        else {
            if( label( ) != "" ) attributes = a_writeInfo.addAttribute( GIDI_labelChars, label( ) );
        }
    }

    if( interpolation( ) != ptwXY_interpolationLinLin ) attributes += a_writeInfo.addAttribute( GIDI_interpolationChars, interpolationString( ) );

    std::string xml = a_writeInfo.nodeStarter( a_indent, moniker( ), attributes );
    xml += nodeWithValuesToDoubles( a_writeInfo,  GIDI_xsChars, m_xs );
    xml += nodeWithValuesToDoubles( a_writeInfo, GIDI_pdfChars, m_pdf );
    xml += nodeWithValuesToDoubles( a_writeInfo, GIDI_cdfChars, m_cdf );
    xml += a_writeInfo.nodeEnder( moniker( ) );

    a_writeInfo.push_back( xml );
}

}               // End namespace Functions.

}               // End namespace GIDI.

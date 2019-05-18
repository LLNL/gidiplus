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

/*! \class XYs3d
 * Class for the GNDS <**XYs3d**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction    [in]    Used to pass user options for parsing.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed and used to construct the XYs2d.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

XYs3d::XYs3d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Function3dForm( a_construction, a_node, f_XYs3d, a_parent ),
        m_interpolationQualifier( a_node.attribute( "interpolationQualifier" ).value( ) ) {

    for( pugi::xml_node child = a_node.first_child( ); child; child = child.next_sibling( ) ) {
        std::string name( child.name( ) );

        if( name == "axes" ) continue;
        if( name == "uncertainty" ) continue;

        Function2dForm *_form = data2dParse( a_construction, child, NULL );
        if( _form == NULL ) throw std::runtime_error( "XYs3d::XYs3d: data2dParse returned NULL." );
        if( m_Xs.size( ) > 0 ) {
            if( _form->outerDomainValue( ) <= m_Xs[m_Xs.size( )-1] ) throw std::runtime_error( "XYs3d::XYs3d: next outerDomainValue <= current outerDomainValue." );
        }
        m_Xs.push_back( _form->outerDomainValue( ) );
        m_function2ds.push_back( _form );
    }

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

XYs3d::~XYs3d( ) {

    for( std::vector<Function2dForm *>::iterator iter = m_function2ds.begin( ); iter < m_function2ds.end( ); ++iter ) delete *iter;
}

/* *********************************************************************************************************//**
 * Returns the domain minimum for the instance.
 *
 * @return          The domain minimum for the instance.
 ***********************************************************************************************************/

double XYs3d::domainMin( ) const {

    if( m_Xs.size( ) == 0 ) throw std::runtime_error( "XYs3d::domainMin: XYs3d has no 2d-functions" );
    return( m_Xs[0] );
}

/* *********************************************************************************************************//**
 * Returns the domain maximum for the instance.
 *
 * @return              The domain maximum for the instance.
 ***********************************************************************************************************/

double XYs3d::domainMax( ) const {

    if( m_Xs.size( ) == 0 ) throw std::runtime_error( "XYs3d::domainMax: XYs3d has no 2d-functions" );
    return( m_Xs[m_Xs.size( )-1] );
}

/* *********************************************************************************************************//**
 * Returns the value of the function *f(x3,x2,x1)* at the specified point *a_x3*, *a_x2* and *a_x1*.
 *
 * @param a_x3              [in]    The value of the **x3** axis.
 * @param a_x2              [in]    The value of the **x2** axis.
 * @param a_x1              [in]    The value of the **x1** axis.
 * @return                          The value of the function evaluated at *a_x3*, *a_x2* and *a_x1*.
 ***********************************************************************************************************/

double XYs3d::evaluate( double a_x3, double a_x2, double a_x1 ) const {

    if( m_Xs.size( ) == 0 ) throw std::runtime_error( "XYs3d::evaluate: XYs3d has no 2d functions." );

    long iX3 = binarySearchVector( a_x3, m_Xs );

    if( iX3 < 0 ) {
        if( iX3 == -1 ) {       /* x3 > last value of Xs. */
            return( m_function2ds.back( )->evaluate( a_x2, a_x1 ) );
        }
        return( m_function2ds[0]->evaluate( a_x2, a_x1 ) ); /* x3 < first value of Xs. */
    }

// Currently does not interpolate;
    return( m_function2ds[iX3]->evaluate( a_x2, a_x1 ) );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 * @param       a_embedded          [in]        If *true*, *this* function is embedded in a higher dimensional function.
 * @param       a_inRegions         [in]        If *true*, *this* is in a Regions2d container.
 ***********************************************************************************************************/

void XYs3d::toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const {

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

    if( m_interpolationQualifier != "" ) attributes = a_writeInfo.addAttribute( "interpolationQualifier", m_interpolationQualifier );

    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );
    axes( ).toXMLList( a_writeInfo, indent2 );
    for( std::vector<Function2dForm *>::const_iterator iter = m_function2ds.begin( ); iter != m_function2ds.end( ); ++iter ) (*iter)->toXMLList_func( a_writeInfo, indent2, true, false );
    a_writeInfo.addNodeEnder( moniker( ) );
}

}

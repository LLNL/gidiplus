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

/*! \class Regions2d
 * Class for the GNDS <**regions2d**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction    [in]     Used to pass user options to the constructor.
 * @param a_node            [in]     The **pugi::xml_node** to be parsed and used to construct the XYs2d.
 * @param a_parent          [in]     The parent GIDI::Suite.
 ***********************************************************************************************************/

Regions2d::Regions2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Function2dForm( a_construction, a_node, f_regions2d, a_parent ) {

    for( pugi::xml_node child = a_node.first_child( ); child; child = child.next_sibling( ) ) {
        std::string name( child.name( ) );

        if( name == "axes" ) continue;
        if( name == "uncertainty" ) continue;

        Function2dForm *_form = data2dParse( a_construction, child, NULL );
        if( _form == NULL ) throw std::runtime_error( "Regions2d::Regions2d: data2dParse returned NULL." );
        append( _form );
    }
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Regions2d::~Regions2d( ) {

    for( std::vector<Function2dForm *>::iterator iter = m_functions2d.begin( ); iter < m_functions2d.end( ); ++iter ) delete *iter;
}

/* *********************************************************************************************************//**
 * Returns the domain minimum for the instance.
 *
 * @return          The domain minimum for the instance.
 ***********************************************************************************************************/

double Regions2d::domainMin( ) const {

    if( m_Xs.size( ) == 0 ) throw std::runtime_error( "Regions2d::domainMin: Regions2d has no regions" );
    return( m_Xs[0] );
}

/* *********************************************************************************************************//**
 * Returns the domain maximum for the instance.
 *
 * @return              The domain maximum for the instance.
 ***********************************************************************************************************/

double Regions2d::domainMax( ) const {

    if( m_Xs.size( ) == 0 ) throw std::runtime_error( "Regions2d::domainMax: Regions2d has no regions" );
    return( m_Xs[m_Xs.size( )-1] );
}

/* *********************************************************************************************************//**
 * Appends the 2d function *a_function* to the region.
 *
 ***********************************************************************************************************/
 
void Regions2d::append( Function2dForm *a_function ) {

    if( dimension( ) != a_function->dimension( ) ) throw std::runtime_error( "Regions2d::append: dimensions differ." );

    double _domainMin = a_function->domainMin( ), _domainMax = a_function->domainMax( );

    if( m_Xs.size( ) == 0 ) {
        m_Xs.push_back( _domainMin ); }
    else {
        if( m_Xs.back( ) != _domainMin ) throw std::runtime_error( "Regions2d::append: regions do not abut." );
    }

    m_Xs.push_back( _domainMax );
    m_functions2d.push_back( a_function );
}

/* *********************************************************************************************************//**
 * The value of *y(x2,x1)* at the point *a_x2*, *a_x1*.
 *
 * @param a_x2          [in]    The point for the *x2* axis.
 * @param a_x1          [in]    The point for the *x1* axis.
 * @return                      The value of the function at the point *a_x2*, *a_x1*.
 ***********************************************************************************************************/

double Regions2d::evaluate( double a_x2, double a_x1 ) const {

    if( m_Xs.size( ) == 0 ) throw std::runtime_error( "Regions2d::evaluate: regions2d has no regions" );

    long iX1 = binarySearchVector( a_x1, m_Xs );

    if( iX1 < 0 ) {
        if( iX1 == -1 ) {       /* x1 > last value of Xs. */
            return( m_functions2d.back( )->evaluate( a_x2, a_x1 ) );
        }
        iX1 = 0;                /* x1 < last value of Xs. */
    }

    return( m_functions2d[iX1]->evaluate( a_x2, a_x1 ) );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 * 
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 * @param       a_embedded          [in]        If *true*, *this* function is embedded in a higher dimensional function.
 * @param       a_inRegions         [in]        If *true*, *this* is in a Regions2d container.
 ***********************************************************************************************************/

void Regions2d::toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const {

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
    for( std::vector<Function2dForm *>::const_iterator iter = m_functions2d.begin( ); iter != m_functions2d.end( ); ++iter ) (*iter)->toXMLList_func( a_writeInfo, indent2, false, true );
    a_writeInfo.addNodeEnder( moniker( ) );
}

}

/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "GIDI.hpp"
#include <HAPI.hpp>

namespace GIDI {

namespace Functions {

/*! \class Regions1d
 * Class for the GNDS <**regions1d**> node.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **HAPI::Node** to be parsed and used to construct the XYs2d.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

Regions1d::Regions1d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent ) :
        Function1dForm( a_construction, a_node, a_setupInfo, FormType::regions1d, a_parent ) {

    if( a_setupInfo.m_formatVersion.format( ) != GNDS_formatVersion_1_10Chars ) {
        data1dListParse( a_construction, a_node.child( GIDI_function1dsChars ), a_setupInfo, m_function1ds );
        checkSequentialDomainLimits1d( m_function1ds, m_Xs );
        return;                                     // Need to add uncertainty parsing.
    }

    for( HAPI::Node child = a_node.first_child( ); !child.empty( ); child.to_next_sibling( ) ) {
        std::string name( child.name( ) );

        if( name == GIDI_axesChars ) continue;
        if( name == GIDI_uncertaintyChars ) continue;

        Function1dForm *_form = data1dParse( a_construction, child, a_setupInfo, nullptr );
        if( _form == nullptr ) throw Exception( "Regions1d::Regions1d: data1dParse returned nullptr." );
        append( _form );
    }
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Regions1d::~Regions1d( ) {

    for( std::vector<Function1dForm *>::iterator iter = m_function1ds.begin( ); iter < m_function1ds.end( ); ++iter ) delete *iter;
}

/* *********************************************************************************************************//**
 * Returns the domain minimum for the instance.
 *
 * @return          The domain minimum for the instance.
 ***********************************************************************************************************/

double Regions1d::domainMin( ) const {

    if( m_Xs.size( ) == 0 ) throw Exception( "Regions1d::domainMin: Regions1d has no regions" );
    return( m_Xs[0] );
}

/* *********************************************************************************************************//**
 * Returns the domain maximum for the instance.
 *
 * @return              The domain maximum for the instance.
 ***********************************************************************************************************/

double Regions1d::domainMax( ) const {

    if( m_Xs.size( ) == 0 ) throw Exception( "Regions1d::domainMax: Regions1d has not regions" );
    return( m_Xs[m_Xs.size( )-1] );
}

/* *********************************************************************************************************//**
 * Appends the 1d function *a_function* to the region.
 *
 * @param a_function            [in]    The 1d function (i.e., 1d region) to append to the Regions1d.
 ***********************************************************************************************************/
 
void Regions1d::append( Function1dForm *a_function ) {

    if( dimension( ) != a_function->dimension( ) ) throw Exception( "Regions1d::append: dimensions differ." );

    double _domainMin = a_function->domainMin( ), _domainMax = a_function->domainMax( );

    if( m_Xs.size( ) == 0 ) {
        m_Xs.push_back( _domainMin ); }
    else {
        if( m_Xs.back( ) != _domainMin ) throw Exception( "Regions1d::append: regions do not abut." );
    }

    m_Xs.push_back( _domainMax );
    m_function1ds.push_back( a_function );
}

/* *********************************************************************************************************//**
 * The value of *y(x1)* at the point *a_x1*.
 *
 * @param a_x1          [in]    Domain value to evaluate this at.
 * @return                      The value of this at the point *a_x1*.
 ***********************************************************************************************************/


double Regions1d::evaluate( double a_x1 ) const {

    if( m_Xs.size( ) == 0 ) throw Exception( "Regions1d::evaluate: Regions1d has not regions" );

    long iX1 = binarySearchVector( a_x1, m_Xs );

    if( iX1 < 0 ) {
        if( iX1 == -1 ) {       /* x1 > last value of Xs. */
            return( m_function1ds.back( )->evaluate( a_x1 ) );
        }
        iX1 = 0;                /* x1 < last value of Xs. */
    }
    return( m_function1ds[iX1]->evaluate( a_x1 ) );
}

/* *********************************************************************************************************//**
 * Evaluates *this* at the X-values in *a_Xs*[*a_offset*:] and adds the results to *a_results*[*a_offset*:].
 * *a_Xs* and *a_results* must be the same size otherwise a throw is executed.
 *
 * @param a_offset          [in]    The offset in *a_Xs* to start.
 * @param a_Xs              [in]    The list of domain values to evaluate *this* at.
 * @param a_results         [in]    The list whose values are added to by the Y-values of *this*.
 * @param a_scaleFactor     [in]    A factor applied to each evaluation before it is added to *a_results*.
 ***********************************************************************************************************/

void Regions1d::mapToXsAndAdd( int a_offset, std::vector<double> const &a_Xs, std::vector<double> &a_results, double a_scaleFactor ) const {

    for( auto iter = m_function1ds.begin( ); iter < m_function1ds.end( ); ++iter ) {
        (*iter)->mapToXsAndAdd( a_offset, a_Xs, a_results, a_scaleFactor );
    }
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 * @param       a_embedded          [in]        If *true*, *this* function is embedded in a higher dimensional function.
 * @param       a_inRegions         [in]        If *true*, *this* is in a Regions1d container.
 ***********************************************************************************************************/

void Regions1d::toXMLList_func( GUPI::WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string attributes;

    if( a_embedded ) {
        attributes += a_writeInfo.addAttribute( GIDI_outerDomainValueChars, LUPI::Misc::doubleToShortestString( outerDomainValue( ) ) ); }
    else {
        if( a_inRegions ) {
            attributes = a_writeInfo.addAttribute( GIDI_indexChars, intToString( index( ) ) ); }
        else {
            if( label( ) != "" ) attributes = a_writeInfo.addAttribute( GIDI_labelChars, label( ) );
        }
    }

    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );
    axes( ).toXMLList( a_writeInfo, indent2 );
    for( std::vector<Function1dForm *>::const_iterator iter = m_function1ds.begin( ); iter != m_function1ds.end( ); ++iter ) (*iter)->toXMLList_func( a_writeInfo, indent2, false, true );
    a_writeInfo.addNodeEnder( moniker( ) );
}

}               // End namespace Functions.

}               // End namespace GIDI.

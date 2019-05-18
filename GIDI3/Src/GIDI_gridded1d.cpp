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

/*! \class Gridded1d
 * Class for the GNDS <**gridded1d**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed and used to construct the XYs2d.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

Gridded1d::Gridded1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Function1dForm( a_construction, a_node, f_gridded1d, a_parent ) {

    Grid const *axis = dynamic_cast<Grid const *>( axes( )[0] );
    m_grid = axis->data( );

    parseFlattened1d( a_construction, a_node.child( "array" ), m_data );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Gridded1d::~Gridded1d( ) {

}

/* *********************************************************************************************************//**
 * Returns the value of the function at the point *a_x1*.
 * Currently not implemented.
 *
 * @param a_x1              [in]    The is ignored.
 * @return                          The value of the function at the point *a_x1*.
 ***********************************************************************************************************/

double Gridded1d::evaluate( double a_x1 ) const {

    throw std::runtime_error( "Gridded1d::evaluate: not implement." );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 * @param       a_embedded          [in]        If *true*, *this* function is embedded in a higher dimensional function.
 * @param       a_inRegions         [in]        If *true*, *this* is in a Regions1d container.
 ***********************************************************************************************************/

void Gridded1d::toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const {

// BRB. This is not correct as it is not converted to a flattened array.

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string indent3 = a_writeInfo.incrementalIndent( indent2 );
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

    attributes = a_writeInfo.addAttribute( "shape", size_t_ToString( m_data.size( ) ) );
    attributes += a_writeInfo.addAttribute( "compression", "flattened" );
    a_writeInfo.addNodeStarter( indent2, "array", attributes );

    std::vector<double> doubles;
    doubles.reserve( m_data.size( ) );
    std::size_t i1;
    for( i1 = 0; i1 < m_data.size( ); ++i1 ) {
        if( m_data[i1] != 0.0 ) break;
    }
    std::size_t start( i1 );
    if( start == m_data.size( ) ) start = 0;
    a_writeInfo.push_back( indent3 + "<values valueType=\"Integer32\" label=\"starts\">" + size_t_ToString( start ) + "</values>" );
    for( ; i1 < m_data.size( ); ++i1 ) doubles.push_back( m_data[i1] );
    a_writeInfo.push_back( indent3 + "<values valueType=\"Integer32\" label=\"lengths\">" + size_t_ToString( doubles.size( ) ) + "</values>" );

    doublesToXMLList( a_writeInfo, indent3, doubles );
    a_writeInfo.addNodeEnder( "array" );
    a_writeInfo.addNodeEnder( moniker( ) );
}

}

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

/*! \class Ys1d
 * Class to store GNDS <**Ys1d**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_domainUnit          [in]    The domain unit for the instance.
 * @param a_rangeUnit           [in]    The range unit for the instance.
 * @param a_interpolation       [in]    The interpolation flag.
 * @param a_index               [in]    If imbedded in a two dimensional function, the index of this instance.
 * @param a_outerDomainValue    [in]    If imbedded in a two dimensional function, the domain value for *x2*.
 ***********************************************************************************************************/

Ys1d::Ys1d( std::string const &a_domainUnit, std::string const &a_rangeUnit, ptwXY_interpolation a_interpolation, int a_index, double a_outerDomainValue ) :
        Function1dForm( f_Ys1d, a_domainUnit, a_rangeUnit, a_interpolation, a_index, a_outerDomainValue ),
        m_start( 0 ) {

}

/* *********************************************************************************************************//**
 * Constructs the instance from a **pugi::xml_node** instance.
 *
 * @param a_construction        [in]    Used to pass user options for parsing.
 * @param a_node                [in]    The Ys1d pugi::xml_node to be parsed and to construct the instance.
 * @param a_parent              [in]    If imbedded in a two dimensional function, its pointers.
 ***********************************************************************************************************/

Ys1d::Ys1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Function1dForm( a_construction, a_node, f_Ys1d, a_parent ),
        m_start( a_node.child( "values" ).attribute( "start" ).as_int( ) ),         // as_int returns 0 if "start" not present.
        m_Ys( ) {

    parseValuesOfDoubles( a_construction, a_node.child( "values" ), m_Ys );
}

/* *********************************************************************************************************//**
 * The Ys1d copy constructor.
 *
 * @param a_Ys1d
 ***********************************************************************************************************/

Ys1d::Ys1d( Ys1d const &a_Ys1d ) :
        Function1dForm( a_Ys1d ),
        m_start( a_Ys1d.start( ) ),
        m_Ys( a_Ys1d.Ys( ) ) {

}

/* *********************************************************************************************************//**
 *
 * @param a_domainUnit          [in]    The domain unit for the instance.
 * @param a_rangeUnit           [in]    The range unit for the instance.
 * @param a_start               [in]    The index of the x1 value the **Ys** data start at.
 * @param a_Ys                  [in]    The list of y values.
 ***********************************************************************************************************/

Ys1d::Ys1d( std::string const &a_domainUnit, std::string const &a_rangeUnit, std::size_t a_start, std::vector<double> const &a_Ys ) :
    Function1dForm( f_Ys1d, a_domainUnit, a_rangeUnit, ptwXY_interpolationLinLin, 0, 0.0 ),
    m_start( a_start ),
    m_Ys( a_Ys ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Ys1d::~Ys1d( ) {

}

/* *********************************************************************************************************//**
 * Adds two **Ys1d** instances and returns the result.
 *
 * @param a_rhs         [in]    The **Ys1d** instance to add to this instance.
 * @return                      An **Ys1d** instance that is the sum of this and *a_rhs*.
 ***********************************************************************************************************/

Ys1d Ys1d::operator+( Ys1d const &a_rhs ) const {

    Ys1d _Ys1d( *this );

    _Ys1d += a_rhs;
    return( _Ys1d );
}

/* *********************************************************************************************************//**
 * Adds an **Ys1d** instance to this.
 *
 * @param a_rhs         [in]    The **Ys1d** instance to add to this instance.
 * @return                      This instance.
 ***********************************************************************************************************/

Ys1d &Ys1d::operator+=( Ys1d const &a_rhs ) {

    if( length( ) == 0 ) m_start = a_rhs.length( );                        // Allow for empty (uninitialized) this.
    if( length( ) != a_rhs.length( ) ) throw std::runtime_error( "Ys1d::operator+=: lengths not equal." );

    long deltaStart = (long) a_rhs.start( );
    deltaStart -= (long)  m_start;
    if( deltaStart >= 0 ) {
        for( std::size_t i1 = 0; i1 < a_rhs.size( ); ++i1 ) m_Ys[i1+deltaStart] += a_rhs[i1]; }
    else {
        std::vector<double> _Ys( a_rhs.Ys( ) );

        for( std::size_t i1 = 0; i1 < size( ); ++i1 ) _Ys[i1-deltaStart] += m_Ys[i1];
        m_Ys = _Ys;
        m_start = a_rhs.start( );
    }
    return( *this );
}

/* *********************************************************************************************************//**
 * This is currently not implemented.
 *
 * @return          The domain minimum for the instance.
 ***********************************************************************************************************/

double Ys1d::domainMin( ) const {

#ifndef __NVCC__
    throw std::runtime_error( "Ys1d::domainMin: not implemented" );
#endif

    return( 0. );
}

/* *********************************************************************************************************//**
 * This is currently not implemented.
 *
 * @return                  The domain maximum for the instance.
 ***********************************************************************************************************/

double Ys1d::domainMax( ) const {

#ifndef __NVCC__
    throw std::runtime_error( "Ys1d::domainMax: not implemented" );
#endif

    return( 0. );
}

/* *********************************************************************************************************//**
 * This is currently not implemented.
 *
 * @param a_x1                  [in]    Domain value to evaluate this at.
 * @return
 ***********************************************************************************************************/

double Ys1d::evaluate( double a_x1 ) const {

#ifndef __NVCC__
    throw std::runtime_error( "Ys1d::evaluate: not implemented" );
#endif

    return( 0. );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 * @param       a_embedded          [in]        If *true*, *this* function is embedded in a higher dimensional function.
 * @param       a_inRegions         [in]        If *true*, *this* is in a Regions1d container.
 ***********************************************************************************************************/

void Ys1d::toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string attributes = a_writeInfo.addAttribute( "label", label( ) );
    
    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );
    axes( ).toXMLList( a_writeInfo, indent2 );
    doublesToXMLList( a_writeInfo, indent2, m_Ys, m_start );
    a_writeInfo.addNodeEnder( moniker( ) );
}

}

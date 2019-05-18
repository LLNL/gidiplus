/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "GIDI.hpp"

static void mutualifyDomains( ptwXYPoints const *a_lhs, ptwXYPoints const *a_rhs, ptwXYPoints **ptwXY1, ptwXYPoints **ptwXY2 );

namespace GIDI {

/*! \class XYs1d
 * Class to store GNDS <**XYs1d**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_values              [in]    The pair of xy values. Must be an even number of values.
 * @param a_domainUnit          [in]    The domain unit for the instance.
 * @param a_rangeUnit           [in]    The range unit for the instance.
 * @param a_interpolation       [in]    The interpolation flag.
 * @param a_index               [in]    If imbedded in a two dimensional function, the index of this instance.
 * @param a_outerDomainValue    [in]    If imbedded in a two dimensional function, the domain value for *x2*.
 ***********************************************************************************************************/

XYs1d::XYs1d( std::vector<double> const &a_values, std::string const &a_domainUnit, std::string const &a_rangeUnit, ptwXY_interpolation a_interpolation, int a_index, double a_outerDomainValue ) :
        Function1dForm( f_XYs1d, a_domainUnit, a_rangeUnit, a_interpolation, a_index, a_outerDomainValue ) {

    moniker( XYs1dMoniker );

    int64_t length = static_cast<int64_t>( a_values.size( ) ) / 2;
    double const *ptr = &a_values[0];

    m_ptwXY = ptwXY_create( NULL, ptwXY_interpolationLinLin, ptwXY_interpolationToString( ptwXY_interpolationLinLin ), 12, 1e-3, 100, 10, length, ptr, 0 );
}

/* *********************************************************************************************************//**
 *
 * @param a_domainUnit          [in]    The domain unit for the instance.
 * @param a_rangeUnit           [in]    The range unit for the instance.
 * @param a_interpolation       [in]    The interpolation flag.
 * @param a_index               [in]    If imbedded in a two dimensional function, the index of this instance.
 * @param a_outerDomainValue    [in]    If imbedded in a two dimensional function, the domain value for *x2*.
 ***********************************************************************************************************/

XYs1d::XYs1d( std::string const &a_domainUnit, std::string const &a_rangeUnit, ptwXY_interpolation a_interpolation, int a_index, double a_outerDomainValue ) :
        Function1dForm( f_XYs1d, a_domainUnit, a_rangeUnit, a_interpolation, a_index, a_outerDomainValue ) {

    m_ptwXY = ptwXY_new( NULL, ptwXY_interpolationLinLin, ptwXY_interpolationToString( ptwXY_interpolationLinLin ), 12, 1e-3, 100, 10, 0 );
}

/* *********************************************************************************************************//**
 * Constructs the instance from a **pugi::xml_node** instance.
 *
 * @param a_construction        [in]    Used to pass user options for parsing.
 * @param a_node                [in]    The XYs1d pugi::xml_node to be parsed and to construct the instance.
 * @param a_parent              [in]    If imbedded in a two dimensional function, its pointers.
 ***********************************************************************************************************/

XYs1d::XYs1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Function1dForm( a_construction, a_node, f_XYs1d, a_parent ) {

    char *endCharacter;

    m_ptwXY = ptwXY_fromString( NULL, a_node.child( "values" ).text( ).get( ), ' ', interpolation( ), 
            interpolationString( ).c_str( ), 12, 1e-3, &endCharacter, a_construction.useSystem_strtod( ) );
    if( m_ptwXY == NULL ) throw std::runtime_error( "XYs1d::XYs1d: ptwXY_fromString failed" );
}

/* *********************************************************************************************************//**
 * The XYs1d copy constructor.
 *
 * @param a_XYs1d               [in]    The XYs1d instance to copy.
 ***********************************************************************************************************/

XYs1d::XYs1d( XYs1d const &a_XYs1d ) :
        Function1dForm( a_XYs1d ),
        m_ptwXY( NULL ) {

    m_ptwXY = ptwXY_clone2( NULL, a_XYs1d.ptwXY( ) );
    if( m_ptwXY == NULL ) throw std::runtime_error( "XYs1d::XYs1d:2: ptwXY_clone2 failed" );
}

/* *********************************************************************************************************//**
 * Constructor that uses an existing **ptwXYPoints** instance. The **m_ptwXY** member is set to **ptwXYPoints** (i.e., this
 * XYs1d instance now owns the inputted **ptwXYPoints** instance).
 *
 * @param a_domainUnit          [in]    The domain unit for the instance.
 * @param a_rangeUnit           [in]    The range unit for the instance.
 * @param a_ptwXY               [in]    **The ptwXYPoints** to set **m_ptwXY** to.
 * @return
 ***********************************************************************************************************/

XYs1d::XYs1d( std::string const &a_domainUnit, std::string const &a_rangeUnit, ptwXYPoints *a_ptwXY ) :
    Function1dForm( f_XYs1d, a_domainUnit, a_rangeUnit, ptwXY_interpolationLinLin, 0, 0.0 ),
    m_ptwXY( a_ptwXY ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

XYs1d::~XYs1d( ) {

    ptwXY_free( m_ptwXY );
}

/* *********************************************************************************************************//**
 * The element access methods that returns a point (i.e., an x1, y pair).
 *
 * @param a_index       [in]    The index of the element to access.
 * @return                      The x1, y values at **a_index**.
 ***********************************************************************************************************/

std::pair<double, double> XYs1d::operator[]( std::size_t a_index ) const {

    if( (int64_t) a_index >= ptwXY_length( NULL, m_ptwXY ) ) throw std::runtime_error( "XYs1d::operator[]: index out of bounds." );

    ptwXYPoint *point = ptwXY_getPointAtIndex_Unsafely( m_ptwXY, (int64_t) a_index );
    std::pair<double, double> CPPPoint( point->x, point->y );

    return( CPPPoint );
}

/* *********************************************************************************************************//**
 * Adds two **XYs1d** instances and returns the result.
 * 
 * @param a_rhs         [in]    The **XYs1d** instance to add to this instance.
 * @return                      An **XYs1d** instance that is the sum of this and *a_rhs*.
 ***********************************************************************************************************/

XYs1d XYs1d::operator+( XYs1d const &a_rhs ) const {

    XYs1d __XYs1d( *this );

    __XYs1d += a_rhs;
    return( __XYs1d );
}

/* *********************************************************************************************************//**
 * Adds an **XYs1d** instance to this.
 *
 * @param a_rhs         [in]    The **XYs1d** instance to add to this instance.
 * @return                      This instance.
 ***********************************************************************************************************/

XYs1d &XYs1d::operator+=( XYs1d const &a_rhs ) {

    ptwXYPoints *sum, *ptwXY1, *ptwXY2;

    mutualifyDomains( m_ptwXY, a_rhs.ptwXY( ), &ptwXY1, &ptwXY2 );
    sum = ptwXY_add_ptwXY( NULL, ptwXY1, ptwXY2 );
    ptwXY_free( ptwXY1 );
    ptwXY_free( ptwXY2 );
    if( sum == NULL ) throw std::runtime_error( "XYs1d::operator+=: ptwXY_clone2 failed for sum" );

    ptwXY_free( m_ptwXY );
    m_ptwXY = sum;

    return( *this );
}

/* *********************************************************************************************************//**
 * Subtracts two **XYs1d** instances and returns the result.
 *
 * @param a_rhs         [in]    The **XYs1d** instance to substract from this instance.
 * @return                      An **XYs1d** instance that is the difference of this and *a_rhs*.
 ***********************************************************************************************************/

XYs1d XYs1d::operator-( XYs1d const &a_rhs ) const {

    XYs1d __XYs1d( *this );

    __XYs1d -= a_rhs;
    return( __XYs1d );
}

/* *********************************************************************************************************//**
 * Subtracts an **XYs1d** instance from this.
 *
 * @param a_rhs         [in]    The **XYs1d** instance to subtract from this instance.
 * @return                      This instance.
 ***********************************************************************************************************/

XYs1d &XYs1d::operator-=( XYs1d const &a_rhs ) {

    ptwXYPoints *sum, *ptwXY1, *ptwXY2;

    mutualifyDomains( m_ptwXY, a_rhs.ptwXY( ), &ptwXY1, &ptwXY2 );
    sum = ptwXY_sub_ptwXY( NULL, ptwXY1, ptwXY2 );
    ptwXY_free( ptwXY1 );
    ptwXY_free( ptwXY2 );
    if( sum == NULL ) throw std::runtime_error( "XYs1d::operator+=: ptwXY_clone2 failed for sum" );

    ptwXY_free( m_ptwXY );
    m_ptwXY = sum;

    return( *this );
}

/* *********************************************************************************************************//**
 * Returns the list of **x1** values of this.
 *
 * @return              The **x1** values of this.
 ***********************************************************************************************************/

std::vector<double> XYs1d::xs( ) const {

    int64_t n1 = size( );
    std::vector<double> _xs( n1, 0. );

    for( int64_t i1 = 0; i1 < n1; ++i1 ) {
        ptwXYPoint const *point = ptwXY_getPointAtIndex_Unsafely( m_ptwXY, i1 );

        _xs[i1] = point->x;
    }
    return( _xs );
}

/* *********************************************************************************************************//**
 * Returns the list of **y** values of this.
 *
 * @return              The **y** values of this.
 ***********************************************************************************************************/

std::vector<double> XYs1d::ys( ) const {

    int64_t n1 = size( );
    std::vector<double> _ys( n1, 0. );

    for( int64_t i1 = 0; i1 < n1; ++i1 ) {
        ptwXYPoint const *point = ptwXY_getPointAtIndex_Unsafely( m_ptwXY, i1 );

        _ys[i1] = point->y;
    }
    return( _ys );
}

/* *********************************************************************************************************//**
 * Returns a list of values that are this **y** mapped to the **x1** values in **a_xs**.
 *
 * @param a_xs              [in]    The list of **x1** values to map this' **y** values to.
 * @param a_offset          [out]   The index of the first value in **a_xs** where this starts.
 * @return                          The liist of **y** values.
 ***********************************************************************************************************/

std::vector<double> XYs1d::ysMappedToXs( std::vector<double> const &a_xs, std::size_t *a_offset ) const {

    int64_t n1 = size( ), i2, n2 = a_xs.size( );
    std::vector<double> _ys;

    *a_offset = 0;
    if( n1 == 0 ) return( _ys );

    ptwXYPoint const *point1 = ptwXY_getPointAtIndex_Unsafely( m_ptwXY, 0 );
    for( i2 = 0; i2 < n2; ++i2 ) if( point1->x <= a_xs[i2] ) break;
    *a_offset = i2;
    if( i2 == n2 ) return( _ys );

    for( int64_t i1 = 1; i1 < n1; ++i1 ) {
        ptwXYPoint const *point2 = ptwXY_getPointAtIndex_Unsafely( m_ptwXY, i1 );

        while( i2 < n2 ) {
            double x = a_xs[i2], y;
            if( x > point2->x ) break;           // Happens because of round off errors. Need to fix.

            ptwXY_interpolatePoint( NULL, ptwXY_interpolationLinLin, x, &y, point1->x, point1->y, point2->x, point2->y );
            _ys.push_back( y );
            ++i2;
            if( x >= point2->x ) break;         // This check can fail hence check above.
        }
        point1 = point2;
    }

    return( _ys );
}

/* *********************************************************************************************************//**
 * Returns an **XYs1d** instance that is this from its domain minimum to **domainMax**.
 *
 * @param a_domainMax           [in]    The maximum domain
 * @return                              An **XYs1d** instance.
 ***********************************************************************************************************/

XYs1d XYs1d::domainSliceMax( double a_domainMax ) const {

    ptwXYPoints *_ptwXY = ptwXY_clone2( NULL, m_ptwXY );
    if( _ptwXY == NULL ) throw std::runtime_error( "domainSliceMax: ptwXY_clone2 failed" );

    ptwXYPoints *ptwXYSliced = ptwXY_domainMaxSlice( NULL, _ptwXY, a_domainMax, 10, 1 );
    ptwXY_free( _ptwXY );
    if( ptwXYSliced == NULL ) throw std::runtime_error( "domainSliceMax: ptwXY_domainMaxSlice failed" );

    return( XYs1d( domainUnit( ), rangeUnit( ), ptwXYSliced ) );
}

/* *********************************************************************************************************//**
 * The **y** value of this at the domain value **a_x1**.
 *
 * @param a_x1          [in]    Domain value to evaluate this at.
 * @return                      The value of this at the domain value **a_x1**.
 ***********************************************************************************************************/

double XYs1d::evaluate( double a_x1 ) const {

    std::size_t length = ptwXY_length( NULL, m_ptwXY );
    if( length == 0 ) throw std::runtime_error( "XYs1d::evaluate: XYs1d has no datum." );

    ptwXYPoint *point = ptwXY_getPointAtIndex_Unsafely( m_ptwXY, 0 );
    if( point->x <= a_x1 ) return( point->y );

    point = ptwXY_getPointAtIndex_Unsafely( m_ptwXY, length - 1 );
    if( point->x >= a_x1 ) return( point->y );

    double y;
    nfu_status status = ptwXY_getValueAtX( NULL, m_ptwXY, a_x1, &y );
    if( status != nfu_Okay ) throw std::runtime_error( "XYs1d::evaluate: status != nfu_Okay" );
    return( y );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 * @param       a_embedded          [in]        If *true*, *this* function is embedded in a higher dimensional function.
 * @param       a_inRegions         [in]        If *true*, *this* is in a Regions1d container.
 ***********************************************************************************************************/

void XYs1d::toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const {

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

    if( interpolation( ) != ptwXY_interpolationLinLin ) attributes += a_writeInfo.addAttribute( "interpolation", interpolationString( ) );

    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );
    axes( ).toXMLList( a_writeInfo, indent2 );

    std::vector<double> doubles( 2 * size( ) );
    for( std::size_t i1 = 0; i1 < size( ); ++i1 ) {
        std::pair<double, double> point = (*this)[i1];
        doubles[2*i1] = point.first;
        doubles[2*i1+1] = point.second;
    }

    doublesToXMLList( a_writeInfo, indent2, doubles );
    a_writeInfo.addNodeEnder( moniker( ) );
}

}

/* *********************************************************************************************************//**
 * Returns to instances of **ptwXYPoints** that are the mutualified domains of **a_lhs** and **a_rhs**.
 *
 * @param a_lhs             [in]    One of the instances used to mutualify domains.
 * @param a_rhs             [in]    One of the instances used to mutualify domains.
 * @param a_ptwXY1          [out]   The mutualified domain for **a_lhs**.
 * @param a_ptwXY2          [out]   The mutualified domain for **a_rhs**.
 ***********************************************************************************************************/

static void mutualifyDomains( ptwXYPoints const *a_lhs, ptwXYPoints const *a_rhs, ptwXYPoints **a_ptwXY1, ptwXYPoints **a_ptwXY2 ) {

    double lowerEps = 1e-12, upperEps = 1e-12;

    *a_ptwXY1 = ptwXY_clone2( NULL, a_lhs );
    if( *a_ptwXY1 == NULL ) throw std::runtime_error( "mutualifyDomains: ptwXY_clone2 failed for a_ptwXY1" );

    *a_ptwXY2 = ptwXY_clone2( NULL, a_rhs );
    if( *a_ptwXY2 == NULL ) {
        ptwXY_free( *a_ptwXY1 );
        throw std::runtime_error( "mutualifyDomains: ptwXY_clone2 failed form a_ptwXY2" );
    }

    nfu_status status = ptwXY_mutualifyDomains( NULL, *a_ptwXY1, lowerEps, upperEps, 1, *a_ptwXY2, lowerEps, upperEps, 1 );
    if( status != nfu_Okay ) {
        ptwXY_free( *a_ptwXY1 );
        ptwXY_free( *a_ptwXY2 );
        throw std::runtime_error( "XYs1d::operator(+|-)=: mutualifyDomains in ptwXY_mutualifyDomains" );
    }
}

/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>
#include <algorithm>

#include "GIDI.hpp"

namespace GIDI {

/* *********************************************************************************************************//**
 * Function to parse a one-d flattened array.
 *
 * @param a_construction        [in]    Used to pass user options for parsing.
 * @param a_node                [in]    The **pugi::xml_node** to be parsed.
 * @param a_data                [out]   An empty GIDI::Vector that is filled with the data.
 *
 * @return                              0 if successfull and 1 otherwise.
 ***********************************************************************************************************/

int parseFlattened1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Vector &a_data ) {

    FlattenedArrayData arrayData( a_node, 1, a_construction.useSystem_strtod( ) );

    std::size_t size = (std::size_t) arrayData.m_shape[0];
    a_data.resize( size );

    std::size_t n1 = 0, n2 = size;
    for( std::size_t i1 = 0; i1 < arrayData.m_numberOfStarts; ++i1 ) {
        std::size_t offset = (std::size_t) arrayData.m_starts[i1];
        for( int32_t i2 = 0; i2 < arrayData.m_lengths[i1]; ++i2, ++n1, ++offset ) {
            if( n1 >= arrayData.m_dValues.size( ) ) throw Exception( "Too many values in flattened array." );
            if( offset >= n2 ) throw Exception( "requested size is too small." );
            a_data[offset] = arrayData.m_dValues[n1];
        }
    }
    return( 0 );
}

/* *********************************************************************************************************//**
 * Function to parse a flattened array of dimension **a_dimensions**.
 *
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 * @param a_dimensions      [int]   The dimension of the flattened array to be parsed.
 * @param a_useSystem_strtod    [in]    Flag passed to the function nfu_stringToListOfDoubles.
 ***********************************************************************************************************/

FlattenedArrayData::FlattenedArrayData( pugi::xml_node const &a_node, int a_dimensions, int a_useSystem_strtod ) :
        Form( a_node, FormType::flattenedArrayData ),
        m_numberOfStarts( 0 ), 
        m_numberOfLengths( 0 ),
        m_starts( NULL ),
        m_lengths( NULL ) {

    bool m_dValuesPresent( false );

    std::string shape( a_node.attribute( "shape" ).value( ) );
    long numberOfDimensions = (long) std::count( shape.begin( ), shape.end( ), ',' ), prior = 0, next;
    while( --numberOfDimensions >= 0 ) {
        next = shape.find( ",", prior );
        std::string value( shape.substr( prior, next - prior ) );
        prior = next + 1;
        m_shape.push_back( atoi( value.c_str( ) ) );
    }
    std::string value( shape.substr( prior ) );
    m_shape.push_back( atoi( value.c_str( ) ) );

    if( a_dimensions != (int) m_shape.size( ) ) throw Exception( "a_dimensions != m_shape.size( )" );

    for( pugi::xml_node child = a_node.first_child( ); child; child = child.next_sibling( ) ) {
        char *endCharacter;
        std::string name( child.name( ) );

        if( name == "values" ) {
            std::string label( child.attribute( "label" ).value( ) );
            char const *text = child.text( ).get( );

            if( label == "starts" ) {
                int64_t numberOfStarts;
                m_starts = nfu_stringToListOfInt32s( NULL, text, ' ', &numberOfStarts, &endCharacter );
                m_numberOfStarts = (std::size_t) numberOfStarts;
                if( m_starts == NULL ) throw Exception( "m_starts == NULL" );
                if( *endCharacter != 0 ) throw Exception( "bad 'starts' string" ); } 
            else if( label == "lengths" ) {
                int64_t numberOfLengths;
                m_lengths = nfu_stringToListOfInt32s( NULL, text, ' ', &numberOfLengths, &endCharacter );
                m_numberOfLengths = (std::size_t) numberOfLengths;
                if( m_lengths == NULL ) throw Exception( "m_lengths == NULL" );
                if( *endCharacter != 0 ) throw Exception( "bad 'lengths' string" ); }
            else if( label == "" ) {
                m_dValuesPresent = true;
                parseValuesOfDoubles( child, m_dValues, a_useSystem_strtod ); 
}
            else {
                throw Exception( "unknown label for flatteded array sub-element" );
            } }
        else {
            throw Exception( "unknown flattened array sub-element" );
        }
    }
    if( m_starts == NULL ) throw Exception( "array missing starts element" );
    if( m_lengths == NULL ) throw Exception( "array missing lengths element" );
    if( !m_dValuesPresent ) throw Exception( "array missing dValues element" );
    if( m_numberOfStarts != m_numberOfLengths ) throw Exception( "m_numberOfStarts != m_numberOfLengths for array" );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

FlattenedArrayData::~FlattenedArrayData( ) {

    smr_freeMemory2( m_starts );
    smr_freeMemory2( m_lengths );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void FlattenedArrayData::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::vector<int> ints;

    std::string shapeString;
    std::string sep = "";
    for( std::size_t i1 = 0; i1 < m_shape.size( ); ++i1 ) {
        shapeString += sep + intToString( m_shape[i1] );
        sep = ",";
    }

    std::string attributes = a_writeInfo.addAttribute( "shape", shapeString ) + a_writeInfo.addAttribute( "compression", "flattened" );

    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );

    for( std::size_t i1 = 0; i1 < m_numberOfStarts; ++i1 ) ints.push_back( m_starts[i1] );
    intsToXMLList( a_writeInfo, indent2, ints, " valueType=\"Integer32\" label=\"starts\"" );

    ints.clear( );
    for( std::size_t i1 = 0; i1 < m_numberOfLengths; ++i1 ) ints.push_back( m_lengths[i1] );
    intsToXMLList( a_writeInfo, indent2, ints, " valueType=\"Integer32\" label=\"lengths\"" );

    doublesToXMLList( a_writeInfo, indent2, m_dValues );
    a_writeInfo.addNodeEnder( moniker( ) );
}

}

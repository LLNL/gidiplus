/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>
#include <sstream>
#include <algorithm>

#include "GIDI.hpp"

namespace GIDI {

/*! \class Rate
 * Class to store GNDS <**rate**> node under a <**delayedNeutron**> node.
 */

/* *********************************************************************************************************//**
 * This function converts the text of a **pugi::xml_node** into a list of doubles.
 *
 * @param a_construction        [in]     Used to pass user options to the constructor.
 * @param a_node                [in]    The **pugi::xml_node** node whoses text is to be converted into a list of doubles.
 * @param a_parent              [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

Rate::Rate( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Form( a_node, f_rate, a_parent ),
        m_value( a_node.attribute( "value" ).as_double( ) ),
        m_unit( a_node.attribute( "unit" ).value( ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Rate::~Rate( ) {

}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void Rate::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string attributes;

    attributes += a_writeInfo.addAttribute( "label", label( ) );
    attributes += a_writeInfo.addAttribute( "value", doubleToShortestString( value( ) ) );
    attributes += a_writeInfo.addAttribute( "unit", unit( ) );

    a_writeInfo.addNodeStarterEnder( a_indent, moniker( ), attributes );
}

/* *********************************************************************************************************//**
 * This function takes a file path and returns its real path. On a Unix system, the system function realPath is called.
 *
 * @param a_path        [in]    The path whose real path is to be determined.
 *
 * @return                      The real path.
 ***********************************************************************************************************/

std::string realPath( char const *a_path ) {

    char *p1 = realpath( a_path, NULL );

    if( p1 == NULL ) {
        std::string errMsg( "realPath: file does not exist: " );
        throw std::runtime_error( errMsg + a_path );
    } 
    std::string basePath( p1 );
    free( p1 );
    return( basePath );
}

/* *********************************************************************************************************//**
 * This function takes a file path and returns its real path. On a Unix system, the system function realPath is called.
 *
 * @param a_path        [in]    The path whose real path is to be determined.
 *
 * @return                      The real path.
 ***********************************************************************************************************/

std::string realPath( std::string const &a_path ) {

    return( realPath( a_path.c_str( ) ) );
}

/* *********************************************************************************************************//**
 * This function splits that string *a_string* into separate strings using the delimiter character *a_delimiter*.
 *
 * @param a_string      [in]    The string to split.
 * @param a_delimiter   [in]    The delimiter character.
 *
 * @return                      The list of strings.
 ***********************************************************************************************************/

std::vector<std::string> splitString( std::string const &a_string, char a_delimiter ) {

    std::stringstream stringStream( a_string );
    std::string segment;
    std::vector<std::string> segments;
    int i1 = 0;

    while( std::getline( stringStream, segment, a_delimiter ) ) {
        if( ( i1 > 0 ) && ( segment.size( ) == 0 ) ) continue;      // Remove sequential "//".
        segments.push_back( segment );
        ++i1;
    }

    return( segments );
}

/* *********************************************************************************************************//**
 * This function searchs the list of ascending values *a_Xs* for the two values that bound *a_x* using a bi-section search.
 * If *a_x* is less than the first value, -2 is returned. If *a_x* is greater than the last value, -1 is returned.
 * Otherwise, the returned index will be such that *a_Xs*[index] <= *a_x* < *a_Xs*[index+1].
 *
 * @param a_x       [in]    The value to search.
 * @param a_Xs      [in]    The list of ascending values to 
 *
 * @return          [in]    The index within the *a_Xs* list that bounds *a_x*.
 ***********************************************************************************************************/

long binarySearchVector( double a_x, std::vector<double> const &a_Xs ) {
/*
*   Returns -2 is a_x < first point of a_Xs, -1 if > last point of a_Xs, and the lower index of a_Xs otherwise.
*/
    long size = a_Xs.size( );
    long imin = 0, imid, imax = size - 1;

    if( a_x < a_Xs[0] ) return( -2 );
    if( a_x > a_Xs[size-1] ) return( -1 );
    while( 1 ) {
        imid = ( imin + imax ) >> 1;
        if( imid == imin ) break;
        if( a_x < a_Xs[imid] ) {
            imax = imid; }
        else {
            imin = imid;
        }
    }
    return( imin );
}

/* *********************************************************************************************************//**
 * Adds the list of integers to the list of XML lines in *a_writeInfo*.
 *
 * @param a_writeInfo           [in/out]    Instance containing incremental indentation, values per line and other information and stores the appended lines.
 * @param a_indent              [in]        The amount to indent *this* node.
 * @param a_values              [in]        The list of integers to convert to strings and add to *a_writeInfo*.
 * @param a_attributes          [in]        String representation of the attributes for the GNDS **values** node.
 ***********************************************************************************************************/

void intsToXMLList( WriteInfo &a_writeInfo, std::string const &a_indent, std::vector<int> a_values, std::string const &a_attributes ) {

    a_writeInfo.addNodeStarter( a_indent, "values", a_attributes );

    std::string intString;
    std::string sep( "" );

    for( std::size_t i1 = 0; i1 < a_values.size( ); ++i1 ) {
        intString += sep + intToString( a_values[i1] );
        if( i1 == 0 ) sep = a_writeInfo.m_sep;
    }

    a_writeInfo.m_lines.back( ) += intString;
    a_writeInfo.addNodeEnder( "values" );
}

/* *********************************************************************************************************//**
 * This function converts the text of a **pugi::xml_node** into a list of doubles.
 *
 * @param a_construction        [in]     Used to pass user options to the constructor.
 * @param a_node        [in]    The **pugi::xml_node** node whoses text is to be converted into a list of doubles.
 * @param a_values      [in]    The list to fill with the converted values.
 ***********************************************************************************************************/

void parseValuesOfDoubles( Construction::Settings const &a_construction, pugi::xml_node const &a_node, std::vector<double> &a_values) {

    parseValuesOfDoubles( a_node, a_values, a_construction.useSystem_strtod( ) );
}

/* *********************************************************************************************************//**
 * This function converts the text of a **pugi::xml_node** into a list of doubles.
 *
 * @param a_node                [in]    The **pugi::xml_node** node whoses text is to be converted into a list of doubles.
 * @param a_values              [in]    The list to fill with the converted values.
 * @param a_useSystem_strtod    [in]    Flag passed to the function nfu_stringToListOfDoubles.
 ***********************************************************************************************************/

void parseValuesOfDoubles( pugi::xml_node const &a_node, std::vector<double> &a_values, int a_useSystem_strtod ) {

    int64_t numberConverted;
    char *endCharacter;

    double *p1 = nfu_stringToListOfDoubles( NULL, a_node.text( ).get( ), ' ', &numberConverted, &endCharacter, a_useSystem_strtod );
    if( p1 == NULL ) throw std::runtime_error( "parseValuesOfDoubles: nfu_stringToListOfDoubles returned NULL." );

    a_values.resize( numberConverted );
    for( int64_t i1 = 0; i1 < numberConverted; ++i1 ) a_values[i1] = p1[i1];
    smr_freeMemory2( p1 );
}

/* *********************************************************************************************************//**
 * Adds the list of doubles to the list of XML lines in *a_writeInfo*.
 *
 * @param a_writeInfo           [in/out]    Instance containing incremental indentation, values per line and other information and stores the appended lines.
 * @param a_indent              [in]        The amount to indent *this* node.
 * @param a_values              [in]        The list of doubles to convert to strings and add to *a_writeInfo*.
 * @param a_start               [in]        The value for the *start* attribute.
 * @param a_newLine             [in]        If *false*, the first *a_writeInfo.m_valuesPerLine* values are added to the last line with no indentation; otherwise, they are put on a new line with indentation.
 * @param a_valueType           [in]        The value for the *valueType* attribute.
 ***********************************************************************************************************/

void doublesToXMLList( WriteInfo &a_writeInfo, std::string const &a_indent, std::vector<double> a_values, std::size_t a_start, bool a_newLine, std::string const &a_valueType ) {

    int valuesPerLine( a_writeInfo.m_valuesPerLine );
    std::string indent( a_indent );
    std::string attributes;
    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string XMLLine;
    std::string sep = "";

    if( !a_newLine ) indent = "";
    if( a_valueType != "" ) attributes += a_writeInfo.addAttribute( "valueType", a_valueType );
    if( a_start != 0 ) attributes += a_writeInfo.addAttribute( "start", size_t_ToString( a_start ) );
    XMLLine = a_writeInfo.nodeStarter( indent, "values", attributes );

    if( valuesPerLine < 1 ) valuesPerLine = 1;
    int numberOfValuesInLine = 0;
    for( std::size_t i1 = 0; i1 < a_values.size( ); ++i1 ) {
        XMLLine += sep + doubleToShortestString( a_values[i1] );
        sep = a_writeInfo.m_sep;
        ++numberOfValuesInLine;

        if( numberOfValuesInLine == valuesPerLine ) {
            if( a_newLine ) {
                a_writeInfo.push_back( XMLLine ); }
            else {
                a_writeInfo.m_lines.back( ) += XMLLine;
            }
            numberOfValuesInLine = 0;
            XMLLine.clear( );
            XMLLine = indent2;
            a_newLine = true;
            sep = "";
        }
    }
    if( numberOfValuesInLine > 0 ) {
        if( a_newLine ) {
            a_writeInfo.push_back( XMLLine ); }
        else {
            a_writeInfo.m_lines.back( ) += XMLLine;
        } }
    else if( a_values.size( ) == 0 ) {
        a_writeInfo.push_back( XMLLine );
    }

    a_writeInfo.addNodeEnder( "values" );
}

/* *********************************************************************************************************//**
 * @param   a_incrementalIndent     [in]    The incremental amount of indentation a node adds to a sub-nodes indentation.
 * @param   a_valuesPerLine         [in]    The maximum number of integer or float values that are written per line before a new line is created.
 * @param   a_sep                   [in]    The separation character to use between integer and float values in a list.
 ***********************************************************************************************************/

WriteInfo::WriteInfo( std::string const &a_incrementalIndent, int a_valuesPerLine, std::string const &a_sep ) :
        m_incrementalIndent( a_incrementalIndent ),
        m_valuesPerLine( a_valuesPerLine ),
        m_sep( a_sep ) {

}

/* *********************************************************************************************************//**
 * This function returns an frame enum representing a **pugi::xml_node**'s attribute with name *a_name*.
 *
 * @param a_node        [in]    The **pugi::xml_node** node whoses attribute named *a_node* is to be parsed to determine the frame.
 * @param a_name        [in]    The name of the attribute to parse.
 *
 * @return                      The *frame* enum representing the node's frame.
 ***********************************************************************************************************/

frame parseFrame( pugi::xml_node const &a_node, std::string const &a_name ) {

    frame _frame = lab;
    if( strcmp( a_node.attribute( a_name.c_str( ) ).value( ), "centerOfMass" ) == 0 ) _frame = centerOfMass;
    return( _frame );
}

/* *********************************************************************************************************//**
 * This function converts the y-values from the Gridded1d into a Ys1d instance.
 *
 * @param a_function1d  [in]    The Gridded1d whoses y-values are converted into a Ys1d instance.
 *
 * @return                      A Ys1d instance of the y-values.
  ***********************************************************************************************************/

Ys1d gridded1d2GIDI_Ys1d( Function1dForm const &a_function1d ) {

    Ys1d ys1d = Ys1d( );

    switch( a_function1d.type( ) ) {
    case f_gridded1d :
        {
            Gridded1d const &gridded1d = static_cast<Gridded1d const &>( a_function1d );
            Vector const &data = gridded1d.data( );
            std::size_t start = 0;

            for( ; start < data.size( ); ++start ) {
                if( data[start] != 0 ) break;
            }
            ys1d.start( start );

            for( std::size_t i1 = start; i1 < data.size( ); ++i1 ) ys1d.push_back( data[i1] );
        }
        break;
    default :
        throw std::runtime_error( "gridded1d2GIDI_Ys1d: unsupported 1d function type " + a_function1d.label( ) );
    }

    return( ys1d );
}

/* *********************************************************************************************************//**
 * This function converts the values of a Vector into a Ys1d instance.
 *
 * @param a_vector      [in]    The Vector whoses values are converted into a Ys1d instance.
 *
 * @return                      A Ys1d instance of the values.
  ***********************************************************************************************************/

Ys1d vector2GIDI_Ys1d( Vector const &a_vector ) {

    std::size_t start = 0;
    Ys1d ys1d = Ys1d( );

    for( ; start < a_vector.size( ); ++start ) {
        if( a_vector[start] != 0 ) break;
    }
    ys1d.start( start );

    for( std::size_t i1 = start; i1 < a_vector.size( ); ++i1 ) ys1d.push_back( a_vector[i1] );

    return( ys1d );
}

/* *********************************************************************************************************//**
 * This function converts an integer gid value (i.e., group id) into the LLNL legacy bdfls label.
 *
 * @param a_gid         [in]    The integer gid used to construct the LLNL legacy bdfls label.
 *
 * @return                      The LLNL legacy bdfls label.
  ***********************************************************************************************************/

std::string LLNL_gidToLabel( int a_gid ) {

    char cLabel[64];

    sprintf( cLabel, "LLNL_gid_%d", a_gid );
    std::string label( cLabel );

    return( label );
}

/* *********************************************************************************************************//**
 * This function converts an integer fid value (i.e., flux id) into the LLNL legacy bdfls label.
 *
 * @param a_fid         [in]    The integer fid used to construct the LLNL legacy bdfls label.
 *
 * @return                      The LLNL legacy bdfls label.
  ***********************************************************************************************************/

std::string LLNL_fidToLabel( int a_fid ) {

    char cLabel[64];

    sprintf( cLabel, "LLNL_fid_%d", a_fid );
    std::string label( cLabel );

    return( label );
}

/* *********************************************************************************************************//**
 * This function returns an instance of *std::vector<std::string>* with only a_string as an item.
 * 
 * @param a_string      [in]    The string to add to the returned *std::vector<std::string>* instance.
 *
 * @return                      A *std::vector<std::string>* instance.
  ***********************************************************************************************************/

std::vector<std::string> vectorOfStrings( std::string const &a_string ) {
    std::vector<std::string> vectorOfStrings1;

    vectorOfStrings1.push_back( a_string );
    return( vectorOfStrings1 );
}

/* *********************************************************************************************************//**
 * This function returns a sorted instance of the strings in *a_strings*.
 *
 * @param a_strings             [in]    The string to add to the returned *std::vector<std::string>* instance.
 * @param a_orderIsAscending    [in]    If *true* the strings are sorted in ascending order; otherwise, descending order.
 *
 * @return                      A *std::vector<std::string>* instance.
  ***********************************************************************************************************/

std::vector<std::string> sortedListOfStrings( std::vector<std::string> const &a_strings, bool a_orderIsAscending ) {

    std::vector<std::string> keys( a_strings );

    std::sort( keys.begin( ), keys.end( ) );

    if( a_orderIsAscending ) return( keys );

    std::vector<std::string> keys2;

    for( std::vector<std::string>::reverse_iterator iter = keys.rbegin( ); iter != keys.rend( ); ++iter ) keys2.push_back( *iter );
    return( keys2 );
}

/* *********************************************************************************************************//**
 * This function returns a std::string representation of a *frame*.
 *
 * @param a_frame       [in]    The frame to convert to a string.
 *
 * @return                      A *std::string* instance.
  ***********************************************************************************************************/

std::string frameToString( frame a_frame ) {

    if( a_frame == lab ) return( "lab" );
    return( "centerOfMass" );
}

/* *********************************************************************************************************//**
 * Create the XML list for xs, pdf or cdf for an Xs_pdf_cdf1d instance.
 *
 * @param a_writeInfo       [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param a_nodeName        [in]        The name of the node (e.g., "xs" );
 * @param a_values          [in]        The list of doubles to wrap.
 *
 * @return                      A *std::string* instance.
  ***********************************************************************************************************/

std::string nodeWithValuesToDoubles( WriteInfo &a_writeInfo, std::string const &a_nodeName, std::vector<double> const &a_values ) {

    std::string xml = a_writeInfo.nodeStarter( "", a_nodeName );
    std::string sep( "" );

    xml += a_writeInfo.nodeStarter( "", "values" );
    for( std::size_t i1 = 0; i1 < a_values.size( ); ++i1 ) {
        xml += sep + doubleToShortestString( a_values[i1] );
        if( i1 == 0 ) sep = " ";
    }
    xml += a_writeInfo.nodeEnder( "values" );
    xml += a_writeInfo.nodeEnder( a_nodeName );

    return( xml );
}

/* *********************************************************************************************************//**
 * Returns a string representation of int *a_value*.
 *
 * @param a_value               [in]        The int value to convert to a string.
 *
 * @return                      A *std::string* instance.
  ***********************************************************************************************************/

std::string intToString( int a_value ) {

    char str[256];

    sprintf( str, "%d", a_value );
    return( std::string( str ) );
}

/* *********************************************************************************************************//**
 * Returns a string representation of std::size_t *a_value*.
 *
 * @param a_value               [in]        The std::size value to convert to a string.
 *
 * @return                      A *std::string* instance.
  ***********************************************************************************************************/

std::string size_t_ToString( std::size_t a_value ) {

    char str[256];

    sprintf( str, "%zu", a_value );
    return( std::string( str ) );
}

/* *********************************************************************************************************//**
 * Returns a string representation of *a_value* that contains the smallest number of character yet still agrees with *a_value*
 * to *a_significantDigits* significant digits. For example, for *a_value* = 1.20000000001, "1.2" will be returned if *a_significantDigits*
 * is less than 11, otherwise "1.20000000001" is returned.
 *
 * @param a_value               [in/out]    The double to convert to a string.
 * @param a_significantDigits   [in]        The number of significant digits the string representation should agree with the double.
 * @param a_favorEFormBy        [in]        The bigger this value the more likely an e-form will be favored in the string representation.
 *
 * @return                      A *std::string* instance.
  ***********************************************************************************************************/

std::string doubleToShortestString( double a_value, int a_significantDigits, int a_favorEFormBy ) {

    char *charValue = nf_floatToShortestString( a_value, a_significantDigits, a_favorEFormBy, nf_floatToShortestString_trimZeros );

    std::string stringValue( charValue );
    free( charValue );

    return( stringValue );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_moniker           [in]        The moniker for the energy type.
 * @param       a_indent            [in]        The amount to indent *this* node.
 * @param       a_function          [in]        The energy function whose information is converted to XML.
 ***********************************************************************************************************/

void energy2dToXMLList( WriteInfo &a_writeInfo, std::string const &a_moniker, std::string const &a_indent, Function1dForm *a_function ) {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );

    if( a_function == NULL ) return;

    a_writeInfo.addNodeStarter( a_indent, a_moniker, "" );
    a_function->toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( a_moniker );
}

}

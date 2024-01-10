/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <limits.h>
#include <stdlib.h>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <ctype.h>

#include <nf_utilities.h>

#include <LUPI.hpp>

namespace LUPI {

/* *********************************************************************************************************//**
 * If the build of GIDI+ defines the MACRO LUPI_printDeprecatedInformation, then all deprecated functions will
 * print a message that they are deprecated.
 * 
 * @param a_functionName        [in]    The name of the function (or method) that is deprecated.
 * @param a_replacementName     [in]    The name of the function that replaces the deprecated function.
 * @param a_asOf                [in]    Specifies the version of GIDI+ for which the function will no longer be available.
 ***********************************************************************************************************/

void deprecatedFunction( std::string const &a_functionName, std::string const &a_replacementName, std::string const &a_asOf ) {

#ifdef LUPI_printDeprecatedInformation
    std::cerr << "The function '" << a_functionName << "' is decreated";
    if( a_asOf != "" ) std::cerr << " and will no longer be available starting with GIDI+ '" << a_asOf << "'";
    std::cerr << ".";
    if( a_replacementName != "" ) std::cerr << " Please use '" << a_replacementName << "' instead.";
    std::cerr << std::endl;
#endif
}

/*! \class Exception
 * Exception class for all GIDI exceptions thrown by GIDI functions.
 */

/* *********************************************************************************************************//**
 * @param a_message         [in]     The message that the function what() will return.
 ***********************************************************************************************************/

Exception::Exception( std::string const & a_message ) :
        std::runtime_error( a_message ) {

}

namespace Misc {

/* *********************************************************************************************************//**
 * This returns a copy of *a_string* with its leading (if *a_left* is **true**) and trailing (if *a_left* is **true**) white spaces removed.
 *
 * @param a_string      [in]    The string to copy and strip leading and trailing white spaces from.
 * @param a_left        [in]    If **true**, white spaces are removed from the beginning of the string.
 * @param a_right       [in]    If **true**, white spaces are removed from the ending of the string.
 *
 * @return                      The list of strings.
 ***********************************************************************************************************/

std::string stripString( std::string const &a_string, bool a_left, bool a_right ) {

    std::string stripped( a_string );
    std::string::iterator beginning = stripped.begin( ), ending = stripped.end( );

    if( a_left ) {
        for( ; beginning != ending; ++beginning )
            if( !std::isspace( *beginning ) ) break;
    }

    if( ( beginning != ending ) && a_right ) {
        --ending;
        for( ; beginning != ending; --ending )
            if( !std::isspace( *ending ) ) break;
        ++ending;
    }

    stripped.erase( ending, stripped.end( ) );
    stripped.erase( stripped.begin( ), beginning );

    return( stripped );
}

/* *********************************************************************************************************//**
 * This function splits that string *a_string* into separate strings using the delimiter character *a_delimiter*.
 * If the delimiter is the space character, consecutive spaces are treated as one space, and leading and trailing
 * white spaces are ignored.
 *
 * @param a_string      [in]    The string to split.
 * @param a_delimiter   [in]    The delimiter character.
 * @param a_strip       [in]    If **true**, white spaces are removed from the begining and ending of each string in the list returned.
 *
 * @return                      The list of strings.
 ***********************************************************************************************************/

std::vector<std::string> splitString( std::string const &a_string, char a_delimiter, bool a_strip ) {

    std::stringstream stringStream( a_string );
    std::string segment;
    std::vector<std::string> segments;

    while( std::getline( stringStream, segment, a_delimiter ) ) {
        if( ( a_delimiter == ' ' ) && ( segment.size( ) == 0 ) ) continue;

        if( a_strip ) segment = stripString( segment );
        segments.push_back( segment );
    }

    return( segments );
}

/* *********************************************************************************************************//**
 * This function splits that string *a_string* into separate strings using the delimiter string *a_delimiter*.
 *
 * @param a_string      [in]    The string to split.
 * @param a_delimiter   [in]    The delimiter string.
 * @param a_strip       [in]    If **true**, white spaces are removed from the begining and ending of each string in the list returned.
 *
 * @return                      The list of strings.
 ***********************************************************************************************************/

std::vector<std::string> splitString( std::string const &a_string, std::string const &a_delimiter, bool a_strip ) {

    std::string segment;
    std::vector<std::string> segments;

    for( std::size_t index1 = 0; ; ) {
        std::size_t index2 = a_string.find( a_delimiter, index1 );

        segment = a_string.substr( index1, index2 - index1 );

        if( a_strip ) segment = stripString( segment );
        segments.push_back( segment );
        if( index2 == std::string::npos ) break;

        index1 = index2 + a_delimiter.size( );
    }

    return( segments );
}

/* *********************************************************************************************************//**
 * This function splits that string *a_string* into separate strings using the delimiter character "/".
 *
 * @param a_string      [in]    The string to split.
 *
 * @return                      The XLink parts as a list of strings.
 ***********************************************************************************************************/

std::vector<std::string> splitXLinkString( std::string const &a_string ) {

    std::vector<std::string> elements = splitString( a_string, '/' ), elements2;

    for( auto iter = elements.begin( ); iter != elements.end( ); ++iter ) {
        if( ( iter == elements.begin( ) ) || ( (*iter).size( ) != 0 ) ) elements2.push_back( *iter );
    }

    return( elements2 );
}

/* *********************************************************************************************************//**
 * Converts a string to an integer. All characteros of the string must be valid int characters except for the trailing 0.
 *
 * @param a_string              [in]        The string to convert to an int.
 * @param a_value               [in]        The converted int value.
 *
 * @return                                  true if successful and false otherwise.
  ***********************************************************************************************************/

bool stringToInt( std::string const &a_string, int &a_value ) {

    char const *digits = a_string.c_str( );
    char *nonDigit;
    long value = strtol( digits, &nonDigit, 10 );

    if( digits == nonDigit ) return( false );
    if( *nonDigit != 0 ) return( false );
    if( ( value < INT_MIN ) || ( value > INT_MAX ) ) return( false );

    a_value = static_cast<int>( value );
    return( true );
}

/* *********************************************************************************************************//**
 * Returns a string that represent the arguments formatted per *a_format*.
 *
 * @param a_format          [in]    A *printf* like format specifier for converting a double to a string.
 *
 * @return                          The string representing the arguments formatted per *a_format*.
 ***********************************************************************************************************/

std::string argumentsToString( char const *a_format, ... ) {

    va_list args;

    va_start( args, a_format );
    char *charStr = smr_vallocateFormatMessage( a_format, &args );
    va_end( args );

    std::string string( charStr );

    free( charStr );
    return( string );
}

/* *********************************************************************************************************//**
 * Returns a string that represent the double **a_value** using a *printf* like format specifier.
 *
 * @param a_format          [in]    A *printf* like format specifier for converting a double to a string.
 * @param a_value           [in]    The **double** to be converted to a string.
 * @param a_reduceBits      [in]    If **true** the lowest digit or two are altered in an attempt to convert numbers like 4.764999999999999 and 4.765 to the same string.
 ***********************************************************************************************************/

std::string doubleToString3( char const *a_format, double a_value, bool a_reduceBits ) {


    if( a_reduceBits ) {    // The next line is an attempt to convert numbers like 4.764999999999999 and 4.765 to the same value.
        a_value = std::stod( LUPI::Misc::argumentsToString("%.14e", a_value ) );
    }

    return( LUPI::Misc::argumentsToString( a_format, a_value ) );
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
 * For internal use only.
 *
 * @param a_indent          [in]    A string containing the help line for an argument up to the description string.
 * @param a_argc            [in]    The number of command arguments.
 * @param a_argv            [in]    The list of command arguments.
 ***********************************************************************************************************/

void printCommand( std::string const &a_indent, int a_argc, char **a_argv ) {

    std::cout << a_indent << a_argv[0];
    for( int iargc = 1; iargc < a_argc; ++iargc ) std::cout << " " << a_argv[iargc];
    std::cout << std::endl;
}

}               // End of namespace Misc.

}               // End of namespace LUPI.

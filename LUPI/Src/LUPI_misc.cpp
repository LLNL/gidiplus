/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>
#include <iostream>
#include <iomanip>

#include <LUPI.hpp>

namespace LUPI {

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

    char Str[256];

    if( a_reduceBits ) {
        sprintf( Str, "%.14e", a_value );         // This line and the next line are an attempt to convert numbers like 4.764999999999999 and 4.765 to the same value.
        a_value = std::stod( Str );
    }

    sprintf( Str, a_format, a_value );

    return( Str );
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

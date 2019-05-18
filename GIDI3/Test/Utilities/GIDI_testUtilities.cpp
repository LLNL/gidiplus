/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>
#include <iomanip>

#include "GIDI_testUtilities.hpp"

#define PRINT_NAME_WIDTH 20

/*
=========================================================
*/
argvOption::argvOption( std::string const &a_name, bool a_needsValue, std::string const &a_descriptor ) :
        m_name( a_name ),
        m_counter( 0 ),
        m_needsValue( a_needsValue ),
        m_descriptor( a_descriptor ) {

}
/*
=========================================================
*/
std::string argvOption::zeroOrOneOption( char **argv, std::string const &a_default ) {

    std::string msg( "ERROR: " );

    if( !m_needsValue ) throw std::runtime_error( msg + m_name + " does not have a value." );
    if( m_counter > 1 ) throw std::runtime_error( msg + m_name + " does not allow multiple arguments." );
    if( m_counter == 0 ) return( a_default );
    return( argv[m_indices[0]] );
}
/*
=========================================================
*/
void argvOption::help( ) {

    std::cout << "    " << std::left << std::setw( PRINT_NAME_WIDTH ) << m_name;
    if( m_needsValue ) {
        std::cout << " VALUE  "; }
    else {
        std::cout << "        ";
    }
    std::cout << m_descriptor << std::endl;
}
/*
=========================================================
*/
void argvOption::print( ) {

    std::cout << std::setw( PRINT_NAME_WIDTH ) << m_name;
    for( std::vector<int>::iterator iter = m_indices.begin( ); iter != m_indices.end( ); ++iter ) std::cout << " " << *iter;
    std::cout << std::endl;
}

/*
=========================================================
*/
argvOptions::argvOptions( std::string const &a_codeName, std::string const &a_descriptor, int a_minimumNonOptions ) :
        m_codeName( a_codeName ),
        m_descriptor( a_descriptor ),
        m_minimumNonOptions( a_minimumNonOptions ) {

    add( argvOption( "-h", false, "Show this help message and exit." ) );
}
/*
=========================================================
*/
void argvOptions::parseArgv( int argc, char **argv ) {

    for( int iargc = 1; iargc < argc; ++iargc ) {
        std::string arg( argv[iargc] );

        if( arg == "-h" ) help( );
        if( arg[0] == '-' ) {
            int index = 0;

            for( ; index < size( ); ++index ) {
                argvOption &option = m_options[index];

                if( option.m_name == arg ) {
                    ++option.m_counter;
                    if( option.m_needsValue ) {
                        ++iargc;
                        if( iargc == argc ) {
                            std::string msg( "ERROR: option '" );

                            throw std::runtime_error( msg + arg + "' has no value." );
                        }
                        option.m_indices.push_back( iargc );
                    }
                    break;
                }
            }

            if( index == size( ) ) {
                std::string msg( "ERROR: invalid option '" );

                throw std::runtime_error( msg + arg + "'." );
            } }
        else {
            m_arguments.push_back( iargc );
        }
    }

    if( static_cast<int>( m_arguments.size( ) ) < m_minimumNonOptions ) throw std::runtime_error( "Insufficient non-option arguments." );
}
/*
=========================================================
*/
argvOption *argvOptions::find( std::string const &a_name ) {

    for( std::vector<argvOption>::iterator iter = m_options.begin( ); iter != m_options.end( ); ++iter ) {
        if( iter->m_name == a_name ) return( &(*iter) );
    }
    return( NULL );
}
/*
=========================================================
*/
void argvOptions::help( ) {

    std::cout << std::endl << "Usage:" << std::endl;
    std::cout << "    " << m_codeName << std::endl;
    if( m_descriptor != "" ) {
        std::cout << std::endl << "Description:" << std::endl;
        std::cout << "    " << m_descriptor << std::endl;
    }

    if( m_options.size( ) > 0 ) {
        std::cout << std::endl << "Options:" << std::endl;
        for( std::vector<argvOption>::iterator iter = m_options.begin( ); iter != m_options.end( ); ++iter ) iter->help( );
    }

    exit( EXIT_SUCCESS );
}
/*
=========================================================
*/
void argvOptions::print( ) {

    std::cout << "Arugment indices:";
    for( std::vector<int>::iterator iter = m_arguments.begin( ); iter != m_arguments.end( ); ++iter ) std::cout << " " << *iter;
    std::cout << std::endl;
    for( std::vector<argvOption>::iterator iter = m_options.begin( ); iter != m_options.end( ); ++iter ) iter->print( );
}

/*
=========================================================
*/
int outputChannelStringMaximumLength( GIDI::Protare const *protare ) {

    std::size_t maximumLength = 0;

    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = protare->reaction( index );

        std::size_t length = outputChannelString( reaction ).size( );
        if( length > maximumLength ) maximumLength = length;
    }
    return( maximumLength );
}
/*
=========================================================
*/
std::string outputChannelString( GIDI::Reaction const *reaction ) {

    if( reaction->hasFission( ) ) return( "fission" );
    return( reaction->label( ) );
}
/*
=========================================================
*/
std::string outputChannelPrefix( int offset, int width, GIDI::Reaction const *reaction ) {

    std::string prefix( outputChannelString( reaction ) );

    prefix.insert( prefix.size( ), width - prefix.size( ), ' ' );
    prefix.insert( 0, offset, ' ' );
    return( prefix + ":" );
}
/*
=========================================================
*/
long integerFromArgv( int iarg, int argc, char **argv ) {

    char *endCharacter;

    if( iarg >= argc ) throw "integerFromArgv: iarg >= argc";

    long value = strtol( argv[iarg], &endCharacter, 10 );

    if( *endCharacter != 0 ) throw "integerFromArgv: invalid integer string.";

    return( value );
}
/*
=========================================================
*/
void printVector( char const *a_prefix, GIDI::Vector &vector ) {

    std::string prefix( a_prefix );

    printVector( prefix, vector );
}
/*
=========================================================
*/
void printVector( std::string &a_prefix, GIDI::Vector &vector ) {

    vector.print( a_prefix );
}
/*
=========================================================
*/
void printVectorOfDoubles( char const *a_prefix, std::vector<double> const &doubles ) {

    GIDI::Vector vector( doubles );

    printVector( a_prefix, vector );
}
/*
=========================================================
*/
void printVectorOfDoubles( std::string &a_prefix, std::vector<double> const &doubles ) {

    GIDI::Vector vector( doubles );

    printVector( a_prefix, vector );
}
/*
=========================================================
*/
void printIDs( char const *a_prefix, std::set<std::string> const &ids ) {

    std::string prefix( a_prefix );

    printIDs( prefix, ids );
}
/*
=========================================================
*/
void printIDs( std::string &a_prefix, std::set<std::string> const &ids ) {

    std::cout << a_prefix;
    for( std::set<std::string>::const_iterator iter = ids.begin( ); iter != ids.end( ); ++iter ) std::cout << " " << *iter;
    std::cout << std::endl;
}
/*
=========================================================
*/
void printMatrix( char const *a_prefix, int maxOrder, GIDI::Matrix &matrix ) {

    std::string prefix( a_prefix );

    printMatrix( prefix, maxOrder, matrix );
}
/*
=========================================================
*/
void printMatrix( std::string &prefix, int maxOrder, GIDI::Matrix &matrix ) {

    std::cout << std::endl << prefix << std::endl;
    if( maxOrder > -2 ) std::cout << "    max. Legendre order = " << maxOrder << std::endl;
    matrix.print( "    ::  " );
}
/*
=========================================================
*/
std::string stripDirectoryBase( std::string const &a_path, std::string const &a_section ) {

    std::string path( a_path );
    std::size_t location( path.find( a_section ) );

    if( location != std::string::npos ) path = "/..." + path.substr( location );

    return( path );
}
/*
=========================================================
*/
std::string doubleToString( char const *format, double value ) {

    char Str[256];

    sprintf( Str, format, value );
    std::string valueAsString( Str );

    return( valueAsString );
}

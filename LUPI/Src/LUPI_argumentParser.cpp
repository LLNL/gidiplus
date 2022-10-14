/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>
#include <libgen.h>
#include <iostream>

#include <LUPI.hpp>

namespace LUPI {

static void printArgumentDescription( std::string const &a_line, std::string const &a_descriptor );
static std::size_t maxPrintLineWidth = 120;

/*! \class ArgumentBase
 * Base class for argument and option sub-classes.
 */

/* *********************************************************************************************************//**
 * ArgumentBase destructor.
 ***********************************************************************************************************/

ArgumentBase::ArgumentBase( ArgumentType a_argumentType, std::string const &a_name, std::string const &a_descriptor, int a_minimumNeeded, int a_maximumNeeded ) :
        m_argumentType( a_argumentType ),
        m_names( ),
        m_descriptor( a_descriptor ),
        m_minimumNeeded( a_minimumNeeded ),
        m_maximumNeeded( a_maximumNeeded ),
        m_numberEntered( 0 ) {

    if( m_minimumNeeded < 0 ) throw std::runtime_error( "ERROR 1000 in ArgumentBase::ArgumentBase: m_minimumNeeded must not be negative." );
    if( m_maximumNeeded > -1 ) {
        if( m_minimumNeeded > m_maximumNeeded )
            throw std::runtime_error( "ERROR 1010 in ArgumentBase::ArgumentBase: for argument '" + a_name + "' m_maximumNeeded less than m_minimumNeeded." );
    }

    if( m_argumentType == ArgumentType::Positional ) {
        if( a_name[0] == '-' )
            throw std::runtime_error( "ERROR 1020 in ArgumentBase::ArgumentBase: positional argument name '" + a_name + "' cannot start with a '-'." );
        m_names.push_back( a_name ); }
    else {
        addAlias( a_name );
    }
}

/* *********************************************************************************************************//**
 * ArgumentBase destructor.
 ***********************************************************************************************************/

ArgumentBase::~ArgumentBase( ) {

}

/* *********************************************************************************************************//**
 * Returns true if *a_name* is one of the names for *this* and false otherwise.
 *
 * @param a_name            [in]    The name to search for.
 *
 * @return                          Returns true if *a_name* is a match for one of the names for *this* and false otherwise.
 ***********************************************************************************************************/

bool ArgumentBase::hasName( std::string const &a_name ) const {

    for( auto iter = m_names.begin( ); iter != m_names.end( ); ++iter ) {
        if( a_name == *iter ) return( true );
    }
    return( false );

}

/* *********************************************************************************************************//**
 * Add *a_name* as an optional name for *this*.
 *
 * @param a_name            [in]    The name to add.
 ***********************************************************************************************************/

void ArgumentBase::addAlias( std::string const &a_name ) {

    if( m_argumentType == ArgumentType::Positional )
        throw std::runtime_error( "ERROR 1100 in ArgumentBase::addAlias: cannot add a name to a positional argument." );
    if( a_name[0] != '-' ) throw std::runtime_error( "ERROR 1110 in ArgumentBase::addAlias: name '" + a_name + "' not a valid optional name." );
    if( hasName( a_name ) ) return;

    m_names.push_back( a_name );
}

/* *********************************************************************************************************//**
 * Counts each time a specific argument/options is found.
 *
 * @param a_index           [in]    The index of the current command argument in *a_argv*.
 * @param a_argc            [in]    The number of command arguments.
 * @param a_argv            [in]    The list of command arguments.
 *
 * @return                          The value of *a_index* + 1.
 ***********************************************************************************************************/

int ArgumentBase::parse( ArgumentParser const &a_argumentParser, int a_index, int a_argc, char **a_argv ) {

    ++m_numberEntered;

    return( a_index + 1 );
}

/* *********************************************************************************************************//**
 * Returns the usage string for *this* option.
 *
 * @param a_requiredOption      [in]    The index of the current command argument in *a_argv*.
 *
 * @return                              The value of *a_index* + 1.
 ***********************************************************************************************************/

std::string ArgumentBase::usage( bool a_requiredOption ) const {

    std::string usageString;

    if( isOptionalArgument( ) ) {
        if( a_requiredOption ) {
            if( m_minimumNeeded != 0 ) return( usageString ); }
        else {
            if( m_minimumNeeded == 0 ) return( usageString );
        }
    }
    usageString += " ";

    std::string value;
    if( isOptionalArgument( ) && requiresAValue( ) ) value = " VALUE";
    if( isOptionalArgument( ) && ( m_minimumNeeded == 0 ) ) {
        usageString += "[" + name( ) + value + "]"; }
    else {
        usageString += name( ) + value;
    }

    if( !isOptionalArgument( ) ) {
        if( ( m_minimumNeeded != 1 ) || ( m_maximumNeeded != 1 ) )
            usageString += "[" + std::to_string( m_minimumNeeded ) + "," + std::to_string( m_maximumNeeded ) + "]";
    }

    return( usageString );
}

/* *********************************************************************************************************//**
 * Prints generic information about the status of *this*.
 *
 * @param a_indent          [in]    The amount of indentation to start the first line with.
 *
 * @return                          The value of *a_index* + 1.
 ***********************************************************************************************************/

void ArgumentBase::printStatus( std::string a_indent ) const {

    std::string name1 = name( );
    if( name1.size( ) < 32 ) name1.resize( 32, ' ' );

    std::cout << a_indent << name1 << ": number entered " << std::to_string( m_numberEntered )
            << "; number needed (" << std::to_string( m_minimumNeeded ) << "," << std::to_string( m_maximumNeeded ) << ")" 
            << printStatus2( ) << std::endl;
    printStatus3( a_indent + "    " );
}

/* *********************************************************************************************************//**
 * Called by *printStatus*. This method returns an empty string. Must be overwritten by argument classes that have value.
 ***********************************************************************************************************/

std::string ArgumentBase::printStatus2( ) const {

    return( "" );

}

/* *********************************************************************************************************//**
 * Called by *printStatus*. This method does nothing. Must be overwritten by argument classes that have value(s).
 *
 * @param a_indent          [in]    The amount of indentation to start the first line with.
 ***********************************************************************************************************/

void ArgumentBase::printStatus3( std::string const &a_indent ) const {

}

/*! \class OptionBoolean
 * Base boolean class.
 */

/* *********************************************************************************************************//**
 * OptionBoolean destructor.
 ***********************************************************************************************************/

OptionBoolean::~OptionBoolean( ) {

}

/* *********************************************************************************************************//**
 * Returns the *true* or *false* status of *this*.
 ***********************************************************************************************************/

bool OptionBoolean::value( ) const {

    if( numberEntered( ) == 0 ) return( m_default );
    return( !m_default );
}

/* *********************************************************************************************************//**
 * Called by *printStatus*. This method return a string representing *this*'s value.
 ***********************************************************************************************************/

std::string OptionBoolean::printStatus2( ) const {

    if( value( ) ) return( ": true" );
    return( ": false" );
}

/*! \class OptionTrue
 * An boolean optional argument whose default is false and changes to true if one or more options are entered.
 */

/*! \class OptionFalse
 * An boolean optional argument whose default is true and changes to false if one or more options are entered.
 */

/*! \class OptionCounter
 * An optional argument the count the number of times the option is entered.
 */

/* *********************************************************************************************************//**
 * Called by *printStatus*. This method return a string representing *this*'s value.
 ***********************************************************************************************************/

std::string OptionCounter::printStatus2( ) const {

    return( " counts = " + std::to_string( counts( ) ) );
}

/*! \class OptionStore
 * An option with a value. If multiple options with the same name are entered, the *m_value* member will represent the last option entered.
 */

/* *********************************************************************************************************//**
 * Calls **ArgumentBase::parse** and sets *m_value* to the next argument.
 *
 * @param a_index           [in]    The index of the current command argument in *a_argv*.
 * @param a_argc            [in]    The total number of command arguments.
 * @param a_argv            [in]    The list of command arguments.
 *
 * @return                          The value of *a_index*.
 ***********************************************************************************************************/

int OptionStore::parse( ArgumentParser const &a_argumentParser, int a_index, int a_argc, char **a_argv ) {

    a_index = ArgumentBase::parse( a_argumentParser, a_index, a_argc, a_argv );
    if( a_index == a_argc ) throw std::runtime_error( "ERROR 1200 in OptionStore::parse: missing value for argument " + name( ) + "." );

    m_value = a_argv[a_index];

    return( ++a_index );
}

/* *********************************************************************************************************//**
 * Prints the value for *this*. Called by *printStatus*. 
 *
 * @param a_indent          [in]    The amount of indentation to start the first line with.
 ***********************************************************************************************************/

void OptionStore::printStatus3( std::string const &a_indent ) const {

    if( numberEntered( ) > 0 ) std::cout << a_indent << m_value << std::endl;
}

/*! \class OptionAppend
 * An option with a value. If multiple options with the same name are entered, the *m_value* member will represent the last option entered.
 */

/* *********************************************************************************************************//**
 * Calls **ArgumentBase::parse** and appends the next argument to *this*.
 *
 * @param a_index           [in]    The index of the current command argument in *a_argv*.
 * @param a_argc            [in]    The total number of command arguments.
 * @param a_argv            [in]    The list of command arguments.
 *
 * @return                          The value of *a_index*.
 ***********************************************************************************************************/

int OptionAppend::parse( ArgumentParser const &a_argumentParser, int a_index, int a_argc, char **a_argv ) {

    a_index = ArgumentBase::parse( a_argumentParser, a_index, a_argc, a_argv );

    if( a_index == a_argc ) throw std::runtime_error( "ERROR 1210 in OptionAppend::parse: missing value for argument " + name( ) + "." );
    if( static_cast<int>( m_values.size( ) ) == maximumNeeded( ) ) throw std::runtime_error( "ERROR 1220 in OptionAppend::parse: too many values for argument " + name( ) + " entered." );

    m_values.push_back( a_argv[a_index] );

    return( ++a_index );
}

/* *********************************************************************************************************//**
 * Prints the values for *this*. Called by *printStatus*. 
 *
 * @param a_indent          [in]    The amount of indentation to start the first line with.
 ***********************************************************************************************************/

void OptionAppend::printStatus3( std::string const &a_indent ) const {

    for( auto valueIterator = m_values.begin( ); valueIterator != m_values.end( ); ++valueIterator ) {
        std::cout << a_indent << *valueIterator << std::endl;
    }
}

/*! \class Positional
 * An option with a value. If multiple options with the same name are entered, the *m_value* member will represent the last option entered.
 */ 
 
/* *********************************************************************************************************//**
 * Calls **ArgumentBase::parse** and appends the next arguments to *this* until *m_maximumNeeded* is reached
 * or there are no more positional arguments (i.e., an option is detected or there are no more arguments).
 * 
 * @param a_index           [in]    The index of the current command argument in *a_argv*.
 * @param a_argc            [in]    The total number of command arguments.
 * @param a_argv            [in]    The list of command arguments.
 *
 * @return                          The value of *a_index*.
 ***********************************************************************************************************/

int Positional::parse( ArgumentParser const &a_argumentParser, int a_index, int a_argc, char **a_argv ) {

            // The following error should never append.
    if( a_index == a_argc ) throw std::runtime_error( "ERROR 1300 in Positional::parse: missing value for positional argument '" + name( ) + "'." );

    int maximumNeeded1 = maximumNeeded( );
    if( maximumNeeded1 < 0 ) maximumNeeded1 = a_argc;
    for( int index = 0; index < maximumNeeded1; ++index ) {
        if( a_index == a_argc ) break;
        if( index >= minimumNeeded( ) ) {                // Break if an option detected and minimumNeeded reached. Need to check for negative number.
            if( a_argumentParser.hasName( a_argv[a_index] ) ) break;
        }

        m_values.push_back( a_argv[a_index] );
        a_index = ArgumentBase::parse( a_argumentParser, a_index, a_argc, a_argv );
    }

    if( static_cast<int>( m_values.size( ) ) < minimumNeeded( ) )
        throw std::runtime_error( "ERROR 1310 in Positional::parse: too few values for positional argument '" + name( )
                + "' entered. Got " + std::to_string( m_values.size( ) ) + " expect " + std::to_string( minimumNeeded( ) ) 
                + " to " + std::to_string( maximumNeeded( ) ) + "." );


    return( a_index );
}

/* *********************************************************************************************************//**
 * Prints the values for *this*. Called by *printStatus*. 
 *
 * @param a_indent          [in]    The amount of indentation to start the first line with.
 ***********************************************************************************************************/

void Positional::printStatus3( std::string const &a_indent ) const {

    for( auto valueIterator = m_values.begin( ); valueIterator != m_values.end( ); ++valueIterator ) {
        std::cout << a_indent << *valueIterator << std::endl;
    }
}

/*! \class ArgumentParser
 * The main argument parser class.
 */

/* *********************************************************************************************************//**
 * ArgumentParser constructor.
 ***********************************************************************************************************/

ArgumentParser::ArgumentParser( std::string const &a_codeName, std::string const &a_descriptor ) :
        m_codeName( FileInfo::basenameWithoutExtension( a_codeName ) ),
        m_descriptor( a_descriptor ) {

}

/* *********************************************************************************************************//**
 * ArgumentParser destructor.
 ***********************************************************************************************************/

ArgumentParser::~ArgumentParser( ) {

    for( auto argumentIterator = m_arguments.begin( ); argumentIterator != m_arguments.end( ); ++argumentIterator ) {
        delete *argumentIterator;
    }
}

/* *********************************************************************************************************//**
 * Add *a_argumentBase* to the list of arguments.
 *
 * @param a_argumentBase   [in]     Pointer to the *ArgumentBase* to add to *this*.
 ***********************************************************************************************************/

void ArgumentParser::add2( ArgumentBase *a_argumentBase ) {

    if( !a_argumentBase->isOptionalArgument( ) ) {
        for( auto argumentIterator = m_arguments.rbegin( ); argumentIterator != m_arguments.rend( ); ++argumentIterator ) {
            if( !(*argumentIterator)->isOptionalArgument( ) ) {
                if( (*argumentIterator)->minimumNeeded( ) == (*argumentIterator)->maximumNeeded( ) ) break;
                throw std::runtime_error( "ERROR 1400 in ArgumentParser::add: request to add postional argument when prior postional argument '" 
                        + (*argumentIterator)->name( ) + "' takes a variable number of values." );
            }
        }
    }

    if( hasName( a_argumentBase->name( ) ) )
        throw std::runtime_error( "ERROR 1500 in ArgumentParser::add: name '" + a_argumentBase->name( ) + "' already present." );

    m_arguments.push_back( a_argumentBase );
}

/* *********************************************************************************************************//**
 * Adds the alias *a_alias* to the argument named *a_name*.
 *
 * @param a_name            [in]    The name of the argument to add the alias to.
 * @param a_alias           [in]    The alias name to add.
 ***********************************************************************************************************/

void ArgumentParser::addAlias( std::string const &a_name, std::string const &a_alias ) {

    if( hasName( a_alias ) )
        throw std::runtime_error( "ERROR 1510 in ArgumentParser::addAlias: name '" + a_alias + "' already present." );

    for( auto argumentIterator = m_arguments.begin( ); argumentIterator != m_arguments.end( ); ++argumentIterator ) {
        if( (*argumentIterator)->hasName( a_name ) ) {
            (*argumentIterator)->addAlias( a_alias );
            return;
        }
    }
    throw std::runtime_error( "ERROR 1520 in ArgumentParser::addAlias: no such argument named '" + a_name + "'." );
}

/* *********************************************************************************************************//**
 * Adds the alias *a_alias* to the argument *a_argumentBase*.
 *
 * @param a_argumentBase    [in]    The argument to add the alias to.
 * @param a_alias           [in]    The name of the argument to add the alias to.
 ***********************************************************************************************************/

void ArgumentParser::addAlias( ArgumentBase const * const a_argumentBase, std::string const &a_alias ) {

    addAlias( a_argumentBase->name( ), a_alias );
}

/* *********************************************************************************************************//**
 * Returns true if name *a_name* is in *this* and false otherwise.
 *
 * @param a_name            [in]    The name to see check if it exists in *this*.
 *
 * @return                          true if name *a_name* is in *this* and false otherwise.
 ***********************************************************************************************************/

bool ArgumentParser::hasName( std::string const &a_name ) const {

    for( auto argumentIterator = m_arguments.begin( ); argumentIterator != m_arguments.end( ); ++argumentIterator ) {
        if( (*argumentIterator)->hasName( a_name ) ) return( true );
    }

    return( false );
}

/* *********************************************************************************************************//**
 * Parses the list of arguments.
 *
 * @param a_argc            [in]    The number of arguments.
 * @param a_argv            [in]    The list of arguments.
 ***********************************************************************************************************/

void ArgumentParser::parse( int a_argc, char **a_argv ) {

    for( int iargc = 1; iargc < a_argc; ++iargc ) {                                       // Check is help requested.
        std::string arg( a_argv[iargc] );

        if( ( arg == "-h" ) || ( arg == "--help" ) ) help( );
    }

    auto argumentIterator = m_arguments.begin( );      // Find first non-option argument.
    for( ; argumentIterator != m_arguments.end( ); ++argumentIterator ) {
        if( !(*argumentIterator)->isOptionalArgument( ) ) break;
    }

    int iargc = 1;
    for( ; iargc < a_argc; ) {
        std::string arg( a_argv[iargc] );

        if( arg[0] == '-' ) {                                                           // Need to check if negative number.
            auto argumentIterator2 = m_arguments.begin( );
            for( ; argumentIterator2 != m_arguments.end( ); ++argumentIterator2 ) {
                if( (*argumentIterator2)->hasName( arg ) ) break;
            }
            if( argumentIterator2 == m_arguments.end( ) ) throw std::runtime_error( "ERROR 1600 in ArgumentParser::parse: invalid option '" + arg + "'." );
            iargc = (*argumentIterator2)->parse( *this, iargc, a_argc, a_argv ); }
        else {
            if( argumentIterator == m_arguments.end( ) )
                throw std::runtime_error( "ERROR 1610 in ArgumentParser::parse: additional positional argument found starting at index " 
                        + std::to_string( iargc ) + " (" + arg + ")" );

            iargc = (*argumentIterator)->parse( *this, iargc, a_argc, a_argv );

            ++argumentIterator;
            for( ; argumentIterator != m_arguments.end( ); ++argumentIterator ) {       // Find next positional arguments.
                if( !(*argumentIterator)->isOptionalArgument( ) ) break;
            }
        }
    }

    for( auto argumentIterator = m_arguments.begin( ); argumentIterator != m_arguments.end( ); ++argumentIterator ) {
        if( (*argumentIterator)->numberEntered( ) < (*argumentIterator)->minimumNeeded( ) ) {
            std::string msg( "arguments for" );

            if( (*argumentIterator)->isOptionalArgument( ) ) msg = "number of option";
            throw std::runtime_error( "ERROR 1620 in ArgumentParser::parse: insufficient " + msg + " '" + (*argumentIterator)->name( ) 
                    + "' entered. Range of " + std::to_string( (*argumentIterator)->minimumNeeded( ) ) + " to " 
                    + std::to_string( (*argumentIterator)->maximumNeeded( ) ) + " required, " 
                    + std::to_string( (*argumentIterator)->numberEntered( ) ) + " entered." );
        }
    }

    std::cerr << "    " << LUPI::FileInfo::basenameWithoutExtension( m_codeName );
    for( int i1 = 1; i1 < a_argc; i1++ ) std::cerr << " " << a_argv[i1];
    std::cerr << std::endl;
}

/* *********************************************************************************************************//**
 * Prints the help for *this*.
 ***********************************************************************************************************/

void ArgumentParser::help( ) const {

    usage( );

    if( m_descriptor != "" ) {
        std::cout << std::endl << "Description:" << std::endl;
        std::cout << "    " << m_descriptor << std::endl;
    }

    bool printHeader = true;
    for( auto argumentIterator = m_arguments.begin( ); argumentIterator != m_arguments.end( ); ++argumentIterator ) {
        if( (*argumentIterator)->isOptionalArgument( ) ) continue;

        if( printHeader ) std::cout << std::endl << "positional arguments:" << std::endl;
        printHeader = false;

        std::string line = (*argumentIterator)->name( );
        if( ( (*argumentIterator)->minimumNeeded( ) != (*argumentIterator)->maximumNeeded( ) ) or ( (*argumentIterator)->maximumNeeded( ) != 1 ) )
                    line += " [" + std::to_string( (*argumentIterator)->minimumNeeded( ) ) + "," + std::to_string( (*argumentIterator)->maximumNeeded( ) ) + "]";
        printArgumentDescription( line, (*argumentIterator)->descriptor( ) );
    }

    std::cout << std::endl << "optional arguments:" << std::endl;
    std::cout << "  -h, --help                  Show this help message and exit." << std::endl;
    for( auto argumentIterator = m_arguments.begin( ); argumentIterator != m_arguments.end( ); ++argumentIterator ) {
        if( !(*argumentIterator)->isOptionalArgument( ) ) continue;

        std::string line;
        std::string sep;
        for( auto namesIterator = (*argumentIterator)->names( ).begin( ); namesIterator != (*argumentIterator)->names( ).end( ); ++namesIterator ) {
            line += sep + *namesIterator;
            sep = ", ";
        }
        if( (*argumentIterator)->requiresAValue( ) ) line += " VALUE";
        if( (*argumentIterator)->argumentType( ) == LUPI::ArgumentType::Append ) {
            if( ( (*argumentIterator)->minimumNeeded( ) != (*argumentIterator)->maximumNeeded( ) ) or ( (*argumentIterator)->maximumNeeded( ) != 1 ) )
                    line += " [" + std::to_string( (*argumentIterator)->minimumNeeded( ) ) + "," + std::to_string( (*argumentIterator)->maximumNeeded( ) ) + "]";
        }
        printArgumentDescription( line, (*argumentIterator)->descriptor( ) );
    }

    exit( EXIT_SUCCESS );
}

/* *********************************************************************************************************//**
 * Prints the usage for *this*.
 ***********************************************************************************************************/

void ArgumentParser::usage( ) const {

    std::string line( "usage: " );
    line += codeName( );
    std::string indent( "" );
    indent.resize( line.size( ), ' ' );

    for( int counter = 0; counter < 2; ++counter ) {
        for( auto argumentIterator = m_arguments.begin( ); argumentIterator != m_arguments.end( ); ++argumentIterator ) {
            if( (*argumentIterator)->isOptionalArgument( ) ) {
                std::string optionUsage( (*argumentIterator)->usage( counter == 0 ) );

                if( ( line.size( ) + optionUsage.size( ) ) > maxPrintLineWidth ) {
                    std::cout << line << std::endl;
                    line = indent;
                }
                line += optionUsage;
            }
        }
    }

    for( auto argumentIterator = m_arguments.begin( ); argumentIterator != m_arguments.end( ); ++argumentIterator ) {
        if( (*argumentIterator)->isOptionalArgument( ) ) continue;
        std::string optionUsage( (*argumentIterator)->usage( false ) );

        if( ( line.size( ) + optionUsage.size( ) ) > maxPrintLineWidth ) {
            std::cout << line << std::endl;
            line = indent;
        }
        line += optionUsage;
    }
    if( line.size( ) > indent.size( ) ) std::cout << line << std::endl;
    std::cout << std::endl;
}

/* *********************************************************************************************************//**
 * Returns the usage string for *this* option.
 *
 * @param a_indent          [in]    The amount of indentation to start the first line with.
 *
 * @return                          The value of *a_index* + 1.
 ***********************************************************************************************************/

void ArgumentParser::printStatus( std::string a_indent ) const {

    for( auto argumentIterator = m_arguments.begin( ); argumentIterator != m_arguments.end( ); ++argumentIterator ) {
        (*argumentIterator)->printStatus( a_indent );
    }
}


/* *********************************************************************************************************//**
 * For internal use only.
 *
 * @param a_line                [in]    A string containing the help line for an argument up to the description string.
 * @param a_descriptor          [in]    The help description string.
 ***********************************************************************************************************/

static void printArgumentDescription( std::string const &a_line, std::string const &a_descriptor ) {

    std::string newLineSpaces = "                            ";
    auto size = newLineSpaces.size( );
    auto maxDescriptionWidth = maxPrintLineWidth - size;
    std::string line = "  " + a_line;
    if( line.size( ) < size ) line.resize( size, ' ' );
    std::cout << line;
    if( line.size( ) > size ) std::cout << std::endl << newLineSpaces;

    std::string descriptor = a_descriptor;
    while( descriptor.size( ) > 0 ) {
        auto length = descriptor.size( );
        if( length > maxDescriptionWidth ) {
            length = maxDescriptionWidth;
            length = descriptor.rfind( ' ', length );
            if( length == 0 ) length = descriptor.find( ' ', length );
        }
    
        std::cout << "  " << descriptor.substr( 0, length ) << std::endl;
        descriptor = descriptor.substr( length );
        auto firstNotOf = descriptor.find_first_not_of( " " );
        if( firstNotOf != descriptor.npos ) descriptor = descriptor.substr( firstNotOf );
        if( descriptor.size( ) > 0 ) std::cout << newLineSpaces;
    }
}

}               // End of namespace LUPI.

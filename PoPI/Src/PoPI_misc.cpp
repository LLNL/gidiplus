/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <limits.h>
#include <sstream>
#include <stdexcept>

#include "PoPI.hpp"

namespace PoPI {

std::string const IDs::photon = "photon";
std::string const IDs::neutron = "n";
std::string const IDs::proton = "p";

/*
============================================================
*/
void appendXMLEnd( std::vector<std::string> &a_XMLList, std::string const &a_label ) {

    std::string theEnd = "</" + a_label + ">";
    std::vector<std::string>::iterator iter = a_XMLList.end( );
    --iter;
    *iter += theEnd;
}
/*
============================================================
*/
int particleZ( Base const &a_particle, bool isNeutronProtonANucleon ) {

    int Z = 0;

    if( a_particle.ID( ) == IDs::proton ) {
        if( isNeutronProtonANucleon ) Z = 1; }
    else if( a_particle.isNuclide( ) ) {
        Nuclide const &particle = (Nuclide const &) a_particle;
        Z = particle.Z( ); }
    else if( a_particle.isNucleus( ) ) {
        Nucleus const &particle = (Nucleus const &) a_particle;
        Z = particle.Z( ); }
    else if( a_particle.isChemicalElement( ) ) {
        ChemicalElement const &object = (ChemicalElement const &) a_particle;
        Z = object.Z( ); }
    else if( a_particle.isIsotope( ) ) {
        Isotope const &object = (Isotope const &) a_particle;
        Z = object.Z( );
    }

    return( Z );
}
/*
============================================================
*/
int particleZ( Database const &a_pops, int a_index, bool isNeutronProtonANucleon ) {

    int Z = 0;
    Base const &base( a_pops.get<Base>( a_pops.final( a_index ) ) );

    if( base.isChemicalElement( ) ) {
        SymbolBase const &object2( static_cast<SymbolBase const &>( base ) );
        Z = particleZ( object2 ); }
    else {
        Particle const &particle( static_cast<Particle const &>( base ) );
        Z = particleZ( particle, isNeutronProtonANucleon );
    }

    return( Z );
}
/*
============================================================
*/
int particleZ( Database const &a_pops, std::string const &a_ID, bool isNeutronProtonANucleon ) {

    Base const &object( a_pops.get<Particle>( a_pops.final( a_ID ) ) );

    return( particleZ( object, isNeutronProtonANucleon ) );
}

/*
============================================================
*/
int particleA( Base const &a_particle, bool isNeutronProtonANucleon ) {

    int A = 0;

    if( a_particle.ID( ) == IDs::neutron ) {
        if( isNeutronProtonANucleon ) A = 1; }
    else if( a_particle.ID( ) == IDs::proton ) {
        if( isNeutronProtonANucleon ) A = 1; }
    else if( a_particle.isNuclide( ) ) {
        Nuclide const &particle = (Nuclide const &) a_particle;
        A = particle.A( ); }
    else if( a_particle.isNucleus( ) ) {
        Nucleus const &particle = (Nucleus const &) a_particle;
        A = particle.A( ); }
    else if( a_particle.isIsotope( ) ) {
        Isotope const &object = (Isotope const &) a_particle;
        A = object.A( );
    }

    return( A );
}
/*
============================================================
*/
int particleA( Database const &a_pops, int a_index, bool isNeutronProtonANucleon ) {

    Base const &particle( a_pops.get<Base>( a_pops.final( a_index ) ) );

    return( particleA( particle, isNeutronProtonANucleon ) );
}
/*
============================================================
*/
int particleA( Database const &a_pops, std::string const &a_ID, bool isNeutronProtonANucleon ) {

    Base const &particle( a_pops.get<Base>( a_pops.final( a_ID ) ) );

    return( particleA( particle, isNeutronProtonANucleon ) );
}

/*
============================================================
*/
int particleZA( Base const &a_particle, bool isNeutronProtonANucleon ) {

    int ZA = 0;

    if( !a_particle.isChemicalElement( ) ) ZA = 1000 * particleZ( a_particle, isNeutronProtonANucleon ) + particleA( a_particle, isNeutronProtonANucleon );

    return( ZA );
}
/*
============================================================
*/
int particleZA( Database const &a_pops, int a_index, bool isNeutronProtonANucleon ) {

    Particle const &particle( a_pops.get<Particle>( a_pops.final( a_index ) ) );
    
    return( particleZA( particle, isNeutronProtonANucleon ) );
}
/*
============================================================
*/
int particleZA( Database const &a_pops, std::string const &a_ID, bool isNeutronProtonANucleon ) {

    Particle const &particle( a_pops.get<Particle>( a_pops.final( a_ID ) ) );

    return( particleZA( particle, isNeutronProtonANucleon ) );
}

/*
============================================================
*/
double getPhysicalQuantityAsDouble( PhysicalQuantity const &a_physicalQuantity ) {

    double value = 0.0;

    switch( a_physicalQuantity.Class( ) ) {
    case PQ_class::Double :
    case PQ_class::shell : {
        PQ_double const &pq_double = static_cast<PQ_double const &>( a_physicalQuantity );
        value = pq_double.value( ); }
        break;
    case PQ_class::integer : {
        PQ_integer const &pq_integer = static_cast<PQ_integer const &>( a_physicalQuantity );
        value = pq_integer.value( ); }
        break;
    default :
        throw Exception( "Cannot convert physical quantitiy to a double." );
    }

    return( value );
}
/*
============================================================
*/
double getPhysicalQuantityOfSuiteAsDouble( PQ_suite const &a_suite, bool a_allowEmpty, double a_emptyValue ) {

    if( a_suite.size( ) == 0 ) {
        if( a_allowEmpty ) return( a_emptyValue );
        throw Exception( "No physical quantitiy in Suite." );
    }

    return( getPhysicalQuantityAsDouble( *a_suite[0] ) );
}

/*
============================================================
*/
Exception::Exception( std::string const & a_message ) :
        std::runtime_error( a_message ) {

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

}

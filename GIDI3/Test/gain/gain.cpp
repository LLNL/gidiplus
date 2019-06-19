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
#include <set>

#include "GIDI.hpp"

void main2( int argc, char **argv );
void gain( GIDI::Protare *a_protare, PoPs::Database &a_pops, GIDI::Settings::MG &a_settings, GIDI::Settings::Particles &a_particles, char const *a_productID );
void printVector( std::string &prefix, GIDI::Vector &vector );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    try {
        main2( argc, argv ); }
    catch ( std::exception exception ) {
        std::cerr << exception.what() << std::endl;
        exit( EXIT_FAILURE );
    }
    exit( EXIT_SUCCESS );
}
/*
=========================================================
*/
void main2( int argc, char **argv ) {

    PoPs::Database pops( "../pops.xml" );
    std::string mapFilename( "../all.map" );
    GIDI::Map map( mapFilename, pops );
    std::string projectileID = "n";
    std::string targetID = "O16";
    GIDI::Protare *protare;

    std::cerr << "    " << __FILE__;
    for( int i1 = 1; i1 < argc; i1++ ) std::cerr << " " << argv[i1];
    std::cerr << std::endl;

    if( argc > 1 ) targetID = argv[1];

    GIDI::Construction::Settings construction( GIDI::Construction::e_all, GIDI::Construction::e_nuclearAndAtomic );
    protare = map.protare( construction, pops, projectileID, targetID );

    std::cout << protare->fileName( ) << std::endl;

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    std::string label( temperatures[0].heatedMultiGroup( ) );
    GIDI::Settings::MG settings( protare->projectile( ).ID( ), label, true );
    GIDI::Settings::Particles particles;

    gain( protare, pops, settings, particles, "n" );
    gain( protare, pops, settings, particles, "H1" );
    gain( protare, pops, settings, particles, "H2" );
    gain( protare, pops, settings, particles, "H3" );
    gain( protare, pops, settings, particles, "He3" );
    gain( protare, pops, settings, particles, "He4" );
    gain( protare, pops, settings, particles, "photon" );

    delete protare;
}
/*
=========================================================
*/
void gain( GIDI::Protare *a_protare, PoPs::Database &a_pops, GIDI::Settings::MG &a_settings, GIDI::Settings::Particles &a_particles, char const *a_productID ) {

    std::string prefix( "Total particle gain" );
    std::string::size_type width = prefix.size( );

    for( std::size_t index = 0; index < a_protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = a_protare->reaction( index );
        std::string string( reaction->label( ) );
        if( string.size( ) > width ) width = string.size( );
    }

    std::cout << std::endl << "Gain for particle '" << a_productID << "':" << std::endl;

    GIDI::Vector multi_group_gain = a_protare->multiGroupGain( a_settings, a_particles, a_productID );

    prefix.insert( prefix.size( ), 4 + width - prefix.size( ), ' ' );
    prefix.insert( 0, 2, ' ' );
    prefix += "::";
    printVector( prefix, multi_group_gain );

    for( std::size_t index = 0; index < a_protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = a_protare->reaction( index );
        GIDI::Vector multi_group_gain = reaction->multiGroupGain( a_settings, a_particles, a_productID, a_protare->projectile( ).ID( ) );

        std::string string( reaction->label( ) );
        string.insert( string.size( ), width - string.size( ), ' ' );
        string = "      " + string + "::";
        printVector( string, multi_group_gain );
    }
}
/*
=========================================================
*/
void printVector( std::string &prefix, GIDI::Vector &vector ) {

    vector.print( prefix );
}

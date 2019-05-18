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

void averageEnergy( GIDI::Protare *a_protare, PoPs::Database &a_pops, GIDI::Settings::MG &a_settings, GIDI::Settings::Particles &a_particles, char const *a_productID );
void printVector( std::string &prefix, GIDI::Vector &vector );
/*
=========================================================
*/
int main( int argc, char **argv ) {

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

    try {
        GIDI::Construction::Settings construction( GIDI::Construction::e_all );
        protare = map.protare( construction, pops, projectileID, targetID ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    std::cout << protare->fileName( ) << std::endl;

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    std::string label( temperatures[0].heatedMultiGroup( ) );
    GIDI::Settings::MG settings( protare->projectile( ).ID( ), label, true );
    GIDI::Settings::Particles particles;

    averageEnergy( protare, pops, settings, particles, "n" );
    averageEnergy( protare, pops, settings, particles, "H1" );
    averageEnergy( protare, pops, settings, particles, "H2" );
    averageEnergy( protare, pops, settings, particles, "H3" );
    averageEnergy( protare, pops, settings, particles, "He3" );
    averageEnergy( protare, pops, settings, particles, "He4" );
    averageEnergy( protare, pops, settings, particles, "photon" );

    GIDI::Vector _averageEnergy;
    for( std::size_t index = 0; index < protare->numberOfOrphanProducts( ); ++index ) {
        GIDI::Reaction const *reaction = protare->orphanProduct( index );
        _averageEnergy += reaction->multiGroupAverageEnergy( settings, particles, PoPs::IDs::photon );
    }

    std::string orphanProductString( "Orphan product gamma average product energy ::" );
    printVector( orphanProductString, _averageEnergy );

    delete protare;
}
/*
=========================================================
*/
void averageEnergy( GIDI::Protare *a_protare, PoPs::Database &a_pops, GIDI::Settings::MG &a_settings, GIDI::Settings::Particles &a_particles, char const *a_productID ) {

    std::string prefix( "Total average product energy" );
    std::string::size_type width = prefix.size( );

    for( std::size_t index = 0; index < a_protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = a_protare->reaction( index );
        std::string string( reaction->label( ) );
        if( string.size( ) > width ) width = string.size( );
    }

    std::cout << std::endl << "Average product energy for product '" << a_productID << "':" << std::endl;
    try {
        GIDI::Vector _averageEnergy = a_protare->multiGroupAverageEnergy( a_settings, a_particles, a_productID );

        prefix.insert( prefix.size( ), 4 + width - prefix.size( ), ' ' );
        prefix.insert( 0, 2, ' ' );
        prefix += "::";
        printVector( prefix, _averageEnergy ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    for( std::size_t index = 0; index < a_protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = a_protare->reaction( index );
        GIDI::Vector _averageEnergy = reaction->multiGroupAverageEnergy( a_settings, a_particles, a_productID );
        std::string string( reaction->label( ) );

        string.insert( string.size( ), width - string.size( ), ' ' );
        string = "      " + string + "::";
        printVector( string, _averageEnergy );
    }
}
/*
=========================================================
*/
void printVector( std::string &prefix, GIDI::Vector &vector ) {

    vector.print( prefix );
}

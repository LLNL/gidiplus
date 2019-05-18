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
#include <set>

#include "GIDI.hpp"

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

    try {
        GIDI::Vector availableMomentum = protare->multiGroupAvailableMomentum( settings, particles );
        std::string prefix( "Total available momentum:: " );
        printVector( prefix, availableMomentum ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = protare->reaction( index );
        GIDI::Vector availableMomentum = reaction->multiGroupAvailableMomentum( settings, particles );
        std::string string( reaction->label( ) );

        string = "    " + string + ":: ";
        printVector( string, availableMomentum );
    }

    delete protare;
}
/*
=========================================================
*/
void printVector( std::string &prefix, GIDI::Vector &vector ) {

    vector.print( prefix );
}

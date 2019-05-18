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

#include <GIDI_testUtilities.hpp>

/*
=========================================================
*/
int main( int argc, char **argv ) {

    PoPs::Database pops( "../pops.xml" );
    std::string mapFilename( "../all.map" );
    GIDI::Map map( mapFilename, pops );
    std::string projectileID( PoPs::IDs::neutron );
    std::string targetID( "O16" );
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
    std::cout << "library = " << protare->libraries( )[0] << std::endl;

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    for( GIDI::Styles::TemperatureInfos::iterator iter = temperatures.begin( ); iter != temperatures.end( ); ++iter ) {
        GIDI::PhysicalQuantity const &temperature = iter->temperature( );

        std::cout << "label = " << iter->heatedMultiGroup( ) << "  temperature = " << temperature.value( ) << std::endl;
    }

    std::string label( temperatures[0].heatedMultiGroup( ) );
    GIDI::Settings::MG settings( protare->projectile( ).ID( ), label, true );
    GIDI::Settings::Particles particles;

    try {
        GIDI::Vector crossSection = protare->multiGroupCrossSection( settings, particles );
        std::string prefix( "Total cross section:: " );
        printVector( prefix, crossSection ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }   

    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = protare->reaction( index );

        GIDI::Vector crossSection = reaction->multiGroupCrossSection( settings, particles );
        std::string string( reaction->label( ) );
        string = "    " + string + ":: ";
        printVector( string, crossSection );
    }

    delete protare;
}

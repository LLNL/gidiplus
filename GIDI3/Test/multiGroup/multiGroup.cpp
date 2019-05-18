/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

/*
    This code loops over all reactions for a protare, gets each reaction's heated cross section (which is guaranteed to be an XYs1d instance),
calculates the cross section's multigroup (via multiGroupXYs1d) and prints the results.
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
    std::string bdflsFileName( "../bdfls" );

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
    for( GIDI::Styles::TemperatureInfos::iterator iter = temperatures.begin( ); iter != temperatures.end( ); ++iter ) {
        GIDI::PhysicalQuantity const &temperature = iter->temperature( );

        std::cout << "label = " << iter->heatedCrossSection( ) << "  temperature = " << temperature.value( ) << std::endl;
    }

    std::string label( temperatures[0].heatedCrossSection( ) );

    GIDI::Settings::Groups_from_bdfls groups( bdflsFileName );
    GIDI::Settings::MultiGroup boundaries = groups.getViaGID( 4 );

    GIDI::Settings::Fluxes_from_bdfls fluxes( bdflsFileName, 0 );
    GIDI::Settings::Flux flux = fluxes.getViaFID( 1 );

    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = protare->reaction( index );

        GIDI::Suite const &crossSection = reaction->crossSection( );
        GIDI::XYs1d crossSectionXY = *crossSection.get<GIDI::XYs1d>( label );
        std::string string( reaction->label( ) );
        string = "    " + string + ":: ";
        GIDI::Vector multiGroup = multiGroupXYs1d( boundaries, crossSectionXY, flux );
        printVector( string, multiGroup );
    }

    delete protare;
}
/*
=========================================================
*/
void printVector( std::string &prefix, GIDI::Vector &vector ) {

    vector.print( prefix );
}

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
    std::string mapFilename( "../all3T.map" );
    GIDI::Map map( mapFilename, pops );
    std::string projectileID = "n";
    std::string targetID = "O16";
    GIDI::Protare *protare;

    std::cerr << "    " << __FILE__;
    for( int i1 = 1; i1 < argc; i1++ ) std::cerr << " " << argv[i1];
    std::cerr << std::endl;

    if( argc > 1 ) targetID = argv[1];

    try {
        GIDI::Construction::Settings construction( GIDI::Construction::e_all, GIDI::Construction::e_nuclearAndAtomic );
        protare = map.protare( construction, pops, projectileID, targetID ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    std::cout << protare->fileName( ) << std::endl;

    std::vector<GIDI::Styles::TemperatureInfo> temperatures = protare->temperatures( );
    std::cout << "number of temperatures = " << temperatures.size( ) << std::endl;
    for( std::size_t i1 = 0; i1 < temperatures.size( ); ++i1 ) {
        GIDI::Styles::TemperatureInfo &temperatureInfo = temperatures[i1];
        GIDI::PhysicalQuantity temperature( temperatureInfo.temperature( ) );

        std::cout << "i1 = " << i1 << ":   temperature = " << temperature.value( ) << " " << temperature.unit( ) << std::endl;
        std::cout << "    heatedCrossSection = <" << temperatureInfo.heatedCrossSection( ) << ">" << std::endl;
        std::cout << "    griddedCrossSection = <" << temperatureInfo.griddedCrossSection( ) << ">" << std::endl;
        std::cout << "    heatedMultiGroup = <" << temperatureInfo.heatedMultiGroup( ) << ">" << std::endl;
        std::cout << "    SnElasticUpScatter = <" << temperatureInfo.SnElasticUpScatter( ) << ">" << std::endl;
    }

    delete protare;
}
/*
=========================================================
*/
void printVector( std::string &prefix, GIDI::Vector &vector ) {

    vector.print( prefix );
}

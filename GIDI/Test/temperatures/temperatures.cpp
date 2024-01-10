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

#include "GIDI_testUtilities.hpp"

static char const *description = "This program prints.";

void main2( int argc, char **argv );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    try {
        main2( argc, argv ); }
    catch (std::exception &exception) {
        std::cerr << exception.what( ) << std::endl;
        exit( EXIT_FAILURE ); }
    catch (char const *str) {
        std::cerr << str << std::endl;
        exit( EXIT_FAILURE ); }
    catch (std::string &str) {
        std::cerr << str << std::endl;
        exit( EXIT_FAILURE );
    }

    exit( EXIT_SUCCESS );
}
/*
=========================================================
*/
void main2( int argc, char **argv ) {

    argvOptions argv_options( "availableEnergy", description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );

    parseTestOptions.m_askGNDS_File = true;

    parseTestOptions.parse( );

    GIDI::Construction::PhotoMode photo_mode = parseTestOptions.photonMode( GIDI::Construction::PhotoMode::nuclearAndAtomic );
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
    PoPI::Database pops;
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../../../TestData/PoPs/pops.xml", "../all3T.map", construction, PoPI::IDs::neutron, "O16" );

    std::cout << stripDirectoryBase( protare->fileName( ), "/GIDI/Test/" ) << std::endl;

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

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

static char const *description = "This test prints the multi-group Q-value for the specified protare and for each of its reactions.";

void main2( int argc, char **argv );
void printQ( GIDI::Protare *protare, bool final );
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

    argvOptions argv_options( "Q", description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );

    parseTestOptions.parse( );

    GIDI::Construction::PhotoMode photo_mode = parseTestOptions.photonMode( GIDI::Construction::PhotoMode::nuclearAndAtomic );
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
    PoPI::Database pops;
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../../../TestData/PoPs/pops.xml", "../all.map", construction, PoPI::IDs::neutron, "O16" );

    std::cout << stripDirectoryBase( protare->fileName( ), "/GIDI/Test" ) << std::endl;

    printQ( protare, false );
    std::cout << std::endl;
    printQ( protare, true );

    delete protare;
}
/*
=========================================================
*/
void printQ( GIDI::Protare *protare, bool final ) {

    LUPI::StatusMessageReporting smr1;

    std::cout << "final = " << final << std::endl;
    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    for( GIDI::Styles::TemperatureInfos::iterator iter = temperatures.begin( ); iter != temperatures.end( ); ++iter ) {
        GIDI::PhysicalQuantity const &temperature = iter->temperature( );

        std::cout << "label = " << iter->heatedMultiGroup( ) << "  temperature = " << temperature.value( ) << std::endl;
    }

    GIDI::Transporting::MG settings( protare->projectile( ).ID( ), GIDI::Transporting::Mode::multiGroup, GIDI::Transporting::DelayedNeutrons::off );
    GIDI::Vector Q = protare->multiGroupQ( smr1, settings, temperatures[0], final );
    std::string prefix = LUPI::Misc::argumentsToString( "%-40s:: ", "Total Q" );
    printVector( prefix, Q );

    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = protare->reaction( index );

        Q = reaction->multiGroupQ( smr1, settings, temperatures[0], final );
        std::string string = LUPI::Misc::argumentsToString( "%-40s:: ", reaction->label( ).c_str( ) );
        printVector( string, Q );
    }
}

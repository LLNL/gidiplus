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

static char const *description = "This code loops over all reactions for a protare, gets each reaction's heated cross section (which is guaranteed to be an XYs1d instance),"
    " calculates the cross section's multigroup (via multiGroupXYs1d) and prints the results.";

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

    argvOptions argv_options( "multiGroup", description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );

    parseTestOptions.m_askGNDS_File = true;

    parseTestOptions.parse( );

    GIDI::Construction::PhotoMode photo_mode = parseTestOptions.photonMode( GIDI::Construction::PhotoMode::nuclearAndAtomic );
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
    PoPI::Database pops;
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../../../TestData/PoPs/pops.xml", "../all.map", construction, PoPI::IDs::neutron, "O16" );

    std::cout << stripDirectoryBase( protare->fileName( ), "/GIDI/Test/" ) << std::endl;

    std::string bdflsFileName( "../bdfls" );

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    for( GIDI::Styles::TemperatureInfos::iterator iter = temperatures.begin( ); iter != temperatures.end( ); ++iter ) {
        GIDI::PhysicalQuantity const &temperature = iter->temperature( );

        std::cout << "label = " << iter->heatedCrossSection( ) << "  temperature = " << temperature.value( ) << std::endl;
    }

    std::string label( temperatures[0].heatedCrossSection( ) );

    GIDI::Transporting::Groups_from_bdfls groups( bdflsFileName );
    GIDI::Transporting::MultiGroup boundaries = groups.getViaGID( 4 );

    GIDI::Transporting::Fluxes_from_bdfls fluxes( bdflsFileName, 0 );
    GIDI::Transporting::Flux flux = fluxes.getViaFID( 1 );

    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction *reaction = protare->reaction( index );

        GIDI::Suite &crossSection = reaction->crossSection( );
        GIDI::Functions::XYs1d crossSectionXY = *crossSection.get<GIDI::Functions::XYs1d>( label );
        std::string string( reaction->label( ) );
        string = "    " + string + ":: ";
        GIDI::Vector multiGroup = multiGroupXYs1d( boundaries, crossSectionXY, flux );
        printVector( string, multiGroup );
    }

    delete protare;
}

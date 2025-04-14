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

#include "GIDI_testUtilities.hpp"

static char const *description = "The program prints the multi-group gain for a protare and its reactions.";

void main2( int argc, char **argv );
void gain( GIDI::Protare *a_protare, LUPI_maybeUnused PoPI::Database &a_pops, GIDI::Transporting::MG &a_settings,
                GIDI::Styles::TemperatureInfos temperatures, char const *a_productID );
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


    argvOptions argv_options( "gain", description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );

    parseTestOptions.parse( );

    GIDI::Construction::PhotoMode photo_mode = parseTestOptions.photonMode( GIDI::Construction::PhotoMode::nuclearAndAtomic );
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
    PoPI::Database pops;
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../../../TestData/PoPs/pops.xml", "../all.map", construction, PoPI::IDs::neutron, "O16" );

    std::cout << stripDirectoryBase( protare->fileName( ), "/GIDI/Test" ) << std::endl;

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    GIDI::Transporting::MG settings( protare->projectile( ).ID( ), GIDI::Transporting::Mode::multiGroup, GIDI::Transporting::DelayedNeutrons::on );

    gain( protare, pops, settings, temperatures, "n" );
    gain( protare, pops, settings, temperatures, "H1" );
    gain( protare, pops, settings, temperatures, "H2" );
    gain( protare, pops, settings, temperatures, "H3" );
    gain( protare, pops, settings, temperatures, "He3" );
    gain( protare, pops, settings, temperatures, "He4" );
    gain( protare, pops, settings, temperatures, "photon" );

    delete protare;
}
/*
=========================================================
*/
void gain( GIDI::Protare *a_protare, LUPI_maybeUnused PoPI::Database &a_pops, GIDI::Transporting::MG &a_settings,
                GIDI::Styles::TemperatureInfos temperatures, char const *a_productID ) {

    LUPI::StatusMessageReporting smr1;
    std::string prefix( "Total particle gain" );
    std::string::size_type width = prefix.size( );

    for( std::size_t index = 0; index < a_protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = a_protare->reaction( index );
        std::string string( reaction->label( ) );
        if( string.size( ) > width ) width = string.size( );
    }

    std::cout << std::endl << "Gain for particle '" << a_productID << "':" << std::endl;

    GIDI::Vector multi_group_gain = a_protare->multiGroupGain( smr1, a_settings, temperatures[0], a_productID );

    prefix.insert( prefix.size( ), 4 + width - prefix.size( ), ' ' );
    prefix.insert( 0, 2, ' ' );
    prefix += "::";
    printVector( prefix, multi_group_gain );

    for( std::size_t index = 0; index < a_protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = a_protare->reaction( index );
        multi_group_gain = reaction->multiGroupGain( smr1, a_settings, temperatures[0], a_productID, a_protare->projectile( ).ID( ) );

        std::string string( reaction->label( ) );
        string.insert( string.size( ), width - string.size( ), ' ' );
        string = "      " + string + "::";
        printVector( string, multi_group_gain );
    }
}

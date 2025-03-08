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

static char const *description = "The program prints the multi-group average product energy for a protare and its reactions for the particle n, H1, H2, H3, He3, He4 and photon.";

void averageEnergy( GIDI::Protare *a_protare, LUPI_maybeUnused PoPI::Database &a_pops, GIDI::Transporting::MG &a_settings, GIDI::Styles::TemperatureInfos a_temperatures, char const *a_productID );
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

    LUPI::StatusMessageReporting smr1;
    argvOptions argv_options( "averageEnergy", description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );

    parseTestOptions.parse( );

    GIDI::Construction::PhotoMode photo_mode = parseTestOptions.photonMode( GIDI::Construction::PhotoMode::nuclearAndAtomic );
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
    PoPI::Database pops;
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../../../TestData/PoPs/pops.xml", "../all.map", construction, PoPI::IDs::neutron, "O16" );

    std::cout << stripDirectoryBase( protare->fileName( ) ) << std::endl;

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    GIDI::Transporting::MG settings( protare->projectile( ).ID( ), GIDI::Transporting::Mode::multiGroup, GIDI::Transporting::DelayedNeutrons::on );

    averageEnergy( protare, pops, settings, temperatures, "n" );
    averageEnergy( protare, pops, settings, temperatures, "H1" );
    averageEnergy( protare, pops, settings, temperatures, "H2" );
    averageEnergy( protare, pops, settings, temperatures, "H3" );
    averageEnergy( protare, pops, settings, temperatures, "He3" );
    averageEnergy( protare, pops, settings, temperatures, "He4" );
    averageEnergy( protare, pops, settings, temperatures, "photon" );

    GIDI::Vector _averageEnergy;
    for( std::size_t index = 0; index < protare->numberOfOrphanProducts( ); ++index ) {
        GIDI::Reaction const *reaction = protare->orphanProduct( index );
        _averageEnergy += reaction->multiGroupAverageEnergy( smr1, settings, temperatures[0], PoPI::IDs::photon );
    }

    std::string orphanProductString( "Orphan product gamma average product energy ::" );
    printVector( orphanProductString, _averageEnergy );

    delete protare;
}
/*
=========================================================
*/
void averageEnergy( GIDI::Protare *a_protare, LUPI_maybeUnused PoPI::Database &a_pops, GIDI::Transporting::MG &a_settings, GIDI::Styles::TemperatureInfos a_temperatures, char const *a_productID ) {

    LUPI::StatusMessageReporting smr1;
    std::string prefix( "Total average product energy" );
    std::string::size_type width = prefix.size( );

    for( std::size_t index = 0; index < a_protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = a_protare->reaction( index );
        std::string string( reaction->label( ) );
        if( string.size( ) > width ) width = string.size( );
    }

    std::cout << std::endl << "Average product energy for product '" << a_productID << "':" << std::endl;
    GIDI::Vector _averageEnergy = a_protare->multiGroupAverageEnergy( smr1, a_settings, a_temperatures[0], a_productID );

    prefix.insert( prefix.size( ), 4 + width - prefix.size( ), ' ' );
    prefix.insert( 0, 2, ' ' );
    prefix += "::";
    printVector( prefix, _averageEnergy );

    for( std::size_t index = 0; index < a_protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = a_protare->reaction( index );
        _averageEnergy = reaction->multiGroupAverageEnergy( smr1, a_settings, a_temperatures[0], a_productID );
        std::string string( reaction->label( ) );

        string.insert( string.size( ), width - string.size( ), ' ' );
        string = "      " + string + "::";
        printVector( string, _averageEnergy );
    }
}

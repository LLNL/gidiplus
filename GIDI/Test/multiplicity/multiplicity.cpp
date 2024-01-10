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

static char const *description = "This program prints the multi-group multiplicity for a protare and its reactions.";

void main2( int argc, char **argv );
void printMultiplicity( GIDI::Protare *protare, std::string const &productID, GIDI::Transporting::DelayedNeutrons delayedNeutrons );
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
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../../../TestData/PoPs/pops.xml", "../all.map", construction, PoPI::IDs::neutron, "O16" );

    std::cout << stripDirectoryBase( protare->fileName( ), "/GIDI/Test/" ) << std::endl;

    printMultiplicity( protare, PoPI::IDs::neutron, GIDI::Transporting::DelayedNeutrons::off );
    printMultiplicity( protare, PoPI::IDs::neutron, GIDI::Transporting::DelayedNeutrons::on );

    delete protare;

    exit( EXIT_SUCCESS );
}
/*
=========================================================
*/
void printMultiplicity( GIDI::Protare *protare, std::string const &productID, GIDI::Transporting::DelayedNeutrons delayedNeutrons ) {

    LUPI::StatusMessageReporting smr1;
    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    for( GIDI::Styles::TemperatureInfos::iterator iter = temperatures.begin( ); iter != temperatures.end( ); ++iter ) {
        GIDI::PhysicalQuantity const &temperature = iter->temperature( );

        std::cout << "label = " << iter->heatedMultiGroup( ) << "  temperature = " << temperature.value( ) << std::endl;
    }


    bool isDelayedNeutrons = ( delayedNeutrons == GIDI::Transporting::DelayedNeutrons::on ? true : false );
    std::cout << "delayedNeutrons = " << isDelayedNeutrons << std::endl;

    std::string label( temperatures[0].heatedMultiGroup( ) );
    GIDI::Transporting::MG settings( protare->projectile( ).ID( ), GIDI::Transporting::Mode::multiGroup, delayedNeutrons );

    std::string prefix( "Total multiplicity:: " );
    try {
        GIDI::Vector multiplicity = protare->multiGroupMultiplicity( smr1, settings, temperatures[0], productID );
        printVector( prefix, multiplicity ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = protare->reaction( index );

        GIDI::Vector multiplicity = reaction->multiGroupMultiplicity( smr1, settings, temperatures[0], productID );
        prefix = reaction->label( );
        prefix = "    " + prefix + ":: ";
        printVector( prefix, multiplicity );
    }

    GIDI::Vector fissionNeutronMultiplicity = protare->multiGroupFissionNeutronMultiplicity( smr1, settings, temperatures[0] );
    prefix = "Fission neutron multiplicity:: ";
    printVector( prefix, fissionNeutronMultiplicity );
}

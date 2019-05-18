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

void printMultiplicity( GIDI::Protare *protare, std::string const &productID, bool delayedNeutrons );
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

    printMultiplicity( protare, PoPs::IDs::neutron, false );
    printMultiplicity( protare, PoPs::IDs::neutron, true );

    delete protare;

    exit( EXIT_SUCCESS );
}
/*
=========================================================
*/
void printMultiplicity( GIDI::Protare *protare, std::string const &productID, bool delayedNeutrons ) {

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    for( GIDI::Styles::TemperatureInfos::iterator iter = temperatures.begin( ); iter != temperatures.end( ); ++iter ) {
        GIDI::PhysicalQuantity const &temperature = iter->temperature( );

        std::cout << "label = " << iter->heatedMultiGroup( ) << "  temperature = " << temperature.value( ) << std::endl;
    }


    std::cout << "delayedNeutrons = " << delayedNeutrons << std::endl;

    std::string label( temperatures[0].heatedMultiGroup( ) );
    GIDI::Settings::MG settings( protare->projectile( ).ID( ), label, delayedNeutrons );
    GIDI::Settings::Particles particles;

    std::string prefix( "Total multiplicity:: " );
    try {
        GIDI::Vector multiplicity = protare->multiGroupMultiplicity( settings, particles, productID );
        printVector( prefix, multiplicity ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = protare->reaction( index );

        GIDI::Vector multiplicity = reaction->multiGroupMultiplicity( settings, particles, productID );
        prefix = reaction->label( );
        prefix = "    " + prefix + ":: ";
        printVector( prefix, multiplicity );
    }

    GIDI::Vector fissionNeutronMultiplicity = protare->multiGroupFissionNeutronMultiplicity( settings, particles );
    prefix = "Fission neutron multiplicity:: ";
    printVector( prefix, fissionNeutronMultiplicity );
}
/*
=========================================================
*/
void printVector( std::string &prefix, GIDI::Vector &vector ) {

    vector.print( prefix );
}

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

void printMatrix( std::string &prefix, int maxOrder, GIDI::Matrix &matrix );
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
    bool delayedNeutrons = false;

    std::cerr << "    " << __FILE__;
    for( int i1 = 1; i1 < argc; i1++ ) std::cerr << " " << argv[i1];
    std::cerr << std::endl;

    if( argc > 1 ) targetID = argv[1];
    if( argc > 2 ) delayedNeutrons = true;

    try {
        GIDI::Construction::Settings construction( GIDI::Construction::e_all, GIDI::Construction::e_nuclearAndAtomic );
        protare = map.protare( construction, pops, projectileID, targetID ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    std::cout << protare->fileName( ) << std::endl;


    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    std::string label( temperatures[0].heatedMultiGroup( ) );
    GIDI::Settings::MG settings( protare->projectile( ).ID( ), label, delayedNeutrons );
    GIDI::Settings::Particles particles;

    try {
        std::string prefix( "Total neutron production matrix: " );
        int maxOrder = protare->maximumLegendreOrder( settings, projectileID );

        GIDI::Matrix m1 = protare->multiGroupProductMatrix( settings, particles, projectileID, 0 );
        printMatrix( prefix, maxOrder, m1 );

        prefix = "Total fission matrix: ";
        m1 = protare->multiGroupFissionMatrix( settings, particles, 0 );
        printMatrix( prefix, -2, m1 ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }   

    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = protare->reaction( index );
        int maxOrder = reaction->maximumLegendreOrder( settings, projectileID );
        GIDI::Matrix m1 = reaction->multiGroupProductMatrix( settings, particles, projectileID, 0 );
        std::string string( reaction->label( ) );

        string += ": ";
        printMatrix( string, maxOrder, m1 );

        m1 = reaction->multiGroupFissionMatrix( settings, particles, 0 );
        string = "  fission:";
        if( m1.size( ) > 0 ) printMatrix( string, -2, m1 );
    }

    delete protare;
}
/*
=========================================================
*/
void printMatrix( std::string &prefix, int maxOrder, GIDI::Matrix &matrix ) {

    std::cout << std::endl << prefix << std::endl;
    if( maxOrder > -2 ) std::cout << "    max. Legendre order = " << maxOrder << std::endl;
    matrix.print( "    ::  " );
}

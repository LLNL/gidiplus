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
    std::map<std::string,int> particle_GID_map;

    particle_GID_map["n"] = 4;
    particle_GID_map["H1"] = 71;
    particle_GID_map["H2"] = 71;
    particle_GID_map["H3"] = 71;
    particle_GID_map["He3"] = 71;
    particle_GID_map["He4"] = 71;
    particle_GID_map["photon"] = 70;

    std::cerr << "    " << __FILE__;
    for( int i1 = 1; i1 < argc; i1++ ) std::cerr << " " << argv[i1];
    std::cerr << std::endl;

    if( argc > 1 ) targetID = argv[1];

    GIDI::Settings::Groups_from_bdfls groups_from_bdfls( "../bdfls" );
    GIDI::Settings::Fluxes_from_bdfls fluxes_from_bdfls( "../bdfls", 0 );

    try {
        GIDI::Construction::Settings construction( GIDI::Construction::e_all );
        protare = map.protare( construction, pops, projectileID, targetID ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    std::cout << protare->fileName( ) << std::endl;

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    std::string label( temperatures[0].heatedMultiGroup( ) );
    GIDI::Settings::MG settings( protare->projectile( ).ID( ), label, true );
    GIDI::Settings::Particles particles;

    for( int i1 = 2; i1 < argc; ++i1 ) {
        std::string particleID( argv[i1] );

        GIDI::Settings::Particle particle( particleID, groups_from_bdfls.getViaGID( particle_GID_map[particleID] ) );
        particle.appendFlux( fluxes_from_bdfls.getViaFID( 1 ) );
        particles.add( particle );
    }

    try {
        GIDI::Vector depositionMomentum = protare->multiGroupDepositionMomentum( settings, particles );
        std::string prefix( "Deposition momentum::" );
        printVector( prefix, depositionMomentum ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    delete protare;
}
/*
=========================================================
*/
void printVector( std::string &prefix, GIDI::Vector &vector ) {

    vector.print( prefix );
}

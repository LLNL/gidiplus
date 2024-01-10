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

static char const *description = "The program prints the multi-group available energy for a protare and its reactions.";

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
    argvOptions argv_options( "availableEnergy", description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );

    parseTestOptions.m_askGNDS_File = true;

    parseTestOptions.parse( );

    GIDI::Construction::PhotoMode photo_mode = parseTestOptions.photonMode( GIDI::Construction::PhotoMode::nuclearAndAtomic );
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
    PoPI::Database pops;
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../../../TestData/PoPs/pops.xml", "../all.map", construction, PoPI::IDs::neutron, "O16" );

    std::cout << stripDirectoryBase( protare->fileName( ), "/GIDI/Test/" ) << std::endl;

    std::map<std::string,int> particle_GID_map;

    particle_GID_map["n"] = 4;
    particle_GID_map["H1"] = 71;
    particle_GID_map["H2"] = 71;
    particle_GID_map["H3"] = 71;
    particle_GID_map["He3"] = 71;
    particle_GID_map["He4"] = 71;
    particle_GID_map["photon"] = 70;

    GIDI::Transporting::Groups_from_bdfls groups_from_bdfls( "../bdfls" );
    GIDI::Transporting::Fluxes_from_bdfls fluxes_from_bdfls( "../bdfls", 0 );

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    GIDI::Transporting::MG settings( protare->projectile( ).ID( ), GIDI::Transporting::Mode::multiGroup, GIDI::Transporting::DelayedNeutrons::on );
    GIDI::Transporting::Particles particles;

    for( std::size_t i1 = 0; i1 < argv_options.m_arguments.size( ); ++i1 ) {
        std::string particleID( argv[argv_options.m_arguments[i1]] );

        GIDI::Transporting::Particle particle( particleID, groups_from_bdfls.getViaGID( particle_GID_map[particleID] ) );
        particle.appendFlux( fluxes_from_bdfls.getViaFID( 1 ) );
        particles.add( particle );
    }

    GIDI::Vector depositionMomentum = protare->multiGroupDepositionMomentum( smr1, settings, temperatures[0], particles );
    std::string prefix( "Deposition momentum::" );
    printVector( prefix, depositionMomentum );

    delete protare;
}

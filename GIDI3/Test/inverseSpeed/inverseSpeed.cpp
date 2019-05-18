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
#include "GIDI_testUtilities.hpp"

static char const *description = "Prints out the multi-group inverse speeds.";

void subMain( int argc, char **argv );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    try {
        subMain( argc, argv ); }
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
void subMain( int argc, char **argv ) {

    PoPs::Database pops( "../pops.xml" );
    GIDI::Construction::PhotoMode photo_mode = GIDI::Construction::e_nuclearOnly;

    std::cerr << "    " << __FILE__;
    for( int i1 = 1; i1 < argc; i1++ ) std::cerr << " " << argv[i1];
    std::cerr << std::endl;

    argvOptions argv_options( "sampleReactions", description );

    argv_options.add( argvOption( "--map", true, "The map file to use." ) );
    argv_options.add( argvOption( "--pid", true, "The PoPs id of the projectile." ) );
    argv_options.add( argvOption( "--tid", true, "The PoPs id of the target." ) );
    argv_options.add( argvOption( "-a", false, "Include photo-atomic protare if relevant. If present, disables photo-nuclear unless *-n* present." ) );
    argv_options.add( argvOption( "-n", false, "Include photo-nuclear protare if relevant. This is the default unless *-a* present." ) );

    argv_options.parseArgv( argc, argv );

    std::string mapFilename = argv_options.find( "--map" )->zeroOrOneOption( argv, "../all.map" );
    std::string projectileID = argv_options.find( "--pid" )->zeroOrOneOption( argv, PoPs::IDs::neutron );
    std::string targetID = argv_options.find( "--tid" )->zeroOrOneOption( argv, "O16" );

    if( argv_options.find( "-a" )->present( ) ) {
        photo_mode = GIDI::Construction::e_atomicOnly;
        if( argv_options.find( "-n" )->present( ) ) photo_mode = GIDI::Construction::e_nuclearAndAtomic;
    }

    GIDI::Map map( mapFilename, pops );
    GIDI::Construction::Settings construction( GIDI::Construction::e_all, photo_mode );
    GIDI::Protare *protare = map.protare( construction, pops, projectileID, targetID );

    std::cout << protare->fileName( ) << std::endl;

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    for( GIDI::Styles::TemperatureInfos::iterator iter = temperatures.begin( ); iter != temperatures.end( ); ++iter ) {
        GIDI::PhysicalQuantity const &temperature = iter->temperature( );

        std::cout << "label = " << iter->heatedMultiGroup( ) << "  temperature = " << temperature.value( ) << std::endl;
    }

    std::string label( temperatures[0].heatedMultiGroup( ) );

    std::cout << "label = " << label << std::endl;

    GIDI::Settings::Groups_from_bdfls groups_from_bdfls( "../bdfls" );
    GIDI::Settings::Fluxes_from_bdfls fluxes_from_bdfls( "../bdfls", 0 );

    GIDI::Settings::MG settings( protare->projectile( ).ID( ), label, true );

    GIDI::Settings::Particles particles;

    int gid = 23;
    if( projectileID == PoPs::IDs::photon ) gid = 70;
    GIDI::Settings::Particle projectile( projectileID, groups_from_bdfls.getViaGID( gid ) );
    projectile.appendFlux( fluxes_from_bdfls.getViaFID( 1 ) );
    particles.add( projectile );

    particles.process( *protare, label );

    GIDI::Vector inverseSpeed = protare->multiGroupInverseSpeed( settings, particles );
    std::string prefix( "inverse speed: " );
    printVector( prefix, inverseSpeed );
    GIDI::Vector inverseSpeedCollapse = GIDI::collapse( inverseSpeed, settings, particles, 0. );
    prefix = "    collapsed: ";
    printVector( prefix, inverseSpeedCollapse );
}

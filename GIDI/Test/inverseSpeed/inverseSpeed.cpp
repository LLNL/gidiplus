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

    LUPI::StatusMessageReporting smr1;
    argvOptions argv_options( "inverseSpeed", description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );

    parseTestOptions.m_askGNDS_File = true;

    parseTestOptions.parse( );

    GIDI::Construction::PhotoMode photo_mode = parseTestOptions.photonMode( GIDI::Construction::PhotoMode::nuclearOnly );
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
    PoPI::Database pops;
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../../../TestData/PoPs/pops.xml", "../all.map", construction, PoPI::IDs::neutron, "O16" );

    std::cout << stripDirectoryBase( protare->fileName( ), "/GIDI/Test/" ) << std::endl;

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    for( GIDI::Styles::TemperatureInfos::iterator iter = temperatures.begin( ); iter != temperatures.end( ); ++iter ) {
        GIDI::PhysicalQuantity const &temperature = iter->temperature( );

        std::cout << "label = " << iter->heatedMultiGroup( ) << "  temperature = " << temperature.value( ) << std::endl;
    }

    std::string label( temperatures[0].heatedMultiGroup( ) );

    std::cout << "label = " << label << std::endl;

    GIDI::Transporting::Groups_from_bdfls groups_from_bdfls( "../bdfls" );
    GIDI::Transporting::Fluxes_from_bdfls fluxes_from_bdfls( "../bdfls", 0 );

    GIDI::Transporting::MG settings( protare->projectile( ).ID( ), GIDI::Transporting::Mode::multiGroup, GIDI::Transporting::DelayedNeutrons::on );

    GIDI::Transporting::Particles particles;

    int gid = 23;
    if( protare->projectile( ).ID( ) == PoPI::IDs::photon ) gid = 70;
    GIDI::Transporting::Particle projectile( protare->projectile( ).ID( ), groups_from_bdfls.getViaGID( gid ) );
    projectile.appendFlux( fluxes_from_bdfls.getViaFID( 1 ) );
    particles.add( projectile );

    particles.process( *protare, label );

    GIDI::Vector inverseSpeed = protare->multiGroupInverseSpeed( smr1, settings, temperatures[0] );
    std::string prefix( "inverse speed: " );
    printVector( prefix, inverseSpeed );
    GIDI::Vector inverseSpeedCollapse = GIDI::collapse( inverseSpeed, settings, particles, 0. );
    prefix = "    collapsed: ";
    printVector( prefix, inverseSpeedCollapse );
}

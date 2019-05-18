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

void printVector( GIDI::Protare *protare, GIDI::Settings::MG &settings, GIDI::Settings::Particles &particles, char const *prefix, 
        GIDI::transportCorrectionType transportCorrectionType, bool delayedNeutrons );
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

    GIDI::Settings::Particle neutron( projectileID, groups_from_bdfls.getViaGID( 4 ) );
    neutron.appendFlux( fluxes_from_bdfls.getViaFID( 1 ) );
    particles.add( neutron );

    particles.process( *protare, label );

    printVector( protare, settings, particles, "transport correction: None", GIDI::transportCorrection_None, true );
    printVector( protare, settings, particles, "transport correction: None", GIDI::transportCorrection_None, false );
    printVector( protare, settings, particles, "transport correction: Pendlebury", GIDI::transportCorrection_Pendlebury, true );
    printVector( protare, settings, particles, "transport correction: Pendlebury", GIDI::transportCorrection_Pendlebury, false );
#if 0
    printVector( protare, settings, particles, "transport correction: LLNL", GIDI::transportCorrection_LLNL, true );
    printVector( protare, settings, particles, "transport correction: LLNL", GIDI::transportCorrection_LLNL, false );
    printVector( protare, settings, particles, "transport correction: Ferguson", GIDI::transportCorrection_Ferguson, true );
    printVector( protare, settings, particles, "transport correction: Ferguson", GIDI::transportCorrection_Ferguson, false );
#endif

    delete protare;
}
/*
=========================================================
*/
void printVector( GIDI::Protare *protare, GIDI::Settings::MG &settings, GIDI::Settings::Particles &particles, char const *prefix,
        GIDI::transportCorrectionType transportCorrectionType, bool delayedNeutrons ) {

    char Str[64];

    for( int order = 0; order < protare->maximumLegendreOrder( settings, protare->projectile( ).ID( ) ); ++order ) {
        try {
            settings.delayedNeutrons( delayedNeutrons );
            GIDI::Vector transportCorrection = protare->multiGroupTransportCorrection( settings, particles, order, transportCorrectionType, 0.0 );
            std::string message( prefix );

            sprintf( Str, ": delayedNeutrons (%s)", ( delayedNeutrons ? "true"  : "false" ) );
            message += Str;

            sprintf( Str, ": order = %d::", order );
            message += Str;
            transportCorrection.print( message ); }
        catch (char const *str) {
            std::cout << str << std::endl;
            exit( EXIT_FAILURE );
        }
    }
}

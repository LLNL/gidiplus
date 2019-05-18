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

#include "GIDI_testUtilities.hpp"

void productMatrixInfo( GIDI::Protare *protare, std::string const &label, int neutronGID,
                        GIDI::Settings::Groups_from_bdfls const &groups_from_bdfls,
                        GIDI::Settings::Fluxes_from_bdfls const &fluxes_from_bdfls );
void productMatrixInfo2( GIDI::Protare *protare, std::string const &label, int neutronGID, int photonGID,
                        GIDI::Settings::Groups_from_bdfls const &groups_from_bdfls,
                        GIDI::Settings::Fluxes_from_bdfls const &fluxes_from_bdfls );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    PoPs::Database pops( "../pops.xml" );
    std::string mapFilename( "../all.map" );
    GIDI::Map map( mapFilename, pops );
    GIDI::Protare *protare;
    std::string targetID( "O" );

    std::cerr << "    " << __FILE__;
    for( int i1 = 1; i1 < argc; i1++ ) std::cerr << " " << argv[i1];
    std::cerr << std::endl;

    if( argc > 1 ) targetID = argv[1];

    try {
        GIDI::Construction::Settings construction( GIDI::Construction::e_all );
        protare = map.protare( construction, pops, PoPs::IDs::photon, targetID ); }
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

    GIDI::Settings::Groups_from_bdfls groups_from_bdfls( "../bdfls" );
    GIDI::Settings::Fluxes_from_bdfls fluxes_from_bdfls( "../bdfls", 0 );

    productMatrixInfo( protare, label, 23, groups_from_bdfls, fluxes_from_bdfls );
    productMatrixInfo( protare, label, 24, groups_from_bdfls, fluxes_from_bdfls );
    productMatrixInfo( protare, label, 25, groups_from_bdfls, fluxes_from_bdfls );
    productMatrixInfo( protare, label, 26, groups_from_bdfls, fluxes_from_bdfls );
    productMatrixInfo( protare, label, 4, groups_from_bdfls, fluxes_from_bdfls );

    delete protare;
}
/*
=========================================================
*/
void productMatrixInfo( GIDI::Protare *protare, std::string const &label, int neutronGID,
                        GIDI::Settings::Groups_from_bdfls const &groups_from_bdfls,
                        GIDI::Settings::Fluxes_from_bdfls const &fluxes_from_bdfls ) {

    productMatrixInfo2( protare, label, neutronGID, 76, groups_from_bdfls, fluxes_from_bdfls );
    productMatrixInfo2( protare, label, neutronGID, 70, groups_from_bdfls, fluxes_from_bdfls );
    productMatrixInfo2( protare, label, neutronGID, 73, groups_from_bdfls, fluxes_from_bdfls );
    productMatrixInfo2( protare, label, neutronGID, 74, groups_from_bdfls, fluxes_from_bdfls );
    productMatrixInfo2( protare, label, neutronGID, 75, groups_from_bdfls, fluxes_from_bdfls );
    productMatrixInfo2( protare, label, neutronGID, 76, groups_from_bdfls, fluxes_from_bdfls );
}
/*
=========================================================
*/
void productMatrixInfo2( GIDI::Protare *protare, std::string const &label, int neutronGID, int photonGID,
                        GIDI::Settings::Groups_from_bdfls const &groups_from_bdfls,
                        GIDI::Settings::Fluxes_from_bdfls const &fluxes_from_bdfls ) {

    int offset = 4;
    int prefixLength = outputChannelStringMaximumLength( protare );
    if( prefixLength < 32 ) prefixLength = 32;

    GIDI::Settings::MG settings( protare->projectile( ).ID( ), label, true );

    GIDI::Settings::Particles particles;

    GIDI::Settings::Particle neutron( PoPs::IDs::neutron, groups_from_bdfls.getViaGID( neutronGID ) );
    neutron.appendFlux( fluxes_from_bdfls.getViaFID( 1 ) );
    particles.add( neutron );

    GIDI::Settings::Particle photon( PoPs::IDs::photon, groups_from_bdfls.getViaGID( photonGID ) );
    photon.appendFlux( fluxes_from_bdfls.getViaFID( 1 ) );
    particles.add( photon );

    particles.process( *protare, label );

    settings.print( );
    particles.print( );
    std::cout << std::endl;

    int maxOrder = protare->maximumLegendreOrder( settings, PoPs::IDs::photon );
    std::string prefix( "Total gamma-gamma matrix:" );
    prefix.insert( prefix.size( ), offset + prefixLength + 1 - prefix.size( ), ' ' );
    GIDI::Matrix uncollapsed = protare->multiGroupProductMatrix( settings, particles, PoPs::IDs::photon, 0 );
    printMatrix( prefix, maxOrder, uncollapsed );

    prefix = "Collapsed gamma-gamma matrix:";
    prefix.insert( prefix.size( ), offset + prefixLength + 1 - prefix.size( ), ' ' );
    GIDI::Matrix collapsed = GIDI::collapse( uncollapsed, settings, particles, 0, PoPs::IDs::photon );
    printMatrix( prefix, maxOrder, collapsed );

    maxOrder = protare->maximumLegendreOrder( settings, PoPs::IDs::neutron );
    prefix = "Total gamma-neutron matrix:";
    prefix.insert( prefix.size( ), offset + prefixLength + 1 - prefix.size( ), ' ' );
    uncollapsed = protare->multiGroupProductMatrix( settings, particles, PoPs::IDs::neutron, 0 );
    printMatrix( prefix, maxOrder, uncollapsed );

    prefix = "Collapsed gamma-neutron matrix:";
    prefix.insert( prefix.size( ), offset + prefixLength + 1 - prefix.size( ), ' ' );
    collapsed = GIDI::collapse( uncollapsed, settings, particles, 0, PoPs::IDs::photon );
    printMatrix( prefix, maxOrder, collapsed );

return;

    GIDI::Matrix summedCollapsed( 0, 0 );
    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = protare->reaction( index );
        int maxOrder = reaction->maximumLegendreOrder( settings, PoPs::IDs::photon );
        GIDI::Matrix m1 = reaction->multiGroupProductMatrix( settings, particles, PoPs::IDs::photon, 0 );
        std::string string( reaction->label( ) );

        string += ":: ";
        printMatrix( string, maxOrder, m1 );
    }
}

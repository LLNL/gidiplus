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

static std::string projectileID;
static std::string photonID;

void crossSectionInfo( GIDI::Protare *protare, std::string const &label, int neutronGID,
                        GIDI::Settings::Groups_from_bdfls const &groups_from_bdfls,
                        GIDI::Settings::Fluxes_from_bdfls const &fluxes_from_bdfls );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    PoPs::Database pops( "../pops.xml" );
    std::string mapFilename( "../all.map" );
    GIDI::Map map( mapFilename, pops );
    std::string targetID( "O16" );
    GIDI::Protare *protare;

    std::cerr << "    " << __FILE__;
    for( int i1 = 1; i1 < argc; i1++ ) std::cerr << " " << argv[i1];
    std::cerr << std::endl;

    projectileID = PoPs::IDs::neutron;
    photonID = PoPs::IDs::photon;
    if( argc > 1 ) targetID = argv[1];

    try {
        GIDI::Construction::Settings construction( GIDI::Construction::e_all, GIDI::Construction::e_nuclearAndAtomic );
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

    GIDI::Settings::Groups_from_bdfls groups_from_bdfls( "../bdfls" );
    GIDI::Settings::Fluxes_from_bdfls fluxes_from_bdfls( "../bdfls", 0 );

    crossSectionInfo( protare, label, 23, groups_from_bdfls, fluxes_from_bdfls );
    crossSectionInfo( protare, label, 24, groups_from_bdfls, fluxes_from_bdfls );
    crossSectionInfo( protare, label, 25, groups_from_bdfls, fluxes_from_bdfls );
    crossSectionInfo( protare, label, 26, groups_from_bdfls, fluxes_from_bdfls );
    crossSectionInfo( protare, label, 4, groups_from_bdfls, fluxes_from_bdfls );

    delete protare;
}
/*
=========================================================
*/
void crossSectionInfo( GIDI::Protare *protare, std::string const &label, int neutronGID,
                        GIDI::Settings::Groups_from_bdfls const &groups_from_bdfls,
                        GIDI::Settings::Fluxes_from_bdfls const &fluxes_from_bdfls ) {

    int offset = 4;
    int prefixLength = outputChannelStringMaximumLength( protare );
    if( prefixLength < 32 ) prefixLength = 32;

    GIDI::Settings::MG settings( protare->projectile( ).ID( ), label, false );

    GIDI::Settings::Particles particles;

    GIDI::Settings::Particle neutron( projectileID, groups_from_bdfls.getViaGID( neutronGID ) );
    neutron.appendFlux( fluxes_from_bdfls.getViaFID( 1 ) );
    particles.add( neutron );

    GIDI::Settings::Particle photon( photonID, groups_from_bdfls.getViaGID( 70 ) );
    photon.appendFlux( fluxes_from_bdfls.getViaFID( 1 ) );
    particles.add( photon );

    particles.process( *protare, label );

    settings.print( );
    particles.print( );
    std::cout << std::endl;

    std::string prefix( "Uncollapsed total cross section:: " );
    prefix.insert( prefix.size( ), offset + prefixLength + 2 - prefix.size( ), ' ' );
    GIDI::Vector crossSection = protare->multiGroupCrossSection( settings, particles );
    printVector( prefix, crossSection );

    GIDI::Vector collapsed = GIDI::collapse( crossSection, settings, particles, 0 );
    prefix = "collapsed cross section::";
    prefix.insert( prefix.size( ), offset + prefixLength + 2 - prefix.size( ), ' ' );
    printVector( prefix, collapsed );

    GIDI::Vector summedCollapsed( 0 );
    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = protare->reaction( index );

        crossSection = reaction->multiGroupCrossSection( settings, particles );
        std::string prefix( outputChannelPrefix( offset, prefixLength, reaction ) );
        prefix += ":";
        printVector( prefix, crossSection );

        std::string::size_type length( prefix.size( ) );
        prefix.erase( );
        prefix.insert( 0, length - 2, ' ' );
        collapsed = GIDI::collapse( crossSection, settings, particles, 0 );
        prefix += "::";
        printVector( prefix, collapsed );
        summedCollapsed += collapsed;
    }
    prefix = "Summed collapsed cross section::";
    prefix.insert( prefix.size( ), offset + prefixLength + 2 - prefix.size( ), ' ' );
    printVector( prefix, summedCollapsed );
}

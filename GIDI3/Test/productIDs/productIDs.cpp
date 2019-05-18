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

void printIDs( char const *prefix, std::set<std::string> &IDs, PoPs::Database &pops );
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
    std::string label( temperatures[0].heatedMultiGroup( ) );
    GIDI::Settings::MG settings( protare->projectile( ).ID( ), label, true );

    GIDI::Settings::Groups_from_bdfls groups_from_bdfls( "../bdfls" );
    GIDI::Settings::Fluxes_from_bdfls fluxes_from_bdfls( "../bdfls", 0 );

    GIDI::Settings::Particles particles;

    GIDI::Settings::Particle neutronParticle( PoPs::IDs::neutron, groups_from_bdfls.getViaGID( 4 ) );
    neutronParticle.appendFlux( fluxes_from_bdfls.getViaFID( 1 ) );
    particles.add( neutronParticle );

    GIDI::Settings::Particle photonParticle( PoPs::IDs::photon, groups_from_bdfls.getViaGID( 70 ) );
    photonParticle.appendFlux( fluxes_from_bdfls.getViaFID( 1 ) );
    particles.add( photonParticle );

    particles.process( *protare, label );

    std::cout << std::endl;
    std::set<std::string> IDs;
    protare->productIDs( IDs, particles, false );
    printIDs( "All particle IDs in protare: ", IDs, pops );

    std::cout << std::endl;
    IDs.clear( );
    protare->productIDs( IDs, particles, true );
    printIDs( "Transportable particle IDs in protare: ", IDs, pops );

    std::cout << std::endl;

    std::cout << "All particle IDs by reaction:" << std::endl;
    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = protare->reaction( index );

        IDs.clear( );
        reaction->productIDs( IDs, particles, false );
        std::string string( reaction->label( ) );
        string = "    " + string + ": ";
        printIDs( string.c_str( ), IDs, pops );
    }
    std::cout << std::endl;

    std::cout << "Transportable particle IDs by reaction:" << std::endl;
    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = protare->reaction( index );

        IDs.clear( );
        reaction->productIDs( IDs, particles, true );
        std::string string( reaction->label( ) );
        string = "    " + string + ": ";
        printIDs( string.c_str( ), IDs, pops );
    }
}
/*
=========================================================
*/
void printIDs( char const *prefix, std::set<std::string> &IDs, PoPs::Database &pops ) {

    std::cout << prefix;

    std::set<std::string>::iterator iter( IDs.find( PoPs::IDs::neutron ) );
    if( iter != IDs.end( ) ) {
        std::cout << " " << *iter;
        IDs.erase( iter );
    }

    iter = IDs.find( PoPs::IDs::photon );
    if( iter != IDs.end( ) ) {
        std::cout << " " << *iter;
        IDs.erase( iter );
    }

    std::vector<std::string> IDs2;
    for( std::set<std::string>::const_iterator iter = IDs.begin( ); iter != IDs.end( ); ++iter ) IDs2.push_back( *iter );
    std::vector<std::string> IDs3( GIDI::sortedListOfStrings( IDs2 ) );

    for( std::vector<std::string>::const_iterator iter = IDs3.begin( ); iter != IDs3.end( ); ++iter ) std::cout << " " << *iter;
    std::cout << std::endl;

}

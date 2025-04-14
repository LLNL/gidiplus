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

static char const *description = "This program prints all and transportable particle ID for a protare and its reactions.";

void main2( int argc, char **argv );
void printIDs( char const *prefix, std::set<std::string> &IDs, LUPI_maybeUnused PoPI::Database &pops );
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

    argvOptions argv_options( "productIDs", description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );

    parseTestOptions.m_askGNDS_File = true;

    argv_options.add( argvOption( "--ENDL99120", false, "If present, ENDL two 99120 products are list for a fission reaction." ) );

    parseTestOptions.parse( );

    GIDI::Construction::PhotoMode photo_mode = parseTestOptions.photonMode( GIDI::Construction::PhotoMode::nuclearAndAtomic );
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
    if( argv_options.find( "--ENDL99120" )->present( ) ) 
        construction.setFissionResiduals( GIDI::Construction::FissionResiduals::ENDL99120 );

    PoPI::Database pops;
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../../../TestData/PoPs/pops.xml", "../all.map", construction, PoPI::IDs::neutron, "O16" );

    std::cout << stripDirectoryBase( protare->fileName( ), "/GIDI/Test/" ) << std::endl;

    GIDI::Transporting::MG settings( protare->projectile( ).ID( ), GIDI::Transporting::Mode::multiGroup, GIDI::Transporting::DelayedNeutrons::on );

    GIDI::Transporting::Groups_from_bdfls groups_from_bdfls( "../bdfls" );
    GIDI::Transporting::Fluxes_from_bdfls fluxes_from_bdfls( "../bdfls", 0 );

    GIDI::Transporting::Particles particles;

    GIDI::Transporting::Particle neutronParticle( PoPI::IDs::neutron, groups_from_bdfls.getViaGID( 4 ) );
    neutronParticle.appendFlux( fluxes_from_bdfls.getViaFID( 1 ) );
    particles.add( neutronParticle );

    GIDI::Transporting::Particle photonParticle( PoPI::IDs::photon, groups_from_bdfls.getViaGID( 70 ) );
    photonParticle.appendFlux( fluxes_from_bdfls.getViaFID( 1 ) );
    particles.add( photonParticle );

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    if( temperatures.size( ) > 0 ) {
        std::string label( temperatures[0].heatedMultiGroup( ) );
        particles.process( *protare, label );
    }

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
void printIDs( char const *prefix, std::set<std::string> &IDs, LUPI_maybeUnused PoPI::Database &pops ) {

    std::cout << prefix;

    std::set<std::string>::iterator iter( IDs.find( PoPI::IDs::neutron ) );
    if( iter != IDs.end( ) ) {
        std::cout << " " << *iter;
        IDs.erase( iter );
    }

    iter = IDs.find( PoPI::IDs::photon );
    if( iter != IDs.end( ) ) {
        std::cout << " " << *iter;
        IDs.erase( iter );
    }

    std::vector<std::string> IDs2;
    for( std::set<std::string>::const_iterator iter2 = IDs.begin( ); iter2 != IDs.end( ); ++iter2 ) IDs2.push_back( *iter2 );
    std::vector<std::string> IDs3( GIDI::sortedListOfStrings( IDs2 ) );

    for( std::vector<std::string>::const_iterator iter2 = IDs3.begin( ); iter2 != IDs3.end( ); ++iter2 ) std::cout << " " << *iter2;
    std::cout << std::endl;

}

/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <set>
#include <exception>
#include <stdexcept>

#include "MCGIDI.hpp"

#include "MCGIDI_testUtilities.hpp"

static bool useSystem_strtod = true;
static GIDI::Construction::Settings *constructionPtr = nullptr;
static bool doParticlesProcessing = true;
static bool doMultiGroup = false;

static char const *description = "Reads in all protares in the specified map file. Besides options, there must be one map file followed by one or more pops files.";

void subMain( int argc, char **argv );
void walk( GIDI::Transporting::Particles &particles, std::string const &mapFilename, PoPI::Database const &pops );
void readProtare( GIDI::Transporting::Particles &particles, std::string const &protareFilename, PoPI::Database const &pops, std::vector<std::string> &a_libraries, bool a_targetRequiredInGlobalPoPs );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    try {
        subMain( argc, argv ); }
    catch (std::exception &exception) {
        std::cerr << exception.what( ) << std::endl;
    }

    exit( EXIT_SUCCESS );
}
/*
=========================================================
*/
void subMain( int argc, char **argv ) {

    PoPI::Database pops;
    GIDI::Transporting::Particles particles;
    std::map<std::string, std::string> particlesAndGIDs;

    particlesAndGIDs[PoPI::IDs::neutron] = "LLNL_gid_4";
    particlesAndGIDs["H1"] = "LLNL_gid_71";
    particlesAndGIDs["H2"] = "LLNL_gid_71";
    particlesAndGIDs["H3"] = "LLNL_gid_71";
    particlesAndGIDs["He3"] = "LLNL_gid_71";
    particlesAndGIDs["He4"] = "LLNL_gid_71";
    particlesAndGIDs[PoPI::IDs::photon] = "LLNL_gid_70";

    argvOptions2 argv_options( "readAllProtaresInMapFile", description );

    argv_options.add( argvOption2( "-f", false, "Use nf_strtod instead of the system stdtod." ) );
    argv_options.add( argvOption2( "--mg", false, "Use multi-group instead continuous energy label for MCGIDI data." ) );

    argv_options.parseArgv( argc, argv );

    doMultiGroup = argv_options.find( "--mg" )->present( );

    if( argv_options.m_arguments.size( ) < 2 ) {
        std::cerr << std::endl << "----- Need map file name and at least one pops file -----" << std::endl << std::endl;
        argv_options.help( );
    }

    for( std::size_t i1 = 1; i1 < argv_options.m_arguments.size( ); ++i1 ) pops.addFile( argv[argv_options.m_arguments[i1]], true );

    GIDI::Transporting::Groups_from_bdfls groups_from_bdfls( "../../GIDI/Test/bdfls" );
    GIDI::Transporting::Fluxes_from_bdfls fluxes_from_bdfls( "../../GIDI/Test/bdfls", 0.0 );

    for( std::map<std::string, std::string>::iterator iter = particlesAndGIDs.begin( ); iter != particlesAndGIDs.end( ); ++iter ) {
        GIDI::Transporting::MultiGroup multi_group = groups_from_bdfls.viaLabel( iter->second );
        GIDI::Transporting::Particle particle( iter->first, multi_group );

        particle.appendFlux( fluxes_from_bdfls.getViaFID( 1 ) );
        particles.add( particle );
    }

    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, GIDI::Construction::PhotoMode::nuclearAndAtomic );
    construction.setUseSystem_strtod( useSystem_strtod );
    constructionPtr = &construction;

    std::string const &mapFilename( argv[argv_options.m_arguments[0]] );
    try {
        walk( particles, mapFilename, pops ); }
    catch (char const *str) {
        std::cout << str << std::endl; }
    catch (std::string str) {
        std::cout << str << std::endl;
    }
}
/*
=========================================================
*/
void walk( GIDI::Transporting::Particles &particles, std::string const &mapFilename, PoPI::Database const &pops ) {

    std::cout << "    " << mapFilename << std::endl;
    GIDI::Map::Map map( mapFilename, pops );

    for( std::size_t i1 = 0; i1 < map.size( ); ++i1 ) {
        GIDI::Map::BaseEntry const *entry = map[i1];

        std::string path = entry->path( GIDI::Map::BaseEntry::PathForm::cumulative );

        if( entry->name( ) == GIDI_importMoniker ) {
            walk( particles, path, pops ); }
        else if( ( entry->name( ) == GIDI_protareMoniker ) || ( entry->name( ) == GIDI_TNSLMoniker ) ) {
            std::vector<std::string> libraries;

            entry->libraries( libraries );
            readProtare( particles, path, pops, libraries, entry->name( ) == GIDI_protareMoniker ); }
        else {
            std::cerr << "    ERROR: unknown map entry name: " << entry->name( ) << std::endl;
        }
    }
}
/*
=========================================================
*/
void readProtare( GIDI::Transporting::Particles &particles, std::string const &protareFilename, PoPI::Database const &pops, std::vector<std::string> &a_libraries, bool a_targetRequiredInGlobalPoPs ) {

    MCGIDI::DomainHash domainHash( 4000, 1e-8, 10 );
    GIDI::ProtareSingle *protare = nullptr;
    MCGIDI::Protare *MCProtare = nullptr;
    std::set<int> reactionsToExclude;

    try {
        std::cout << "        " << protareFilename << std::endl;

        protare = new GIDI::ProtareSingle( *constructionPtr, protareFilename, GIDI::FileType::XML, pops, a_libraries, a_targetRequiredInGlobalPoPs );
        GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );


        std::string label( temperatures[0].griddedCrossSection( ) );
        if( doMultiGroup ) label = temperatures[0].heatedMultiGroup( );

        if( doParticlesProcessing ) particles.process( *protare, temperatures[0].heatedMultiGroup( ) );
        doParticlesProcessing = false;

        MCGIDI::Transporting::MC MC( pops, protare->projectile( ).ID( ), &protare->styles( ), label, GIDI::Transporting::DelayedNeutrons::on, 30.0 );
        MCProtare = new MCGIDI::ProtareSingle( *protare, pops, MC, particles, domainHash, temperatures, reactionsToExclude ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    delete MCProtare;
    delete protare;
}

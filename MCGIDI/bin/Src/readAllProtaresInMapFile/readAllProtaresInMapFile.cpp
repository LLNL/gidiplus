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
static bool doMultiGroup = false;
static int errCount = 0;

static char const *description = "Reads in all protares in the specified map file. Besides options, there must be one map file followed by \n"
    "one or more pops files. If an error occurs when reading a protare, the C++ 'throw' message is printed. Also,\n"
    "for each protare all standard LLNL transportable particles with missing or unspecified distribution data will \n"
    "be printed. The standard LLNL transportable particles are n, p, d, t, h, a and g";

void subMain( int argc, char **argv );
void walk( GIDI::Transporting::Particles &particles, std::string const &mapFilename, PoPI::Database const &pops );
void readProtare( GIDI::Transporting::Particles &particles, std::string const &protareFilename, PoPI::Database const &pops, std::vector<std::string> &a_libraries, bool a_targetRequiredInGlobalPoPs );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    subMain( argc, argv );

    if( errCount > 0 ) exit( EXIT_FAILURE );
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

//    doMultiGroup = argv_options.find( "--mg" )->present( ); // Currently not working as need to have option to get multi-group and flux information.

    if( argv_options.m_arguments.size( ) < 2 ) {
        std::cerr << std::endl << "----- Need map file name and at least one pops file -----" << std::endl << std::endl;
        argv_options.help( );
    }

    for( std::size_t i1 = 1; i1 < argv_options.m_arguments.size( ); ++i1 ) pops.addFile( argv[argv_options.m_arguments[i1]], true );

    for( std::map<std::string, std::string>::iterator iter = particlesAndGIDs.begin( ); iter != particlesAndGIDs.end( ); ++iter ) {
//        GIDI::Transporting::MultiGroup multi_group = groups_from_bdfls.viaLabel( iter->second );
//        GIDI::Transporting::Particle particle( iter->first, multi_group );
        GIDI::Transporting::Particle particle( iter->first );

        particles.add( particle );
    }

    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, GIDI::Construction::PhotoMode::nuclearAndAtomic );
    construction.setUseSystem_strtod( useSystem_strtod );
    constructionPtr = &construction;

    std::string const &mapFilename( argv[argv_options.m_arguments[0]] );
    walk( particles, mapFilename, pops );
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

        if( entry->name( ) == GIDI_importChars ) {
            walk( particles, path, pops ); }
        else if( ( entry->name( ) == GIDI_protareChars ) || ( entry->name( ) == GIDI_TNSLChars ) ) {
            std::vector<std::string> libraries;

            entry->libraries( libraries );
            readProtare( particles, path, pops, libraries, entry->name( ) == GIDI_protareChars ); }
        else {
            std::cout << "    ERROR: unknown map entry name: " << entry->name( ) << std::endl;
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
    std::string throwMessage;
    std::string throwFunction( "GIDI::ProtareSingle" );

    try {
        std::cout << "        " << protareFilename << std::endl;

        GIDI::ParticleSubstitution particleSubstitution;
        protare = new GIDI::ProtareSingle( *constructionPtr, protareFilename, GIDI::FileType::XML, pops, particleSubstitution, a_libraries, 
                GIDI_MapInteractionNuclearChars, a_targetRequiredInGlobalPoPs );
        throwFunction = "post GIDI::ProtareSingle";                     // Unlikely for a throw to occur before "MCGIDI::ProtareSingle".
        GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );


        std::string label( temperatures[0].griddedCrossSection( ) );
        if( doMultiGroup ) label = temperatures[0].heatedMultiGroup( );

        GIDI::Transporting::Settings baseSetting( protare->projectile( ).ID( ), GIDI::Transporting::DelayedNeutrons::on );
        GIDI::Transporting::Particles particles2;
        std::set<std::string> incompleteParticles;
        protare->incompleteParticles( baseSetting, incompleteParticles );
        std::vector<std::string> transportableIncompleteParticles;
        for( auto particle = particles.particles( ).begin( ); particle != particles.particles( ).end( ); ++particle ) {
            if( incompleteParticles.count( particle->first ) == 0 ) {
                particles2.add( particle->second ); }
            else {
                transportableIncompleteParticles.push_back( particle->first );
            }
        }
        if( transportableIncompleteParticles.size( ) > 0 ) {
            std::cout << "            Incomplete transportable particles:";
            for( auto iter = transportableIncompleteParticles.begin( ); iter != transportableIncompleteParticles.end( ); ++iter ) std::cout << " " << *iter;
            std::cout << std::endl;
        }

        MCGIDI::Transporting::MC MC( pops, protare->projectile( ).ID( ), &protare->styles( ), label, GIDI::Transporting::DelayedNeutrons::on, 30.0 );
        throwFunction = "MCGIDI::ProtareSingle";
        MCProtare = new MCGIDI::ProtareSingle( *protare, pops, MC, particles2, domainHash, temperatures, reactionsToExclude ); }
    catch (char const *str) {
        throwMessage = str; }
    catch (std::string str) {
        throwMessage = str; }
    catch (std::exception &exception) {
        throwMessage = exception.what( );
    }

    if( throwMessage != "" ) {
        ++errCount;
        std::cout << "ERROR: throw from " << throwFunction << " with message '" << throwMessage << "'" << std::endl;
    }

    delete MCProtare;
    delete protare;
}

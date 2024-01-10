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
static int maxDepth = 999999;
static int nth = 1;
static int countDown = 1;
static bool printTiming = false;
static bool useSlowerContinuousEnergyConversion = false;
static long numberOfTemperatures = -1;

static char const *description = "Reads in all protares in the specified map file. Besides options, there must be one map file followed by \n"
    "one or more pops files. If an error occurs when reading a protare, the C++ 'throw' message is printed. Also,\n"
    "for each protare all standard LLNL transportable particles with missing or unspecified distribution data will \n"
    "be printed. The standard LLNL transportable particles are n, p, d, t, h, a and g";

void subMain( int argc, char **argv );
void walk( std::string const &a_indent, GIDI::Transporting::Particles const &a_particles, std::string const &mapFilename, 
                PoPI::Database const &pops, int depth );
void readProtare( std::string const &a_indent, GIDI::Transporting::Particles const &a_particles, std::string const &protareFilename, 
                PoPI::Database const &pops, std::vector<std::string> const &a_libraries, bool a_targetRequiredInGlobalPoPs );
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
    argv_options.add( argvOption2( "-n",  true, "If present, only every nth protare is read where 'n' is the next argument." ) );
    argv_options.add( argvOption2( "-t", false, "Prints read time information for each protare read." ) );
    argv_options.add( argvOption2( "--maxDepth", true, "The maximum nesting depth that the map file will be descended. Default is no maximum depth." ) );
    argv_options.add( argvOption2( "--lazyParsing", false, "If true, does lazy parsing for all read protares." ) );
    argv_options.add( argvOption2( "--useSlowerContinuousEnergyConversion", false, "If present, old continuous energy conversion logic is used." ) );
    argv_options.add( argvOption2( "--numberOfTemperatures", true, "The number of temperatures whose data are accessed by MCGIDI." ) );

    argv_options.parseArgv( argc, argv );

//    doMultiGroup = argv_options.find( "--mg" )->present( ); // Currently not working as need to have option to get multi-group and flux information.

    nth = static_cast<int>( argv_options.find( "-n" )->asLong( argv, 1 ) );
    if( nth < 0 ) nth = 1;

    printTiming = argv_options.find( "-t" )->m_counter > 0;

    maxDepth = argv_options.find( "--maxDepth" )->asLong( argv, maxDepth );

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

    numberOfTemperatures = argv_options.find( "--numberOfTemperatures" )->asLong( argv, -1 );

    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, GIDI::Construction::PhotoMode::nuclearAndAtomic );
    construction.setUseSystem_strtod( useSystem_strtod );
    construction.setLazyParsing( argv_options.find( "--lazyParsing" )->m_counter > 0 );
    constructionPtr = &construction;

    useSlowerContinuousEnergyConversion = argv_options.find( "--useSlowerContinuousEnergyConversion" )->present( );

    std::string const &mapFilename( argv[argv_options.m_arguments[0]] );
    LUPI::Timer timer;
    walk( "    ", particles, mapFilename, pops, 0 );

    if( printTiming ) std::cout << std::endl << LUPI::Misc::doubleToString3( "Total CPU %7.3f s", timer.deltaTime( ).CPU_time( ) )
            << LUPI::Misc::doubleToString3( " wall %6.3f s", timer.deltaTime( ).wallTime( ) ) << std::endl;
}
/*
=========================================================
*/
void walk( std::string const &a_indent, GIDI::Transporting::Particles const &a_particles, std::string const &mapFilename, 
                PoPI::Database const &pops, int depth ) {

    if( depth > maxDepth ) return;

    std::string const &indent2 = a_indent + "    ";

    std::cout << a_indent << mapFilename << std::endl;
    GIDI::Map::Map map( mapFilename, pops );

    for( std::size_t i1 = 0; i1 < map.size( ); ++i1 ) {
        GIDI::Map::BaseEntry const *entry = map[i1];

        std::string path = entry->path( GIDI::Map::BaseEntry::PathForm::cumulative );

        if( entry->name( ) == GIDI_importChars ) {
            walk( indent2, a_particles, path, pops, depth + 1 ); }
        else if( ( entry->name( ) == GIDI_protareChars ) || ( entry->name( ) == GIDI_TNSLChars ) ) {
            std::vector<std::string> libraries;

            entry->libraries( libraries );
            readProtare( indent2, a_particles, path, pops, libraries, entry->name( ) == GIDI_protareChars ); }
        else {
            std::cout << "ERROR: unknown map entry name: " << entry->name( ) << std::endl;
        }
    }
}
/*
=========================================================
*/
void readProtare( std::string const &a_indent, GIDI::Transporting::Particles const &a_particles, std::string const &protareFilename, 
                PoPI::Database const &pops, std::vector<std::string> const &a_libraries, bool a_targetRequiredInGlobalPoPs ) {

    --countDown;
    if( countDown != 0 ) return;
    countDown = nth;

    MCGIDI::DomainHash domainHash( 4000, 1e-8, 10 );
    GIDI::ProtareSingle *protare = nullptr;
    MCGIDI::Protare *MCProtare = nullptr;
    std::set<int> reactionsToExclude;
    std::string throwMessage;
    std::string throwFunction( "GIDI::ProtareSingle" );

    try {
        std::cout << a_indent << protareFilename;

        GIDI::ParticleSubstitution particleSubstitution;

        LUPI::Timer timer;
        protare = new GIDI::ProtareSingle( *constructionPtr, protareFilename, GIDI::FileType::XML, pops, particleSubstitution, a_libraries, 
                GIDI_MapInteractionNuclearChars, a_targetRequiredInGlobalPoPs );
        throwFunction = "post GIDI::ProtareSingle";                     // Unlikely for a throw to occur before "MCGIDI::ProtareSingle".
        GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
        long numberOfTemperatures2 = numberOfTemperatures;
        long numberOfTemperatures3 = static_cast<long>( temperatures.size( ) );
        if( numberOfTemperatures2 < 0 ) numberOfTemperatures2 = numberOfTemperatures3;
        if( numberOfTemperatures2 > numberOfTemperatures3 ) numberOfTemperatures2 = numberOfTemperatures3;
        temperatures.resize( numberOfTemperatures2 );

        std::string label( temperatures[0].griddedCrossSection( ) );
        if( doMultiGroup ) label = temperatures[0].heatedMultiGroup( );

        GIDI::Transporting::Settings baseSetting( protare->projectile( ).ID( ), GIDI::Transporting::DelayedNeutrons::on );
        GIDI::Transporting::Particles particles2;
        std::set<std::string> incompleteParticles;
        protare->incompleteParticles( baseSetting, incompleteParticles );
        std::vector<std::string> transportableIncompleteParticles;
        for( auto particle = a_particles.particles( ).begin( ); particle != a_particles.particles( ).end( ); ++particle ) {
            if( incompleteParticles.count( particle->first ) == 0 ) {
                particles2.add( particle->second ); }
            else {
                transportableIncompleteParticles.push_back( particle->first );
            }
        }

        MCGIDI::Transporting::MC MC( pops, protare->projectile( ).ID( ), &protare->styles( ), label, GIDI::Transporting::DelayedNeutrons::on, 30.0 );
        MC.setUseSlowerContinuousEnergyConversion( useSlowerContinuousEnergyConversion );

        LUPI::StatusMessageReporting smr1;
        throwFunction = "MCGIDI::ProtareSingle";
        MCProtare = new MCGIDI::ProtareSingle( smr1, *protare, pops, MC, particles2, domainHash, temperatures, reactionsToExclude );
        if( printTiming ) std::cout << LUPI::Misc::doubleToString3( " CPU %6.3f s", timer.deltaTime( ).CPU_time( ) )
                << LUPI::Misc::doubleToString3( " wall %6.3f s", timer.deltaTime( ).wallTime( ) );
        std::cout << std::endl;

        if( transportableIncompleteParticles.size( ) > 0 ) {
            std::cout << a_indent << "  -- Incomplete transportable particles:";
            for( auto iter = transportableIncompleteParticles.begin( ); iter != transportableIncompleteParticles.end( ); ++iter )
                    std::cout << " " << *iter;
            std::cout << std::endl;
        } }
    catch (char const *str) {
        throwMessage = str; }
    catch (std::string str) {
        throwMessage = str; }
    catch (std::exception &exception) {
        throwMessage = exception.what( );
    }

    if( throwMessage != "" ) {
        ++errCount;
        std::cout << std::endl << "ERROR: throw from " << throwFunction << " with message '" << throwMessage << "'" << std::endl;
    }

    delete MCProtare;
    delete protare;
}

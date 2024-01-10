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

#include "MCGIDI.hpp"

#include "MCGIDI_testUtilities.hpp"

static char const *description = "Loops over energy at the specified temperature, sampling reactions. If projectile is a photon, see options *-pa* and *-pn*.";

void main2( int argc, char **argv );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    try {
        main2( argc, argv );
        exit( EXIT_SUCCESS ); }
    catch (std::exception &exception) {
        std::cerr << exception.what( ) << std::endl; }
    catch (char const *str) {
        std::cout << str << std::endl; }
    catch (std::string &str) {
        std::cout << str << std::endl;
    }
}
/*
=========================================================
*/
void main2( int argc, char **argv ) {

    PoPI::Database pops( "../../../TestData/PoPs/pops.xml" );
    GIDI::Transporting::Particles particles;
    char *endChar;
    std::size_t numberOfSamples = 1000 * 1000;
    void *rngState = nullptr;
    unsigned long long seed = 1;
    std::set<int> reactionsToExclude;
    LUPI::StatusMessageReporting smr1;
    GIDI::Construction::PhotoMode photo_mode = GIDI::Construction::PhotoMode::nuclearOnly;

    std::cerr << "    " << __FILE__;
    for( int i1 = 1; i1 < argc; i1++ ) std::cerr << " " << argv[i1];
    std::cerr << std::endl;

    MCGIDI_test_rngSetup( seed );

    argvOptions2 argv_options( "sampleReactions", description );

    argv_options.add( argvOption2( "--pid", true, "The PoPs id of the projectile." ) );
    argv_options.add( argvOption2( "--tid", true, "The PoPs id of the target." ) );
    argv_options.add( argvOption2( "--map", true, "The map file to use." ) );
    argv_options.add( argvOption2( "-pa", false, "Include photo-atomic protare if relevant. If present, disables photo-nuclear unless *-pn* present." ) );
    argv_options.add( argvOption2( "-pn", false, "Include photo-nuclear protare if relevant. This is the default unless *-pa* present." ) );
    argv_options.add( argvOption2( "--fixedGrid", false, "Set fixed grid data. Only used if protare is only photo-atomic protare." ) );
    argv_options.add( argvOption2( "--temperature", true, "The temperature of the target material." ) );

    argv_options.parseArgv( argc, argv );

    std::string projectileID = argv_options.find( "--pid" )->zeroOrOneOption( argv, PoPI::IDs::neutron );
    std::string targetID = argv_options.find( "--tid" )->zeroOrOneOption( argv, "O16" );
    std::string mapFilename = argv_options.find( "--map" )->zeroOrOneOption( argv, "../../../GIDI/Test/all3T.map" );

    if( argv_options.find( "-pa" )->present( ) ) {
        photo_mode = GIDI::Construction::PhotoMode::atomicOnly;
        if( argv_options.find( "-pn" )->present( ) ) photo_mode = GIDI::Construction::PhotoMode::nuclearAndAtomic;
    }

    GIDI::Map::Map map( mapFilename, pops );

    std::string temperatureString = argv_options.find( "--temperature" )->zeroOrOneOption( argv, "2.58522e-8" );
    double temperature = strtod( temperatureString.c_str( ), &endChar );
    if( *endChar != 0 ) throw "Invalid temperature input.";

    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
    GIDI::Protare *protare = map.protare( construction, pops, projectileID, targetID );

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    for( GIDI::Styles::TemperatureInfos::const_iterator iter = temperatures.begin( ); iter != temperatures.end( ); ++iter ) {
        std::cout << "label = " << iter->heatedCrossSection( ) << "  temperature = " << iter->temperature( ).value( ) << std::endl;
    }

    std::string label( temperatures[0].griddedCrossSection( ) );
    MCGIDI::Transporting::MC MC( pops, projectileID, &protare->styles( ), label, GIDI::Transporting::DelayedNeutrons::on, 20.0 );

    if( argv_options.find( "--fixedGrid" )->present( ) ) {
        GIDI::Groups groups( "../../../GIDI/Test/groups.xml" );
        MC.fixedGridPoints( groups.get<GIDI::Group>( "LLNL_gid_80" )->data( ) );
    }

    MCGIDI::DomainHash domainHash( 4000, 1e-8, 10 );
    MCGIDI::Protare *MCProtare = MCGIDI::protareFromGIDIProtare( smr1, *protare, pops, MC, particles, domainHash, temperatures, reactionsToExclude );

    MCGIDI::Vector<MCGIDI::Protare *> protares( 1 );
    protares[0] = MCProtare;
    MCGIDI::URR_protareInfos URR_protare_infos( protares );

    int numberOfReactions = (int) MCProtare->numberOfReactions( );

    if( argv_options.find( "--fixedGrid" )->present( ) ) {
        for( std::size_t reactionIndex = 0; reactionIndex < MCProtare->numberOfReactions( ); ++reactionIndex ) {
            MCGIDI::Reaction const *reaction = MCProtare->reaction( reactionIndex );

            std::cout << reaction->label( ).c_str( ) << std::endl;
        }
    }

    std::cout << "temperature = " << temperature << std::endl;
    std::cout << "numberOfSamples = " << numberOfSamples << std::endl;
    for( double energy = 1e-12; energy < 25.0; energy *= 1.2 ) {
        int hashIndex = domainHash.index( energy );

        double crossSection = MCProtare->crossSection( URR_protare_infos, hashIndex, temperature, energy );
        std::cout << "energy = " << energy << " " << "cross section = " << doubleToString2( "%13.6e", crossSection ) << std::endl;
        if( crossSection == 0.0 ) continue;

        std::vector<double> reactionCrossSections( numberOfReactions );
        std::cout << "      ";
        for( int i1 = 0; i1 < numberOfReactions; ++i1 ) {
            double reactionCrossSection = MCProtare->reactionCrossSection( i1, URR_protare_infos, hashIndex, temperature, energy );
            std::cout << LUPI::Misc::argumentsToString( " %9.6f", reactionCrossSection / crossSection );
        }
        std::cout << std::endl;

        std::vector<long> counts( numberOfReactions + 1, 0 );
        for( std::size_t i1 = 0; i1 < numberOfSamples; ++i1 ) {
            int reactionIndex = MCProtare->sampleReaction( URR_protare_infos, hashIndex, temperature, energy, crossSection, float64RNG64, rngState );
            if( reactionIndex > numberOfReactions ) reactionIndex = numberOfReactions;
            ++counts[reactionIndex];
        }

        std::cout << "      ";
        for( int i1 = 0; i1 < numberOfReactions; ++i1 ) {
            double ratio = counts[i1];
            std::cout << LUPI::Misc::argumentsToString( " %9.6f", ratio / numberOfSamples );
        }
        std::cout << std::endl;

        std::cout << "      ";
        for( int i1 = 0; i1 < numberOfReactions; ++i1 ) {
            std::cout << LUPI::Misc::argumentsToString( " %9ld", counts[i1] );
        }
        std::cout << std::endl;
    }

    delete protare;

    delete MCProtare;
}

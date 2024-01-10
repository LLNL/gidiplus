/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <set>

#include <statusMessageReporting.h>

#include <GIDI_testUtilities.hpp>

static char const *description = "Displays a list of all reactions for protares matching projectileID and  targetID in the specified map file.";

void main2( int argc, char **argv );
void printUsage( );
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

    std::vector<std::string> libraries;

    argvOptions argv_options( "reactions", description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );

    parseTestOptions.parse( );

    PoPI::Database pops;
    parseTestOptions.pops( pops, "../../../../TestData/PoPs/pops.xml" );

    GIDI::Construction::PhotoMode photo_mode = parseTestOptions.photonMode( GIDI::Construction::PhotoMode::nuclearAndAtomic );
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
    GIDI::ParticleSubstitution particleSubstitution;

    std::string mapFilename = argv_options.find( "--map" )->zeroOrOneOption( argv, "../../../Test/all.map" );
    GIDI::Map::Map map( mapFilename, pops );

    std::string projectileID = argv_options.find( "--pid" )->zeroOrOneOption( argv, "" );
    std::string targetID = argv_options.find( "--tid" )->zeroOrOneOption( argv, "" );

    std::vector<GIDI::Map::ProtareBase const *> protareEntries = map.directory( projectileID, targetID );
    for( std::size_t i1 = 0; i1 < protareEntries.size( ); ++i1 ) {
        GIDI::Map::ProtareBase const *protareEntry = protareEntries[i1];
        std::string path( protareEntry->path( ) );

        printf( "%-8s %-8s %-24s %s\n", protareEntry->projectileID( ).c_str( ), protareEntry->targetID( ).c_str( ), protareEntry->evaluation( ).c_str( ),
            protareEntry->path( ).c_str( ) );
        GIDI::ProtareSingle protare( construction, path, GIDI::FileType::XML, pops, particleSubstitution, libraries, GIDI_MapInteractionNuclearChars, false, false );

        GIDI::Suite const &reactions = protare.reactions( );
        for( std::size_t reactionIndex = 0; reactionIndex < reactions.size( ); ++reactionIndex ) {
            GIDI::Reaction const &reaction = *reactions.get<GIDI::Reaction>( reactionIndex );

            printf( "   %3d: %-32s   %3d   %3d   %3d\n", (int) reactionIndex, reaction.label( ).c_str( ), reaction.ENDF_MT( ), reaction.ENDL_C( ), reaction.ENDL_S( ) );
        }
    }

    exit( EXIT_SUCCESS );
}

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

#include "GIDI.hpp"

static int mode = 0;
static bool printLibraries = false;
static bool useSystem_strtod = true;
static GIDI::Construction::Settings *constructionPtr = nullptr;

void subMain( int argc, char **argv );
void walk( std::string const &mapFilename, PoPI::Database const &pops );
void readProtare( std::string const &protareFilename, PoPI::Database const &pops, std::vector<std::string> &a_libraries, bool a_targetRequiredInGlobalPoPs );
void printUsage( );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    subMain( argc, argv );
    exit( EXIT_SUCCESS );
}
/*
=========================================================
*/
void subMain( int argc, char **argv ) {

    int iarg = 1;
    PoPI::Database pops = PoPI::Database( );
    char *mapFilePtr = nullptr;

    for( ; iarg < argc; ++iarg ) {

        if( argv[iarg][0] == '-' ) {
            std::string arg( argv[iarg] );

            if( arg == "-h" ) {
                printUsage( ); }
            else if( arg == "-f" ) {
                useSystem_strtod = false; }
            else if( arg == "-m" ) {
                ++iarg;
                if( iarg == argc ) throw std::range_error( "-m options needs integer argument" );

                char *endPtr;
                long i2 = strtol( argv[iarg], &endPtr, 10 );
                if( ( i2 < 0 ) || ( i2 > static_cast<int>( GIDI::Construction::ParseMode::outline ) ) ) throw std::range_error( "-m argument out of range" );
                mode = (int) i2; }
            else if( arg == "-l" ) {
                printLibraries = true; }
            else {
                std::cerr << "Unsupported options '" << arg << "'" << std::endl;
                exit( EXIT_FAILURE );
            } }
        else {
            try {
                if( mapFilePtr == nullptr ) {
                    mapFilePtr = argv[iarg]; }
                else {
                    pops.addFile( argv[iarg], true );
                } }
            catch (char const *str) {
                std::cerr << "PoPI::Database failed while reading '" << argv[iarg] << "'" << std::endl;
                std::cerr << str << std::endl;
                exit( EXIT_FAILURE );
            }
        }
    }

    if( mapFilePtr == nullptr ) throw std::range_error( "need map file name" );
    if( argc < iarg ) printUsage( );

    GIDI::Construction::ParseMode parseMode( GIDI::Construction::ParseMode::all );
    if( mode == 1 ) {
        parseMode = GIDI::Construction::ParseMode::multiGroupOnly; }
    else if( mode == 2 ) {
        parseMode = GIDI::Construction::ParseMode::MonteCarloContinuousEnergy ; }
    else if( mode == 3 ) {
        parseMode = GIDI::Construction::ParseMode::excludeProductMatrices; }
    else if( mode == 4 ) {
        parseMode = GIDI::Construction::ParseMode::outline;
    }
    GIDI::Construction::Settings construction( parseMode, GIDI::Construction::PhotoMode::nuclearAndAtomic );
    construction.setUseSystem_strtod( useSystem_strtod );
    constructionPtr = &construction;

    std::string const &mapFilename( mapFilePtr );
    try {
        walk( mapFilename, pops ); }
    catch (char const *str) {
        std::cout << str << std::endl; }
    catch (std::string str) {
        std::cout << str << std::endl;
    }
}
/*
=========================================================
*/
void walk( std::string const &mapFilename, PoPI::Database const &pops ) {

    std::cout << "    " << mapFilename << std::endl;
    GIDI::Map::Map map( mapFilename, pops );

    for( std::size_t i1 = 0; i1 < map.size( ); ++i1 ) {
        GIDI::Map::BaseEntry const *entry = map[i1];

        std::string path = entry->path( GIDI::Map::BaseEntry::PathForm::cumulative );

        if( entry->name( ) == GIDI_importChars ) {
            walk( path, pops ); }
        else if( ( entry->name( ) == GIDI_protareChars ) || ( entry->name( ) == GIDI_TNSLChars ) ) {
            std::vector<std::string> libraries;

            entry->libraries( libraries );
            readProtare( path, pops, libraries, entry->name( ) == GIDI_protareChars ); }
        else {
            std::cerr << "    ERROR: unknown map entry name: " << entry->name( ) << std::endl;
        }
    }
}
/*
=========================================================
*/
void readProtare( std::string const &protareFilename, PoPI::Database const &pops, std::vector<std::string> &a_libraries, bool a_targetRequiredInGlobalPoPs ) {

    GIDI::Protare *protare = nullptr;
    GIDI::ParticleSubstitution particleSubstitution;

    try {
        std::cout << "        " << protareFilename;

        protare = new GIDI::ProtareSingle( *constructionPtr, protareFilename, GIDI::FileType::XML, pops, particleSubstitution, a_libraries, 
                GIDI_MapInteractionNuclearChars, a_targetRequiredInGlobalPoPs );

        if( printLibraries ) {
            std::cout << ": libraries =";
            std::vector<std::string> libraries( protare->libraries( ) );
            for( std::vector<std::string>::iterator iter = libraries.begin( ); iter != libraries.end( ); ++iter ) std::cout << " " << *iter;
        }
        std::cout << std::endl; }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    delete protare;
}
/*
=========================================================
*/
void printUsage( ) {

    printf( "\nUSAGE:\n" );
    printf( "    readAllProtaresInMapFile [-h] [-f] [-m MFLAG] mapFile popsFile [popsFile, [popsFile, ...]]\n\n" );

    printf( "    mapFile        Map file name to use.\n" );
    printf( "    popsFile       One or more PoPs file names. At least 1 is required.\n" );
    printf( "\n" );

    printf( "    -h         Print usage information.\n" );
    printf( "    -f         Use nf_strtod instead of the system stdtod.\n" );
    printf( "    -m         Which GIDI::Construction::Settings flag to use,\n" );
    printf( "                   0 is GIDI::Construction::ParseMode::all,\n" );
    printf( "                   1 is GIDI::Construction::ParseMode::multiGroupOnly,\n" );
    printf( "                   2 is GIDI::Construction::ParseMode::MonteCarloContinuousEnergy,\n" );
    printf( "                   3 is GIDI::Construction::ParseMode::excludeProductMatrices,\n" );
    printf( "                   4 is GIDI::Construction::ParseMode::outline.\n" );
    printf( "    -l         Print libraries for each Protare.\n" );
    printf( "\n" );

    exit( EXIT_SUCCESS );
}

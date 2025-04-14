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

static char const *description = "This program traverses a map file, printing each protare, and its library and resolved library.";

GIDI::Map::Map *mapRoot = nullptr;

void main2( int argc, char **argv );
void walk( std::string const &mapFilename, PoPI::Database const &pops );
void readProtare( std::string const &protareFilename, PoPI::Database const &pops, LUPI_maybeUnused std::string const &a_resolvedLibrary );
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

    argvOptions argv_options( "readAll", description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );

    parseTestOptions.m_askPid = false;
    parseTestOptions.m_askTid = false;
    parseTestOptions.m_askPhotoAtomic = false;
    parseTestOptions.m_askPhotoNuclear = false;

    parseTestOptions.parse( );

    PoPI::Database pops;
    parseTestOptions.pops( pops, "../../../TestData/PoPs/pops.xml" );

    std::string mapFilename = argv_options.find( "--map" )->zeroOrOneOption( argv, "../all.map" );

    walk( mapFilename, pops );
}
/*
=========================================================
*/
void walk( std::string const &mapFilename, PoPI::Database const &pops ) {

    std::cout << "    " << stripDirectoryBase( mapFilename, "/GIDI/" ) << std::endl;
    GIDI::Map::Map map( mapFilename, pops );

    if( mapRoot == nullptr ) mapRoot = &map;

    for( std::size_t i1 = 0; i1 < map.size( ); ++i1 ) {
        GIDI::Map::BaseEntry const *entry = map[i1];

        std::string path = entry->path( GIDI::Map::BaseEntry::PathForm::cumulative );

        if( entry->name( ) == GIDI_importChars ) {
            walk( path, pops ); }
        else if( ( entry->name( ) == GIDI_protareChars ) || ( entry->name( ) == GIDI_TNSLChars ) ) {
            readProtare( path, pops, map.resolvedLibrary( ) ); }
        else {
            std::cerr << "    ERROR: unknown map entry name: " << entry->name( ) << std::endl;
        }
    }
}
/*
=========================================================
*/
void readProtare( std::string const &protareFilename, PoPI::Database const &pops, LUPI_maybeUnused std::string const &a_resolvedLibrary ) {

    std::cout << std::endl;
    std::cout << "        " << stripDirectoryBase( protareFilename, "/GIDI/" ) << std::endl;

    GIDI::Protare *protare;
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, GIDI::Construction::PhotoMode::nuclearAndAtomic );
    GIDI::ParticleSubstitution particleSubstitution;
    std::vector<std::string> libraries;

    protare = new GIDI::ProtareSingle( construction, protareFilename, GIDI::FileType::XML, pops, particleSubstitution, libraries, GIDI_MapInteractionNuclearChars );

    GIDI::Map::ProtareBase const *protareEntry = mapRoot->findProtareEntry( protare->projectile( ).ID( ), protare->target( ).ID( ), "", protare->evaluation( ) );
    if( protareEntry == nullptr ) {
        std::cout << "ERROR: Could not find protare '" << protare->projectile( ).ID( ) << " + " << protare->target( ).ID( ) 
                << "' with evaluation " << protare->evaluation( ) << " in map file using findProtareEntry."; }
    else {
        std::cout << "        library          = " << protareEntry->parent( )->library( ) << std::endl;
        std::cout << "        resolved library = " << protareEntry->parent( )->resolvedLibrary( ) << std::endl;

        GIDI::stringAndDoublePairs labelsAndMuCutoffs = protare->muCutoffForCoulombPlusNuclearElastic( );
        for( std::size_t i1 = 0; i1 < labelsAndMuCutoffs.size( ); ++i1 ) {
            GIDI::stringAndDoublePair labelAndMuCutoff = labelsAndMuCutoffs[i1];

            std::cout << "    label = " << labelAndMuCutoff.first << " mu = " << labelAndMuCutoff.second << std::endl;
        }
    }

    delete protare;
}

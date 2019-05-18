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

#include "GIDI_testUtilities.hpp"

GIDI::Map *mapRoot = NULL;

void walk( std::string const &mapFilename, PoPs::Database const &pops );
void readProtare( std::string const &protareFilename, PoPs::Database const &pops, std::string const &a_resolvedLibrary );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    PoPs::Database pops( "../pops.xml" );
    std::string mapFilename( "../all.map" );

    if( argc > 1 ) mapFilename = "../Data/MG_MC/all_maps.map";

    std::cerr << "    " << __FILE__;
    for( int i1 = 1; i1 < argc; i1++ ) std::cerr << " " << argv[i1];
    std::cerr << std::endl;

    walk( mapFilename, pops );
}
/*
=========================================================
*/
void walk( std::string const &mapFilename, PoPs::Database const &pops ) {

    std::cout << "    " << stripDirectoryBase( mapFilename, "/GIDI3/Test/Data/MG_MC" ) << std::endl;
    GIDI::Map map( mapFilename, pops );

    if( mapRoot == NULL ) mapRoot = &map;

    for( std::size_t i1 = 0; i1 < map.size( ); ++i1 ) {
        GIDI::MapBaseEntry const *entry = map[i1];

        std::string path = entry->path( GIDI::MapBaseEntry::e_cumulative );

        if( entry->name( ) == GIDI_importMoniker ) {
            walk( path, pops ); }
        else if( ( entry->name( ) == GIDI_protareMoniker ) || ( entry->name( ) == GIDI_TNSLMoniker ) ) {
            readProtare( path, pops, map.resolvedLibrary( ) ); }
        else {
            std::cerr << "    ERROR: unknown map entry name: " << entry->name( ) << std::endl;
        }
    }
}
/*
=========================================================
*/
void readProtare( std::string const &protareFilename, PoPs::Database const &pops, std::string const &a_resolvedLibrary ) {

    std::cout << std::endl;
    std::cout << "        " << protareFilename.substr( protareFilename.find( "GIDI3" ) ) << std::endl;

    GIDI::Protare *protare;
    try {
        GIDI::Construction::Settings construction( GIDI::Construction::e_all );
        std::vector<std::string> libraries;

        protare = new GIDI::ProtareSingleton( construction, protareFilename, GIDI::XML, pops, libraries );

        GIDI::ProtareBaseEntry const *protareEntry = mapRoot->findProtareEntry( protare->projectile( ).ID( ), protare->target( ).ID( ), protare->evaluation( ) );
        std::cout << "        library          = " << protareEntry->parent( )->library( ) << std::endl;
        std::cout << "        resolved library = " << protareEntry->parent( )->resolvedLibrary( ) << std::endl;

        GIDI::stringAndDoublePairs labelsAndMuCutoffs = protare->muCutoffForCoulombPlusNuclearElastic( );
        for( std::size_t i1 = 0; i1 < labelsAndMuCutoffs.size( ); ++i1 ) {
            GIDI::stringAndDoublePair labelAndMuCutoff = labelsAndMuCutoffs[i1];

            std::cout << "    label = " << labelAndMuCutoff.first << " mu = " << labelAndMuCutoff.second << std::endl;
        } }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    delete protare;
}

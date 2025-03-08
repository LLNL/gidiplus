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

void walk( PoPI::Database &pops, char const *fileName );
bool mapWalkCallBack( GIDI::Map::ProtareBase const *a_protareEntry, LUPI_maybeUnused std::string const &a_library, LUPI_maybeUnused void *a_data, int a_level );
void printList( char const *prefix, std::vector<std::string> &list );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    PoPI::Database pops( "../../../TestData/PoPs/pops.xml" );

    printCodeArguments( __FILE__, argc, argv );

    if( argc > 1 ) throw "No arguments are allowed";

    walk( pops, "neutrons1.map" );
    walk( pops, "neutrons2.map" );
    walk( pops, "all.map" );
    walk( pops, "level3.map" );
    walk( pops, "level4.map" );

    exit( EXIT_SUCCESS );
}
/*
=========================================================
*/
void walk( PoPI::Database &pops, char const *fileName ) {

    GIDI::Map::Map *map;

    try {
        map = new GIDI::Map::Map( fileName, pops ); }
    catch ( char const *str ) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    std::cout << std::endl;
    std::cout << "library = " << map->library( ) << std::endl;
    std::cout << std::endl;

    map->walk( mapWalkCallBack, map );

    delete map;
}
/*
=========================================================
*/
bool mapWalkCallBack( GIDI::Map::ProtareBase const *a_protareEntry, LUPI_maybeUnused std::string const &a_library, LUPI_maybeUnused void *a_data, int a_level ) {

    std::string path( stripDirectoryBase( a_protareEntry->path( ) ) );

    for( int i1 = 0; i1 < a_level + 1; ++i1 ) std::cout << "    ";
    std::cout << "path = " << path << ":: evaluation = " << a_protareEntry->evaluation( ) << ":: libraries are: ";
    std::vector<std::string> libraries;
    a_protareEntry->libraries( libraries );
    for( std::vector<std::string>::iterator iter = libraries.begin( ); iter != libraries.end( ); ++iter ) std::cout << " " << *iter;
    std::cout << std::endl;

    return( true );
}
/*
=========================================================
*/
void printList( char const *prefix, std::vector<std::string> &list ) {

    std::string sep( "" );

    std::cout << prefix << " <";
    for( std::vector<std::string>::const_iterator iter = list.begin( ); iter != list.end( ); ++iter ) {
        std::cout << sep << "'" << *iter << "'";
        sep = ", ";
    }
    std::cout << ">" << std::endl;
}

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

static char const *description = "Test the library argument for the Map interface.";

void main2( int argc, char **argv );
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
        std::cout << str << std::endl;
        exit( EXIT_FAILURE ); }
    catch (std::string &str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    exit( EXIT_SUCCESS );
}
/*
=========================================================
*/
void main2( int argc, char **argv ) {

    argvOptions argv_options( "libraries", description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );
    parseTestOptions.m_askPid = false;
    parseTestOptions.m_askTid = false;
    parseTestOptions.m_askPhotoAtomic = false;
    parseTestOptions.m_askPhotoNuclear = false;

    parseTestOptions.parse( );

    PoPI::Database pops( "../../../TestData/PoPs/pops.xml" );
    parseTestOptions.pops( pops, "../pops.xml" );
    std::string mapFilename = argv_options.find( "--map" )->zeroOrOneOption( argv, "libraries.map" );
    GIDI::Map::Map map( mapFilename, pops );

    std::cout << "    isProtareAvailable( PoPI::IDs::neutron, H1 ) " << map.isProtareAvailable( PoPI::IDs::neutron, "H1" ) << std::endl;

    std::vector<GIDI::Map::ProtareBase const *> list = map.directory( "n", "H1" );

    std::cout << "    Printing all libraries that contain 'n + H1'." << std::endl;
    for( auto iter = list.begin( ); iter != list.end( ); ++iter ) {
        std::string library = (*iter)->parent( )->library( );

        std::cout << "        " << library << "  " << map.isProtareAvailable( PoPI::IDs::neutron, "H1", library ) << std::endl;
    }

    std::cout << "    Printing a libraries that does not contain 'n + H1'." << std::endl;
    std::string library = "junk";
    std::cout << "        " << library << "  " << map.isProtareAvailable( PoPI::IDs::neutron, "H1", library ) << std::endl;
}

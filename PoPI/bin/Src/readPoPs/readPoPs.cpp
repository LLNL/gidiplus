/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdio.h>
#include <iostream>

#include "LUPI.hpp"
#include "PoPI.hpp"

static char const *description = "Reads in a list of pops files and prints out timing information. Note, with the '-x' option\n"
    "the type of file is not verified (e.g., if the file is not a PoPs file not 'ERROR' is reported).";

/*
=========================================================
*/
int main( int argc, char **argv ) {

    PoPI::Database pops;
    pugi::xml_document doc;

    LUPI::ArgumentParser argumentParser( __FILE__, description );

    LUPI::Positional *fileNames = argumentParser.add<LUPI::Positional>( "fileNames", "List of pops files to read.", 1, -1 );
    LUPI::OptionTrue *xmlParseOnly = argumentParser.add<LUPI::OptionTrue>( "-xmlParseOnly", "If entered, only pugixml parsing is done." );
    argumentParser.addAlias( "-xmlParseOnly", "-x" );

    argumentParser.parse( argc, argv );

    LUPI::Timer timer;
    for( auto fileName = fileNames->values( ).begin( ); fileName != fileNames->values( ).end( ); ++fileName ) {
        std::cout << "      reading " << *fileName << std::endl;
        if( xmlParseOnly->counts( ) ) {
            pugi::xml_parse_result result = doc.load_file( fileName->c_str( ) );
            if( result.status != pugi::status_ok ) {
                std::cerr << "    " << *fileName << std::endl;
                std::cerr << "        ERROR: in file " << *fileName << ": " << result.description( ) << std::endl;
            } }
        else {
            try {
                pops.addFile( *fileName, false ); }
            catch (std::exception &exception) {
                std::cerr << "        ERROR: " << exception.what( ) << std::endl;
                continue;
            }
        }
        std::cout << "        " << timer.deltaTime( ).toString( ) << "." << std::endl;
    }
}

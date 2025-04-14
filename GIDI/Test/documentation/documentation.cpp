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

static char const *description = "This test prints the multi-group Q-value for the specified protare and for each of its reactions.";

void main2( int argc, char **argv );
void printDocumentation( GIDI::Protare *protare );
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

    argvOptions argv_options( "documentation", description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );

    parseTestOptions.parse( );

    GIDI::Construction::PhotoMode photo_mode = parseTestOptions.photonMode( GIDI::Construction::PhotoMode::nuclearAndAtomic );
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
    PoPI::Database pops;
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../../../TestData/PoPs/pops.xml", "../all.map", construction, PoPI::IDs::neutron, "O16" );

    std::cout << stripDirectoryBase( protare->fileName( ), "/GIDI/Test" ) << std::endl;

    printDocumentation( protare );
    std::cout << std::endl;

    delete protare;
}
/*
=========================================================
*/
void printDocumentation( GIDI::Protare *protare ) {

    LUPI::StatusMessageReporting smr1;

    for( std::size_t i1 = 0; i1 < protare->styles( ).size( ); ++i1 ) {
        GIDI::Styles::Base *style = protare->styles( ).get<GIDI::Styles::Base>( i1 );
        std::cout<< i1 << " " << style->label() << " <- " << style->derivedStyle() << std::endl;
        GUPI::Documentation *documentation = style->documentation();
        if (documentation != nullptr) {
            std::cout << "  " << documentation->publicationDate()  << std::endl;
            std::cout << "  " << documentation->title().body()  << std::endl;
            std::cout << "  " << documentation->abstract().body()  << std::endl;
            std::cout << "  " << documentation->body().body()  << std::endl;
        }
    }

}

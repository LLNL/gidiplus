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

#include <GIDI_testUtilities.hpp>

static char const *description = "Prints the label for each **crossSectionSum** and **multiplicitySum** node.";

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

    argvOptions argv_options( "sumsCheck", description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );

    parseTestOptions.parse( );

    GIDI::Construction::PhotoMode photo_mode = parseTestOptions.photonMode( GIDI::Construction::PhotoMode::nuclearAndAtomic );
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
    PoPI::Database pops;
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../../../TestData/PoPs/pops.xml", "../all.map", construction, PoPI::IDs::neutron, "O16" );

    GIDI::ProtareSingle *protareSingle( protare->protare( 0 ) );

    std::cout << stripDirectoryBase( protareSingle->fileName( ), "/GIDI/Test/" ) << std::endl;

    GIDI::Sums::Sums const &sums( protareSingle->sums( ) );

    std::cout << std::endl;
    std::cout << "crossSectionSum labels:" << std::endl;
    GIDI::Suite const &crossSectionSums = sums.crossSectionSums( );
    for( std::size_t index = 0; index < crossSectionSums.size( ); ++index ) {
        GIDI::Sums::CrossSectionSum const *crossSectionSum = crossSectionSums.get<GIDI::Sums::CrossSectionSum>( index );

        std::cout << "    " << crossSectionSum->label( ) << std::endl;
    }

    std::cout << std::endl;
    std::cout << "multiplicitySum labels:" << std::endl; 
    GIDI::Suite const &multiplicitySums = sums.multiplicitySums( );
    for( std::size_t index = 0; index < multiplicitySums.size( ); ++index ) {
        GIDI::Sums::MultiplicitySum const *multiplicitySum = multiplicitySums.get<GIDI::Sums::MultiplicitySum>( index );

        std::cout << "    " << multiplicitySum->label( ) << std::endl;
    }
//    for( auto iter = sums.multiplicitySums( ).begin( ); iter != sums.multiplicitySums( ).end( ); iter++ ) std::cout << "    " << (*iter)->label( ) << std::endl;

    delete protare;
}

/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>
#include <math.h>
#include <iostream>

#include <GIDI_testUtilities.hpp>

void main2( int argc, char **argv );
void readProtare( PoPI::Database const &a_pops, GIDI::Map::Map const &a_map, GIDI::Construction::Settings const &construction, std::string const &a_pid, std::string const &a_tid );
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

    printCodeArguments( __FILE__, argc, argv );

    std::string tid( "HinCH2" );
    PoPI::Database pops( "../../../TestData/PoPs/pops.xml" );
    GIDI::Map::Map map( "../Data/MG_MC/all_maps.map", pops );
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, GIDI::Construction::PhotoMode::nuclearAndAtomic );

    readProtare( pops, map, construction, "n",      tid );
    readProtare( pops, map, construction, "H1",     tid );
    readProtare( pops, map, construction, "photon", tid );
}
/*
=========================================================
*/
void readProtare( PoPI::Database const &a_pops, GIDI::Map::Map const &a_map, GIDI::Construction::Settings const &construction, std::string const &a_pid, std::string const &a_tid ) {

    GIDI::Protare *protare = a_map.protare( construction, a_pops, a_pid, a_tid );

    std::cout << stripDirectoryBase( protare->fileName( ), "/GIDI/Test/" ) << std::endl;
    std::cout << "    projectile = " << protare->projectile( ).ID( ) << ";    target = " << protare->target( ).ID( ) << ";    evaluation = " << protare->evaluation( ) << std::endl;

    bool isTNSL = protare->protareType( ) == GIDI::ProtareType::TNSL;
    std::cout << "    Is TNSL protare = " << isTNSL << std::endl;

    std::cout << "        " << "Number of reactions = " << protare->numberOfReactions( ) << std::endl;

    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = protare->reaction( index );

        std::cout << "            " << reaction->label( ) << std::endl;
    }

    delete protare;
}

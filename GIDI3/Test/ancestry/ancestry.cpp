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

void printReaction( GIDI::Reaction const *a_reaction, GIDI::Ancestry const *a_protare );
void printSuite( GIDI::Suite const &a_suite, GIDI::Ancestry const *a_protare, GIDI::Ancestry const *a_reaction, bool a_isProduct = false );
void printOutputChannel( GIDI::OutputChannel const *a_outputChannel, GIDI::Ancestry const *a_protare, GIDI::Ancestry const *a_reaction );
std::string checkXLink( GIDI::Ancestry const &ancestry, GIDI::Ancestry const *a_protare, GIDI::Ancestry const *a_reaction );
void checkXLink2( GIDI::Ancestry const &a_ancestry, std::string const &xlink, GIDI::Ancestry const *from );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    PoPs::Database pops( "../pops.xml" );
    std::string mapFilename( "../all.map" );
    GIDI::Map map( mapFilename, pops );
    std::string projectileID = "photon";
    std::string targetID = "O";
    GIDI::Protare *protare;

    std::cerr << "    " << __FILE__;
    for( int i1 = 1; i1 < argc; i1++ ) std::cerr << " " << argv[i1];
    std::cerr << std::endl;

    if( argc > 1 ) targetID = argv[1];

    try {
        GIDI::Construction::Settings construction( GIDI::Construction::e_all );
        protare = map.protare( construction, pops, projectileID, targetID ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    std::cout << stripDirectoryBase( protare->fileName( ), "/GIDI3/Test/Data/MG_MC/" ) << std::endl;

    GIDI::Reaction const *reaction = protare->reaction( 0 );
    std::cout << checkXLink( *protare, protare, reaction ) << std::endl;

    printSuite( protare->styles( ), protare, reaction );

    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = protare->reaction( index );
        printReaction( reaction, protare );
    }

    for( std::size_t index = 0; index < protare->numberOfOrphanProducts( ); ++index ) {
        GIDI::Reaction const *reaction = protare->orphanProduct( index );
        printReaction( reaction, protare );
    }

    delete protare;
}
/*
=========================================================
*/
void printReaction( GIDI::Reaction const *a_reaction, GIDI::Ancestry const *a_protare ) {

    std::cout << "" << checkXLink( *a_reaction, a_protare, a_reaction ) << std::endl;

    printSuite( a_reaction->doubleDifferentialCrossSection( ), a_protare, a_reaction );
    printSuite( a_reaction->crossSection( ), a_protare, a_reaction );
    printSuite( a_reaction->availableEnergy( ), a_protare, a_reaction );
    printSuite( a_reaction->availableMomentum( ), a_protare, a_reaction );

    printOutputChannel( a_reaction->outputChannel( ), a_protare, a_reaction );
}
/*
=========================================================
*/
void printSuite( GIDI::Suite const &a_suite, GIDI::Ancestry const *a_protare, GIDI::Ancestry const *a_reaction, bool a_isProduct ) {

    std::cout << "    " << checkXLink( a_suite, a_protare, a_reaction ) << std::endl;
    for( std::size_t i1 = 0; i1 < a_suite.size( ); ++i1 ) {
        GIDI::Form const *form = a_suite.get<GIDI::Form>( i1 );
        std::cout << "    " << checkXLink( *form, a_protare, a_reaction ) << std::endl;

        if( a_isProduct ) {
            GIDI::Product const *product = static_cast<GIDI::Product const *>( form );

                printSuite( product->multiplicity( ), a_protare, a_reaction );
                printSuite( product->distribution( ), a_protare, a_reaction );
                printSuite( product->averageEnergy( ), a_protare, a_reaction );
                printSuite( product->averageMomentum( ), a_protare, a_reaction );

                GIDI::OutputChannel const *outputChannel = product->outputChannel( );
                if( outputChannel != NULL ) printOutputChannel( outputChannel, a_protare, a_reaction );
        }
    }
}
/*
=========================================================
*/
void printOutputChannel( GIDI::OutputChannel const *a_outputChannel, GIDI::Ancestry const *a_protare, GIDI::Ancestry const *a_reaction ) {

    std::cout << "    " << checkXLink( *a_outputChannel, a_protare, a_reaction ) << std::endl;
    printSuite( a_outputChannel->Q( ), a_protare, a_reaction );
    printSuite( a_outputChannel->products( ), a_protare, a_reaction, true );
}
/*
=========================================================
*/
std::string checkXLink( GIDI::Ancestry const &a_ancestry, GIDI::Ancestry const *a_protare, GIDI::Ancestry const *a_reaction ) {

    std::string xlink = a_ancestry.toXLink( );

    checkXLink2( a_ancestry, xlink, a_protare );
    checkXLink2( a_ancestry, xlink, a_reaction );

    return( xlink );
}
/*
=========================================================
*/
void checkXLink2( GIDI::Ancestry const &a_ancestry, std::string const &a_xlink, GIDI::Ancestry const *a_from ) {

    GIDI::Ancestry const *link = a_from->findInAncestry( a_xlink );

    if( link == NULL ) {
        std::cout << " LINK NOT FOUND "; }
    else {
        if( &a_ancestry != link ) {
            std::cout << " a_ancestry DOES NOT MATCH link "; }
        else {
            std::string toXLink( link->toXLink( ) );
            if( a_xlink != toXLink ) {
                std::cout << " XLINK DOES NOT MATCH link->toXLink( ) ";
            }
        }
    }
}

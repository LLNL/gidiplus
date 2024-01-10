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

static char const *description = "This code checks that the Ancestry class functions findInAncestry and toXLink are working, and that GIDI set setAncestry in all the required objects.";

void main2( int argc, char **argv );
void printReaction( GIDI::Reaction *a_reaction, GUPI::Ancestry *a_protare );
void printSuite( GIDI::Suite &a_suite, GUPI::Ancestry *a_protare, GUPI::Ancestry *a_reaction, bool a_isProduct = false );
void printOutputChannel( GIDI::OutputChannel *a_outputChannel, GUPI::Ancestry *a_protare, GUPI::Ancestry *a_reaction );
std::string checkXLink( GUPI::Ancestry &ancestry, GUPI::Ancestry *a_protare, GUPI::Ancestry *a_reaction );
void checkXLink2( GUPI::Ancestry &a_ancestry, std::string &xlink, GUPI::Ancestry *from );
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

    argvOptions argv_options( "ancestry", description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );

    parseTestOptions.m_askGNDS_File = true;

    parseTestOptions.parse( );

    GIDI::Construction::PhotoMode photo_mode = parseTestOptions.photonMode( GIDI::Construction::PhotoMode::atomicOnly );
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
    PoPI::Database pops;
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../../../TestData/PoPs/pops.xml", "../all.map", construction, PoPI::IDs::photon, "O" );

    std::cout << stripDirectoryBase( protare->fileName( ), "/GIDI/Test/" ) << std::endl;

    GIDI::Reaction *reaction = protare->reaction( 0 );
    std::cout << checkXLink( *protare, protare, reaction ) << std::endl;

    printSuite( protare->styles( ), protare, reaction );

    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        reaction = protare->reaction( index );
        printReaction( reaction, protare );
    }

    for( std::size_t index = 0; index < protare->numberOfOrphanProducts( ); ++index ) {
        reaction = protare->orphanProduct( index );
        printReaction( reaction, protare );
    }

    delete protare;
}
/*
=========================================================
*/
void printReaction( GIDI::Reaction *a_reaction, GUPI::Ancestry *a_protare ) {

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
void printSuite( GIDI::Suite &a_suite, GUPI::Ancestry *a_protare, GUPI::Ancestry *a_reaction, bool a_isProduct ) {

    std::cout << "    " << checkXLink( a_suite, a_protare, a_reaction ) << std::endl;
    for( std::size_t i1 = 0; i1 < a_suite.size( ); ++i1 ) {
        GIDI::Form *form = a_suite.get<GIDI::Form>( i1 );
        std::cout << "    " << checkXLink( *form, a_protare, a_reaction ) << std::endl;

        if( a_isProduct ) {
            GIDI::Product *product = static_cast<GIDI::Product *>( form );

                printSuite( product->multiplicity( ), a_protare, a_reaction );
                printSuite( product->distribution( ), a_protare, a_reaction );
                printSuite( product->averageEnergy( ), a_protare, a_reaction );
                printSuite( product->averageMomentum( ), a_protare, a_reaction );

                GIDI::OutputChannel *outputChannel = product->outputChannel( );
                if( outputChannel != nullptr ) printOutputChannel( outputChannel, a_protare, a_reaction );
        }
    }
}
/*
=========================================================
*/
void printOutputChannel( GIDI::OutputChannel *a_outputChannel, GUPI::Ancestry *a_protare, GUPI::Ancestry *a_reaction ) {

    std::cout << "    " << checkXLink( *a_outputChannel, a_protare, a_reaction ) << std::endl;
    printSuite( a_outputChannel->Q( ), a_protare, a_reaction );
    printSuite( a_outputChannel->products( ), a_protare, a_reaction, true );
}
/*
=========================================================
*/
std::string checkXLink( GUPI::Ancestry &a_ancestry, GUPI::Ancestry *a_protare, GUPI::Ancestry *a_reaction ) {

    std::string xlink = a_ancestry.toXLink( );

    checkXLink2( a_ancestry, xlink, a_protare );
    checkXLink2( a_ancestry, xlink, a_reaction );

    return( xlink );
}
/*
=========================================================
*/
void checkXLink2( GUPI::Ancestry &a_ancestry, std::string &a_xlink, GUPI::Ancestry *a_from ) {

    GUPI::Ancestry *link = a_from->findInAncestry( a_xlink );

    if( link == nullptr ) {
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

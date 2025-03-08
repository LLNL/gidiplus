/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <set>

#include <statusMessageReporting.h>

#include <GIDI_testUtilities.hpp>

static char const *description = "Displays the list of incomplete particles for the requested protare and each of its reactions.";
static std::string indent( "      " );

void main2( int argc, char **argv );
void printProtareSingle( GIDI::Transporting::Settings const &a_settings, GIDI::ProtareSingle const *protare );
void printSet( std::string const &a_indent, std::string const &a_label, std::set<std::string> const &a_incompleteParticles );
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

    std::vector<std::string> libraries;
    GIDI::Transporting::DelayedNeutrons delayedNeutrons( GIDI::Transporting::DelayedNeutrons::on );

    argvOptions argv_options( "incompleteParticles", description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );

    parseTestOptions.m_askGNDS_File = true;

    parseTestOptions.parse( );

    GIDI::Construction::PhotoMode photo_mode = parseTestOptions.photonMode( GIDI::Construction::PhotoMode::nuclearAndAtomic );
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
    PoPI::Database pops;
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../../../../TestData/PoPs/pops.xml", "../../../Test/all.map", construction, PoPI::IDs::neutron, "O16" );
    GIDI::Transporting::Settings settings( PoPI::IDs::neutron, delayedNeutrons );

    for( std::size_t protareIndex = 0; protareIndex < protare->numberOfProtares( ); ++protareIndex ) {
        GIDI::ProtareSingle const *protareSingle = protare->protare( protareIndex );

        printProtareSingle( settings, protareSingle );
    }

    delete protare;

    exit( EXIT_SUCCESS );
}
/*
=========================================================
*/
void printProtareSingle( GIDI::Transporting::Settings const &a_settings, GIDI::ProtareSingle const *protare ) {

    std::set<std::string> incompleteParticles;

    std::cout << indent << "For protare:" << std::endl;
    protare->incompleteParticles( a_settings, incompleteParticles );
    printSet( indent, protare->realFileName( ), incompleteParticles );

    std::cout << indent << "By reaction:" << std::endl;
    std::string indent2 = indent + "  ";
    GIDI::Suite const &reactions = protare->reactions( );
    for( std::size_t reactionIndex = 0; reactionIndex < reactions.size( ); ++reactionIndex ) {
        GIDI::Reaction const &reaction = *reactions.get<GIDI::Reaction>( reactionIndex );

        incompleteParticles.clear( );
        reaction.incompleteParticles( a_settings, incompleteParticles );
        printSet( indent2, reaction.label( ), incompleteParticles );
    }
}
/*
=========================================================
*/
void printSet( std::string const &a_indent, std::string const &a_label, std::set<std::string> const &a_incompleteParticles ) {

    std::string indent2 = a_indent + "  ";

    std::cout << indent2 << a_label << std::endl;

    for( auto iter = a_incompleteParticles.begin( ); iter != a_incompleteParticles.end( ); ++iter ) {
        std::cout << indent2 << "  " << *iter << std::endl;
    }
}

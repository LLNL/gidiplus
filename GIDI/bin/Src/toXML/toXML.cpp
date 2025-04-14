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

#include "GIDI.hpp"

void printMsg( char const *message );
void printUsage( );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    if (argc < 2) {
        std::cerr << "Usage: toXML <filename>" << std::endl;
	exit( EXIT_FAILURE );
    }

    PoPI::Database pops( argv[1] );
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, GIDI::Construction::PhotoMode::nuclearAndAtomic );
    std::vector<std::string> libraries;
    GIDI::ProtareSingle *protare;
    GIDI::ParticleSubstitution particleSubstitution;

    try {
        protare = new GIDI::ProtareSingle( construction, argv[2], GIDI::FileType::XML, pops, particleSubstitution, libraries, GIDI_MapInteractionNuclearChars ); }
    catch (char const *str) {
        std::cerr << str << std::endl;
        exit( EXIT_FAILURE );
    }

    protare->saveAs( "test.xml" );

    delete protare;

    exit( EXIT_SUCCESS );
}
/*
=========================================================
*/
void printUsage( ) {

    printf( "\nUsage:\n" );
    printf( "directory -h -p PoPsFile [-p PoPsFile] mapFile [projectileID] [targetID] [evaluation]\n" );
    printf( "\n" );
    printf( "   mapFile         path to the map file.\n" );
    printf( "   projectileID    projectile's ID (e.g, 'n', 'photo').\n" );
    printf( "   targetID        target's ID (e.g, 'O16', 'Pu238').\n" );
    printf( "   evaluation      evaluation (e.g, 'endl2009.3', 'ENDF/B-VII.1').\n" );
    printf( "   -h              This help message.\n" );
    printf( "   -p              The next argument is the name of a PoPs file to use.\n" );
    printf( "   -w              Indicates to match any value.\n" );
    printf( "\n" );
    printf( "Displays a list of all protares matching projectileID, targetID and evaluation:\n" );
    printf( "\n" );
    printf( "Notes:\n" );
    printf( "The -w argument is only needed when an argument is to be wild and a following argument is specified. For example,\n" );
    printf( "    directory -p pops.xml all.map -w U236\n" );
    printf( "    directory -p pops.xml all.map n -w -w\n" );
    printf( "    directory -p pops.xml all.map n\n" );
    printf( "The last two commands are the same since the '-w' argument is only needed if a later argument is entered.\n" );
    exit( EXIT_SUCCESS );
}

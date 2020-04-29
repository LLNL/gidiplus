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

    std::vector<std::string> inputs;
    std::vector<std::string> popsFiles;
    std::vector<std::string> *option2;

    int mode = 0;
    for( int i1 = 1; i1 < argc; ++i1 ) {
        char *arg = argv[i1];
        if( mode == 0 ) {
            if( arg[0] == '-' ) {
                if( strlen( arg ) != 2 ) {
                    printMsg( smr_allocateFormatMessage( "Invalid option = %s", arg ) ); }
                else {
                    if(      arg[1] == 'h' ) {
                        printUsage( ); }
                    else if( arg[1] == 'p' ) {
                        mode = 2;
                        option2 = &popsFiles; }
                    else if( arg[1] == 'w' ) {
                        inputs.push_back( std::string( "" ) ); }
                    else {
                        printMsg( smr_allocateFormatMessage( "Invalid option = %s", arg ) );
                    }
                } }
            else {
                inputs.push_back( std::string( arg ) );
            } }
        else {
            if( mode == 2 ) {
                option2->push_back( std::string( arg ) ); }
            else {
                printMsg( smr_allocateFormatMessage( "Oops, this should not have happened: mode = %d", mode ) );
            }
            mode = 0;
        }
    }

    if( inputs.size( ) < 2 ) printMsg( "Need map file, optionally a projectile id and target id, and at least one '-p' option." );
    if( popsFiles.size( ) == 0 ) printMsg( "Need at least one '-p' option." );

    PoPI::Database pops;
    try {
        for( std::size_t i1 = 0; i1 < popsFiles.size( ); ++i1 ) pops.addFile( popsFiles[i1], 1 ); }
    catch (char const *str) {
        printMsg( str );
    }

    std::string mapFilename( inputs[0] );
    std::string projectileID( "" );
    std::string targetID( "" );
    std::string evaluation( "" );
    if( inputs.size( ) > 1 ) {
        if( inputs[1] != "" ) projectileID = inputs[1];
        if( inputs.size( ) > 2 ) {
            if( inputs[2] != "" ) targetID = inputs[2];
            if( inputs.size( ) > 3 ) evaluation = inputs[3];
        }
    }

    GIDI::Map::Map map( mapFilename, pops );

    std::vector<GIDI::Map::ProtareBase const *> protareEntries = map.directory( projectileID, targetID, evaluation );

    for( std::size_t i1 = 0; i1 < protareEntries.size( ); ++i1 ) {
        GIDI::Map::ProtareBase const *protareEntry = protareEntries[i1];

        printf( "%-8s %-8s %s\n", protareEntry->projectileID( ).c_str( ), protareEntry->targetID( ).c_str( ), protareEntry->evaluation( ).c_str( ) );
    }

    exit( EXIT_SUCCESS );
}
/*
=========================================================
*/
void printMsg( char const *message ) {

    fprintf( stderr, "%s", message );
    fprintf( stderr, "\n" );
    exit( EXIT_FAILURE );
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

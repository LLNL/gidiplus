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

static char const *description = "Reads in a pop file and checks ";

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

    int counter;
    int counterInc = 400;
    PoPI::Database pops;
    pugi::xml_document doc;

    LUPI::ArgumentParser argumentParser( __FILE__, description );

    LUPI::Positional *fileNames = argumentParser.add<LUPI::Positional>( "popsPath", "List of pops files to read.", 1, -1 );

    argumentParser.parse( argc, argv );

    for( auto fileName = fileNames->values( ).begin( ); fileName != fileNames->values( ).end( ); ++fileName ) {
        pops.addFile( *fileName, false );
    }

    PoPI::ParticleList const &list = pops.list( );
    counter = 0;
    counterInc = 1;
    for( auto particleIter = list.begin( ); particleIter != list.end( ); ++particleIter, ++counter ) {
        PoPI::Base const *base = *particleIter;
        if( counter % counterInc == 0 ) {
            if( base->isParticle( ) ) {
                PoPI::IDBase const *idbase = static_cast<PoPI::IDBase const *>( base );

                std::string id = idbase->ID( );
                int intid = idbase->intid( );

                std::string finalId = pops.final( id );
                PoPI::Particle const &particle = pops.particle( finalId );
                int finalIntid = particle.intid( );

                std::cout << id << " " << intid;
                if( ( id != finalId ) || ( intid != finalIntid ) ) std::cout << " " << finalId << " " << finalIntid; 
                std::cout  << std::endl;
            }
            counter = 0;
        }
    }
}

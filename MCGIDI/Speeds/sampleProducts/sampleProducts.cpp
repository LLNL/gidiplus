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
#include <iomanip>

#include "MCGIDI.hpp"

#include "utilities4Speed.hpp"

/*
=========================================================
*/
int main( int argc, char **argv ) {

    std::string mapFilename( "../../../GIDI/Test/all3T.map" );
    PoPI::Database pops( "../../../GIDI/Test/pops.xml" );
    GIDI::Map map( mapFilename, pops );
    std::string protareFilename( map.protareFilename( "n", "O16" ) );
    GIDI::ProtareSingle *protare;
    std::vector<std::string> libraries;
    GIDI::Transporting::Particles particles;
    std::set<int> reactionsToExclude;
    clock_t time0, time1;
    long numberOfSamples = 100 * 1000, sampleTemperatures = 0, sampleEnergies = 0;
    char label[1024];
    void *rngState = nullptr;

    std::cout << __FILE__;
    for( int i1 = 1; i1 < argc; i1++ ) std::cout << " " << argv[i1];
    std::cout << std::endl;

    time0 = clock( );
    time1 = time0;
    try {
        GIDI::Construction::Settings construction( GIDI::Construction::e_all, GIDI::Construction::e_nuclearAndAtomic );
        protare = new GIDI::ProtareSingle( construction, protareFilename, GIDI::XML, pops, libraries ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }
    printTime( "    load GIDI: ", time1 );

    std::string label1( "MonteCarlo" );
    MCGIDI::Settings::MC MC( pops, "n", &protare->styles( ), label1, true, 20 );

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    MCGIDI::DomainHash domainHash( 4000, 1e-8, 100.0 );
    MCGIDI::ProtareSingle *MCProtare;
    try {
        MCProtare = new MCGIDI::ProtareSingle( *protare, pops, MC, particles, domainHash, temperatures, reactionsToExclude ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }
    printTime( "    load MCGIDI: ", time1 );

    MCGIDI::Sampling::Input input( true, MCGIDI::Sampling::Upscatter::None );
    MCGIDI::Sampling::StdVectorProductHandler products;
    std::size_t numberOfReactions = (int) MCProtare->numberOfReactions( );

    std::cout << "Reactions:" << std::endl;
    for( int reactionIndex = 0; reactionIndex < numberOfReactions; ++reactionIndex ) {
        MCGIDI::Reaction const *reaction = MCProtare->reaction( reactionIndex );

        std::cout << "    " << reactionIndex << "  " << reaction->label( ).c_str( ) << std::endl;
    }
    std::cout << std::endl;

    for( double temperature = 1e-8; temperature < 2e-3; temperature *= 10.1, ++sampleTemperatures ) {
        clock_t time1_1 = clock( );
        clock_t time2_1 = time1_1;
        clock_t time3_1 = time1_1;

        for( int reactionIndex = 0; reactionIndex < numberOfReactions; ++reactionIndex ) {
            MCGIDI::Reaction const *reaction = MCProtare->reaction( reactionIndex );
            double threshold = MCProtare->threshold( reactionIndex );

            long energyIndex = 0;
            if( threshold < 1e-12 ) threshold = 1e-12;
            for( double energy = threshold; energy < 100.1; energy *= 10, ++energyIndex ) {
                for( long sampleIndex = 0; sampleIndex <= numberOfSamples; ++sampleIndex ) {
                    products.clear( );
                    reaction->sampleProducts( MCProtare, energy, input, myRNG, rngState, products );
                }
                printTime_energy( "                energy: ", energyIndex, energy, time3_1 );
            }
            sampleEnergies = energyIndex;
            std::cout << std::endl;
            sprintf( label, "            reaction %d: ", reactionIndex );
            printTime( label, time2_1 );
        }
        printTime_double( "        temperature: ", temperature, time1_1 );
        std::cout << std::endl;
    }

    long sampled = sampleTemperatures * sampleEnergies * numberOfReactions * numberOfSamples;

    printSpeeds( __FILE__, time1, sampled );
    printTime( "total sample time : ", time1 );
    printTime( "total time : ", time0 );

    std::cout << "total sampled = " << sampled << "  (" << std::setprecision( 3 ) << (double) sampled << ")" << std::endl;

    delete protare;

    delete MCProtare;
}


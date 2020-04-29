/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <iomanip>

#include "utilities4Speed.hpp"

/*
=========================================================
*/
double myRNG( void *state ) {

    return( drand48( ) );
}
/*
=========================================================
*/
void printTime_reaction( char const *label, long index, clock_t &time0 ) {

    char label2[1024] = "";

    if( index == 0 ) strcpy( label2, label );
    printTime( label2, time0, false );
}
/*
=========================================================
*/
void printTime_energy( char const *label, long index, double value, clock_t &time0 ) {

    char label2[1024] = "";
    char label3[1024];

    if( index == 0 ) strcpy( label2, label );
    sprintf( label3, "%s %.2e: ", label2, value );
    printTime( label3, time0, false );
}
/*
=========================================================
*/
void printTime_double( char const *label, double value, clock_t &time0 ) {

    char label2[1024];

    sprintf( label2, "%s %.6e: ", label, value );
    printTime( label2, time0 );
}
/*
=========================================================
*/
void printTime( char const *label, clock_t &time0, bool printEndOfLine ) {

    clock_t time1 = clock( );
    double cpuTime = ( time1 - time0 ) / ( (double) CLOCKS_PER_SEC );

    std::cout << label << std::setprecision( 3 ) << cpuTime;
    if( printEndOfLine ) {
        std::cout << std::endl; }
    else {
        std::cout << ", ";
    }
    time0 = time1;
}
/*
=========================================================
*/
void printSpeeds( char const *label, clock_t &time0, long samples ) {

    double cpuTime = ( clock( ) - time0 ) / ( (double) CLOCKS_PER_SEC );
    double speed = samples / cpuTime;
    double _samples = samples;

    std::cerr << std::showpoint;
    std::cerr << std::left << std::setw( 30 ) << label << ": time = " << std::setprecision( 3 ) << cpuTime << " s: samples = " << 
            std::right << std::setw( 10 ) << samples << " (" << std::setprecision( 3 ) << _samples << "): samples/s = " << speed << std::endl;
}

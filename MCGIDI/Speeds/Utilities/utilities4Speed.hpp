/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <time.h>

double myRNG( void *state );
void printTime_reaction( char const *label, long index, clock_t &time0 );
void printTime_energy( char const *label, long index, double value, clock_t &time0 );
void printTime_double( char const *label, double value, clock_t &time0 );
void printTime( char const *label, clock_t &time0, bool printEndOfLine = true );
void printSpeeds( char const *label, clock_t &time0, long sampled );

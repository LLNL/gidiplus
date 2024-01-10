/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <time.h>

#include <LUPI.hpp>

double myRNG( void *state );
void printTime_reaction( std::string const &label, long index, clock_t &time0 );
void printTime_energy( std::string const &label, long index, double value, clock_t &time0 );
void printTime_double( std::string const &label, double value, clock_t &time0 );
void printTime( std::string const &label, clock_t &time0, bool printEndOfLine = true );
void printSpeeds( std::string const &label, clock_t &time0, long sampled );

/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

/*
    Code to read in a file containing a fluxes node.
*/

#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <set>

#include "GIDI.hpp"

void compare( std::string const &a_msg, GIDI::Group &a_group, GIDI::Settings::MultiGroup &a_boundaries );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    std::cerr << "    " << __FILE__;
    for( int i1 = 1; i1 < argc; i1++ ) std::cerr << " " << argv[i1];
    std::cerr << std::endl;

    GIDI::Fluxes fluxes( "../fluxes.xml" );

    std::cout << fluxes.size( ) << std::endl;

    for( std::size_t i1 = 0; i1 < fluxes.size( ); ++i1 ) {
        GIDI::Function3dForm const *function3d = fluxes.get<GIDI::Function3dForm>( i1 );

        std::cout << function3d->label( ) << std::endl;
    }
}

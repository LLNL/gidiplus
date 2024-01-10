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

#include "GIDI_testUtilities.hpp"

static char const *description = "Reads a flux file and prints each flux's info.";

void compare( std::string const &a_msg, GIDI::Group &a_group, GIDI::Transporting::MultiGroup &a_boundaries );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    printCodeArguments( __FILE__, argc, argv );

    argvOptions argv_options( "fluxes", description );

    argv_options.add( argvOption( "--fluxFile", true, "Specifies the flux file to use." ) );
    argv_options.parseArgv( argc, argv );


    GIDI::Fluxes fluxes( argv_options.find( "--fluxFile" )->zeroOrOneOption( argv, "../fluxes.xml" ) );

    std::cout << fluxes.size( ) << std::endl;

    for( std::size_t i1 = 0; i1 < fluxes.size( ); ++i1 ) {
        GIDI::Functions::Function3dForm const *function3d = fluxes.get<GIDI::Functions::Function3dForm>( i1 );

        std::cout << function3d->label( ) << std::endl;

        std::vector<GIDI::Transporting::Flux> setting_fluxes = GIDI::settingsFluxesFromFunction3d( *function3d );
        for( std::size_t i2 = 0; i2 < setting_fluxes.size( ); ++i2 ) setting_fluxes[i2].print( "" );
    }

}

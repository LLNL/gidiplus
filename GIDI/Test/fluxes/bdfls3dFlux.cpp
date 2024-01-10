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

static char const *description = "Test the conversion of a bdfls flux file to a 3d-function object (i.e., f(T,E,mu)).";

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
        std::cout << str << std::endl;
        exit( EXIT_FAILURE ); }
    catch (std::string &str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    exit( EXIT_SUCCESS );
}
/*
=========================================================
*/
void main2( int argc, char **argv ) {

    argvOptions argv_options( "bdfls3dFlux", description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );

    parseTestOptions.m_askPoPs = false;
    parseTestOptions.m_askMap = false;
    parseTestOptions.m_askPid = false;
    parseTestOptions.m_askTid = false;
    parseTestOptions.m_askPhotoAtomic = false;
    parseTestOptions.m_askPhotoNuclear = false;

    argv_options.add( argvOption( "--bdfls", true, "Specified the bdfls file to use." ) );

    parseTestOptions.parse( );

    std::string bdflsFileName = argv_options.find( "--bdfls" )->zeroOrOneOption( argv, "../bdfls" );
    GIDI::Transporting::Fluxes_from_bdfls fluxes_from_bdfls( bdflsFileName, 0 );

    std::vector<int> fids = fluxes_from_bdfls.FIDs( );
    for( auto fid = fids.begin( ); fid != fids.end( ); ++fid ) {
        std::cout << "        fid = " << *fid << std::endl;
        GIDI::Functions::Function3dForm *flux = fluxes_from_bdfls.get3dViaFID( *fid );
        std::cout << "            " << flux->label( ) << std::endl;

        GUPI::WriteInfo writeInfo;
        flux->toXMLList_func( writeInfo, "", false, false );
        writeInfo.print( );

        delete flux;
    }
}

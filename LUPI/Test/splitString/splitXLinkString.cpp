/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>

#include <LUPI.hpp>

static char const *description = "This code checks the splitXLinkString function.";

void main2( int argc, char **argv );
void splitAndPrint( std::string a_string );
void output( std::vector<std::string> a_elements );

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

    LUPI::ArgumentParser argumentParser( __FILE__, description );

    argumentParser.parse( argc, argv );

    splitAndPrint( "/" );
    splitAndPrint( "./" );
    splitAndPrint( "../" );
    splitAndPrint( "/this///is/an/example//of/a/full/path" );
    splitAndPrint( "/this//../..///is/an/example//of/a/full/path/" );
    splitAndPrint( "/reactionSuite/styles/griddedCrossSection" );
    splitAndPrint( "./reactionSuite/styles/griddedCrossSection" );
    splitAndPrint( "../../styles/griddedCrossSection" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + C{2p1/2}']/outputChannel/products/product[@label='e-']" );
    splitAndPrint( "/reactionSuite/styles/griddedCrossSection[@label='MonteCarlo_000']/grid[@index='1']" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='C + photon [coherent]']/doubleDifferentialCrossSection/coherentPhotonScattering[@label='eval']" );
    splitAndPrint( "/reactionSuite/styles/griddedCrossSection[@label='MonteCarlo_000']/grid[@index='1']" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='C + photon [incoherent]']/doubleDifferentialCrossSection/incoherentPhotonScattering[@label='eval']" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='C + photon [coherent]']/crossSection" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='C + photon [incoherent]']/crossSection" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + e- + e-_anti + C [pair production: electron field]']/crossSection" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + e-_anti + C [pair production: nuclear field]']/crossSection" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + C{1s1/2}']/crossSection" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + C{2s1/2}']/crossSection" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + C{2p1/2}']/crossSection" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + C{2p3/2}']/crossSection" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + e- + e-_anti + C [pair production: electron field]']/crossSection" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + e-_anti + C [pair production: nuclear field]']/crossSection" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='C + photon [coherent]']/outputChannel/products/product[@label='C']" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='C + photon [incoherent]']/outputChannel/products/product[@label='C']" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + e- + e-_anti + C [pair production: electron field]']/outputChannel/products/product[@label='C']" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + e-_anti + C [pair production: nuclear field]']/outputChannel/products/product[@label='C']" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + C{1s1/2}']/outputChannel/products/product[@label='C{1s1/2}']" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + C{2s1/2}']/outputChannel/products/product[@label='C{2s1/2}']" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + C{2p1/2}']/outputChannel/products/product[@label='C{2p1/2}']" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + C{2p3/2}']/outputChannel/products/product[@label='C{2p3/2}']" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + e- + e-_anti + C [pair production: electron field]']/outputChannel/products/product[@label='e-']" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + e- + e-_anti + C [pair production: electron field]']/outputChannel/products/product[@label='e-__a']" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + e- + e-_anti + C [pair production: electron field]']/outputChannel/products/product[@label='e-_anti']" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + e-_anti + C [pair production: nuclear field]']/outputChannel/products/product[@label='e-']" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + e-_anti + C [pair production: nuclear field]']/outputChannel/products/product[@label='e-_anti']" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + C{1s1/2}']/outputChannel/products/product[@label='e-']" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + C{2s1/2}']/outputChannel/products/product[@label='e-']" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + C{2p1/2}']/outputChannel/products/product[@label='e-']" );
    splitAndPrint( "/reactionSuite/reactions/reaction[@label='e- + C{2p3/2}']/outputChannel/products/product[@label='e-']" );
    splitAndPrint( "/reactionSuite/styles/griddedCrossSection[@label='MonteCarlo_000']/grid[@index='1']" );
    splitAndPrint( "/reactionSuite/applicationData/institution[@label='LLNL::photoAtomicIncoherentDoppler']/reactions/reaction[@label='e- + C{2p1/2} + photon']/doubleDifferentialCrossSection/incoherentBoundToFreePhotonScattering[@label='eval']" );
    splitAndPrint( "/reactionSuite/applicationData/institution[@label='LLNL::photoAtomicIncoherentDoppler']/reactions/reaction[@label='e- + C{2p1/2} + photon']/doubleDifferentialCrossSection/incoherentBoundToFreePhotonScattering[@label='MonteCarlo_cdf']" );

    exit( 0 );
}

/*
=========================================================
*/
void splitAndPrint( std::string a_string ) {

    std::vector<std::string> elements = LUPI::Misc::splitXLinkString( a_string );

    std::cout << std::endl;
    std::cout << a_string << std::endl;
    output( elements );
}

/*
=========================================================
*/
void output( std::vector<std::string> a_elements ) {

    for( auto iter = a_elements.begin( ); iter != a_elements.end( ); ++iter ) {
        std::cout << "    '" << *iter << "'" << std::endl;
    }
}

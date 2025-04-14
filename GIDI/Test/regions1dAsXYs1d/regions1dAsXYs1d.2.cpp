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

#include "GIDI_testUtilities.hpp"

static char const *description = "This test runs various check of converting a Regions1d instance to an XYs1d instance using the Regions1d::asXYs1d method.";
static char const *format = " %20.12e %20.12e\n";

void main2( int argc, char **argv );
void toXYs1d( GIDI::Functions::Regions1d const &regions1d, double epsLower, double epsUpper );
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
    LUPI::Positional *inputPathArgument = argumentParser.add<LUPI::Positional>( "inputPath", "The path to the regions1d file to read.", 1, 1 );
    LUPI::Positional *epsilonArgument = argumentParser.add<LUPI::Positional>( "epsilon", "The epsilon to cycle through.", 1, 1 );

    argumentParser.parse( argc, argv );

    std::string path( inputPathArgument->value( ) );
    double epsilon( std::stod( epsilonArgument->value( ) ) );


    HAPI::PugiXMLFile *doc = new HAPI::PugiXMLFile( path.c_str( ), __FILE__ );
    if( doc == nullptr ) throw std::runtime_error( "Failure to read " + path + "." );

    HAPI::Node node = doc->first_child( );

    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, GIDI::Construction::PhotoMode::nuclearAndAtomic );
    GIDI::SetupInfo setupInfo( nullptr );
    GIDI::Functions::Regions1d regions1d( construction, node, setupInfo, nullptr );

    delete doc;

    std::cout << "# regions = " << path << std::endl;

    toXYs1d( regions1d, 0.0, epsilon );
    toXYs1d( regions1d, epsilon, 0.0 );
    toXYs1d( regions1d, epsilon, epsilon );
}

/*
=========================================================
*/
void toXYs1d( GIDI::Functions::Regions1d const &regions1d, double epsLower, double epsUpper ) {

    GIDI::Functions::XYs1d *xys1d = regions1d.asXYs1d( true, 1e-4, epsLower, epsUpper );

    std::cout << std::endl << std::endl;
    std::cout << "# length = " << xys1d->size( ) << "; " << epsLower << " " << epsUpper << std::endl;
    xys1d->print( format );
    delete xys1d;
}

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
void toXYs1d( GIDI::Functions::Function1dForm const *function1d, double accuracy );
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
    LUPI::Positional *accuracyArgument = argumentParser.add<LUPI::Positional>( "accuracy", "The accuracy of the conversion to an XYs1d instance.", 1, 1 );

    argumentParser.parse( argc, argv );

    std::string path( inputPathArgument->value( ) );
    double accuracy( std::stod( accuracyArgument->value( ) ) );

    HAPI::PugiXMLFile *doc = new HAPI::PugiXMLFile( path.c_str( ), __FILE__ );
    if( doc == nullptr ) throw std::runtime_error( "Failure to read " + path + "." );

    HAPI::Node node = doc->first_child( );

    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, GIDI::Construction::PhotoMode::nuclearAndAtomic );
    GIDI::SetupInfo setupInfo( nullptr );
    GIDI::Functions::Function1dForm *function1d;

    if(      node.name( ) == GIDI_constant1dChars ) {
        function1d = new GIDI::Functions::Constant1d( construction, node, setupInfo, nullptr ); }
    else if( node.name( ) == GIDI_polynomial1dChars ) {
        function1d = new GIDI::Functions::Polynomial1d( construction, node, setupInfo, nullptr ); }
    else if( node.name( ) == GIDI_LegendreChars ) {
        function1d = new GIDI::Functions::Legendre1d( construction, node, setupInfo, nullptr ); }
    else {
        throw "Unsupported data type = " + node.name( );
    }

    delete doc;

    toXYs1d( function1d, accuracy );
    delete function1d;
}

/*
=========================================================
*/
void toXYs1d( GIDI::Functions::Function1dForm const *function1d, double accuracy ) {

    GIDI::Functions::XYs1d *xys1d = function1d->asXYs1d( true, accuracy, 0.0, 0.0 );

    std::cout << std::endl << std::endl;
    std::cout << "# length = " << xys1d->size( ) << "; " << accuracy << std::endl;
    xys1d->print( format );
    delete xys1d;
}

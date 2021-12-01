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

#include <LUPI.hpp>

static char const *description = "The alias checker.";

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

    LUPI::OptionTrue *optionTrue = argumentParser.add<LUPI::OptionTrue>( "--true", "First option with aliases." );
    argumentParser.addAlias( "--true", "-t" );
    argumentParser.addAlias( "--true", "-y" );
    argumentParser.addAlias( "--true", "-yes" );
    argumentParser.addAlias( optionTrue, "--yeap" );

    LUPI::OptionFalse *optionFalse = argumentParser.add<LUPI::OptionFalse>( "--false", "Testing the 'OptionFalse' class." );

    LUPI::OptionCounter *optionCounter = argumentParser.add<LUPI::OptionCounter>( "--veryVeryLongCounterNameVeryVeryLong", 
            "A very long counter name, very long." );
    argumentParser.addAlias( optionCounter, "-v" );

    LUPI::OptionStore *optionStore = argumentParser.add<LUPI::OptionStore>( "--store", "The path to a pops file to load." );

    LUPI::OptionAppend *optionAppend = argumentParser.add<LUPI::OptionAppend>( "--pops", "Second option with aliases.", 0, -1 );
    argumentParser.addAlias( "--pops", "-p" );
    argumentParser.addAlias( "--pops", "--pop" );

    argumentParser.parse( argc, argv );

    std::cout << "    " << optionTrue->name( )  << " option: number entered " << optionTrue->numberEntered( ) << ",   value = " << optionTrue->value( ) << std::endl;
    std::cout << "    " << optionFalse->name( ) << " option: number entered " << optionFalse->numberEntered( ) << ",   value = " << optionFalse->value( ) << std::endl;
    std::cout << "    " << optionCounter->name( ) << " option: number entered " << optionCounter->numberEntered( ) << ",   counts = " << optionCounter->counts( ) << std::endl;
    std::cout << "    " << optionStore->name( ) << " option: number entered " << optionStore->numberEntered( ) << ",   value = '" << optionStore->value( ) << "'" << std::endl;

    std::cout << "    " << optionAppend->name( ) << " option: number entered " << optionAppend->numberEntered( ) << std::endl;
    std::cout << "         ";
    for( int index = 0; index < optionAppend->numberEntered( ); ++index ) {
        std::cout << " '" << optionAppend->value( index ) << "'";
    }
    std::cout << std::endl;
}

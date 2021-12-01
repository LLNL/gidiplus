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

    std::string fileName( __FILE__ );

    LUPI::ArgumentParser argumentParser( __FILE__, "Parser for a simple positional argument." );

    argumentParser.add<LUPI::Positional>( "P1", "A simple positional argument." );
    argumentParser.add<LUPI::Positional>( "P2", "A second simple positional argument. Must have 2 to 3 values.", 2, 3 );

    argumentParser.add<LUPI::OptionStore>( "--store", "OptionStore for missing value check." );
    argumentParser.addAlias( "--store", "-s" );

    argumentParser.add<LUPI::OptionAppend>( "--as", "OptionAppend for missing value check. Allows any number.", 0, -1 );

    argumentParser.add<LUPI::OptionAppend>( "-a", "OptionAppend for missing value check. Must have only 1 to 2 values.", 1, 2 );

    argumentParser.parse( argc, argv );

    argumentParser.printStatus( "        " );
}

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

static char const *description = "Test that when more than one positional argument has variable needed arguments, a throw is raised.";

void main2( int argc, char **argv );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    try {
        main2( argc, argv ); }
    catch (std::exception &exception) {
        std::cerr << __FILE__ << ": " << exception.what( ) << std::endl;
        exit( EXIT_FAILURE ); }
    catch (char const *str) {
        std::cerr << __FILE__ << ": " << str << std::endl;
        exit( EXIT_FAILURE ); }
    catch (std::string &str) {
        std::cerr << __FILE__ << ": " << str << std::endl;
        exit( EXIT_FAILURE );
    }

    exit( EXIT_SUCCESS );
}
/*
=========================================================
*/
void main2( LUPI_maybeUnused int argc, LUPI_maybeUnused char **argv ) {

    LUPI::ArgumentParser argumentParser( __FILE__, description );

    argumentParser.add<LUPI::Positional>( "PA1", "First positional argument with 1 required value" );
    argumentParser.add<LUPI::Positional>( "PA2", "Second positional argument with 1 required value" );
    argumentParser.add<LUPI::Positional>( "PA3", "Third positional argument with 3 required value", 3, 5 );

    try {
        argumentParser.add<LUPI::Positional>( "PA4", "Fourth positional argument with 1 required value" ); }
    catch (std::exception &exception) {
        std::cerr << exception.what( ) << std::endl;
    }
}

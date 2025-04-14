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
void main2( LUPI_maybeUnused int argc, LUPI_maybeUnused char **argv ) {

    std::string fileName( __FILE__ );

    try {
        LUPI::OptionAppend( "-a", "A simple append.", -10, 20 );
        throw "FAILURE (ERROR 1000) in " + fileName + " check for negative minimum required test."; }
    catch (std::exception &exception) {
        std::cerr << exception.what( ) << std::endl;
    }

    try {
        LUPI::OptionAppend( "-a", "A simple append.", 3, 2 );
        throw "FAILURE (ERROR 1010) in " + fileName + " check for minimum required greater than maximum required test."; }
    catch (std::exception &exception) {
        std::cerr << exception.what( ) << std::endl;
    }

    try {
        LUPI::Positional( "-a", "A simple positional argument." );
        throw "FAILURE (ERROR 1020) in " + fileName + " check for positional argument starting with '-'."; }
    catch (std::exception &exception) {
        std::cerr << exception.what( ) << std::endl;
    }

    try {
        LUPI::OptionAppend( "a", "A simple append." );
        throw "FAILURE (ERROR 1110) in " + fileName + " check for option not starting with '-'."; }
    catch (std::exception &exception) {
        std::cerr << exception.what( ) << std::endl;
    }

    LUPI::ArgumentParser argumentParser( __FILE__, "Parser for a simple positional argument." );
    LUPI::Positional *positional = argumentParser.add<LUPI::Positional>( "a", "A simple positional argument." );
    try {
        argumentParser.addAlias( positional, "oops" );
        throw "FAILURE (ERROR 1100) in " + fileName + " check for adding alias to positional."; }
    catch (std::exception &exception) {
        std::cerr << exception.what( ) << std::endl;
    }
}

/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <iostream>

#include "PoPI.hpp"

void main2( int argc, char **argv );
void printParts( std::string const &a_id, std::string &a_base, std::string &a_anti, std::string &a_qualifier );

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

    std::cerr << "    " << LUPI::FileInfo::basenameWithoutExtension( __FILE__ ) << std::endl;

    std::string base;
    std::string anti;
    std::string qualifier;

    printParts( "Co", base, anti, qualifier );
    printParts( "Co58", base, anti, qualifier );
    printParts( "Co58_e1", base, anti, qualifier );
    printParts( "Co58_m1", base, anti, qualifier );
    printParts( "gamma", base, anti, qualifier );
    printParts( "photon", base, anti, qualifier );
    printParts( "p", base, anti, qualifier );
    printParts( "p_anti", base, anti, qualifier );
    printParts( "e-", base, anti, qualifier );
    printParts( "e-_anti", base, anti, qualifier );
    printParts( "e-_anti_anti", base, anti, qualifier );
    printParts( "H2{1s1/2}", base, anti, qualifier );
    printParts( "H2{1s1/2}", base, anti, qualifier );
    printParts( "H21s1/2}", base, anti, qualifier );
    printParts( "H2{1s1/2", base, anti, qualifier );
    printParts( "H2_anti{1s1/2}", base, anti, qualifier );
    printParts( "H2{1s1/2}_anti", base, anti, qualifier );
}

/*
=========================================================
*/
void printParts( std::string const &a_id, std::string &a_base, std::string &a_anti, std::string &a_qualifier ) {

    std::cout << std::endl;
    std::cout << a_id << ": ";
    try {
        a_base = PoPI::baseAntiQualifierFromID( a_id, a_anti );
        std::cout << "'" << a_base << "' '" << a_anti << "'" << std::endl;
        a_base = PoPI::baseAntiQualifierFromID( a_id, a_anti, &a_qualifier );
        for( std::size_t index = 0; index < a_id.size( ) + 2; ++index ) std::cout << " ";
        std::cout << "'" << a_base << "' '" << a_anti << "' '" << a_qualifier << "'" << std::endl; }
    catch (...) {
        std::cout << "FAILED: for id = " << a_id << std::endl;
    }
}

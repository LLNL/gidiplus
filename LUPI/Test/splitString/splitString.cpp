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
#include <fstream>

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

    std::cerr << "    " << LUPI::FileInfo::basenameWithoutExtension( __FILE__ );
    for( int iargv = 1; iargv < argc; ++iargv ) std::cerr << " " << argv[iargv];
    std::cerr << std::endl;

    std::ifstream inputFile;
    inputFile.open( "test.ris" );

    bool stripWhiteSpaces = argc > 1;
    std::string line;
    while( getline( inputFile, line ) ) {
        std::cout << "<" << line << ">" << std::endl;
        std::vector<std::string> elements = LUPI::Misc::splitString( line, ':', stripWhiteSpaces );
        for( auto iter = elements.begin( ); iter != elements.end( ); ++iter ) std::cout << "    <" << *iter << ">" << std::endl;
    }

    inputFile.close( );
}

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

#include "GIDI.hpp"

void splitAndPrint( std::string const &string, char delimiter );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    splitAndPrint( "/", '/' );
    splitAndPrint( "./", '/' );
    splitAndPrint( "../", '/' );

    splitAndPrint( "/This/Is/An/Example/Of/A/Full/Path", '/' );
    splitAndPrint( "/this///is/an/example//of/a/full/path", '/' );

    splitAndPrint( "./This/Is/An/Example/Of/A/relative/Path", '/' );
    splitAndPrint( ".///this/is/an/example///of/a//relative/path", '/' );

    splitAndPrint( "../This/Is/An/Example/Of/A/Path/Relative/To/Parent/Directory", '/' );
    splitAndPrint( "..///this/is/an//example/of/a/path///relative/to/parent/directory", '/' );


    splitAndPrint( "../../../This/Is/An/Example/Of/A/Path/Relative/To/Parent/Directory/Three/Up", '/' );
}
/*
=========================================================
*/
void splitAndPrint( std::string const &string, char delimiter ) {

    std::cout << "String to split is '" << string << "'" << std::endl;
    std::cout << "Delimiter is '" << delimiter << "'" << std::endl;

    std::vector<std::string> segmets = GIDI::splitString( string, delimiter );
    std::cout << "    ";
    for( std::size_t i1 = 0; i1 < segmets.size( ); ++i1 ) {
        if( i1 > 0 ) std::cout << ", ";
        std::cout << "'" << segmets[i1] << "'";
    }
    std::cout << std::endl;
}

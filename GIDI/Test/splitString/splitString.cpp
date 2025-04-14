/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

/*
This needs to be moved to PoPI/Test when tests are implemented in PoPI.
*/

#include <stdlib.h>
#include <iostream>

#include "PoPI.hpp"

void splitAndPrint( std::string const &string );
/*
=========================================================
*/
int main( LUPI_maybeUnused int argc, LUPI_maybeUnused char **argv ) {

    splitAndPrint( "/" );
    splitAndPrint( "./" );
    splitAndPrint( "../" );

    splitAndPrint( "/This/Is/An/Example/Of/A/Full/Path" );
    splitAndPrint( "/this///is/an/example//of/a/full/path" );

    splitAndPrint( "./This/Is/An/Example/Of/A/relative/Path" );
    splitAndPrint( ".///this/is/an/example///of/a//relative/path" );

    splitAndPrint( "../This/Is/An/Example/Of/A/Path/Relative/To/Parent/Directory" );
    splitAndPrint( "..///this/is/an//example/of/a/path///relative/to/parent/directory" );


    splitAndPrint( "../../../This/Is/An/Example/Of/A/Path/Relative/To/Parent/Directory/Three/Up" );
}
/*
=========================================================
*/
void splitAndPrint( std::string const &string ) {

    std::cout << "XLink to split is '" << string << "'" << std::endl;

    std::vector<std::string> segmets = LUPI::Misc::splitXLinkString( string );
    std::cout << "    ";
    for( std::size_t i1 = 0; i1 < segmets.size( ); ++i1 ) {
        if( i1 > 0 ) std::cout << ", ";
        std::cout << "'" << segmets[i1] << "'";
    }
    std::cout << std::endl;
}

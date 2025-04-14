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

/*
=========================================================
*/
int main( LUPI_maybeUnused int argc, LUPI_maybeUnused char **argv ) {

    std::cerr << "    " << LUPI::FileInfo::basenameWithoutExtension( __FILE__ ) << std::endl;

    std::string fileName( "../../../TestData/PoPs/pops.xml" );

    try {
        PoPI::Database database( fileName );
        database.saveAs( "Outputs/pops3.xml.out" );
        database.print( false ); }
    catch (char const *str) {
        std::cout << str << std::endl;
    }
}

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
int main( int argc, char **argv ) {

    std::cerr << "    " << LUPI::FileInfo::basenameWithoutExtension( __FILE__ ) << std::endl;

    std::string fileName( "../pops.xml" );
    std::string aliasFileName( "../LLNL_alias.xml" );

    try {
        PoPI::Database database( fileName );
        database.addFile( aliasFileName, false );
        database.print( false );

        }
    catch (char const *str) {
        std::cout << str << std::endl;
    }
}

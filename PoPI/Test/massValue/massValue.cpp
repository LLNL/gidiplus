/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <iostream>
#include <iomanip>

#include "PoPI.hpp"

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

    std::cerr << "    " << LUPI::FileInfo::basenameWithoutExtension( __FILE__ ) << std::endl;

    PoPI::Database pops( "../../../TestData/PoPs/pops.xml" );
    pops.addFile( "../../../TestData/PoPs/metastables_alias.xml", false );

    for( auto aliasIter = pops.aliases( ).begin( ); aliasIter != pops.aliases( ).end( ); ++aliasIter ) {
        std::string pid = pops.final( (*aliasIter)->ID( ) );
        std::cout << (*aliasIter)->ID( ) << " " << pops.massValue( (*aliasIter)->ID( ), "amu" ) 
                << " " << pid << " " << pops.massValue( pid, "amu" ) << std::endl;
    }
}

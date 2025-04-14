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
void printSymbol( PoPI::Database const & a_pops, std::string const &a_id );

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
    pops.addFile( "../../../TestData/PoPs/LLNL_alias.xml", false );

    printSymbol( pops, "O" );
    printSymbol( pops, "O16" );
    printSymbol( pops, "8016" );
    printSymbol( pops, "12000" );
    printSymbol( pops, "Co" );
    printSymbol( pops, "Co58" );
    printSymbol( pops, "Co58_e1" );
    printSymbol( pops, "Co58_m1" );
    printSymbol( pops, "Co58_m2" );
    printSymbol( pops, "gamma" );
    printSymbol( pops, "photon" );
    printSymbol( pops, "e-" );
}

/*
=========================================================
*/
void printSymbol( PoPI::Database const & a_pops, std::string const &a_id ) {

    std::string isotope = a_pops.isotopeSymbol( a_id );
    std::string chemicalElement = a_pops.chemicalElementSymbol( a_id );

    std::cout << a_id << " <"  << isotope << "> <" << chemicalElement << ">" << std::endl;
}

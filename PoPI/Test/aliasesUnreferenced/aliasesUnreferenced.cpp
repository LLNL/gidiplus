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
void printUnreferencedAliasData( PoPI::Database &database, char const *filePath );
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

    PoPI::Database database;

    printUnreferencedAliasData( database, "../metastables_alias.xml" );
    printUnreferencedAliasData( database, "../../../TestData/PoPs/pops.xml" );
}

/*
=========================================================
*/
void printUnreferencedAliasData( PoPI::Database &database, char const *filePath ) {

    if( filePath[0] != 0 ) database.addFile( filePath, false );

    std::cout << "      Number of unreferenced aliases = " << database.numberOfUnresolvedAliases( ) << std::endl;

    auto ids = database.unresolvedAliasIds( );
    for( auto iter = ids.begin( ); iter != ids.end( ); ++iter ) {
        std::cout << "        " << *iter << std::endl;
    }
}

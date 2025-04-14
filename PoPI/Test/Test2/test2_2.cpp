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

void printInfo( PoPI::Database const &a_pops, char const *a_id );
/*
=========================================================
*/
int main( LUPI_maybeUnused int argc, LUPI_maybeUnused char **argv ) {

    std::cerr << "    " << LUPI::FileInfo::basenameWithoutExtension( __FILE__ ) << std::endl;

    std::string fileName( "../../../TestData/PoPs/pops.xml" );
    std::string aliasFileName( "../../../TestData/PoPs/LLNL_alias.xml" );

    try {
        PoPI::Database pops( fileName );
        pops.addFile( aliasFileName, false );
        printInfo( pops, "U235" );
        printInfo( pops, "92235" );
        }
    catch (char const *str) {
        std::cout << str << std::endl;
    }
}
/*
=========================================================
*/
void printInfo( PoPI::Database const &a_pops, char const *a_id ) {

    PoPI::IDBase const &particle = a_pops.get<PoPI::IDBase>( a_id );
    std::cout << "particle info: id = " << a_id << " index = " << particle.index( ) << " isAlias = " << particle.isAlias( ) << " final = " << a_pops.final( particle.index( ) );
    if( particle.Class( ) == PoPI::Particle_class::alias ) {
        PoPI::Alias const &alias = dynamic_cast<PoPI::Alias const &>( particle );
        std::cout << " parent index = " << alias.pidIndex( );
    }
    std::cout << std::endl;
}

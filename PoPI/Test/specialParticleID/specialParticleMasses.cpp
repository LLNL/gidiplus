/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdio.h>
#include <iostream>

#include "PoPI.hpp"

void main2( int argc, char **argv );
void massesFor( PoPI::Database const &a_database, std::string const &a_id );
void massFor( PoPI::Database const &a_database, std::string const &a_id, PoPI::SpecialParticleID_mode a_mode );
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

    std::string fileName( "../../../TestData/PoPs/pops.xml" );
    std::string aliasFileName( "../../../TestData/PoPs/LLNL_alias.xml" );
    std::string metaStableFileName( "../../../TestData/PoPs/metastables_alias.xml" );

    PoPI::Database database( fileName );
    database.addFile( aliasFileName, false );
    database.addFile( metaStableFileName, false );

    massesFor( database, "p" );
    massesFor( database, "d" );
    massesFor( database, "t" );
    massesFor( database, "h" );
    massesFor( database, "a" );
}
/*
=========================================================
*/
void massesFor( PoPI::Database const &a_database, std::string const &a_id ) {

    printf( "  %s\n", a_id.c_str( ) );
    massFor( a_database, a_id, PoPI::SpecialParticleID_mode::familiar );
    massFor( a_database, a_id, PoPI::SpecialParticleID_mode::nucleus );
    massFor( a_database, a_id, PoPI::SpecialParticleID_mode::nuclide );
}
/*
=========================================================
*/
void massFor( PoPI::Database const &a_database, std::string const &a_id, PoPI::SpecialParticleID_mode a_mode ) {

    std::string id = PoPI::specialParticleID( a_mode, a_id.c_str( ) );
    double mass = a_database.massValue( id, "amu" );

    printf( "    %-8s %20.12e\n", id.c_str( ), mass );
}

/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <iostream>

#include <PoPI.hpp>

static bool terse( true );

void main2( int argc, char **argv );
void checkId( std::string const &a_id );

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

    terse = argc == 1;
    std::cerr << "    " << LUPI::FileInfo::basenameWithoutExtension( __FILE__ ) << std::endl;

    checkId( "n" );
    checkId( "n_anti" );
    checkId( "p" );
    checkId( "p_anti" );
    checkId( "d" );
    checkId( "h2" );
    checkId( "H2" );
    checkId( "H" );
    checkId( "Am242" );
    checkId( "Am242_e2" );
    checkId( "Am242_m2" );
    checkId( "Am" );
    checkId( "am242" );
    checkId( "am242_e2" );
    checkId( "am242_m2" );
    checkId( "am" );
    checkId( PoPI::IDs::FissionProductENDL99120 );
}

/*
=========================================================
*/
void checkId( std::string const &a_id ) {

    PoPI::ParseIdInfo parseIdInfo( a_id );
    parseIdInfo.print( terse, "" );
}

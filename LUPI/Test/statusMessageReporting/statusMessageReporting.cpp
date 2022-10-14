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

#include <LUPI.hpp>

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
void main2( int argc, char **argv ) {

    std::string fileName = LUPI::FileInfo::basenameWithoutExtension( __FILE__ );

    std::cerr << "    " << fileName;
    for( int iargv = 1; iargv < argc; ++iargv ) std::cerr << " " << argv[iargv];
    std::cerr << std::endl;

    LUPI::StatusMessageReporting smr;
    int numberOfReports = argc - 1;

    statusMessageReporting *c_smr = smr.smr( );

    for( int index = 0; index < numberOfReports; ++index ) {
        smr_setReportError2( c_smr, 0, 10 * index, "Index = %d", index );
    }

    for( int index = 0; index < numberOfReports + 1; ++index ) {
        std::string message( smr.constructFullMessage( "Test", index, false ) );
        std::cout << message << std::endl << std::endl;
    }

    smr.clear( );
}

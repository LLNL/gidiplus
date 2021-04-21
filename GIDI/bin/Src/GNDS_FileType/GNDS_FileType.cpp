/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <iostream>

#include <GIDI_testUtilities.hpp>

static char const descriptor[] = "Loops through all arguments and prints the type of GNDS each is.";

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

    argvOptions argvOptions1( __FILE__, descriptor );
    argvOptions1.parseArgv( argc, argv );

    for( int iarg = 1; iarg < argc; ++iarg ) {
        GIDI::GNDS_FileTypeInfo GNDS_fileTypeInfo;
        std::cout << argv[iarg];

        GIDI::GNDS_FileType GNDS_FileType = GNDS_fileType( argv[iarg], GNDS_fileTypeInfo );
        switch( GNDS_FileType ) {
        case GIDI::GNDS_FileType::uninitialized :
            std::cout << " error read file." << std::endl;
            break;
        case GIDI::GNDS_FileType::unknown :
            std::cout << " is not a GNDS type file." << std::endl;
            break;
        case GIDI::GNDS_FileType::pops :
            std::cout << " is a pops file." << std::endl;
            break;
        case GIDI::GNDS_FileType::protare :
            std::cout << " is a protare file :: ";
            std::cout << "projectile '" << GNDS_fileTypeInfo.projectileID( ) << "', target '" << GNDS_fileTypeInfo.targetID( ) 
                      << "', evaluation '" << GNDS_fileTypeInfo.evaluation( ) << "', interaction '" << GNDS_fileTypeInfo.interaction( ) << "'." << std::endl;
            break;
        case GIDI::GNDS_FileType::covarianceSuite :
            std::cout << " is a covarianceSuite file." << std::endl;
            break;
        case GIDI::GNDS_FileType::map :
            std::cout << " is a map file." << std::endl;
            break;
        }
    }
}

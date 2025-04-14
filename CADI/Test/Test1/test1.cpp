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

#include <PoPI.hpp>
#include <CADI.hpp>

// static char const *description = "Reads an isotopic abundance file.";

void main2( LUPI_maybeUnused int argc, LUPI_maybeUnused char **argv );
void chemicalElementPrint( PoPI::Database const &pops, CADI::Map::IAM &a_IAM, std::string const &a_symbol, std::string const &a_evaluation );
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

    PoPI::Database pops( "../../../TestData/PoPs/pops.xml" );
    std::string fileName( "../../../TestData/isotopicAbundances/COGZA2a.xml" );
    CADI::IsotopicAbundancesByChemicalElement isotopicAbundancesByChemicalElement( fileName );

    GUPI::WriteInfo writeInfo;
    isotopicAbundancesByChemicalElement.toXMLList( writeInfo );
//    writeInfo.print( );

    CADI::ChemicalElement const *chemicalElement = isotopicAbundancesByChemicalElement.chemicalElements( ).get<CADI::ChemicalElement>( "O" );
    GUPI::Suite const &isotopes = chemicalElement->isotopes( );
    for( auto iter = isotopes.begin( ); iter != isotopes.end( ); ++iter ) {
        CADI::Isotope *isotope = static_cast<CADI::Isotope *>( *iter );
        std::cout << isotope->id( ) << " " << isotope->atomFraction( ) << " " << isotope->uncertainty( ) << std::endl;
    }

    CADI::Map::IAM iam( "../../../TestData/isotopicAbundances/all.iam" );

    chemicalElementPrint( pops, iam, "O", "" );
    chemicalElementPrint( pops, iam, "O", "COGZA2a" );
    chemicalElementPrint( pops, iam, "O", "FUDGE" );
    chemicalElementPrint( pops, iam, "O16", "COGZA2a" );
    chemicalElementPrint( pops, iam, "O0", "COGZA2a" );
    chemicalElementPrint( pops, iam, "O", "A" );
}
/*
=========================================================
*/
void chemicalElementPrint( PoPI::Database const &pops, CADI::Map::IAM &a_IAM, std::string const &a_symbol, std::string const &a_evaluation ) {

    std::cout << std::endl;

    std::string symbol = pops.chemicalElementSymbol( a_symbol );
    std::cout << a_symbol << " " << a_evaluation << ": " << symbol << std::endl;
    CADI::ChemicalElement const *chemicalElement = a_IAM.find( symbol, a_evaluation );
    if( chemicalElement == nullptr ) return;

    GUPI::Suite const &isotopes = chemicalElement->isotopes( );
    for( auto iter = isotopes.begin( ); iter != isotopes.end( ); ++iter ) {
        CADI::Isotope *isotope = static_cast<CADI::Isotope *>( *iter );
        std::cout << "    " << isotope->id( ) << " " << isotope->atomFraction( ) << " " << isotope->uncertainty( ) << std::endl;
    }
}

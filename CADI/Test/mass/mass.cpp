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

#include <CADI.hpp>

static char const *description = "Reads an iam file and serializes it. Then, either prints the original or serialized part of it.";

void main2( int argc, char **argv );
void chemicalElementPrint( CADI::Map::IAM &a_IAM, std::string const &a_symbol, std::string const &a_evaluation );
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

    std::string symbol = "O";
    std::string massUnit = "amu";
    LUPI::ArgumentParser argumentParser( __FILE__, description );

    argumentParser.parse( argc, argv );

    PoPI::Database pops( "../../../TestData/PoPs/pops.xml" );

    CADI::Map::IAM iam( "../../../TestData/isotopicAbundances/all.iam" );

    CADI::IsotopicAbundancesByChemicalElement const &isotopicAbundancesByChemicalElement = *iam.findEvaluation( "FUDGE" );
    CADI::ChemicalElement const &chemicalElement = *isotopicAbundancesByChemicalElement[symbol];

    std::cout << "Elemental mass for " + symbol + " is " << chemicalElement.mass( pops, massUnit ) << std::endl;

    std::cout << "Isotope atom fractions and masses:" << std::endl;
    for( auto isotopeIter = chemicalElement.isotopes( ).begin( ); isotopeIter != chemicalElement.isotopes( ).end( ); ++isotopeIter ) {
        CADI::Isotope *isotope = static_cast<CADI::Isotope *>( *isotopeIter );
        PoPI::Particle const &particle = pops.particle( isotope->id( ) );
        std::cout << isotope->id( ) << " " << isotope->atomFraction( ) << " " << particle.massValue( massUnit ) << std::endl;
    }
}

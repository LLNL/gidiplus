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

    LUPI::ArgumentParser argumentParser( __FILE__, description );

    LUPI::ArgumentBase *printSerialized = argumentParser.add( LUPI::ArgumentType::True, "--serialized", 
            "If present, serialized IAM is printed; otherwise, read one is printed." );
    argumentParser.addAlias( printSerialized, "-s" );
    argumentParser.parse( argc, argv );

    CADI::Map::IAM iam("../../../TestData/isotopicAbundances/all.iam");

    LUPI::DataBuffer dataBuffer;

    iam.serialize( dataBuffer, LUPI::DataBuffer::Mode::Count );
    dataBuffer.allocateBuffers( );
    dataBuffer.zeroIndexes( );
    iam.serialize( dataBuffer, LUPI::DataBuffer::Mode::Pack );
    dataBuffer.zeroIndexes( );

    CADI::Map::IAM iam2;
    iam2.serialize( dataBuffer, LUPI::DataBuffer::Mode::Unpack );

    CADI::IsotopicAbundancesByChemicalElement const *isotopicAbundancesByChemicalElement = nullptr;
    if( printSerialized->counts( ) == 0 ) {
        isotopicAbundancesByChemicalElement = iam.findEvaluation( "COGZA2a" ); }
    else {
        isotopicAbundancesByChemicalElement = iam2.findEvaluation( "COGZA2a" );
    }

    GUPI::WriteInfo writeInfo( "  ", 1, " " );
    isotopicAbundancesByChemicalElement->toXMLList( writeInfo );
    writeInfo.print( );
}

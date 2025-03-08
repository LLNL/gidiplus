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

    PoPI::Database database( "../../../TestData/PoPs/pops.xml" );
    database.addFile( "../metastables_alias.xml", false );

    for( std::size_t index = 0; index < database.size( ); ++index ) {
        PoPI::Base const *base = &database.get<PoPI::Base>( index );

        if( base->isParticle( ) ) {
            PoPI::Particle const *particle = static_cast<PoPI::Particle const *>( base );
            std::vector<std::string> aliasReferences = database.aliasReferences( particle->ID( ) );
            if( aliasReferences.size( ) > 0 ) {
                std::cout << std::setw( 12 ) << std::left << particle->ID( );
                for( auto referenceIter = aliasReferences.begin( ); referenceIter != aliasReferences.end( ); ++referenceIter ) {
                    std::cout << " " << (*referenceIter);
                }
                std::cout << std::endl;
            }
        }
    }
}

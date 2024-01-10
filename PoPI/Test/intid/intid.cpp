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

    std::cerr << "    " << LUPI::FileInfo::basenameWithoutExtension( __FILE__ ) << std::endl;

    PoPI::Database pops( "../../../TestData/PoPs/pops.xml" );

    PoPI::ParticleList const &particleList = pops.particleList( );
    for( std::size_t i1 = 0; i1 < particleList.size( ); ++i1 ) {
        PoPI::Base const &base = pops.get<PoPI::Base>( i1 );

        if( base.isParticle( ) ) {
            PoPI::IDBase const &particle = static_cast<PoPI::IDBase const &>( base );

            int intid = pops.intid( particle.ID( ) );
            PoPI::ParseIntidInfo intidInfo( intid );
            std::cout << particle.ID( )
                        << " " << intid
                        << " " << intidInfo.id( )
                        << std::endl;
        }
    }
}

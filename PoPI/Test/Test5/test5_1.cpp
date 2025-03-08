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

/*
=========================================================
*/
int main( LUPI_maybeUnused int argc, LUPI_maybeUnused char **argv ) {

    std::cerr << "    " << LUPI::FileInfo::basenameWithoutExtension( __FILE__ ) << std::endl;

    std::string fileName( "../../../TestData/PoPs/pops.xml" );
    std::string aliasFileName( "../../../TestData/PoPs/LLNL_alias.xml" );
    std::string metaStableFileName( "../../../TestData/PoPs/metastables_alias.xml" );

    PoPI::Database *pops;

    try {
        pops = new PoPI::Database( fileName );
        pops->addFile( aliasFileName, false );
        pops->addFile( metaStableFileName, false ); }
    catch (char const *str) {
        std::cout << str << std::endl;
    }

    PoPI::ParticleList const &particleList = pops->particleList( );

    printf( "ID                       :  isParticle\n" );
    printf( "                                isAlias\n" );
    printf( "                                    isMetaStableAlias\n" );
    printf( "                                        isGaugeBoson\n" );
    printf( "                                            isLepton\n" );
    printf( "                                                isBaryon\n" );
    printf( "                                                    isUnorthodox\n" );
    printf( "                                                        isNucleus\n" );
    printf( "                                                            isNuclide\n" );
    printf( "                                                                isIsotope\n" );
    printf( "                                                                    isChemicalElement\n" );
    printf( "                                                                         Z\n" );
    printf( "                                                                             A\n" );
    printf( "                                                                                   ZA\n" );

    for( std::size_t i1 = 0; i1 < particleList.size( ); ++i1 ) {
        PoPI::Base const &base = pops->get<PoPI::Base>( i1 );

        printf( "%-25s:  %d   %d   %d   %d   %d   %d   %d   %d   %d   %d   %d  %3d %3d %6d\n", 
                base.ID( ).c_str( ),
                base.isParticle( ),     base.isAlias( ),    base.isMetaStableAlias( ),  base.isGaugeBoson( ),   base.isLepton( ),           base.isBaryon( ), 
                base.isUnorthodox( ),   base.isNucleus( ),  base.isNuclide( ),          base.isIsotope( ),      base.isChemicalElement( ),
                PoPI::particleZ( base ), PoPI::particleA( base ), PoPI::particleZA( base ) );

        if( base.isAlias( ) ) {
            
        }
    }

    printf( "\n\n" );
    printf( "ID        :    Z   A     ZA\n" );

    PoPI::Lepton const &electron = pops->get<PoPI::Lepton>( "e-" );
    printf( "%-10s:  %3d %3d %6d\n", electron.ID( ).c_str( ), PoPI::particleZ( electron ), PoPI::particleA( electron ), PoPI::particleZA( electron ) );

    PoPI::GaugeBoson const &photon = pops->get<PoPI::GaugeBoson>( "photon" );
    printf( "%-10s:  %3d %3d %6d\n", photon.ID( ).c_str( ), PoPI::particleZ( photon ), PoPI::particleA( photon ), PoPI::particleZA( photon ) );

    PoPI::ChemicalElement const &Fe = pops->get<PoPI::ChemicalElement>( "Fe" );
    printf( "%-10s:  %3d %3d %6d\n", Fe.ID( ).c_str( ), PoPI::particleZ( Fe ), PoPI::particleA( Fe ), PoPI::particleZA( Fe ) );

    PoPI::Nuclide const &O16 = pops->get<PoPI::Nuclide>( "O16" );
    printf( "%-10s:  %3d %3d %6d\n", O16.ID( ).c_str( ), PoPI::particleZ( O16 ), PoPI::particleA( O16 ), PoPI::particleZA( O16 ) );

    PoPI::Nucleus const &pu239 = pops->get<PoPI::Nucleus>( "pu239" );
    printf( "%-10s:  %3d %3d %6d\n", pu239.ID( ).c_str( ), PoPI::particleZ( pu239 ), PoPI::particleA( pu239 ), PoPI::particleZA( pu239 ) );

    PoPI::Baryon const *baryon = &pops->get<PoPI::Baryon>( "n" );
    printf( "%-10s:  %3d %3d %6d\n", baryon->ID( ).c_str( ), PoPI::particleZ( *baryon ), PoPI::particleA( *baryon ), PoPI::particleZA( *baryon ) );
    printf( "%-10s:  %3d %3d %6d\n", baryon->ID( ).c_str( ), PoPI::particleZ( *baryon, true ), PoPI::particleA( *baryon, true ), PoPI::particleZA( *baryon, true ) );

    baryon = &pops->get<PoPI::Baryon>( "p" );
    printf( "%-10s:  %3d %3d %6d\n", baryon->ID( ).c_str( ), PoPI::particleZ( *baryon ), PoPI::particleA( *baryon ), PoPI::particleZA( *baryon ) );
    printf( "%-10s:  %3d %3d %6d\n", baryon->ID( ).c_str( ), PoPI::particleZ( *baryon, true ), PoPI::particleA( *baryon, true ), PoPI::particleZA( *baryon, true ) );
}

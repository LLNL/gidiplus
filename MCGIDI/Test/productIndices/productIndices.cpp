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
#include <iomanip>
#include <set>

#include "MCGIDI.hpp"

#include "MCGIDI_testUtilities.hpp"

static char const *description = "Loops over temperature and energy, printing the total cross section. If projectile is a photon, see options *--pa* and *--pn*.";

void main2( int argc, char **argv );
void read_MCGIDI_protare( PoPI::Database const &a_pops, GIDI::Protare *a_protare, GIDI::Styles::TemperatureInfos const & a_temperatures, MCGIDI::Transporting::MC &a_settings, 
                int productBinary );
void printProductList( PoPI::Database const &a_pops, MCGIDI::Protare * a_protare, bool a_transportablesOnly );
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

    PoPI::Database pops;
    pops.addFile( "../../../GIDI/Test/pops.xml", false );
    GIDI::Construction::PhotoMode photo_mode = GIDI::Construction::PhotoMode::nuclearOnly;

    std::cerr << "    " << __FILE__;
    for( int i1 = 1; i1 < argc; i1++ ) std::cerr << " " << argv[i1];
    std::cerr << std::endl;

    argvOptions2 argv_options( "crossSections", description );

    argv_options.add( argvOption2( "--map", true, "The map file to use." ) );
    argv_options.add( argvOption2( "--pid", true, "The PoPs id of the projectile." ) );
    argv_options.add( argvOption2( "--tid", true, "The PoPs id of the target." ) );
    argv_options.add( argvOption2( "--pa", false, "Include photo-atomic protare if relevant. If present, disables photo-nuclear unless *--pn* also present." ) );
    argv_options.add( argvOption2( "--pn", false, "Include photo-nuclear protare if relevant. This is the default unless *--pa* present." ) );

    argv_options.parseArgv( argc, argv );

    std::string mapFilename = argv_options.find( "--map" )->zeroOrOneOption( argv, "../../../GIDI/Test/all3T.map" );
    std::string projectileID = argv_options.find( "--pid" )->zeroOrOneOption( argv, PoPI::IDs::neutron );
    std::string targetID = argv_options.find( "--tid" )->zeroOrOneOption( argv, "O16" );

    if( argv_options.find( "--pa" )->present( ) ) {
        photo_mode = GIDI::Construction::PhotoMode::atomicOnly;
        if( argv_options.find( "--pn" )->present( ) ) photo_mode = GIDI::Construction::PhotoMode::nuclearAndAtomic;
    }

    GIDI::Map::Map map( mapFilename, pops );
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
    GIDI::Protare *protare = map.protare( construction, pops, projectileID, targetID );

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    std::string label( temperatures[0].heatedCrossSection( ) );
    MCGIDI::Transporting::MC settings( pops, projectileID, &protare->styles( ), label, GIDI::Transporting::DelayedNeutrons::on, 20.0 );

    for( int i1 = 0; i1 < 2 << 7; ++i1 ) read_MCGIDI_protare( pops, protare, temperatures, settings, i1 );

    delete protare;
}
/*
=========================================================
*/
void read_MCGIDI_protare( PoPI::Database const &a_pops, GIDI::Protare *a_protare, GIDI::Styles::TemperatureInfos const & a_temperatures, MCGIDI::Transporting::MC &a_settings, 
                int productBinary ) {

    std::set<int> reactionsToExclude;
    std::map<std::string, std::string> particlesAndGIDs;

    particlesAndGIDs[PoPI::IDs::neutron] = "LLNL_gid_4";
    particlesAndGIDs["H1"] = "LLNL_gid_71";
    particlesAndGIDs["H2"] = "LLNL_gid_71";
    particlesAndGIDs["H3"] = "LLNL_gid_71";
    particlesAndGIDs["He3"] = "LLNL_gid_71";
    particlesAndGIDs["He4"] = "LLNL_gid_71";
    particlesAndGIDs[PoPI::IDs::photon] = "LLNL_gid_70";

    GIDI::Transporting::Particles particles;

    GIDI::Transporting::Groups_from_bdfls groups_from_bdfls( "../../../GIDI/Test/bdfls" );
    GIDI::Transporting::Fluxes_from_bdfls fluxes_from_bdfls( "../../../GIDI/Test/bdfls", 0.0 );

    int productDigit = 1;
    std::cout << std::endl;
    for( std::map<std::string, std::string>::iterator iter = particlesAndGIDs.begin( ); iter != particlesAndGIDs.end( ); ++iter, productDigit <<= 1 ) {
        if( productDigit & productBinary ) {
            GIDI::Transporting::MultiGroup multi_group = groups_from_bdfls.viaLabel( iter->second );
            GIDI::Transporting::Particle particle( iter->first, multi_group );

            particle.appendFlux( fluxes_from_bdfls.getViaFID( 1 ) );
            particles.add( particle );

            std::cout << iter->first << " ";
        }
    }
    std::cout << std::endl;

    MCGIDI::DomainHash domainHash( 4000, 1e-8, 10 );
    MCGIDI::Protare *MCProtare;
    MCProtare = MCGIDI::protareFromGIDIProtare( *a_protare, a_pops, a_settings, particles, domainHash, a_temperatures, reactionsToExclude );

    MCProtare->setUserParticleIndex( a_pops[PoPI::IDs::neutron], 0 );
    MCProtare->setUserParticleIndex( a_pops["H2"], 10 );
    MCProtare->setUserParticleIndex( a_pops[PoPI::IDs::photon], 11 );

    printProductList( a_pops, MCProtare, true );
    printProductList( a_pops, MCProtare, false );

    delete MCProtare;
}
/*
=========================================================
*/
void printProductList( PoPI::Database const &a_pops, MCGIDI::Protare * a_protare, bool a_transportablesOnly ) {

    MCGIDI::Vector<int> indices = a_protare->productIndices( a_transportablesOnly );
    MCGIDI::Vector<int> userIndices = a_protare->userProductIndices( a_transportablesOnly );

    std::cout << "    transportables only = " << a_transportablesOnly << std::endl;
    for( auto i1 = 0; i1 < indices.size( ); ++i1 ) {
        PoPI::Base const &particle = a_pops.get<PoPI::Base>( indices[i1] );

        std::cout << "        " << indices[i1] << " " << particle.ID( ) << " " << userIndices[i1] << std::endl;
    }
}

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

static char const *description = "Loops over temperature and energy, printing the total cross section. If projectile is a photon, see options *-a* and *-n*.";

void printVector( std::string &prefix, GIDI::Vector &vector );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    PoPI::Database pops( "../../../GIDI/Test/pops.xml" );
    GIDI::Protare *protare;
    GIDI::Transporting::Particles particles;
    std::set<int> reactionsToExclude;
    GIDI::Construction::PhotoMode photo_mode = GIDI::Construction::PhotoMode::nuclearOnly;

    std::cerr << "    " << __FILE__;
    for( int i1 = 1; i1 < argc; i1++ ) std::cerr << " " << argv[i1];
    std::cerr << std::endl;

    argvOptions2 argv_options( "crossSections", description );

    argv_options.add( argvOption2( "--map", true, "The map file to use." ) );
    argv_options.add( argvOption2( "--pid", true, "The PoPs id of the projectile." ) );
    argv_options.add( argvOption2( "--tid", true, "The PoPs id of the target." ) );
    argv_options.add( argvOption2( "-a", false, "Include photo-atomic protare if relevant. If present, disables photo-nuclear unless *-n* present." ) );
    argv_options.add( argvOption2( "-n", false, "Include photo-nuclear protare if relevant. This is the default unless *-a* present." ) );

    argv_options.parseArgv( argc, argv );

    std::string mapFilename = argv_options.find( "--map" )->zeroOrOneOption( argv, "../../../GIDI/Test/all3T.map" );
    std::string projectileID = argv_options.find( "--pid" )->zeroOrOneOption( argv, PoPI::IDs::neutron );
    std::string targetID = argv_options.find( "--tid" )->zeroOrOneOption( argv, "O16" );

    if( argv_options.find( "-a" )->present( ) ) {
        photo_mode = GIDI::Construction::PhotoMode::atomicOnly;
        if( argv_options.find( "-n" )->present( ) ) photo_mode = GIDI::Construction::PhotoMode::nuclearAndAtomic;
    }

    GIDI::Map::Map map( mapFilename, pops );

    try {
        GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
        protare = map.protare( construction, pops, projectileID, targetID ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    for( GIDI::Styles::TemperatureInfos::const_iterator iter = temperatures.begin( ); iter != temperatures.end( ); ++iter ) {
        std::cout << "label = " << iter->heatedMultiGroup( ) << "  temperature = " << iter->temperature( ).value( ) << std::endl;
    }

    std::string label( temperatures[0].heatedMultiGroup( ) );

    GIDI::Transporting::Groups_from_bdfls groups_from_bdfls( "../../../GIDI/Test/bdfls" );
    GIDI::Transporting::Fluxes_from_bdfls fluxes_from_bdfls( "../../../GIDI/Test/bdfls", 0.0 );

    std::string gid( "LLNL_gid_4" );
    if( projectileID == PoPI::IDs::photon ) gid = "LLNL_gid_70";
    GIDI::Transporting::MultiGroup multi_group = groups_from_bdfls.viaLabel( gid );
    GIDI::Transporting::Particle projectile( projectileID, multi_group );
    projectile.appendFlux( fluxes_from_bdfls.getViaFID( 1 ) );
    particles.add( projectile );
    particles.process( *protare, label );

    MCGIDI::Transporting::MC MC( pops, projectileID, &protare->styles( ), label, GIDI::Transporting::DelayedNeutrons::on, 20.0 );
    MC.crossSectionLookupMode( MCGIDI::Transporting::LookupMode::Data1d::multiGroup );
    MCGIDI::DomainHash domainHash( 4000, 1e-8, 10 );
    MCGIDI::Protare *MCProtare;
    try {
        MCProtare = MCGIDI::protareFromGIDIProtare( *protare, pops, MC, particles, domainHash, temperatures, reactionsToExclude ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    std::cout << std::endl;
    MCGIDI::Vector<double> MC_temperatures = MCProtare->temperatures( 0 );
    std::cout << "MCGIDI temperatures:" << std::endl;
    for( auto iter = MC_temperatures.begin( ); iter != MC_temperatures.end( ); ++iter ) {
        std::cout << "    " << *iter << std::endl;
    }
    std::cout << std::endl;

    MCGIDI::Vector<MCGIDI::Protare *> protares( 1 );
    protares[0] = MCProtare;
    MCGIDI::URR_protareInfos URR_protare_infos( protares );

    MCGIDI::MultiGroupHash multiGroupHash( *protare, temperatures[0] );

    for( MCGIDI_VectorSizeType i1 = 0; i1 < (MCGIDI_VectorSizeType) MCProtare->numberOfReactions( ); ++i1 ) {
        MCGIDI::Reaction const &reaction = *MCProtare->reaction( i1 );

        std::cout << std::setw( 40 ) << reaction.label( ).c_str( ) << "  threshold = " << std::setw( 12 ) << reaction.crossSectionThreshold( ) <<
                "  threshold = " << std::setw( 12 ) << reaction.crossSectionThreshold( ) << std::endl;
    }
    std::cout << std::endl;

    std::cout << "List of reactions" << std::endl;
    for( MCGIDI_VectorSizeType i1 = 0; i1 < (MCGIDI_VectorSizeType) MCProtare->numberOfReactions( ); ++i1 ) {
        MCGIDI::Reaction const &reaction = *MCProtare->reaction( i1 );

        std::cout << "    reaction: " << reaction.label( ).c_str( ) << std::endl;
    }
    std::cout << std::endl;

    for( double temperature = 1e-8; temperature < 2e-3; temperature *= 10.1 ) {
        std::cout << "temperature = " << temperature << std::endl;
        for( double energy = 1e-12; energy < 20; energy *= 1.2 ) {
            int hashIndex = multiGroupHash.index( energy );

            double crossSection = MCProtare->crossSection( URR_protare_infos, hashIndex, temperature, energy );
            std::cout << "    energy = " << std::setw( 16 ) << energy << " index = " << std::setw( 6 ) << hashIndex << "   crossSection = " << crossSection << std::endl;
        }
    }

    delete protare;

    delete MCProtare;
}

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
#include <math.h>

#include "GIDI_testUtilities.hpp"

static char const *description = "The program prints the multi-group deposition energy for a protare and its reactions.";

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

    LUPI::StatusMessageReporting smr1;
    argvOptions argv_options( "depositionEnergy", description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );

    parseTestOptions.m_askGNDS_File = true;
    argv_options.add( argvOption( "-z", false, "If present, setZeroDepositionIfAllProductsTracked is called with **true**, otherwise **false**." ) );

    parseTestOptions.parse( );

    GIDI::Construction::PhotoMode photo_mode = parseTestOptions.photonMode( GIDI::Construction::PhotoMode::nuclearAndAtomic );
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
    PoPI::Database pops;
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../../../TestData/PoPs/pops.xml", "../all.map", construction, PoPI::IDs::neutron, "O16" );

    std::cout << stripDirectoryBase( protare->fileName( ), "/GIDI/Test/" ) << std::endl;

    std::map<std::string,int> particle_GID_map;

    particle_GID_map["n"] = 4;
    particle_GID_map["H1"] = 71;
    particle_GID_map["H2"] = 71;
    particle_GID_map["H3"] = 71;
    particle_GID_map["He3"] = 71;
    particle_GID_map["He4"] = 71;
    particle_GID_map["photon"] = 70;

    GIDI::Transporting::Groups_from_bdfls groups_from_bdfls( "../bdfls" );
    GIDI::Transporting::Fluxes_from_bdfls fluxes_from_bdfls( "../bdfls", 0 );

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    GIDI::Styles::TemperatureInfo const &temperature = temperatures[0];
    GIDI::Transporting::MG settings( protare->projectile( ).ID( ), GIDI::Transporting::Mode::multiGroup, GIDI::Transporting::DelayedNeutrons::on );
    settings.setZeroDepositionIfAllProductsTracked( argv_options.find( "-z" )->present( ) || true );
    GIDI::Transporting::Particles particles;

    for( std::size_t i1 = 0; i1 < argv_options.m_arguments.size( ); ++i1 ) {
        std::string particleID( argv[argv_options.m_arguments[i1]] );

        GIDI::Transporting::Particle particle( particleID, groups_from_bdfls.getViaGID( particle_GID_map[particleID] ) );
        particle.appendFlux( fluxes_from_bdfls.getViaFID( 1 ) );
        particles.add( particle );
    }

    std::vector<double> groupBoundaries = protare->groupBoundaries( settings, temperature, protare->projectile( ).ID( ) );
    printVectorOfDoubles( "Group boundaries        ::", groupBoundaries );

    GIDI::Vector depositionEnergy;
    depositionEnergy += protare->multiGroupDepositionEnergy( smr1, settings, temperature, particles );
    std::string prefix( "Deposition energy       ::" );
    printVector( prefix, depositionEnergy );

    GIDI::Vector depositionEnergySum;
    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = protare->reaction( index );

        depositionEnergySum += reaction->multiGroupDepositionEnergy( smr1, settings, temperature, particles );
    }

    GIDI::Vector photonAverageEnergy;
    for( std::size_t index = 0; index < protare->numberOfOrphanProducts( ); ++index ) {
        GIDI::Reaction const *reaction = protare->orphanProduct( index );

        depositionEnergySum -= reaction->multiGroupAverageEnergy( smr1, settings, temperature, PoPI::IDs::photon );
        photonAverageEnergy += reaction->multiGroupAverageEnergy( smr1, settings, temperature, PoPI::IDs::photon );
    }

    prefix = "Deposition energy sum   ::";
    printVector( prefix, depositionEnergySum );

    depositionEnergy -= depositionEnergySum;
    double maxDiff = 0.0;
    for( std::size_t i1 = 0; i1 < depositionEnergy.size( ); ++i1 ) {
        if( fabs( depositionEnergy[i1] ) > maxDiff ) maxDiff = fabs( depositionEnergy[i1] );
    }
    if( maxDiff > 1e-12 ) {
        prefix = "Deposition energy diff :::";
        printVector( prefix, depositionEnergy );
    }

    prefix = "photon average energy   ::";
    printVector( prefix, photonAverageEnergy );

    delete protare;
}

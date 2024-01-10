/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <set>

#include <statusMessageReporting.h>

#include <MCGIDI.hpp>

#include <GIDI_testUtilities.hpp>
#include <MCGIDI_testUtilities.hpp>
#include <bins.hpp>

static char const *description = "Loops over energy at the specified temperature, sampling reactions. If projectile is a photon, see options *-pa* and *-pn*.";

double myRNG( void *state );
void main2( int argc, char **argv );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    try {
        main2( argc, argv );
        exit( EXIT_SUCCESS ); }
    catch (std::exception &exception) {
        std::cerr << exception.what( ) << std::endl; }
    catch (char const *str) {
        std::cout << str << std::endl; }
    catch (std::string &str) {
        std::cout << str << std::endl;
    }

    exit( EXIT_SUCCESS );
}
/*
=========================================================
*/
void main2( int argc, char **argv ) {

    PoPI::Database pops;
    unsigned long long seed = 1;
    std::set<int> reactionsToExclude;
    GIDI::Transporting::Particles particles;
    double temperature_keV_K = 2.582e-5;
    LUPI::StatusMessageReporting smr1;

    std::map<std::string, std::string> particlesAndGIDs;

    particlesAndGIDs[PoPI::IDs::neutron] = "LLNL_gid_4";
    particlesAndGIDs["H1"] = "LLNL_gid_71";
    particlesAndGIDs["H2"] = "LLNL_gid_71";
    particlesAndGIDs["H3"] = "LLNL_gid_71";
    particlesAndGIDs["He3"] = "LLNL_gid_71";
    particlesAndGIDs["He4"] = "LLNL_gid_71";
    particlesAndGIDs[PoPI::IDs::photon] = "LLNL_gid_70";

    argvOptions argv_options( "angleBiasing", description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );

    parseTestOptions.m_askOid= true;
    parseTestOptions.m_askGNDS_File = true;

    argv_options.add( argvOption( "-r", true, "Specifies the reaction index for sampling angle biasing. If negative, all reactions are listed and then the code exits." ) );
    argv_options.add( argvOption( "-n", true, "Number of samples. If value is negative, it is multiplied by -1000." ) );
    argv_options.add( argvOption( "--numberOfBins", true, "Number of sample bins. Default is 1000." ) );

    parseTestOptions.parse( );

    if( argv_options.m_arguments.size( ) != 2 ) throw "Need projectile energy and outgoing particle angle (in lab frame).";

    MCGIDI_test_rngSetup( seed );

    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, parseTestOptions.photonMode( ) );
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../../../../TestData/PoPs/pops.xml", "../../../../GIDI/Test/Data/MG_MC/all_maps.map", construction, PoPI::IDs::neutron, "O16" );

    std::string productID = argv_options.find( "--oid" )->zeroOrOneOption( argv, protare->projectile( ).ID( ) );
    long numberOfSamples = argv_options.find( "-n" )->asLong( argv, -1000 );
    if( numberOfSamples < 0 ) numberOfSamples *= -1000;
    long numberOfBins = argv_options.find( "--numberOfBins" )->asLong( argv, 1000 );

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    std::string label( temperatures[0].griddedCrossSection( ) );
    MCGIDI::Transporting::MC MC( pops, protare->projectile( ).ID( ), &protare->styles( ), label, GIDI::Transporting::DelayedNeutrons::on, 20.0 );

    GIDI::Transporting::Groups_from_bdfls groups_from_bdfls( "../../../../GIDI/Test/bdfls" );
    GIDI::Transporting::Fluxes_from_bdfls fluxes_from_bdfls( "../../../../GIDI/Test/bdfls", 0.0 );

    for( std::map<std::string, std::string>::iterator iter = particlesAndGIDs.begin( ); iter != particlesAndGIDs.end( ); ++iter ) {
        if( iter->first == productID ) {
            GIDI::Transporting::MultiGroup multi_group = groups_from_bdfls.viaLabel( iter->second );
            GIDI::Transporting::Particle particle( iter->first, multi_group );

            particle.appendFlux( fluxes_from_bdfls.getViaFID( 1 ) );
            particles.add( particle );
        }
    }

    MCGIDI::DomainHash domainHash( 4000, 1e-8, 10 );
    MCGIDI::Protare *MCProtare = MCGIDI::protareFromGIDIProtare( smr1, *protare, pops, MC, particles, domainHash, temperatures, reactionsToExclude );

    int productIndex = pops[productID];

    double energy_in = argv_options.asDouble( argv, 0 );
    double mu_lab = argv_options.asDouble( argv, 1 );
    if( mu_lab < -1.0 ) throw "mu_lab must be greater than or equal to -1.0";
    if( mu_lab > 1.0 ) throw "mu_lab must be less than or equal to 1.0";

    int reactionIndex = argv_options.find( "-r" )->asLong( argv, 0 );
    if( reactionIndex >= static_cast<int>( MCProtare->numberOfReactions( ) ) ) throw "Invalid reaction index.";
    if( reactionIndex < 0 ) {
        std::cout << "List of reaction indices, thresholds and labels are:" << std::endl;
        for( reactionIndex = 0; reactionIndex < static_cast<int>( MCProtare->numberOfReactions( ) ); ++reactionIndex ) {
            MCGIDI::Reaction const *reaction = MCProtare->reaction( reactionIndex );

            std::cout << std::setw( 4 ) << reactionIndex << "  " << doubleToString( "%14.6e", reaction->crossSectionThreshold( ) ) << "  " << reaction->label( ).c_str( ) << std::endl;
        } }
    else {

        MCGIDI::Reaction const *reaction = MCProtare->reaction( reactionIndex );
        MCGIDI::Sampling::ClientCodeRNGData clientCodeRNGData( float64RNG64, nullptr );

        double energy_out, weight;
        weight = reaction->angleBiasing( productIndex, temperature_keV_K, energy_in, mu_lab, energy_out, float64RNG64, nullptr );
        double energyMin = energy_out, energyMax = energy_out;

        for( long sampleIndex = 0; sampleIndex < numberOfSamples; ++sampleIndex ) {
            weight = reaction->angleBiasing( productIndex, temperature_keV_K, energy_in, mu_lab, energy_out, float64RNG64, nullptr );
            if( energy_out < energyMin ) energyMin = energy_out;
            if( energy_out > energyMax ) energyMax = energy_out;
        }

        energyMin *= 0.5;
        energyMax *= 1.2;
        double deltaEnergy = energyMax - energyMin;
        if( deltaEnergy > energyMin ) energyMin = 0.0;

        if( energyMin == energyMax ) energyMax = 1.0 + 1.1 * energyMin;

        Bins bins( numberOfBins, energyMin, energyMax );

        for( long sampleIndex = 0; sampleIndex < numberOfSamples; ++sampleIndex ) {
            weight = reaction->angleBiasing( productIndex, temperature_keV_K, energy_in, mu_lab, energy_out, float64RNG64, nullptr );
            bins.accrue( energy_out, weight );
        }

        bins.print( stdout, "", true );
    }

    delete protare;
    delete MCProtare;

    exit( EXIT_SUCCESS );
}

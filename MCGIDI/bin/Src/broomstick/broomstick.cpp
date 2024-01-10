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
#include <fstream>
#include <iomanip>
#include <set>

#include <statusMessageReporting.h>

#include <MCGIDI.hpp>

#include <GIDI_testUtilities.hpp>
#include <MCGIDI_testUtilities.hpp>
#include <bins.hpp>

static char const *description = "For a protare, samples specified reaction (or all if specified reaction index is negative) many times (see option '-n') "
    "at the specified projectile energy, and creates an energy and angular spectrum for the specified outgoing particle (see options '--oid').";

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

    argvOptions argv_options( __FILE__, description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );
    parseTestOptions.m_askOid = true;

    parseTestOptions.m_askGNDS_File = true;

    argv_options.add( argvOption( "-n", true, "Number of samples. If value is negative, it is multiplied by -1000." ) );
    argv_options.add( argvOption( "--numberOfBins", true, "Number of sample bins. Default is 1000." ) );
    argv_options.add( argvOption( "-r", true, 
            "Specifies a reaction index to sample. If negative, all reactions are sampled using the protare's sampleReaction method." ) );
    argv_options.add( argvOption( "--recordPath", true, 
            "If present, each sampled event's outgoing particle data are written to file 'broomstick.out'." ) );

    parseTestOptions.parse( );

    if( argv_options.m_arguments.size( ) != 1 ) throw "Need projectile energy.";

    std::string recordPath = argv_options.find( "--recordPath" )->zeroOrOneOption(argv, "");
    std::ofstream recordStream;
    if( recordPath != "" ) {
        recordStream.open(recordPath, std::ios::out);
    }

    MCGIDI_test_rngSetup( seed );

    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, parseTestOptions.photonMode( ) );
    GIDI::Protare *protare = parseTestOptions.protare( pops, "/usr/gapps/data/nuclear/common/PoPs/pops.xml", "../../../../GIDI/Test/Data/MG_MC/all_maps.map", construction, PoPI::IDs::neutron, "O16" );

    GIDI::Transporting::Settings incompleteParticlesSetting( protare->projectile( ).ID( ), GIDI::Transporting::DelayedNeutrons::on );
    std::set<std::string> incompleteParticles;
    protare->incompleteParticles( incompleteParticlesSetting, incompleteParticles );
    std::cout << "# List of incomplete particles:";
    for( auto iter = incompleteParticles.begin( ); iter != incompleteParticles.end( ); ++iter ) {
        std::cout << " " << *iter;
    }
    std::cout << std::endl;

    std::string productID = argv_options.find( "--oid" )->zeroOrOneOption( argv, protare->projectile( ).ID( ) );
    long numberOfSamples = argv_options.find( "-n" )->asLong( argv, -1000 );
    if( numberOfSamples < 0 ) numberOfSamples *= -1000;
    long numberOfBins = argv_options.find( "--numberOfBins" )->asLong( argv, 1000 );

    int reactionIndex = static_cast<int>( argv_options.find( "-r" )->asLong( argv, 999999 ) );

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    std::string label( temperatures[0].griddedCrossSection( ) );
    MCGIDI::Transporting::MC MC( pops, protare->projectile( ).ID( ), &protare->styles( ), label, GIDI::Transporting::DelayedNeutrons::on, 20.0 );
    MC.setThrowOnError( false );

    for( std::map<std::string, std::string>::iterator iter = particlesAndGIDs.begin( ); iter != particlesAndGIDs.end( ); ++iter ) {
        GIDI::Transporting::Particle particle( iter->first );
        particles.add( particle );
    }

    MCGIDI::DomainHash domainHash( 4000, 1e-8, 10 );
    MCGIDI::Protare *MCProtare = MCGIDI::protareFromGIDIProtare( smr1, *protare, pops, MC, particles, domainHash, temperatures, reactionsToExclude );

    if( reactionIndex > static_cast<int>( MCProtare->numberOfReactions( ) ) ) {
        std::cout << "List of reaction indices, thresholds and labels are:" << std::endl;
        for( reactionIndex = 0; reactionIndex < static_cast<int>( MCProtare->numberOfReactions( ) ); ++reactionIndex ) {
            MCGIDI::Reaction const *reaction = MCProtare->reaction( reactionIndex );

            std::cout << std::setw( 4 ) << reactionIndex << "  " << doubleToString( "%14.6e", reaction->crossSectionThreshold( ) ) << "  " << reaction->label( ).c_str( ) << std::endl;
        }
        delete protare;
        delete MCProtare;
        exit( EXIT_SUCCESS );
    }

    int oidIndex = -1;
    int maxProductIndex = 0;
    for( auto particleIter = particles.particles( ).begin( ); particleIter != particles.particles( ).end( );  ++particleIter, ++maxProductIndex ) {
        MCProtare->setUserParticleIndex( pops[(*particleIter).first], maxProductIndex );
        if( (*particleIter).first == productID ) oidIndex = maxProductIndex;
        std::cout << "# particle ID/index " << (*particleIter).first << " " << maxProductIndex << std::endl;
    }

    double energy_in = argv_options.asDouble( argv, 0 );
    int hashIndex = domainHash.index( energy_in );

    std::cout << "# path is " << protare->realFileName( ) << std::endl;
    std::cout << "# projectile is " << MCProtare->projectileID( ).c_str( ) << std::endl;
    std::cout << "# target is " << MCProtare->targetID( ).c_str( ) << std::endl;
    std::cout << "# projectile energy is " << energy_in << " MeV" << std::endl;
    if( reactionIndex >= 0 ) {
        MCGIDI::Reaction const *reaction = MCProtare->reaction( reactionIndex );
        std::cout << "# Reaction Info:" << std::endl;
        std::cout << "#     index: " << reactionIndex << std::endl;
        std::cout << "#     label: " << reaction->label( ).c_str( ) << std::endl;
        std::cout << "#     threshold: " << doubleToString( "%14.6e MeV", reaction->crossSectionThreshold( ) ) << std::endl;
    }

    MCGIDI::Sampling::ClientCodeRNGData clientCodeRNGData( float64RNG64, nullptr );

    double energyMin = 1e-11;
    double energyMax = 20;
    Bins energyBins( numberOfBins, energyMin, energyMax, true );
    Bins muBins( numberOfBins, -1.0, 1.0 );

    MCGIDI::URR_protareInfos URR_protareInfos;
    double totalCrossSection = 0.0;
    MCGIDI::Sampling::Input input( false, MCGIDI::Sampling::Upscatter::Model::none );           // This should be an input option.
    MCGIDI::Sampling::StdVectorProductHandler products;
    if( reactionIndex < 0 ) {
        totalCrossSection = MCProtare->crossSection( URR_protareInfos, hashIndex, temperature_keV_K, energy_in, true ); }
    else {
        MCGIDI::Reaction const *reaction = MCProtare->reaction( reactionIndex );
        if( reaction->crossSectionThreshold( ) > energy_in ) {
            delete protare;
            delete MCProtare;

            exit( EXIT_SUCCESS );
        }
    }

    for( long sampleIndex = 0; sampleIndex < numberOfSamples; ++sampleIndex ) {
        int reactionIndex2 = reactionIndex;
        if( reactionIndex2 < 0 ) reactionIndex2 = MCProtare->sampleReaction( URR_protareInfos, hashIndex, temperature_keV_K, energy_in, totalCrossSection, float64RNG64, nullptr );
        MCGIDI::Reaction const *reaction = MCProtare->reaction( reactionIndex2 );

        if( recordPath != "" ) recordStream << "Event: " << reactionIndex2 << std::endl;

        products.clear( );
        reaction->sampleProducts( MCProtare, energy_in, input, float64RNG64, nullptr, products );
        for( std::size_t productIndex = 0; productIndex < products.size( ); ++productIndex ) {
            MCGIDI::Sampling::Product const &product = products[productIndex];
            int userProductIndex = product.m_userProductIndex;
            if( recordPath != "" ) {
                recordStream << "    product index: " << userProductIndex << " " << product.m_kineticEnergy << " " << product.m_px_vx << " " << product.m_py_vy << " " << product.m_pz_vz << std::endl;
            }
            if( userProductIndex != oidIndex ) continue;
            
            energyBins.accrue( product.m_kineticEnergy, 1.0 );
            double speed = sqrt( product.m_px_vx * product.m_px_vx + product.m_py_vy * product.m_py_vy + product.m_pz_vz * product.m_pz_vz );
            double mu = 0.0;
            if( speed != 0.0 ) mu = product.m_pz_vz / speed;
            muBins.accrue( mu, 1.0 );
        }
    }

    energyBins.print( stdout, "# energy", true );
    muBins.print( stdout, "# mu", true );

    delete protare;
    delete MCProtare;
    if( recordPath != "" ) {
        recordStream.close();
    }

    exit( EXIT_SUCCESS );
}

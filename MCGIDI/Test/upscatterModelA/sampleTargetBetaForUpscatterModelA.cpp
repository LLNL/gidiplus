/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "MCGIDI.hpp"

#define nBins 501

/*
=========================================================
*/
int main( int argc, char **argv ) {

    std::string mapFilename( "../../../GIDI/Test/all3T.map" );
    PoPI::Database pops( "../../../GIDI/Test/pops.xml" );
    GIDI::Map::Map map( mapFilename, pops );
    std::string neutronID( PoPI::IDs::neutron );
    std::string targetID( "O16" );
    GIDI::Protare *protare;
    GIDI::Transporting::Particles particles;
    double temperature_MeV = 1e-3;
    long numberOfSamples = 10 * 1000 * 1000;
    long bins[nBins+1];
    std::set<int> reactionsToExclude;
    LUPI::StatusMessageReporting smr1;

    for( int i1 = 0; i1 <= nBins; ++i1 ) bins[i1] = 0;

    std::cerr << "    " << __FILE__;
    for( int i1 = 1; i1 < argc; i1++ ) std::cerr << " " << argv[i1];
    std::cerr << std::endl;

    if( argc > 1 ) targetID = argv[1];

    try {
        GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, GIDI::Construction::PhotoMode::nuclearAndAtomic );
        protare = map.protare( construction, pops, neutronID, targetID ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }
    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );

    for( GIDI::Styles::TemperatureInfos::iterator temperature = temperatures.begin( ); temperature < temperatures.end( ); ++temperature ) {
        temperature->print( );
    }

    std::string label( temperatures[0].heatedCrossSection( ) );
    MCGIDI::Transporting::MC MC( pops, neutronID, &protare->styles( ), label, GIDI::Transporting::DelayedNeutrons::on, 20.0 );

    std::string upScatteringLabel = temperatures[0].heatedMultiGroup( );
    std::cout << "upScatteringLabel = " << upScatteringLabel << std::endl;
    MC.setUpscatterModelA( upScatteringLabel );
    std::cout << "upscatterModel = " << MC.upscatterModel( ) << ",   upscatterModelALabel = " << MC.upscatterModelALabel( ) << std::endl;

    MCGIDI::DomainHash domainHash( 4000, 1e-8, 10 );
    MCGIDI::Protare *MCProtare;

    try {
        MCProtare = MCGIDI::protareFromGIDIProtare( smr1, *protare, pops, MC, particles, domainHash, temperatures, reactionsToExclude ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    PoPI::Base const &target = pops.get<PoPI::Base>( targetID );
    char Str[512];
    sprintf( Str, "sampleTargetBetaForUpscatterModelA.%s.dat", target.ID( ).c_str( ) );
    FILE *fOut;
    if( ( fOut = fopen( Str, "w" ) ) == nullptr ) throw "error opening output file";

    MCGIDI::Sampling::Input input( true, MCGIDI::Sampling::Upscatter::Model::B );
    input.m_temperature = temperature_MeV * 1e3;

    MCGIDI::Reaction const *reaction;
    for( std::size_t i1 = 0; i1 < MCProtare->numberOfReactions( ); ++i1 ) {
        reaction = MCProtare->reaction( i1 );
        if( reaction->ENDF_MT( ) == 102 ) break;
    }
    input.m_reaction = reaction;

    double targetMass = MCProtare->targetMass( );
    double betaTargetThermal = MCGIDI_particleBeta( targetMass, temperature_MeV );
    double betaTargetMax = 5.0 * betaTargetThermal;
    double projectileEnergy = temperature_MeV;
    for( long i1 = 0; i1 < numberOfSamples; ++i1 ) {
        if( sampleTargetBetaForUpscatterModelA( MCProtare, projectileEnergy, input, drand48, nullptr ) ) {
            int bin = (int) ( nBins * input.m_targetBeta / betaTargetMax );
            if( bin > nBins ) bin = nBins;
            bins[bin]++;
        }
    }

    if( ( fOut = fopen( Str, "w" ) ) == nullptr ) throw "error opening output file";

    double betaPerBin = betaTargetMax / nBins;
    std::cout << "# betaTargetMax = " << betaTargetMax << std::endl;
    std::cout << "# betaTargetThermal = " << betaTargetThermal << std::endl;
    for( int i1 = 0; i1 <= nBins; ++i1 ) {
        double binVelocity = ( i1 + 0.5 ) * betaPerBin;
        double binEnergy = 0.5 * targetMass * binVelocity * binVelocity;
        double probability = bins[i1] / (double) numberOfSamples;
        double dEnergy = binVelocity * targetMass * betaPerBin;

        fprintf( fOut, "%e %e %e %e %e %9ld\n", binEnergy, probability / dEnergy, binVelocity, binVelocity / betaTargetThermal, probability, bins[i1] );
    }
    fclose( fOut );

    exit( EXIT_SUCCESS );
}

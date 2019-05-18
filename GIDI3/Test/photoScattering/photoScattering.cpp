/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <set>

#include <GIDI_testUtilities.hpp>

void readProtare( GIDI::Map &map, PoPs::Database const &pops, std::string const &targetID, GIDI::Construction::PhotoMode photoMode, GIDI::Settings::Particle const &photonParticle );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    PoPs::Database pops( "../pops.xml" );
    std::string mapFilename( "../all.map" );
    GIDI::Map map( mapFilename, pops );
    std::string targetID( "O16" );

    if( argc > 1 ) targetID = argv[1];

    std::cerr << "    " << __FILE__;
    for( int i1 = 1; i1 < argc; i1++ ) std::cerr << " " << argv[i1];
    std::cerr << std::endl;

    GIDI::Settings::Groups_from_bdfls groups_from_bdfls( "../bdfls" );
    GIDI::Settings::Fluxes_from_bdfls fluxes_from_bdfls( "../bdfls", 0 );

    GIDI::Settings::Particle photonParticle( PoPs::IDs::photon, groups_from_bdfls.getViaGID( 70 ) );
    photonParticle.appendFlux( fluxes_from_bdfls.getViaFID( 1 ) );

    std::cout << "Atomic only." << std::endl;
    readProtare( map, pops, targetID, GIDI::Construction::e_atomicOnly, photonParticle );

    std::cout << std::endl;
    std::cout << "Nuclear only." << std::endl;
    readProtare( map, pops, targetID, GIDI::Construction::e_nuclearOnly, photonParticle );

    std::cout << std::endl;
    std::cout << "Nuclear and atomic." << std::endl;
    readProtare( map, pops, targetID, GIDI::Construction::e_nuclearAndAtomic, photonParticle );
}
/*
=========================================================
*/
void readProtare( GIDI::Map &map, PoPs::Database const &pops, std::string const &targetID, GIDI::Construction::PhotoMode photoMode, GIDI::Settings::Particle const &photonParticle ) {

    GIDI::Protare *protare;

    try {
        GIDI::Construction::Settings construction( GIDI::Construction::e_all, photoMode );
        protare = map.protare( construction, pops, PoPs::IDs::photon, targetID); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }
    if( protare == NULL ) {
        std::cout << "protare for " << targetID << " not found." << std::endl;
        exit( EXIT_FAILURE );
    }

    std::string fileName( protare->fileName( ) );
    std::size_t offset = fileName.rfind( "GIDI" );
    std::cout << fileName.substr( offset ) << std::endl;
    if( photoMode == GIDI::Construction::e_nuclearAndAtomic ) {
        fileName = protare->fileName( 1 );
        offset = fileName.rfind( "GIDI" );
        std::cout << fileName.substr( offset ) << std::endl;
    }

    std::cout << "  " << "Number of reactions = " << protare->numberOfReactions( ) << std::endl;

    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = protare->reaction( index );

        GIDI::OutputChannel const *outputChannel = reaction->outputChannel( );
        GIDI::Product const *product = static_cast<GIDI::Product const *>( outputChannel->products( ).get<GIDI::Product>( 0 ) );
        GIDI::Suite const &distribution = product->distribution( );

        std::cout << "    " << reaction->label( ) << std::endl;
        std::cout << "      " << product->particle( ).ID( ) << std::endl;

        for( std::size_t i2 = 0; i2 < distribution.size( ); ++i2 ) {
            GIDI::Distribution const *form = distribution.get<GIDI::Distribution>( i2 );

            std::cout << "       distribution form moniker = " << form->moniker( ) << std::endl;
        }
    }

    std::cout << std::endl;

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    for( std::size_t temperatureIndex = 0; temperatureIndex < temperatures.size( ); ++temperatureIndex ) {
        GIDI::Styles::TemperatureInfo &temperatureInfo = temperatures[temperatureIndex];
        GIDI::PhysicalQuantity temperature( temperatureInfo.temperature( ) );

        std::cout << "temperatureIndex = " << temperatureIndex << ":   temperature = " << temperature.value( ) << " " << temperature.unit( ) << std::endl;
        std::cout << "    heatedCrossSection = <" << temperatureInfo.heatedCrossSection( ) << ">" << std::endl;
        std::cout << "    griddedCrossSection = <" << temperatureInfo.griddedCrossSection( ) << ">" << std::endl;
        std::cout << "    heatedMultiGroup = <" << temperatureInfo.heatedMultiGroup( ) << ">" << std::endl;
        std::cout << "    SnElasticUpScatter = <" << temperatureInfo.SnElasticUpScatter( ) << ">" << std::endl;
    }

    std::string label( temperatures[0].heatedMultiGroup( ) );
    GIDI::Settings::MG settings( protare->projectile( ).ID( ), label, true );

    GIDI::Settings::Particles particles;
    particles.add( photonParticle );
    particles.process( *protare, label );

    std::vector<double> doubles = protare->groupBoundaries( settings, PoPs::IDs::neutron );
    printVectorOfDoubles( "Neutron group boundaries :: ", doubles );

    doubles = protare->groupBoundaries( settings, PoPs::IDs::photon );
    printVectorOfDoubles( "Neutron group boundaries :: ", doubles );

    GIDI::Vector vector = protare->multiGroupCrossSection( settings, particles );
    printVector( "Total cross section :: ", vector );

    vector = protare->multiGroupInverseSpeed( settings, particles );
    printVector( "Inverse speed :: ", vector );

    vector = protare->multiGroupQ( settings, particles, false );
    printVector( "Q (initial) :: ", vector );
    vector = protare->multiGroupQ( settings, particles, true );
    printVector( "Q (final)   :: ", vector );

    vector = protare->multiGroupAvailableEnergy( settings, particles );
    printVector( "Available energy   :: ", vector );

    vector = protare->multiGroupAvailableMomentum( settings, particles );
    printVector( "Available momentum :: ", vector );

    vector = protare->multiGroupAverageEnergy( settings, particles, PoPs::IDs::neutron );
    printVector( "Neutron average energy :: ", vector );

    vector = protare->multiGroupAverageEnergy( settings, particles, PoPs::IDs::photon );
    printVector( "Photon average energy :: ", vector );

    vector = protare->multiGroupAverageMomentum( settings, particles, PoPs::IDs::neutron );
    printVector( "Neutron average momentum :: ", vector );

    vector = protare->multiGroupAverageMomentum( settings, particles, PoPs::IDs::photon );
    printVector( "Photon average momentum :: ", vector );

    vector = protare->multiGroupMultiplicity( settings, particles, PoPs::IDs::neutron );
    printVector( "Neutron multiplicity :: ", vector );

    vector = protare->multiGroupMultiplicity( settings, particles, PoPs::IDs::photon );
    printVector( "Photon multiplicity :: ", vector );

    std::set<std::string> ids;
    protare->productIDs( ids, particles, false );
    printIDs( "Product IDs (all)           : ", ids );

    ids.clear( );
    protare->productIDs( ids, particles, true );
    printIDs( "Product IDs (transportable) : ", ids );

    for( int order = 0; order < protare->maximumLegendreOrder( settings, PoPs::IDs::photon ); ++order ) {

        std::cout << "Data for Legendre order " << order << std::endl;

        vector = protare->multiGroupTransportCorrection( settings, particles, order, GIDI::transportCorrection_Pendlebury, 0.0 );
        printVector( "    Transport correction ::", vector );

        GIDI::Matrix matrix = protare->multiGroupProductMatrix( settings, particles, PoPs::IDs::photon, order );
        printMatrix( "    Photon product matrix", -2, matrix );
    }
// depositionData

    delete protare;
}

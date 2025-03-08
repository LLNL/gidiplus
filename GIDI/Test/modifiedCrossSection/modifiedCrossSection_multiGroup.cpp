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

#include <GIDI_testUtilities.hpp>

static char const *description = "Thid program modifies multi-group data and then prints the data.";

void main2( int argc, char **argv );
void printMultiGroup1d( std::string const &a_label, std::vector<double> const & a_groupBoundaries, GIDI::Vector const &a_vector );

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

    argvOptions argv_options( __FILE__, description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );
    double offset, slope, domainMin, domainMax;
    bool inputPresent = false;
    LUPI::StatusMessageReporting smr;
    std::vector<std::string> particleIds { PoPI::IDs::neutron, PoPI::IDs::photon };

    parseTestOptions.m_askGNDS_File = true;

    argv_options.add( argvOption( "--domainMin", true, "The domain minimum for the offset and slope functions." ) );
    argv_options.add( argvOption( "--domainMax", true, "The domain maximum for the offset and slope functions." ) );
    argv_options.add( argvOption( "--offset", true, "The constant value for the offset functions over the domain." ) );
    argv_options.add( argvOption( "--slope", true, "The constant value for the slope functions over the domain." ) );

    parseTestOptions.parse( );

    inputPresent = argv_options.find( "--domainMin" )->present( ) || argv_options.find( "--domainMax" )->present( ) 
                    || argv_options.find( "--offset" )->present( ) || argv_options.find( "--slope" )->present( );

    GIDI::Construction::PhotoMode photo_mode = parseTestOptions.photonMode( GIDI::Construction::PhotoMode::nuclearAndAtomic );
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
    PoPI::Database pops;
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../../../TestData/PoPs/pops.xml", "../all3T.map", construction, PoPI::IDs::neutron, "H1" );

    std::cout << stripDirectoryBase( protare->fileName( ), "/GIDI/Test/" ) << std::endl;

    GIDI::Transporting::Particles particles;
    for( auto iterProductId = particleIds.begin( ); iterProductId != particleIds.end( ); ++iterProductId ) {
        GIDI::Transporting::Particle particle( *iterProductId );
        particles.add( particle );
    }

    offset = argv_options.find( "--offset" )->asDouble( argv, 0.0 );
    slope = argv_options.find( "--slope" )->asDouble( argv, 1.0 );
    domainMin = argv_options.find( "--domainMin" )->asDouble( argv, 0.0 );
    domainMax = argv_options.find( "--domainMax" )->asDouble( argv, 20.0 );
    if( domainMin >= domainMax ) throw( "domainMin must be less than domainMax." );

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );

    GIDI::Transporting::MG settings( protare->projectile( ).ID( ), GIDI::Transporting::Mode::multiGroup, GIDI::Transporting::DelayedNeutrons::on );
    settings.setThrowOnError( false );

    std::vector<std::pair<std::string, std::string>> labelsAndUnits;
    labelsAndUnits.push_back( std::pair<std::string, std::string>( "Energy_in", "MeV" ) );
    labelsAndUnits.push_back( std::pair<std::string, std::string>( "Cross section", "b" ) );
    GIDI::Axes offsetAxes = GIDI::Axes::makeAxes( labelsAndUnits );

    GUPI::WriteInfo writeInfo = GUPI::WriteInfo( );
    offsetAxes.toXMLList( writeInfo, "  " );
    std::cout << std::endl << "Offset Axes" << std::endl;
    writeInfo.print( );

    labelsAndUnits[1] = std::pair<std::string, std::string>( "Cross section", "b/MeV" );
    GIDI::Axes slopeAxes = GIDI::Axes::makeAxes( labelsAndUnits );

    writeInfo.clear( );
    slopeAxes.toXMLList( writeInfo, "  " );
    std::cout << std::endl << "Slope Axes" << std::endl;
    writeInfo.print( );

    GIDI::Functions::XYs1d *offsetXYs1d = GIDI::Functions::XYs1d::makeConstantXYs1d( offsetAxes, domainMin, domainMax, offset );
    writeInfo.clear( );
    offsetXYs1d->toXMLList( writeInfo, "  " );
    std::cout << std::endl << "Offset" << std::endl;
    writeInfo.print( );

    GIDI::Functions::XYs1d *slopeXYs1d = GIDI::Functions::XYs1d::makeConstantXYs1d( slopeAxes, domainMin, domainMax, slope );
    writeInfo.clear( );
    slopeXYs1d->toXMLList( writeInfo, "  " );
    std::cout << std::endl << "Slope" << std::endl;
    writeInfo.print( );

    std::cout << std::endl;
    GIDI::Styles::Suite &styles = protare->styles( );
    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction *reaction = protare->reaction( index );
        GIDI::Suite &crossSectionSuite = reaction->crossSection( );

        std::string string( reaction->label( ) );
        std::cout << std::endl << "# Reaction:: " << string << std::endl;
        if( inputPresent ) reaction->modifyCrossSection( offsetXYs1d, slopeXYs1d, true );

        for( auto temperatureInfo = temperatures.begin( ); temperatureInfo != temperatures.end( ); ++temperatureInfo ) {
            std::cout << std::endl << std::endl;
            string = "#   " + temperatureInfo->heatedCrossSection( ) + ":: ";
            std::cout << string << std::endl;

            GIDI::Functions::XYs1d const *xys1d = static_cast<GIDI::Functions::XYs1d *>(
                crossSectionSuite.findInstanceOfTypeInLineage( styles, temperatureInfo->heatedCrossSection( ), GIDI_XYs1dChars ) );
            xys1d->print( "    %16.8e %16.8e\n" );

            string = "#   " + temperatureInfo->heatedMultiGroup( ) + ":: ";
            std::cout << std::endl << std::endl << string << std::endl;

            GIDI::Functions::Gridded1d *gridded1d = crossSectionSuite.get<GIDI::Functions::Gridded1d>( temperatureInfo->heatedMultiGroup( ) );
            gridded1d->print( "    %14.6e %14.6e\n" );
        }
    }


    std::cout << std::endl << std::endl << "# Temperature information:" << std::endl;
    for( auto iter = temperatures.begin( ); iter != temperatures.end( ); ++iter ) {
        auto temperatureInfo = *iter;
        GIDI::PhysicalQuantity const &temperature = iter->temperature( );

        std::cout << "#   label = " << iter->heatedMultiGroup( ) << "  temperature = " << temperature.value( ) << std::endl;
        std::vector<double> groupBoundaries = protare->groupBoundaries( settings, temperatureInfo, protare->projectile( ).ID( ) );

        printMultiGroup1d( "Cross section", groupBoundaries, protare->multiGroupCrossSection( smr, settings, temperatureInfo ) );
        printMultiGroup1d( "Q", groupBoundaries, protare->multiGroupQ( smr, settings, temperatureInfo, true ) );
        printMultiGroup1d( "Available energy", groupBoundaries, protare->multiGroupAvailableEnergy( smr, settings, temperatureInfo ) );
        printMultiGroup1d( "Fission neutron multiplicity", groupBoundaries, protare->multiGroupFissionNeutronMultiplicity( smr, settings, temperatureInfo ) );

        printMultiGroup1d( "Deposition energy", groupBoundaries, protare->multiGroupDepositionEnergy( smr, settings, temperatureInfo, particles ) );

        for( auto iterProductId = particleIds.begin( ); iterProductId != particleIds.end( ); ++iterProductId ) {
            std::string &productID = *iterProductId;
            printMultiGroup1d( "Multiplicty for " + productID, groupBoundaries, protare->multiGroupMultiplicity( smr, settings, temperatureInfo, productID ) );
            printMultiGroup1d( "Gain for " + productID, groupBoundaries, protare->multiGroupGain( smr, settings, temperatureInfo, productID ) );
            printMultiGroup1d( "Average energy for " + productID, groupBoundaries, protare->multiGroupAverageEnergy( smr, settings, temperatureInfo, productID ) );
        }
    }

    delete offsetXYs1d;
    delete slopeXYs1d;
    delete protare;
}

/*
=========================================================
*/

void printMultiGroup1d( std::string const &a_label, std::vector<double> const & a_groupBoundaries, GIDI::Vector const &a_vector ) {

    std::cout << "#   " << a_label << ":" << std::endl;
    for( std::size_t index = 0; index < a_vector.size( ); ++index ) {
        std::cout << LUPI::Misc::argumentsToString( "        %12.5e %13.5e", a_groupBoundaries[index], a_vector[index] ) << std::endl;
    }
}

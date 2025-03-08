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

static char const *description = "The program prints the multi-group cross section for a protare and its reactions.";

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

    argvOptions argv_options( __FILE__, description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );
    double offset, slope, domainMin, domainMax;
    bool inputPresent = false;

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

    offset = argv_options.find( "--offset" )->asDouble( argv, 0.0 );
    slope = argv_options.find( "--slope" )->asDouble( argv, 1.0 );
    domainMin = argv_options.find( "--domainMin" )->asDouble( argv, 0.0 );
    domainMax = argv_options.find( "--domainMax" )->asDouble( argv, 20.0 );
    if( domainMin >= domainMax ) throw( "domainMin must be less than domainMax." );

    GIDI::Styles::Suite const &styles = protare->styles( );

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    std::cout << "Temperature information" << std::endl;
    for( GIDI::Styles::TemperatureInfos::iterator iter = temperatures.begin( ); iter != temperatures.end( ); ++iter ) {
        GIDI::PhysicalQuantity const &temperature = iter->temperature( );

        std::cout << "    label = " << iter->heatedMultiGroup( ) << "  temperature = " << temperature.value( ) << std::endl;
    }
    std::cout << std::endl;

    GIDI::Transporting::MG settings( protare->projectile( ).ID( ), GIDI::Transporting::Mode::multiGroup, GIDI::Transporting::DelayedNeutrons::on );

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
    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction *reaction = protare->reaction( index );
        GIDI::Suite &crossSectionSuite = reaction->crossSection( );

        std::string string( reaction->label( ) );
        std::cout << "# Reaction: " << string << std::endl;
        if( inputPresent ) reaction->modifiedCrossSection( offsetXYs1d, slopeXYs1d );

        for( GIDI::Styles::TemperatureInfos::iterator temperatureInfo = temperatures.begin( ); temperatureInfo != temperatures.end( ); ++temperatureInfo ) {
            std::cout << std::endl << std::endl;
            string = "#   " + temperatureInfo->heatedCrossSection( ) + ":: ";
            std::cout << string << std::endl;

            GIDI::Functions::XYs1d *xys1d = static_cast<GIDI::Functions::XYs1d *>( crossSectionSuite.findInstanceOfTypeInLineage( 
                    styles, temperatureInfo->heatedCrossSection( ), GIDI_XYs1dChars ) );
            xys1d->print( "    %16.8e %16.8e\n" );

            std::cout << std::endl << std::endl;
            GIDI::Functions::Ys1d *ys1d = crossSectionSuite.get<GIDI::Functions::Ys1d>( temperatureInfo->griddedCrossSection( ) );
            std::vector<double> &Ys = ys1d->Ys( );
            GIDI::Styles::GriddedCrossSection const griddedCrossSection = *styles.get<GIDI::Styles::GriddedCrossSection>( temperatureInfo->griddedCrossSection( ) );
            nf_Buffer<double> const &grid = griddedCrossSection.grid( ).values( );
            std::size_t start = ys1d->start( ), size = ys1d->size( ), index2;
            for( index2 = 0; index2 < size; ++index2, ++start ) printf( "    %16.8e %16.8e\n", grid[start], Ys[index2] );
        }
    }

    delete offsetXYs1d;
    delete slopeXYs1d;
    delete protare;
}

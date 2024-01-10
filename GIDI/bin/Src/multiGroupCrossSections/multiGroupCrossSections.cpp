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
#include <vector>

#include <LUPI.hpp>
#include <GIDI_testUtilities.hpp>

static char const *description = 
    "This program write to files the multi-group cross section for a protare and its reactions.\n"
    "It outputs the multi-group data for each temperature into a sub-directory of the output\n"
    "path specified with the '-o' option.";
static int verbose = 0;

void main2( int argc, char **argv );
void writeBoundaries( std::string a_outputDir, std::vector<double> const &a_boundaries );
void write( std::string a_outputDir, int a_index, int a_ENDL_C, std::vector<double> const &a_groupBoundaries, 
                GIDI::Vector a_vector, std::string const &a_fileName = "" );
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

    parseTestOptions.m_askGNDS_File = true;

    argv_options.add( argvOption( "-o", true, "The directory to send all output to." ) );
    argv_options.add( argvOption( "-v", false, "Determines the verbosity." ) );

    parseTestOptions.parse( );

    verbose = argv_options.find( "-v" )->m_counter;
    std::string outputRoot = argv_options.find( "-o" )->zeroOrOneOption( argv, "multiGroupCrossSections.out" );

    GIDI::Construction::PhotoMode photo_mode = parseTestOptions.photonMode( GIDI::Construction::PhotoMode::nuclearAndAtomic );
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
    PoPI::Database pops;
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../../../../TestData/PoPs/pops.xml", "../all.map", construction, PoPI::IDs::neutron, "O16" );
    if( protare == nullptr ) throw "No matching protare.";

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    for( GIDI::Styles::TemperatureInfos::iterator temperatureIter = temperatures.begin( ); temperatureIter != temperatures.end( ); ++temperatureIter ) {
        if( temperatureIter->heatedMultiGroup( ) == "" ) continue;          // No data for this temperature.

        GIDI::PhysicalQuantity const &temperature = temperatureIter->temperature( );
        if( verbose > 0 ) std::cout << "    label = " << temperatureIter->heatedMultiGroup( ) 
                << "  temperature = " << temperature.value( ) << " " << temperature.unit( ) << std::endl;

        std::string outputDir( outputRoot + LUPI_FILE_SEPARATOR + temperatureIter->heatedMultiGroup( ) );

        LUPI::FileInfo::createDirectories( outputDir );

        GIDI::Transporting::MG settings( protare->projectile( ).ID( ), GIDI::Transporting::Mode::multiGroup, GIDI::Transporting::DelayedNeutrons::on );

        std::vector<double> const &groupBoundaries = protare->groupBoundaries( settings, *temperatureIter, protare->projectile( ).ID( ) );
        writeBoundaries( outputDir, groupBoundaries );

        GIDI::Vector total = protare->multiGroupCrossSection( settings, *temperatureIter );
        write( outputDir, -1, 0, groupBoundaries, total );

        std::map<int,GIDI::Vector> CValues;
        GIDI::Vector sum;
        for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
            GIDI::Reaction const *reaction = protare->reaction( index );
            int C = reaction->ENDL_C( );

            GIDI::Vector crossSection = reaction->multiGroupCrossSection( settings, *temperatureIter );
            write( outputDir, index, C, groupBoundaries, crossSection, reaction->label( ) );

            if( CValues.find( C ) == CValues.end( ) ) CValues[C] = GIDI::Vector( );
            CValues[C] += crossSection;

            sum += crossSection;
        }

        for( auto CValue = CValues.begin( ); CValue != CValues.end( ); ++CValue ) {
            if( verbose > 1 ) std::cout << "        C = " << CValue->first << std::endl;
            write( outputDir, -1, CValue->first, groupBoundaries, CValue->second );
        }

        write( outputDir, -1, -1, groupBoundaries, sum, "sumCrossSection" );

        GIDI::Vector diff = total - sum;
        write( outputDir, -1, -1, groupBoundaries, diff, "totalMinusSumCrossSection" );
    }

    delete protare;
}
/*
=========================================================
*/
void writeBoundaries( std::string a_outputDir, std::vector<double> const &a_boundaries ) {

    std::string fileName = a_outputDir + "/Boundaries";

    FILE *file = fopen( fileName.c_str( ), "w" );
    if( file == nullptr ) throw( "Could not open file '" + fileName + "'." );

    for( std::size_t index = 0; index < a_boundaries.size( ); ++index ) fprintf( file, "%16.8e\n", a_boundaries[index] );

    fclose( file );
}
/*
=========================================================
*/
void write( std::string a_outputDir, int a_index, int a_ENDL_C, std::vector<double> const &a_groupBoundaries, 
                GIDI::Vector a_vector, std::string const &a_fileName ) {

    std::string Str = LUPI::Misc::argumentsToString( "C_%.2d", a_ENDL_C );

    if( a_index >= 0 ) {
        Str = LUPI::Misc::argumentsToString( "reactionIndex_%.4d_C%.2d", a_index, a_ENDL_C ); }
    else {
        if( a_fileName != "" ) Str = a_fileName;
    }
    std::string fileName = a_outputDir + "/" + Str + ".dat";

    FILE *file = fopen( fileName.c_str( ), "w" );
    if( file == nullptr ) throw( "Could not open file '" + fileName + "'." );

    a_vector.writeWithBoundaries( file, "  %16.8e %16.8e\n", a_groupBoundaries, 1e-7 );

    fclose( file );
}

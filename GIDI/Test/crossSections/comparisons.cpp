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

static char const *description = "This program prints the continuous energy and multi-group forms of all \n(i.e., each reaction and total) the cross sections "
    "for a protare. Each dataset \n(i.e., reaction/form) is written to a separate file.";

void main2( int argc, char **argv );
void writeVector( std::string a_fileName, std::string a_label, GIDI::Vector const &a_data, std::vector<double> const &a_boundaries );
void writeXYs1d( std::string a_fileName, std::string a_label, GIDI::Functions::XYs1d &a_data );
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
    argvOptions argv_options( "crossSections", description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );

    parseTestOptions.m_askGNDS_File = true;

    parseTestOptions.parse( );

    GIDI::Construction::PhotoMode photo_mode = parseTestOptions.photonMode( GIDI::Construction::PhotoMode::nuclearAndAtomic );
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
    PoPI::Database pops;
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../../../TestData/PoPs/pops.xml", "../all.map", construction, PoPI::IDs::neutron, "O16" );
    GIDI::Styles::Suite const &styles = protare->protare( 0 )->styles( );

    std::cout << stripDirectoryBase( protare->fileName( ), "/GIDI/Test/" ) << std::endl;

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    GIDI::Transporting::MG settings( protare->projectile( ).ID( ), GIDI::Transporting::Mode::multiGroup, GIDI::Transporting::DelayedNeutrons::on );
    GIDI::Styles::TemperatureInfo temperature = temperatures[0];

    std::vector<double> groupBoundaries = protare->groupBoundaries( settings, temperature, protare->projectile( ).ID( ) );

    GIDI::Vector crossSection = protare->multiGroupCrossSection( smr1, settings, temperature );
    writeVector( "total", "total", crossSection, groupBoundaries );

    GIDI::Functions::XYs1d total;
    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = protare->reaction( index );

        crossSection = reaction->multiGroupCrossSection( smr1, settings, temperature );
        std::string label( reaction->label( ) );
        std::string name = LUPI::Misc::argumentsToString( "r%.4d", (int) index);
        writeVector( name, label, crossSection, groupBoundaries );

            // The next line is needed because FUDGE currently does not create a heated style for the large angle Coulomb scattering reaction cross section.
        std::string styleLabel = *styles.findLabelInLineage( reaction->crossSection( ), temperature.heatedCrossSection( ) );
        GIDI::Functions::XYs1d reactionCrossSection = *reaction->crossSection( ).get<GIDI::Functions::XYs1d>( styleLabel );
        writeXYs1d( name, label, reactionCrossSection );
        total += reactionCrossSection;
    }

    writeXYs1d( "total", "total", total );

    delete protare;
}
/*
=========================================================
*/
void writeVector( std::string a_fileName, std::string a_label, GIDI::Vector const &a_data, std::vector<double> const &a_boundaries ) {

    FILE *fOut;
    a_fileName += ".mg";

    fOut = fopen( a_fileName.c_str( ), "w" );
    fprintf( fOut, "# %s\n", a_label.c_str( ) );
    a_data.writeWithBoundaries( fOut, "%14.5e %14.5e\n", a_boundaries, 1e-6 );

    fclose( fOut );
}
/*
=========================================================
*/
void writeXYs1d( std::string a_fileName, std::string a_label, GIDI::Functions::XYs1d &a_data ) {

    FILE *fOut;
    a_fileName += ".ce";

    fOut = fopen( a_fileName.c_str( ), "w" );
    fprintf( fOut, "# %s\n", a_label.c_str( ) );
    a_data.write( fOut, "%14.5e %14.5e\n" );

    fclose( fOut );
}

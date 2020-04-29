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

#include "GIDI_testUtilities.hpp"

static char const *description = "This program prints the multi-group boundaries for each transportable particle in a GNDS file.";

void main2( int argc, char **argv );
void printVector( std::string const &prefix, std::string const &indent, std::vector<double> const &vector );
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

    argvOptions argv_options( "groupBoundaries", description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );

    parseTestOptions.m_askGNDS_File = true;

    parseTestOptions.parse( );

    GIDI::Construction::PhotoMode photo_mode = parseTestOptions.photonMode( GIDI::Construction::PhotoMode::nuclearAndAtomic );
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
    PoPI::Database pops;
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../pops.xml", "../all.map", construction, PoPI::IDs::neutron, "O16" );

    std::cout << stripDirectoryBase( protare->fileName( ), "/GIDI/Test/" ) << std::endl;

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    for( GIDI::Styles::TemperatureInfos::iterator iter = temperatures.begin( ); iter != temperatures.end( ); ++iter ) {
        GIDI::PhysicalQuantity const &temperature = iter->temperature( );

        std::cout << "label = " << iter->heatedMultiGroup( ) << "  temperature = " << temperature.value( ) << std::endl;
    }

    std::string label( temperatures[0].heatedMultiGroup( ) );
    std::cout << "label = " << label << std::endl;
    GIDI::Transporting::MG settings( protare->projectile( ).ID( ), GIDI::Transporting::Mode::multiGroup, GIDI::Transporting::DelayedNeutrons::on );

    try {
        std::vector<double> groupBoundaries = protare->groupBoundaries( settings, temperatures[0], protare->projectile( ).ID( ) );
        std::string prefix( "group boundaries" );
        printVector( prefix, "    ", groupBoundaries ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    GIDI::Styles::HeatedMultiGroup const &style = *(protare->styles( ).get<GIDI::Styles::HeatedMultiGroup>( label ));
    std::cout << style.moniker( ) << std::endl;
    GIDI::Styles::MultiGroup const &multiGroupStyle = style.multiGroup( );
    std::cout << multiGroupStyle.moniker( ) << std::endl;
    label = multiGroupStyle.label( );
    std::cout << "label = " << label << std::endl;
    try {
        std::vector<double> groupBoundaries = protare->groupBoundaries( settings, temperatures[0], protare->projectile( ).ID( ) );
        std::string prefix( "group boundaries" );
        printVector( prefix, "    ", groupBoundaries ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    std::cout << std::endl;
    GIDI::Styles::MultiGroup const &multiGroup = *(protare->multiGroup( temperatures[0].heatedMultiGroup( ) ));
    std::cout << "multiGroup for label " << temperatures[0].heatedMultiGroup( ) << std::endl;
    std::cout << "    maximumLegendreOrder = " << multiGroup.maximumLegendreOrder( ) << std::endl;

    GIDI::Suite const &transportables = multiGroup.transportables( );
    for( std::size_t i1 = 0; i1 < transportables.size( ); ++i1 ) {
        GIDI::Transportable const &transportable = *(transportables.get<GIDI::Transportable const>( i1 ));

        std::cout << "    " << transportable.label( ) << std::endl;
        std::cout << "        pid = " << transportable.pid( ) << std::endl;

        GIDI::Group const &group = transportable.group( );
        std::cout << "        group name = " << group.label( ) << std::endl;

        GIDI::Grid const &grid = group.grid( );
        std::cout << "        grid unit = " << grid.unit( ) << std::endl;
        printVector( "        values", "            ", grid.data( ) );
    }

    delete protare;
}
/*
=========================================================
*/
void printVector( std::string const &prefix, std::string const &indent, std::vector<double> const &vector ) {

    std::cout << prefix << ": size = " << vector.size( ) << std::endl;
    std::cout << indent;
    for( std::string::size_type index = 0; index < vector.size( ); ++index ) std::cout << " " << vector[index];
    std::cout << std::endl;
}

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
#include <set>

#include "GIDI.hpp"

void printVector( std::string const &prefix, std::string const &indent, std::vector<double> const &vector );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    PoPs::Database pops( "../pops.xml" );
    std::string mapFilename( "../all.map" );
    GIDI::Map map( mapFilename, pops );
    std::string projectileID = "n";
    std::string targetID = "O16";
    GIDI::Protare *protare;

    std::cerr << "    " << __FILE__;
    for( int i1 = 1; i1 < argc; i1++ ) std::cerr << " " << argv[i1];
    std::cerr << std::endl;

    if( argc > 1 ) targetID = argv[1];

    try {
        GIDI::Construction::Settings construction( GIDI::Construction::e_all, GIDI::Construction::e_nuclearAndAtomic );
        protare = map.protare( construction, pops, projectileID, targetID ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    std::cout << protare->fileName( ) << std::endl;

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    for( GIDI::Styles::TemperatureInfos::iterator iter = temperatures.begin( ); iter != temperatures.end( ); ++iter ) {
        GIDI::PhysicalQuantity const &temperature = iter->temperature( );

        std::cout << "label = " << iter->heatedMultiGroup( ) << "  temperature = " << temperature.value( ) << std::endl;
    }

    std::string label( temperatures[0].heatedMultiGroup( ) );
    std::cout << "label = " << label << std::endl;
    GIDI::Settings::MG settings( protare->projectile( ).ID( ), label, true );

    try {
        std::vector<double> groupBoundaries = protare->groupBoundaries( settings, projectileID );
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
    settings.label( label );
    try {
        std::vector<double> groupBoundaries = protare->groupBoundaries( settings, projectileID );
        std::string prefix( "group boundaries" );
        printVector( prefix, "    ", groupBoundaries ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    std::cout << std::endl;
    GIDI::Styles::MultiGroup const &multiGroup = *(protare->multiGroup( settings.label( ) ));
    std::cout << "multiGroup for label " << settings.label( ) << std::endl;
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

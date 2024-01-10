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
void printVector( std::string const &prefix, std::string const &indent, nf_Buffer<double> const &vector );
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
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../../../TestData/PoPs/pops.xml", "../all.map", construction, PoPI::IDs::neutron, "O16" );

    std::cout << stripDirectoryBase( protare->fileName( ), "/GIDI/Test/" ) << std::endl;

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    for( GIDI::Styles::TemperatureInfos::iterator iter = temperatures.begin( ); iter != temperatures.end( ); ++iter ) {
        GIDI::PhysicalQuantity const &temperature = iter->temperature( );

        std::cout << "label = " << iter->heatedMultiGroup( ) << "  temperature = " << temperature.value( ) << std::endl;
    }

    std::string label( temperatures[0].heatedMultiGroup( ) );
    std::cout << "label = " << label << std::endl;
    GIDI::Transporting::MG settings( protare->projectile( ).ID( ), GIDI::Transporting::Mode::multiGroup, GIDI::Transporting::DelayedNeutrons::on );

    GIDI::Styles::HeatedMultiGroup const &heatedMultiGroup = *(protare->styles( ).get<GIDI::Styles::HeatedMultiGroup>( label ));
    std::cout << heatedMultiGroup.moniker( ) << std::endl;
    std::cout << "label = " << label << std::endl;

    std::cout << std::endl;

    GIDI::Suite const &transportables = heatedMultiGroup.transportables( );
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
void printVector( std::string const &prefix, std::string const &indent, nf_Buffer<double> const &vector ) {

    std::cout << prefix << ": size = " << vector.size( ) << std::endl;
    std::cout << indent;
    for( std::string::size_type index = 0; index < vector.size( ); ++index ) std::cout << " " << vector[index];
    std::cout << std::endl;
}

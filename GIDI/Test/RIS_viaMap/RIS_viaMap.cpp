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

static char const *description = "Hi";

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

    std::string mapFilename( "../Data/MG_MC/all_maps.map" );
    PoPI::Database pops;

    LUPI::ArgumentParser argumentParser( __FILE__, description );
    LUPI::OptionStore *mapOption = argumentParser.add<LUPI::OptionStore>( "--map", "The path to the map file to use.", 0, -1 );
    LUPI::OptionAppend *popsOption = argumentParser.add<LUPI::OptionAppend>( "--pops", "Optional PoPs files to load.", 0, -1 );
    LUPI::OptionStore *levelOption = argumentParser.add<LUPI::OptionStore>( "--maxLevel", "Level.", 0, -1 );
    LUPI::OptionStore *energyMaxOption = argumentParser.add<LUPI::OptionStore>( "--energyMax", "Level.", 0, -1 );
    LUPI::Positional *seedTargetsArgument = argumentParser.add<LUPI::Positional>( "Targets", "List of seed targets.", 0, -1 );

    argumentParser.parse( argc, argv );

    if( popsOption->values( ).size( ) == 0 ) pops.addFile( "../../../TestData/PoPs/pops.xml", false );
    for( std::size_t index = 0; index < popsOption->values( ).size( ); ++index ) pops.addFile( popsOption->value( index ), false );

    if( mapOption->counts( ) > 0 ) mapFilename = mapOption->value( );
    GIDI::Map::Map map( mapFilename, pops );

    int maxLevel = 99;
    if( levelOption->counts( ) > 0 ) maxLevel = std::stoi( levelOption->value( ) );
    std::cout << "maxLevel = " << maxLevel << std::endl;

    double energyMax = 20;
    if( energyMaxOption->counts( ) > 0 ) energyMax = std::stod( energyMaxOption->value( ) );
    std::cout << "energyMax = " << energyMax << " MeV" << std::endl;

    std::vector<std::string> seedTargets = seedTargetsArgument->values( );
    std::cout << "Seed targets:" << std::endl;
    for( auto iter = seedTargets.begin( ); iter != seedTargets.end( ); ++iter ) std::cout << "    " << (*iter) << std::endl;

    std::cout << "RIS file name " << map.RIS_fileName( ) << std::endl;
    if( !map.RIS_fileExist( ) ) {
        std::cout << "RIS " + map.RIS_fileName( ) << " does not exists." << std::endl; }
    else {
        GIDI::RISI::Projectiles const &projectiles = map.RIS_load( "MeV" );
        std::cout << std::endl;
        projectiles.print( );

        std::vector<std::string> products = projectiles.products( PoPI::IDs::neutron, seedTargets, maxLevel, energyMax );
        std::cout << std::endl << "Products:" << std::endl;
        for( auto product = products.begin( ); product != products.end( ); ++product ) std::cout << "    " << (*product) << std::endl;

        std::string target( "U239" );
        std::cout << target << " replaced with " << map.replacementTarget( pops, PoPI::IDs::neutron, target ) << std::endl;

        target = "U242";
        std::cout << target << " replaced with " << map.replacementTarget( pops, PoPI::IDs::neutron, target ) << std::endl;

        target = "U229";
        std::cout << target << " replaced with " << map.replacementTarget( pops, PoPI::IDs::neutron, target ) << std::endl;
    }
}

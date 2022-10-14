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
    LUPI::OptionAppend *popsOption = argumentParser.add<LUPI::OptionAppend>( "--pops", "Optional PoPs files to load.", 1, -1 );
    LUPI::OptionStore *projectileOption = argumentParser.add<LUPI::OptionStore>( "--projectile", "The projectile for each protare. Default is .", 0, -1 );
    LUPI::OptionStore *levelOption = argumentParser.add<LUPI::OptionStore>( "--maxLevel", "Only recursively inspect target to this level. Default is 99.", 0, -1 );
    LUPI::OptionStore *energyMaxOption = argumentParser.add<LUPI::OptionStore>( "--energyMax", "All reactions with thresholds greater than this value are ignored. Default is 20 MeV.", 0, -1 );
    LUPI::OptionStore *energyUnitOption = argumentParser.add<LUPI::OptionStore>( "--energyUnit", "The energy unit of '--energyMax'. Default is MeV.", 0, -1 );
    LUPI::Positional *mapOption = argumentParser.add<LUPI::Positional>( "map", "The path to the map file to use.", 1, 1 );
    LUPI::Positional *seedTargetsArgument = argumentParser.add<LUPI::Positional>( "Targets", "List of seed targets.", 0, -1 );

    argumentParser.parse( argc, argv );

    for( std::size_t index = 0; index < popsOption->values( ).size( ); ++index ) pops.addFile( popsOption->value( index ), false );

    GIDI::Map::Map map( mapOption->value( 0 ), pops );

    std::string projectile( PoPI::IDs::neutron );
    if( projectileOption->value( ) != "" ) projectile = projectileOption->value( );

    int maxLevel = 99;
    if( levelOption->value( ) != "" ) maxLevel = std::stoi( levelOption->value( ) );
    std::cout << "maxLevel = " << maxLevel << std::endl;

    double energyMax = 20;
    std::string energyUnit( energyUnitOption->value( ) );
    if( energyUnit == "" ) {
        energyUnit = "MeV"; }
    else if( energyUnit != "MeV" ) {
        if( energyUnit != "eV" ) throw LUPI::Exception( "Energy unit must be 'eV' or 'MeV' and not '" + energyUnit + "'." );
        energyMax *= 1e6;
    }
    
    if( energyMaxOption->value( ) != "" ) energyMax = std::stod( energyMaxOption->value( ) );
    std::cout << "energyMax = " << energyMax << " " << energyUnit << std::endl;

    std::vector<std::string> seedTargets = seedTargetsArgument->values( );
    std::cout << "Seed targets:" << std::endl;
    for( auto iter = seedTargets.begin( ); iter != seedTargets.end( ); ++iter ) std::cout << "    " << (*iter) << std::endl;

    GIDI::RISI::Projectiles const &projectiles = map.RIS_load( energyUnit );

    std::vector<std::string> products = projectiles.products( projectile, seedTargets, maxLevel, energyMax );
    std::cout << std::endl << "Products:" << std::endl;
    for( auto product = products.begin( ); product != products.end( ); ++product ) {
        std::string replacementTarget = map.replacementTarget( pops, projectile, *product );
        if( replacementTarget == *product ) replacementTarget = "";
        std::cout << "    " << *product << " " << replacementTarget << std::endl;
    }
}

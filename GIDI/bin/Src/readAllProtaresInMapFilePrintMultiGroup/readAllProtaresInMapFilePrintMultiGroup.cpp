/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <set>
#include <exception>
#include <stdexcept>

#include "GIDI.hpp"
#include "GIDI_testUtilities.hpp"

static char const *description = 
    "Reads evergy 'nth' protare (including TNSL) in the specified map file.\n"
    "In addition to a map file, one or more PoPs files must be specified.\n\n"
    "Example:\n"
    "    readAllProtaresInMapFile.cpp all.map pops.xml metaStable.xml\n\n"
    "The default for 'nth' is 1 and can be set with the '-n' options.";

static int nth = 1;
static int countDown = 1;
static GIDI::Construction::Settings *constructionPtr = nullptr;
static int errCount = 0;
static long numberOfTemperatures = -1;
static bool printTiming = false;
static bool printData = false;
static GIDI::Transporting::DelayedNeutrons includeDelayedNeutrons = GIDI::Transporting::DelayedNeutrons::off;

void main2( int argc, char **argv );
void walk( std::string const &mapFilename, PoPI::Database const &pops );
void readProtare( std::string const &protareFilename, PoPI::Database const &pops, std::vector<std::string> &a_libraries, bool a_targetRequiredInGlobalPoPs );
void printVector2( LUPI::StatusMessageReporting &a_smr, std::string const &a_label, GIDI::Vector const &a_vector );
void printMatrix2( LUPI::StatusMessageReporting &a_smr, std::string const &a_label, GIDI::Matrix const &a_matrix );
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

    if( errCount > 0 ) exit( EXIT_FAILURE );
    exit( EXIT_SUCCESS );
}
/*
=========================================================
*/
void main2( int argc, char **argv ) {

    PoPI::Database pops;
    argvOptions argv_options( "readAllProtaresInMapFile", description );

    argv_options.add( argvOption( "-n",  true, "If present, only every nth protare is read where 'n' is the next argument." ) );
    argv_options.add( argvOption( "-t", false, "Prints read time information for each protare read." ) );
    argv_options.add( argvOption( "--numberOfTemperatures", true, "The number of temperatures whose multi-group data are accessed." ) );
    argv_options.add( argvOption( "--print", false, "Prints multi-group data to the terminal." ) );
    argv_options.add( argvOption( "--lazyParsing", false, "If true, does lazy parsing for all read protares." ) );
    argv_options.add( argvOption( "--delayed", false, "If present, fission delayed neutrons are included with product sampling." ) );

    if( argv_options.m_arguments.size( ) < 2 ) {
        std::cerr << std::endl << "----- Need map file name and at least one pops file -----" << std::endl << std::endl;
        argv_options.help( );
    }

    argv_options.parseArgv( argc, argv );

    if( argv_options.find( "--delayed" )->present( ) ) includeDelayedNeutrons = GIDI::Transporting::DelayedNeutrons::on;
    printData = argv_options.find( "--print" )->m_counter > 0;
    printTiming = argv_options.find( "-t" )->m_counter > 0;

    std::string numberOfTemperaturesString = argv_options.find( "--numberOfTemperatures" )->zeroOrOneOption( argv, "-1" );
    numberOfTemperatures = asLong( numberOfTemperaturesString.c_str( ) );

    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, GIDI::Construction::PhotoMode::nuclearAndAtomic );
    constructionPtr = &construction;
    construction.setUseSystem_strtod( true );
    construction.setLazyParsing( argv_options.find( "--lazyParsing" )->m_counter > 0 );

    nth = argv_options.find( "-n" )->asInt( argv, 1 );
    if( nth < 0 ) nth = 1;

    std::string const &mapFilename( argv[argv_options.m_arguments[0]] );

    for( std::size_t index = 1; index < argv_options.m_arguments.size( ); ++index ) pops.addFile( argv[argv_options.m_arguments[index]], false );

    LUPI::Timer timer;
    walk( mapFilename, pops );
    if( printTiming ) std::cout << std::endl << LUPI::Misc::doubleToString3( "Total CPU %7.3f s", timer.deltaTime( ).CPU_time( ) )
            << LUPI::Misc::doubleToString3( " wall %6.3f s", timer.deltaTime( ).wallTime( ) ) << std::endl;
}
/*
=========================================================
*/
void walk( std::string const &mapFilename, PoPI::Database const &pops ) {

    std::cout << "    " << mapFilename << std::endl;
    GIDI::Map::Map map( mapFilename, pops );

    for( std::size_t i1 = 0; i1 < map.size( ); ++i1 ) {
        GIDI::Map::BaseEntry const *entry = map[i1];

        std::string path = entry->path( GIDI::Map::BaseEntry::PathForm::cumulative );

        if( entry->name( ) == GIDI_importChars ) {
            walk( path, pops ); }
        else if( ( entry->name( ) == GIDI_protareChars ) || ( entry->name( ) == GIDI_TNSLChars ) ) {
            std::vector<std::string> libraries;

            entry->libraries( libraries );
            readProtare( path, pops, libraries, entry->name( ) == GIDI_protareChars ); }
        else {
            std::cerr << "    ERROR: unknown map entry name: " << entry->name( ) << std::endl;
        }
    }
}
/*
=========================================================
*/
void readProtare( std::string const &protareFilename, PoPI::Database const &pops, std::vector<std::string> &a_libraries, bool a_targetRequiredInGlobalPoPs ) {

    --countDown;
    if( countDown != 0 ) return;
    countDown = nth;

    std::string throwMessage;
    GIDI::Protare *protare = nullptr;
    GIDI::ParticleSubstitution particleSubstitution;
    LUPI::StatusMessageReporting smr;

    try {
        std::cout << "        " << protareFilename;

        LUPI::Timer timer;
        protare = new GIDI::ProtareSingle( *constructionPtr, protareFilename, GIDI::FileType::XML, pops, particleSubstitution, a_libraries, 
                GIDI_MapInteractionNuclearChars, a_targetRequiredInGlobalPoPs );
        GIDI::ProtareSingle *protareSingle = protare->protare( 0 );
        if( printData ) std::cout << std::endl;

        std::string projectileID = protare->projectile( ).ID( );

        GIDI::Transporting::MG settings( projectileID, GIDI::Transporting::Mode::multiGroup, includeDelayedNeutrons );

        GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
        long numberOfTemperatures2 = numberOfTemperatures;
        long numberOfTemperatures3 = static_cast<long>( temperatures.size( ) );
        if( numberOfTemperatures2 < 0 ) numberOfTemperatures2 = numberOfTemperatures3;
        if( numberOfTemperatures2 > numberOfTemperatures3 ) numberOfTemperatures2 = numberOfTemperatures3;

        std::vector<double> angularFlux( 1, 1.0 );
        GIDI::Functions::XYs2d *flux2d = new GIDI::Functions::XYs2d( GIDI::Axes( ), ptwXY_interpolationLinLin );

        GIDI::Functions::Legendre1d *flux1d_1 = new GIDI::Functions::Legendre1d( GIDI::Axes( ), 0, protareSingle->projectileEnergyMin( ) );
        flux1d_1->coefficients( ) = angularFlux;
        flux2d->append( flux1d_1 );

        GIDI::Functions::Legendre1d *flux1d_2 = new GIDI::Functions::Legendre1d( GIDI::Axes( ), 0, protareSingle->projectileEnergyMax( ) );
        flux1d_2->coefficients( ) = angularFlux;
        flux2d->append( flux1d_2 );

        GIDI::Functions::XYs3d flux3d( GIDI::Axes( ), ptwXY_interpolationLinLin );
        flux3d.append( flux2d );

        GIDI::Transporting::Particles particles;

        std::string productID( protare->projectile( ).ID( ) );
        GIDI::Transporting::MultiGroup multiGroup( productID, protare->groupBoundaries( settings, temperatures[0], productID ) );
        GIDI::Transporting::Particle projectile( productID, multiGroup, flux3d );
        particles.add( projectile );

        productID = PoPI::IDs::neutron;
        if( projectileID != productID ) {
            GIDI::Transporting::MultiGroup multiGroup2( productID, protare->groupBoundaries( settings, temperatures[0], productID ) );
            GIDI::Transporting::Particle neutron( productID, multiGroup2, flux3d );
            particles.add( neutron );
        }

        particles.process( *protare, temperatures[0].heatedMultiGroup( ) );

        long temperatureIndex = 0;
        for( auto temperatureIter = temperatures.begin( ); temperatureIndex < numberOfTemperatures2; ++temperatureIter, ++temperatureIndex ) {
            printVector2( smr, "Inverse speed", protare->multiGroupInverseSpeed( smr, settings, *temperatureIter ) );
            printVector2( smr, "Cross section", protare->multiGroupCrossSection( smr, settings, *temperatureIter ) );
            printVector2( smr, "Q", protare->multiGroupQ( smr, settings, *temperatureIter, true ) );
            printVector2( smr, "Multiplity", protare->multiGroupMultiplicity( smr, settings, *temperatureIter, projectileID ) );
            printVector2( smr, "Fission neutron multiplity", protare->multiGroupFissionNeutronMultiplicity( smr, settings, *temperatureIter ) );
            printMatrix2( smr, "Projectile l=0 matrix", protare->multiGroupProductMatrix( smr, settings, *temperatureIter, particles, projectileID, 0 ) );
            printMatrix2( smr, "Fission l=0 matrix", protare->multiGroupFissionMatrix( smr, settings, *temperatureIter, particles, 0 ) );
            printVector2( smr, "Energy deposition", protare->multiGroupDepositionEnergy( smr, settings, *temperatureIter, particles ) );
            printVector2( smr, "Momentum deposition", protare->multiGroupDepositionMomentum( smr, settings, *temperatureIter, particles ) );
            printVector2( smr, "Projectile gain", protare->multiGroupGain( smr, settings, *temperatureIter, projectileID ) );
        }

        std::cout << "    lazy parsing done " << protare->numberOfLazyParsingHelperForms( ) << " and replaced " << protare->numberOfLazyParsingHelperFormsReplaced( ) << ":";
        if( printData ) std::cout << std::endl << "    timing: ";
        if( printTiming ) std::cout << LUPI::Misc::doubleToString3( " CPU %6.3f s", timer.deltaTime( ).CPU_time( ) )
                << LUPI::Misc::doubleToString3( " wall %6.3f s", timer.deltaTime( ).wallTime( ) );
        std::cout << std::endl; }
    catch (char const *str) {
        throwMessage = str; }
    catch (std::string str) {
        throwMessage = str; }
    catch (std::exception &exception) {
        throwMessage = exception.what( );
    }

    if( throwMessage != "" ) {
        ++errCount;
        std::cout << "ERROR: throw with message '" << throwMessage << "'" << std::endl;
    }

    delete protare;
}
/*
=========================================================
*/
void printVector2( LUPI::StatusMessageReporting &a_smr, std::string const &a_label, GIDI::Vector const &a_vector ) {

    std::string label = "          " + a_label + "::";
    if( printData ) a_vector.print( label );
    a_smr.clear( ); 
}
/*
=========================================================
*/
void printMatrix2( LUPI::StatusMessageReporting &a_smr, std::string const &a_label, GIDI::Matrix const &a_matrix ) {

    std::string label = "          " + a_label + "::";
    if( printData ) a_matrix.print( label );
    a_smr.clear( ); 
}

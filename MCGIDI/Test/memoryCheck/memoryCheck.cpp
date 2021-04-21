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

#include "MCGIDI.hpp"

#include "GIDI_testUtilities.hpp"
#include "MCGIDI_testUtilities.hpp"

static char const *description = "Loops over energy at the specified temperature, sampling reactions. If projectile is a photon, see options *-pa* and *-pn*.";

void main2( int argc, char **argv );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    try {
        main2( argc, argv );
        exit( EXIT_SUCCESS ); }
    catch (std::exception &exception) {
        std::cerr << exception.what( ) << std::endl; }
    catch (char const *str) {
        std::cout << str << std::endl; }
    catch (std::string &str) {
        std::cout << str << std::endl;
    }

    exit( EXIT_SUCCESS );
}
/*
=========================================================
*/
void main2( int argc, char **argv ) {

    char Str[1024];
    PoPI::Database pops;
    GIDI::Transporting::Particles particles;
    GIDI::Groups groups( "../../../GIDI/Test/groups.xml" );
    GIDI::Fluxes fluxFile( "../../../GIDI/Test/fluxes.xml" );
    std::set<int> reactionsToExclude;
    GIDI::Transporting::Mode transportingMode( GIDI::Transporting::Mode::MonteCarloContinuousEnergy );

    argvOptions argv_options( "memoryCheck", description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );

    argv_options.add( argvOption( "--fixedGrid", false, "Set fixed grid data. Only used if protare is only photo-atomic protare." ) );
    argv_options.add( argvOption( "--multiGroup", false, "Set fixed grid data. Only used if protare is only photo-atomic protare." ) );

    parseTestOptions.parse( );

    if( argv_options.find( "--multiGroup" )->present( ) ) transportingMode = GIDI::Transporting::Mode::multiGroup;

    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, parseTestOptions.photonMode( ) );
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../../../GIDI/Test/pops.xml", "../../../GIDI/Test/all3T.map", construction, PoPI::IDs::neutron, "O16" );

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );

    std::string label( temperatures[0].griddedCrossSection( ) );
    MCGIDI::Transporting::MC MC( pops, protare->projectile( ).ID( ), &protare->styles( ), label, GIDI::Transporting::DelayedNeutrons::on, 20.0 );

    if( argv_options.find( "--fixedGrid" )->present( ) ) {
        MC.fixedGridPoints( groups.get<GIDI::Group>( "LLNL_gid_80" )->data( ) );
    }

    GIDI::Functions::Function3dForm const *fluxes = fluxFile.get<GIDI::Functions::Function3dForm>( "LLNL_fid_1" );
    GIDI::Transporting::Particle neutron( PoPI::IDs::neutron, *groups.get<GIDI::Group>( "LLNL_gid_4" ), *fluxes, transportingMode );
    particles.add( neutron );

    GIDI::Transporting::Particle photon( PoPI::IDs::photon, *groups.get<GIDI::Group>( "LLNL_gid_70" ), *fluxes, transportingMode );
    particles.add( photon );

    MCGIDI::DomainHash domainHash( 4000, 1e-8, 10 );
    MCGIDI::Protare *MCProtare = MCGIDI::protareFromGIDIProtare( *protare, pops, MC, particles, domainHash, temperatures, reactionsToExclude );

    MCGIDI::Vector<MCGIDI::Protare *> protares( 1 );
    protares[0] = MCProtare;
    MCGIDI::URR_protareInfos URR_protare_infos( protares );

    MCGIDI::DataBuffer dataBuffer;

// Count phase
    int protareType = 0;
    if( MCProtare->protareType( ) == MCGIDI::ProtareType::composite ) { protareType = 1; }
    if( MCProtare->protareType( ) == MCGIDI::ProtareType::TNSL ) { protareType = 2; }
    dataBuffer.m_intIndex++;                                                        // Add 1 for storing the protare type
    MCProtare->serialize( dataBuffer, MCGIDI::DataBuffer::Mode::Count );
    dataBuffer.allocateBuffers( );
    dataBuffer.zeroIndexes( );

// Pack phase
    dataBuffer.m_intData[dataBuffer.m_intIndex++] = protareType;                    // Protare type is special
    MCProtare->serialize( dataBuffer, MCGIDI::DataBuffer::Mode::Pack );

// Memory phase
    dataBuffer.m_maxPlacementSize = MCProtare->memorySize( );
    dataBuffer.m_placementStart = reinterpret_cast<char *>( malloc( dataBuffer.m_maxPlacementSize ) );
    dataBuffer.m_placement = dataBuffer.m_placementStart;

    delete MCProtare;

// Unpack phase
    dataBuffer.zeroIndexes( );

    protareType = dataBuffer.m_intData[(dataBuffer.m_intIndex)++];                  // Protare type is special
    if( protareType == 0 ) {
        MCProtare = new(dataBuffer.m_placementStart) MCGIDI::ProtareSingle( );
        dataBuffer.m_placement += sizeof( MCGIDI::ProtareSingle ); }
    else if( protareType == 1 ) {
        MCProtare = new(dataBuffer.m_placementStart) MCGIDI::ProtareComposite( );
        dataBuffer.m_placement += sizeof( MCGIDI::ProtareComposite ); }
    else if( protareType == 2 ) {
        MCProtare = new(dataBuffer.m_placementStart) MCGIDI::ProtareTNSL( );
        dataBuffer.m_placement += sizeof( MCGIDI::ProtareTNSL ); }
    else {
        sprintf( Str, "Bad protare type %d.", protareType );
        throw std::runtime_error( Str );
    }

    MCProtare->serialize( dataBuffer, MCGIDI::DataBuffer::Mode::Unpack );

    long actual = (long) ( dataBuffer.m_placement - dataBuffer.m_placementStart );
    long diff = (long) dataBuffer.m_maxPlacementSize - actual;
    if( diff != 0 ) {                                                               // diff = 0 is success.
        sprintf( Str, "Protare %s + %s, predicted size %ld, actual size %ld, diff %ld.", MCProtare->projectileID( ).c_str( ), 
                MCProtare->targetID( ).c_str( ), (long) dataBuffer.m_maxPlacementSize, actual, diff );
        throw std::runtime_error( Str );
    }

    delete protare;

    free( MCProtare );
}

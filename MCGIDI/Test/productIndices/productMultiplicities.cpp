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
#include <iomanip>
#include <set>

#include "MCGIDI.hpp"

#include "GIDI_testUtilities.hpp"

static char const *description = "Loops over temperature and energy, printing the total cross section. If projectile is a photon, see options *-a* and *-n*.";

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

    PoPI::Database pops;
    argvOptions argv_options( __FILE__, description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );

    parseTestOptions.parse( );

    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, parseTestOptions.photonMode( ) );
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../../../GIDI/Test/pops.xml", "../../../GIDI/Test/Data/MG_MC/all_maps.map",
        construction, PoPI::IDs::neutron, "O16" );

    GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
    std::string label( temperatures[0].heatedCrossSection( ) );
    MCGIDI::Transporting::MC settings( pops, protare->projectile( ).ID( ), &protare->styles( ), label, GIDI::Transporting::DelayedNeutrons::on, 20.0 );

    GIDI::Transporting::Particles particles;

    MCGIDI::DomainHash domainHash( 4000, 1e-8, 10 );
    std::set<int> reactionsToExclude;
    LUPI::StatusMessageReporting smr1;
    MCGIDI::Protare *MCProtare = MCGIDI::protareFromGIDIProtare( smr1, *protare, pops, settings, particles, domainHash, temperatures, reactionsToExclude );

    MCProtare->setUserParticleIndex( pops[PoPI::IDs::neutron], 0 );
    MCProtare->setUserParticleIndex( pops["H2"], 10 );
    MCProtare->setUserParticleIndex( pops[PoPI::IDs::photon], 11 );

    for( std::size_t reactionIndex = 0; reactionIndex < MCProtare->numberOfReactions( ); ++reactionIndex ) {
        MCGIDI::Reaction const *reaction = MCProtare->reaction( reactionIndex );

        std::cout << reaction->label( ).c_str( ) << std::endl;

        auto indices = reaction->productIndices( );
        auto userIndices = reaction->userProductIndices( );
        for( MCGIDI_VectorSizeType productIndex = 0; productIndex < indices.size( ); ++productIndex ) {
            int index = indices[productIndex];

            std::cout << "    " << index << "  " << reaction->productMultiplicity( index ) << " " << userIndices[productIndex] << std::endl;
        }
    }

    delete MCProtare;

    delete protare;
}

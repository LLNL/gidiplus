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
#include <math.h>

#include "GIDI_testUtilities.hpp"

static char const *description = "This program loops over each arrangement of the LLNL 7 transportable particles and prints each reaction where only the list transportable particles are present in the reaction.";
static std::map<std::string,int> particle_GID_map;
static std::map<int,std::string> particle_int_id;

void main2( int argc, char **argv );
void areAllProductsTracked( GIDI::Protare *protare, int mode, GIDI::Transporting::Groups_from_bdfls const &a_groups_from_bdfls,
                GIDI::Transporting::Fluxes_from_bdfls const &a_fluxes_from_bdfls );
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
    argvOptions argv_options( "areAllProductsTracked", description );
    ParseTestOptions parseTestOptions( argv_options, argc, argv );

    parseTestOptions.m_askGNDS_File = true;

    parseTestOptions.parse( );

    GIDI::Construction::PhotoMode photo_mode = parseTestOptions.photonMode( GIDI::Construction::PhotoMode::nuclearAndAtomic );
    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
    PoPI::Database pops;
    GIDI::Protare *protare = parseTestOptions.protare( pops, "../../../TestData/PoPs/pops.xml", "../all.map", construction, PoPI::IDs::neutron, "O16" );

    std::cout << stripDirectoryBase( protare->fileName( ), "/GIDI/Test/" ) << std::endl;

    std::cout << "  List of reactions:" << std::endl;
    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = protare->reaction( index );

        std::cout << "    " << reaction->label( ) << std::endl;
    }
    particle_GID_map["n"] = 4;
    particle_GID_map["H1"] = 71;
    particle_GID_map["H2"] = 71;
    particle_GID_map["H3"] = 71;
    particle_GID_map["He3"] = 71;
    particle_GID_map["He4"] = 71;
    particle_GID_map["photon"] = 70;

    particle_int_id[1] = "n";
    particle_int_id[2] = "H1";
    particle_int_id[3] = "H2";
    particle_int_id[4] = "H3";
    particle_int_id[5] = "He3";
    particle_int_id[6] = "He4";
    particle_int_id[7] = "photon";

    GIDI::Transporting::Groups_from_bdfls groups_from_bdfls( "../bdfls" );
    GIDI::Transporting::Fluxes_from_bdfls fluxes_from_bdfls( "../bdfls", 0 );

    GIDI::Transporting::MG settings( protare->projectile( ).ID( ), GIDI::Transporting::Mode::multiGroup, GIDI::Transporting::DelayedNeutrons::on );

    for( int mode = 0; mode < 128; ++mode ) {
        areAllProductsTracked( protare, mode, groups_from_bdfls, fluxes_from_bdfls );
    }

    delete protare;
}

/*
=========================================================
*/
void areAllProductsTracked( GIDI::Protare *protare, int mode, GIDI::Transporting::Groups_from_bdfls const &a_groups_from_bdfls,
                GIDI::Transporting::Fluxes_from_bdfls const &a_fluxes_from_bdfls ) {

    std::cout << std::endl;
    GIDI::Transporting::Particles particles;

    std::cout << "  Tracked particles:";
    for( int index = 1; index < 8; ++index ) {
        if( mode % 2 != 0 ) {
            std::string id = particle_int_id[index];
            std::cout << " " << id;
            GIDI::Transporting::Particle particle( id, a_groups_from_bdfls.getViaGID( particle_GID_map[id] ) );
            particle.appendFlux( a_fluxes_from_bdfls.getViaFID( 1 ) );
            particles.add( particle );
        }
        mode >>= 1;
    }
    std::cout << std::endl;

    for( std::size_t index = 0; index < protare->numberOfReactions( ); ++index ) {
        GIDI::Reaction const *reaction = protare->reaction( index );

        if( reaction->areAllProductsTracked( particles ) ) std::cout << "    " << reaction->label( ) << std::endl;
    }
}

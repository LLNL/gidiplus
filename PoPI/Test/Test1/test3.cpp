/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <iostream>

#include "PoPI.hpp"

void printParticleInfo( PoPI::Database const &database, std::string const &ID, bool isNeutronProtonANucleon = false );
std::string check3Ints( int i1, int i2, int i3 );
/*
=========================================================
*/
int main( LUPI_maybeUnused int argc, LUPI_maybeUnused char **argv ) {

    std::cerr << "    " << LUPI::FileInfo::basenameWithoutExtension( __FILE__ ) << std::endl;

    std::string fileName( "../../../TestData/PoPs/pops.xml" );

    try {
        PoPI::Database database( fileName );

        printParticleInfo( database, "photon" );
        printParticleInfo( database, PoPI::IDs::photon );

        printParticleInfo( database, "n", true );
        printParticleInfo( database, PoPI::IDs::neutron );

        printParticleInfo( database, "p", true );
        printParticleInfo( database, PoPI::IDs::proton );

        printParticleInfo( database, "H1" );
        printParticleInfo( database, "O16" );
        printParticleInfo( database, "Fe54" );
        printParticleInfo( database, "Pu236" ); }
    catch (char const *str) {
        std::cout << str << std::endl;
    }
}
/*
=========================================================
*/
void printParticleInfo( PoPI::Database const &database, std::string const &ID, bool isNeutronProtonANucleon ) {

    PoPI::Particle const &particle = database.get<PoPI::Particle>( ID );
    int Z = PoPI::particleZ( particle, isNeutronProtonANucleon );
    int A = PoPI::particleA( particle, isNeutronProtonANucleon );
    int ZA = PoPI::particleZA( particle, isNeutronProtonANucleon );
    std::cout << ID << " -> " << particle.ID( ) << "  Z = " << Z << " A = " << A << 
        "  ZA = " << ZA << std::endl;

    int Z2 = PoPI::particleZ( database, ID, isNeutronProtonANucleon );
    int Z3 = PoPI::particleZ( database, particle.index( ), isNeutronProtonANucleon );
    std::string message( check3Ints( Z, Z2, Z3 ) );
    std::cout << "   Z test: " << Z << " " << Z2 << " " << Z3 << message << std::endl;

    int A2 = PoPI::particleA( database, ID, isNeutronProtonANucleon );
    int A3 = PoPI::particleA( database, particle.index( ), isNeutronProtonANucleon );
    message = check3Ints( A, A2, A3 );
    std::cout << "   A test: " << A << " " << A2 << " " << A3 << message << std::endl;

    int ZA2 = PoPI::particleZA( database, ID, isNeutronProtonANucleon );
    int ZA3 = PoPI::particleZA( database, particle.index( ), isNeutronProtonANucleon );
    message = check3Ints( ZA, ZA2, ZA3 );
    std::cout << "   ZA test: " << ZA << " " << ZA2 << " " << ZA3 << message << std::endl;
}
/*
=========================================================
*/
std::string check3Ints( int i1, int i2, int i3 ) {

    std::string message;

    if( ( i1 == i2 ) and ( i1 == i3 ) ) {
        }
    else {
        message = LUPI::Misc::argumentsToString( ": check3Ints FAILURE; %4d, %4d, %4d", i1, i2, i3 );
    }
    return( message );
}

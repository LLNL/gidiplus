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

#include "GIDI.hpp"

#include "GIDI_testUtilities.hpp"

void printList( char const *prefix, std::vector<std::string> &list );
bool mapWalkCallBack( GIDI::ProtareBaseEntry const *a_protareEntry, void *a_data, int a_level );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    PoPs::Database pops( "../pops.xml" );
    std::string fileName( "all.map" );
    GIDI::Map *map;
    GIDI::MapBaseEntry::pathForm entered = GIDI::MapBaseEntry::e_entered;
    GIDI::MapBaseEntry::pathForm cumulative = GIDI::MapBaseEntry::e_cumulative;
    GIDI::MapBaseEntry::pathForm l_real = GIDI::MapBaseEntry::e_real;

    std::cerr << "    " << __FILE__;
    for( int i1 = 1; i1 < argc; i1++ ) std::cerr << " " << argv[i1];
    std::cerr << std::endl;

    if( argc > 1 ) fileName = argv[1];

    try {
        map = new GIDI::Map( fileName, pops ); }
    catch ( char const *str ) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }
    catch ( std::string str ) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    std::cout << std::endl;

    std::cout << "library = " << map->library( ) << std::endl;

    std::cout << "isProtareAvailable( PoPs::IDs::neutron, H1 ) " << map->isProtareAvailable( PoPs::IDs::neutron, "H1" ) << std::endl;
    std::cout << "isProtareAvailable( PoPs::IDs::neutron, H2 ) " << map->isProtareAvailable( PoPs::IDs::neutron, "H2" ) << std::endl;
    std::cout << "isProtareAvailable( PoPs::IDs::neutron, H2 ) " << map->isProtareAvailable( PoPs::IDs::neutron, "H3" ) << std::endl;

    std::cout << "isProtareAvailable( PoPs::IDs::neutron, H1, 'ENDF/BVII.1' ) " << map->isProtareAvailable( PoPs::IDs::neutron, "H1", "ENDF/BVII.1" ) << std::endl;
    std::cout << "isProtareAvailable( PoPs::IDs::neutron, H1, 'ENDF/BVII.2' ) " << map->isProtareAvailable( PoPs::IDs::neutron, "H1", "ENDF/BVII.2" ) << std::endl;
    std::cout << "isProtareAvailable( PoPs::IDs::neutron, H1, 'ENDF/BVIII.1' ) " << map->isProtareAvailable( PoPs::IDs::neutron, "H1", "ENDF/BVIII.1" ) << std::endl;

    std::cout << "protareFilename( PoPs::IDs::neutron, H1 ) = <" << 
            stripDirectoryBase( map->protareFilename( PoPs::IDs::neutron, "H1" ) ) << ">" << std::endl;
    std::cout << "protareFilename( PoPs::IDs::neutron, H1 ) = <" << map->protareFilename( PoPs::IDs::neutron, "H1", "", entered ) << ">" << std::endl;
    std::cout << "protareFilename( PoPs::IDs::neutron, H1 ) = <" << 
            stripDirectoryBase( map->protareFilename( PoPs::IDs::neutron, "H1", "", cumulative ) ) << ">" << std::endl;
    std::cout << "protareFilename( PoPs::IDs::neutron, H1 ) = <" << 
            stripDirectoryBase( map->protareFilename( PoPs::IDs::neutron, "H1", "", l_real ) ) << ">" << std::endl;
    std::cout << "protareFilename( PoPs::IDs::neutron, H1, 'ENDF/BVII.1' ) = <" << map->protareFilename( PoPs::IDs::neutron, "H1", "ENDF/BVII.1", entered ) << ">" << std::endl;
    std::cout << "protareFilename( PoPs::IDs::neutron, H1, 'ENDF/BVII.2' ) = <" << map->protareFilename( PoPs::IDs::neutron, "H1", "ENDF/BVII.2", cumulative ) << ">" << std::endl;
    std::cout << "protareFilename( PoPs::IDs::neutron, H1, 'ENDF/BVIII.1' ) = <" << 
            stripDirectoryBase( map->protareFilename( PoPs::IDs::neutron, "H1", "ENDF/BVIII.1", l_real ) ) << ">" << std::endl;

    std::vector<std::string> list = map->availableEvaluations( PoPs::IDs::neutron, "H1" );
    printList( "availableEvaluations( PoPs::IDs::neutron, H1 ) =", list );

    list = map->availableEvaluations( PoPs::IDs::neutron, "H2" );
    printList( "availableEvaluations( PoPs::IDs::neutron, H2 ) =", list );

    list = map->availableEvaluations( PoPs::IDs::neutron, "O16" );
    printList( "availableEvaluations( PoPs::IDs::neutron, O16 ) =", list );

    list = map->availableEvaluations( PoPs::IDs::neutron, "O17" );
    printList( "availableEvaluations( PoPs::IDs::neutron, O17 ) =", list );

    std::cout << std::endl;
    map->walk( mapWalkCallBack, map );

    std::cout << std::endl;
    PoPs::Particle const &O16 = pops.get<PoPs::Particle>( "O16" );
    PoPs::PhysicalQuantity const *pMass = O16.mass( )[0];
    PoPs::PQ_double const &mass = dynamic_cast<PoPs::PQ_double const &>( *pMass );
    std::cout << "O16 mass = " << mass.value( ) << " " << mass.unit( ) << std::endl;

    delete map;
}
/*
=========================================================
*/
void printList( char const *prefix, std::vector<std::string> &list ) {

    std::string sep( "" );

    std::cout << prefix << " <";
    for( std::vector<std::string>::const_iterator iter = list.begin( ); iter != list.end( ); ++iter ) {
        std::cout << sep << "'" << *iter << "'";
        sep = ", ";
    }
    std::cout << ">" << std::endl;
}
/*
=========================================================
*/
bool mapWalkCallBack( GIDI::ProtareBaseEntry const *a_protareEntry, void *a_data, int a_level ) {

    GIDI::Map const *map = (GIDI::Map const *) a_data;
    std::string path = stripDirectoryBase( a_protareEntry->path( ) );

    std::cout << path << std::endl;

    GIDI::ProtareBaseEntry const *protareEntry = map->findProtareEntry( a_protareEntry->projectileID( ), a_protareEntry->targetID( ), a_protareEntry->evaluation( ) );
    if( protareEntry == NULL ) {
        std::cout << "Oops" << std::endl; }
    else {
        std::cout << "    " << protareEntry->parent( )->library( ) << "    " << protareEntry->evaluation( ) << std::endl;
    }

    return( true );
}

/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

/*
    Code to read in a file containing a groups node.
*/

#include <stdlib.h>
#include <math.h>
#include <iostream>

#include "GIDI_testUtilities.hpp"

void compare( std::string const &a_msg, GIDI::Group &a_group, GIDI::Transporting::MultiGroup &a_boundaries );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    printCodeArguments( __FILE__, argc, argv );

    GIDI::Transporting::Groups_from_bdfls bdfls_groups( "../bdfls" );
    std::vector<std::string> labels = bdfls_groups.labels( );

    GIDI::Groups groups( "../groups.xml" );

    std::cout << groups.size( ) << std::endl;

    for( std::size_t i1 = 0; i1 < groups.size( ); ++i1 ) {
        GIDI::Group *group = (GIDI::Group *) groups.get<GIDI::Group>( i1 );
        GIDI::Transporting::MultiGroup boundaries = bdfls_groups.viaLabel( group->label( ) );

        compare( "original", *group, boundaries );

        GIDI::Transporting::MultiGroup boundaries2( *group );
        compare( "converted", *group, boundaries2 );
        std::cout << std::endl;
    }
}
/*
=========================================================
*/
void compare( std::string const &a_msg, GIDI::Group &a_group, GIDI::Transporting::MultiGroup &a_boundaries ) {

    std::cout << a_msg << " " << a_group.label( ) << " " << a_group.size( ) << " " << a_boundaries.size( ) << std::endl;

    if( a_group.size( ) == a_boundaries.size( ) ) {
        for( std::size_t i1 = 0; i1 < a_group.size( ); ++i1 ) {
            double diff = a_group[i1] - a_boundaries[i1];
            if( fabs( diff ) > 1e-15 * ( a_group[i1] + a_boundaries[i1] ) ) std::cout << "        " << i1 << " " << a_group[i1] << " " << a_boundaries[i1] << " " << diff << std::endl;
        }
    }
}

/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>
#include <stdexcept>

#include "PoPI.hpp"

namespace PoPI {

// FIXME - Must be removed once unit conversion is supported.
#define AMU2MeV 931.494028

/*
=========================================================
*/
Nuclide::Nuclide( HAPI::Node const &a_node, Database *a_DB, Isotope *a_isotope ) :
        Particle( a_node, Particle_class::nuclide, PoPI_nuclideChars, -1 ),
        m_isotope( a_isotope ),
        m_nucleus( a_node.child( PoPI_nucleusChars ), a_DB, this ) {

    addToDatabase( a_DB );
}
/*
=========================================================
*/
Nuclide::~Nuclide( ) {

}
/*
=========================================================
*/
int Nuclide::Z( void ) const {

    return( m_isotope->Z( ) );
}
/*
=========================================================
*/
int Nuclide::A( void ) const {

    return( m_isotope->A( ) );
}
/*
=========================================================
*/
std::string const &Nuclide::atomsID( void ) const {

    return( m_isotope->symbol( ) );
}
/*
=========================================================
*/
PQ_suite const &Nuclide::baseMass( void ) const {

    return( (*m_isotope).nuclides( )[0].mass( ) );
}
/*
=========================================================
*/
double Nuclide::massValue( char const *a_unit ) const {

    std::string unit_c2( a_unit );
    unit_c2 += " * c**2";
    PQ_double const *pq_mass;

    if( mass( ).size( ) > 0 ) {
        pq_mass = dynamic_cast<PQ_double const *>( mass( )[0] ); }
    else {
        if( baseMass( ).size( ) == 0 ) throw Exception( "nuclide::massValue: no mass in level 0." );
        pq_mass = dynamic_cast<PQ_double const *>( baseMass( )[0] );
    }
    double _mass = pq_mass->value( a_unit );

    double v_levelEnergy = levelEnergy( unit_c2 ) / AMU2MeV;

    return( _mass + v_levelEnergy );
}
/*
=========================================================
*/
void Nuclide::calculateNuclideGammaBranchStateInfos( PoPI::Database const &a_pops, NuclideGammaBranchStateInfos &a_nuclideGammaBranchStateInfos ) const {

    NuclideGammaBranchStateInfo *nuclideGammaBranchStateInfo = new NuclideGammaBranchStateInfo( ID( ) );

    decayData( ).calculateNuclideGammaBranchStateInfo( a_pops, *nuclideGammaBranchStateInfo );    

    if( nuclideGammaBranchStateInfo->branches( ).size( ) > 0 ) {
        a_nuclideGammaBranchStateInfos.add( nuclideGammaBranchStateInfo ); }
    else {
        delete nuclideGammaBranchStateInfo;
    }
}
/*
=========================================================
*/
void Nuclide::toXMLListExtraElements( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const {

    m_nucleus.toXMLList( a_XMLList, a_indent1 );
}

}

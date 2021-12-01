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

#define PoPI_energyChars "energy"

/*
=========================================================
*/
Nucleus::Nucleus( HAPI::Node const &a_node, Database *a_DB, Nuclide *a_nuclide ) :
        Particle( a_node, Particle_class::nucleus, PoPI_nucleusChars, -1 ),
        m_nuclide( a_nuclide ),
        m_Z( a_nuclide->Z( ) ),
        m_A( a_nuclide->A( ) ),
        m_levelName( a_node.attribute( PoPI_indexChars ).value( ) ),                // The string version of m_levelIndex.
        m_levelIndex( a_node.attribute( PoPI_indexChars ).as_int( ) ),              // The int version of m_levelName.
        m_energy( a_node.child( PoPI_energyChars ) ) {

    if( a_node.empty( ) ) throw Exception( "nuclide is missing nuclues" );

    addToDatabase( a_DB );
}
/*
=========================================================
*/
Nucleus::~Nucleus( ) {

}
/*
=========================================================
*/
std::string const &Nucleus::atomsID( void ) const {

    return( m_nuclide->atomsID( ) );
}
/*
=========================================================
*/
double Nucleus::massValue( char const *a_unit ) const {

// FIXME: still need to correct for electron masses and binding energy.
    return( m_nuclide->massValue( a_unit ) );
}
/*
=========================================================
*/
double Nucleus::energy( std::string const &a_unit ) const {

    PQ_double *pq = dynamic_cast<PQ_double *>( m_energy[0] );
    if( pq->unit( ) == "eV" ) return( pq->value( ) * 1e-6 );        // Kludge until units are functional.
    return( pq->value( a_unit ) );
}
/*
=========================================================
*/
std::string Nucleus::toXMLListExtraAttributes( void ) const {

    return( std::string( " index=\"" + m_levelName + "\"" ) );
}
/*
=========================================================
*/
void Nucleus::toXMLListExtraElements( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const {

    m_energy.toXMLList( a_XMLList, a_indent1 );
}

}

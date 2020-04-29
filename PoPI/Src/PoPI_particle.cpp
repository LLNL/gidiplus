/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "PoPI.hpp"

namespace PoPI {

/*
============================================================
========================= Particle =========================
============================================================
*/


/*
=========================================================
*/
Particle::Particle( pugi::xml_node const &a_node, Particle_class a_class, std::string const &a_family, int a_hasNucleus ) :
        IDBase( a_node, a_class ),
        m_family( a_family ),
        m_hasNucleus( a_hasNucleus ),
        m_mass( a_node.child( "mass" ) ),
        m_spin( a_node.child( "spin" ) ),
        m_parity( a_node.child( "parity" ) ),
        m_charge( a_node.child( "charge" ) ),
        m_halflife( a_node.child( "halflife" ) ),
        m_decayData( a_node.child( "decayData" ) ) {

}
/*
=========================================================
*/
Particle::~Particle( ) {

}
/*
=========================================================
*/
double Particle::massValue( char const *a_unit ) const {

    if( m_mass.size( ) == 0 ) throw Exception( "Particle does not have any mass data." );

    PQ_double const *pq_mass = dynamic_cast<PQ_double const *>( mass( )[0] );

    if( pq_mass == NULL ) throw Exception( "Particle does not have a PoPI::PQ_double mass." );
    return( pq_mass->value( a_unit ) );
}
/*
=========================================================
*/
void Particle::toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const {

    std::string indent2 = a_indent1 + "  ";

    std::string header = a_indent1 + "<" + family( ) + " id=\"" + ID( ) + "\"" + toXMLListExtraAttributes( ) + ">";
    a_XMLList.push_back( header );

    m_mass.toXMLList( a_XMLList, indent2 );
    m_spin.toXMLList( a_XMLList, indent2 );
    m_parity.toXMLList( a_XMLList, indent2 );
    m_charge.toXMLList( a_XMLList, indent2 );
    m_halflife.toXMLList( a_XMLList, indent2 );
    toXMLListExtraElements( a_XMLList, indent2 );
    m_decayData.toXMLList( a_XMLList, indent2 );

    appendXMLEnd( a_XMLList, family( ) );
}
/*
=========================================================
*/
std::string Particle::toXMLListExtraAttributes( void ) const {

    return( "" );
}
/*
=========================================================
*/
void Particle::toXMLListExtraElements( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const {

    return;
}

}

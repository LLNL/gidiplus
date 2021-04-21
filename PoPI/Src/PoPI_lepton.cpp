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

#define PoPI_generationChars "generation"

/*
=========================================================
*/
Lepton::Lepton( pugi::xml_node const &a_node, Database *a_DB, Database *a_parent ) :
        Particle( a_node, Particle_class::lepton, PoPI_leptonChars ),
        m_generation( a_node.attribute( PoPI_generationChars ).value( ) ) {

    addToDatabase( a_DB );
}
/*
=========================================================
*/
Lepton::~Lepton( ) {

}
/*
=========================================================
*/
std::string Lepton::toXMLListExtraAttributes( void ) const {

    return( std::string( " generation=\"" + m_generation + "\"" ) );
}

}

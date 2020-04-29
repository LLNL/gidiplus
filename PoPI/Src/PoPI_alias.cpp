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
=========================================================
*/
Alias::Alias( pugi::xml_node const &a_node, Database *a_DB, Particle_class a_class ) :
        IDBase( a_node, a_class ),
        m_pid( a_node.attribute( "pid" ).value( ) ),
        m_pidIndex( -1 ) {

    if( a_class == Particle_class::alias ) addToDatabase( a_DB );
}
/*
=========================================================
*/
Alias::~Alias( ) {

}
/*
=========================================================
*/
void Alias::toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const {

    std::string header = a_indent1 + "<particle id=\"" + ID( ) + "\" pid=\"" + m_pid + "\"/>";
    a_XMLList.push_back( header );
}

/*
=========================================================
*/
MetaStable::MetaStable( pugi::xml_node const &a_node, Database *a_DB ) :
        Alias( a_node, a_DB, Particle_class::metaStable ),
        m_metaStableIndex( a_node.attribute( "metaStableIndex" ).as_int( ) ) {

    addToDatabase( a_DB );
}
/*
=========================================================
*/
MetaStable::~MetaStable( ) {

}
/*
=========================================================
*/
void MetaStable::toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const {

    char indexStr[18];

    sprintf( indexStr, "%d", m_metaStableIndex );
    std::string header = a_indent1 + "<metaStable id=\"" + ID( ) + "\" pid=\"" + pid( ) + "\" metaStableIndex=\"" + indexStr + "\"/>";
    a_XMLList.push_back( header );
}

}

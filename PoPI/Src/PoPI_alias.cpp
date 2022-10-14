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

#define PoPI_metaStableIndexChars "metaStableIndex"

/*! \class Alias
 * This class represents a **PoPs** alias instance.
 */

/* *********************************************************************************************************//**
 * Constructor that parses an **HAPI** instance to create a **GNDS** alias node.
 *
 * @param a_node            [in]    The **HAPI::Node** to be parsed.
 * @param a_DB              [in]    The **PoPI::Database:: instance to add the constructed **Alias** to.
 * @param a_class           [in]    The particle class for **Alias**.
 ***********************************************************************************************************/

Alias::Alias( HAPI::Node const &a_node, Database *a_DB, Particle_class a_class ) :
        IDBase( a_node, a_class ),
        m_pid( a_node.attribute( PoPI_pidChars ).value( ) ),
        m_pidIndex( -1 ) {

    if( a_class == Particle_class::alias ) addToDatabase( a_DB );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Alias::~Alias( ) {

}

/* *********************************************************************************************************//**
 * This method uses *a_DB* to get the final particle and returns its mass in units of *a_unit*.
 *
 * @param       a_DB        [in]    The PoPs database to look up the final particle.
 * @param       a_unit      [in]    The unit of the returned mass.
 *
 * @return                          The mass in units of *a_unit*.
 ***********************************************************************************************************/

double Alias::massValue2( Database const &a_DB, std::string const &a_unit ) const {

    Particle const &particle = a_DB.particle( a_DB.final( m_pid ) );

    return( particle.massValue( a_unit ) );
}

/* *********************************************************************************************************//**
 * Adds the contents of *this* to *a_XMLList* where each item in *a_XMLList* is one line (without linefeeds) to output as an XML representation of *this*.
 *
 * @param a_XMLList                     [in]    The list to add an XML output representation of *this* to.
 * @param a_indent1                     [in]    The amount of indentation to added to each line added to *a_XMLList*.
 ***********************************************************************************************************/

void Alias::toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const {

    std::string header = a_indent1 + "<particle id=\"" + ID( ) + "\" pid=\"" + m_pid + "\"/>";
    a_XMLList.push_back( header );
}

/*! \class MetaStable
 * This class represents **PoPs** metaStable instance.
 */

/* *********************************************************************************************************//**
 * Constructor that parses an **HAPI** instance to create a **GNDS** metastable alias node.
 *
 * @param a_node            [in]    The **HAPI::Node** to be parsed.
 * @param a_DB              [in]    The **PoPI::Database:: instance to add the constructed **MetaStable** to.
 ***********************************************************************************************************/

MetaStable::MetaStable( HAPI::Node const &a_node, Database *a_DB ) :
        Alias( a_node, a_DB, Particle_class::metaStable ),
        m_metaStableIndex( a_node.attribute( PoPI_metaStableIndexChars ).as_int( ) ) {

    addToDatabase( a_DB );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

MetaStable::~MetaStable( ) {

}

/* *********************************************************************************************************//**
 * Adds the contents of *this* to *a_XMLList* where each item in *a_XMLList* is one line (without linefeeds) to output as an XML representation of *this*.
 *
 * @param a_XMLList                     [in]    The list to add an XML output representation of *this* to.
 * @param a_indent1                     [in]    The amount of indentation to added to each line added to *a_XMLList*.
 ***********************************************************************************************************/

void MetaStable::toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const {

    char indexStr[18];

    sprintf( indexStr, "%d", m_metaStableIndex );
    std::string header = a_indent1 + "<metaStable id=\"" + ID( ) + "\" pid=\"" + pid( ) + "\" metaStableIndex=\"" + indexStr + "\"/>";
    a_XMLList.push_back( header );
}

}

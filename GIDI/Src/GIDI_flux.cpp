/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "GIDI.hpp"

namespace GIDI {

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options for parsing.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed and used to construct the Protare.
 ***********************************************************************************************************/

Flux::Flux( Construction::Settings const &a_construction, pugi::xml_node const &a_node ) :
        Form( a_node, FormType::flux ),
        m_flux( data2dParse( a_construction, a_node.first_child( ), NULL ) ) {

    if( m_flux != NULL ) m_flux->setAncestor( this );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Flux::~Flux( ) {

    delete m_flux;
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void Flux::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string attributes = a_writeInfo.addAttribute( "label", label( ) );

    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );
    if( m_flux != NULL ) m_flux->toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class Fluxes
 * Class for the GNDS <**fluxes**> node that contains a list of flux nodes each as a 3d function.
 */

/* *********************************************************************************************************//**
 * @param a_fileName            [in]    File containing a fluxes node to be parsed.
 ***********************************************************************************************************/

Fluxes::Fluxes( ) :
        Suite( fluxesMoniker ) {

}

/* *********************************************************************************************************//**
 * @param a_fileName            [in]    File containing a fluxes node to be parsed.
 ***********************************************************************************************************/

Fluxes::Fluxes( std::string const &a_fileName ) :
        Suite( fluxesMoniker ) {

    addFile( a_fileName );
}

/* *********************************************************************************************************//**
* Adds the contents of the specified file to *this*.
 *
 ***********************************************************************************************************/

void Fluxes::addFile( std::string const &a_fileName ) {

    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file( a_fileName.c_str( ) );
    if( result.status != pugi::status_ok ) throw Exception( result.description( ) );

    pugi::xml_node fluxes = doc.first_child( );

    std::string name( fluxes.name( ) );
    Construction::Settings construction( Construction::ParseMode::all, GIDI::Construction::PhotoMode::atomicOnly );
    PoPI::Database pops;

    for( pugi::xml_node child = fluxes.first_child( ); child; child = child.next_sibling( ) ) {
        Functions::Function3dForm *function3d = data3dParse( construction, child, NULL );

        add( function3d );
    }
}

}

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
        Form( a_node, f_flux ),
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

}

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

/*! \class Transportable
 * Class for the GNDS <**transportable**> node that resides under the <**transportables**> node.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed to construct a Transportable instance.
 * @param a_pops            [in]    A PoPs::Database instance used to get particle indices and possibly other particle information.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

Transportable::Transportable( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, Suite *a_parent ) :
        Form( a_node, f_transportable, a_parent ),
        m_conserve( a_node.attribute( "conserve" ).value( ) ),
        m_group( a_construction, a_node.child( groupMoniker ), a_pops ) {
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void Transportable::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string attributes = a_writeInfo.addAttribute( "label", label( ) );

    attributes += a_writeInfo.addAttribute( "conserve", m_conserve );
    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );
    m_group.toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( moniker( ) );
}

}

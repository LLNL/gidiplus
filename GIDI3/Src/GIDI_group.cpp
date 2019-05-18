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

/*! \class Group
 * Class for the GNDS <**group**> node that resides under the <**transportable**> node.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed to construct a Group instance.
 * @param a_pops            [in]    A PoPs::Database instance used to get particle indices and possibly other particle information.
 ***********************************************************************************************************/

Group::Group( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops ) :
        Form( a_node, f_group ),
        m_grid( a_node.child( "grid" ), a_construction.useSystem_strtod( ) ) {

}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void Group::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );

    a_writeInfo.addNodeStarter( a_indent, moniker( ), a_writeInfo.addAttribute( "label", label( ) ) );
    m_grid.toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( moniker( ) );
}

}

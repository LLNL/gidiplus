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

namespace Functions {

/*! \class Gridded3d
 * Class for the GNDS <**gridded3d**> node.
 */

/* *********************************************************************************************************//**
 * Constructed from data in a <**product**> node.
 *
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed and used to construct the Gridded3d.
 ***********************************************************************************************************/

Gridded3d::Gridded3d( Construction::Settings const &a_construction, pugi::xml_node const &a_node ) :
        Function3dForm( a_construction, a_node, FormType::gridded3d ),
        m_data( a_node.child( "array" ), a_construction.useSystem_strtod( ) ) {

    m_domain3Unit = axes( )[3]->unit( );
    m_domain2Unit = axes( )[2]->unit( );
    m_domain1Unit = axes( )[1]->unit( );
    m_rangeUnit = axes( )[0]->unit( );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Gridded3d::~Gridded3d( ) {

}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void Gridded3d::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );

    a_writeInfo.addNodeStarter( a_indent, moniker( ) );
    axes( ).toXMLList( a_writeInfo, indent2 );
    m_data.toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( moniker( ) );
}

}               // End namespace Functions.

}               // End namespace GIDI.

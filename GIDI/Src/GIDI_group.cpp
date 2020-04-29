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
 * @param a_pops            [in]    A PoPI::Database instance used to get particle indices and possibly other particle information.
 ***********************************************************************************************************/

Group::Group( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPI::Database const &a_pops ) :
        Form( a_node, FormType::group ),
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

/*! \class Groups
 * Class for the GNDS <**groups**> node that contains a list of flux nodes each as a 3d function.
 */

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Groups::Groups( ) :
        Suite( groupsMoniker ) {

}

/* *********************************************************************************************************//**
 * @param a_fileName            [in]    File containing a groups node to be parsed.
 ***********************************************************************************************************/

Groups::Groups( std::string const &a_fileName ) :
        Suite( groupsMoniker ) {

    addFile( a_fileName );
}

/* *********************************************************************************************************//**
 * Adds the contents of the specified file to *this*.
 *
 * @param a_fileName            [in]    File containing a groups node to be parsed.
 ***********************************************************************************************************/

void Groups::addFile( std::string const &a_fileName ) {

    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file( a_fileName.c_str( ) );
    if( result.status != pugi::status_ok ) throw Exception( result.description( ) );

    pugi::xml_node groups = doc.first_child( );

    std::string name( groups.name( ) );
    if( name != groupsMoniker ) throw Exception( "Invalid groups node file: file node name is '" + name + "'." );

    Construction::Settings construction( Construction::ParseMode::all, GIDI::Construction::PhotoMode::atomicOnly );
    PoPI::Database pops;

    for( pugi::xml_node child = groups.first_child( ); child; child = child.next_sibling( ) ) {
        Group *group = new Group( construction, child, pops );

        add( group );
    }
}

}

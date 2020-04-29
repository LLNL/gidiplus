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

/*! \class Ancestry
 * This is a base class inherit by most other GIDI classes. It allows one to construct a node's *xlink* or get another
 * node from its *xlink*.
 */

/* *********************************************************************************************************//**
 * @param a_moniker             [in]    The **GNDS** node's name (i.e., moniker).
 * @param a_attribute           [in]    Currently not used.
 ***********************************************************************************************************/

Ancestry::Ancestry( std::string const &a_moniker, std::string const &a_attribute ) :
        m_moniker( a_moniker ),
        m_ancestor( NULL ),
        m_attribute( a_attribute ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Ancestry::~Ancestry( ) {

}

/* *********************************************************************************************************//**
 * Returns the root node, ascending all parent nodes until one is found without an ancester. That node is returned.
 *
 * @return                              Returns the root node (i.e., the top level node).
 ***********************************************************************************************************/

Ancestry *Ancestry::root( ) {

    Ancestry *_root = this;

    while( _root->m_ancestor != NULL ) _root = _root->m_ancestor;
    return( _root );
}

/* *********************************************************************************************************//**
 * Returns the root node, ascending all parent nodes until one is found without an ancester. That node is returned.
 *
 * @return                              Returns the root node (i.e., the top level node).
 ***********************************************************************************************************/

Ancestry const *Ancestry::root( ) const {

    Ancestry const *_root = this;

    while( _root->m_ancestor != NULL ) _root = _root->m_ancestor;
    return( _root );
}

/* *********************************************************************************************************//**
 * Returns a pointer to the node whose *xlink* (i.e., *a_href*) is *a_href*.
 *
 * @param a_href                [in]    The *xlink* whose node is to be returned.
 * @return                              Returns the root node (i.e., the top level node).
 ***********************************************************************************************************/

Ancestry *Ancestry::findInAncestry( std::string const &a_href ) {

    std::vector<std::string> segments = splitString( a_href, '/' );

    return( findInAncestry2( 0, segments ) );
}

/* *********************************************************************************************************//**
 * Returns a pointer to the node whose *xlink* (i.e., *a_href*) is *a_href*.
 *
 * @param a_href                [in]    The *xlink* whose node is to be returned.
 * @return                              Returns the root node (i.e., the top level node).
 ***********************************************************************************************************/

Ancestry const *Ancestry::findInAncestry( std::string const &a_href ) const {

    std::vector<std::string> segments = splitString( a_href, '/' );

    return( findInAncestry2( 0, segments ) );
}

/* *********************************************************************************************************//**
 * Returns a pointer to the node whose *xlink* is defined by the *a_segments* argument. The *a_segments* is the *xlink*
 * divided into segments separated by the '/' character.
 *
 * @param a_index               [in]    An index into the *a_segments* whose segment is to be found at this level.
 * @param a_segments            [in]    The list of *xlink* segments.
 * @return                              Returns the root node (i.e., the top level node).
 ***********************************************************************************************************/

Ancestry *Ancestry::findInAncestry2( std::size_t a_index, std::vector<std::string> const &a_segments ) {

    Ancestry *item = this;

    if( a_index == a_segments.size( ) ) return( item );

    std::string segment( a_segments[a_index] );

    if( segment == "" ) {
        item = this->root( );
        ++a_index;
        if( a_segments[a_index] != item->moniker( ) ) return( NULL ); }
    else if( segment == "." ) {
        }
    else if( segment == ".." ) {
        item = this->ancestor( ); }
    else {
        item = this->findInAncestry3( segment );
    }

    if( item == NULL ) return( item );

    ++a_index;
    return( item->findInAncestry2( a_index, a_segments ) );
}

/* *********************************************************************************************************//**
 * Returns a pointer to the node whose *xlink* is defined by the *a_segments* argument. The *a_segments* is the *xlink*
 * divided into segments separated by the '/' character.
 *
 * @param a_index               [in]    An index into the *a_segments* whose segment is to be found at this level.
 * @param a_segments            [in]    The list of *xlink* segments.
 * @return                              Returns the root node (i.e., the top level node).
 ***********************************************************************************************************/

Ancestry const *Ancestry::findInAncestry2( std::size_t a_index, std::vector<std::string> const &a_segments ) const {

    Ancestry const *item = this;

    if( a_index == a_segments.size( ) ) return( item );

    std::string segment( a_segments[a_index] );

    if( segment == "" ) {
        item = this->root( );
        ++a_index;
        if( a_segments[a_index] != item->moniker( ) ) return( NULL ); }
    else if( segment == "." ) {
        }
    else if( segment == ".." ) {
        item = this->ancestor( ); }
    else {
        item = this->findInAncestry3( segment );
    }

    if( item == NULL ) return( item );

    ++a_index;
    return( item->findInAncestry2( a_index, a_segments ) );
}

/* *********************************************************************************************************//**
 * Constructs and returns the *xlink* for *this*.
 *
 * @return          The constructed *xlink*.
 ***********************************************************************************************************/

std::string Ancestry::toXLink( ) {

    std::string xlink( "/" + m_moniker + xlinkItemKey( ) );

    if( isRoot( ) ) return( xlink );
    return( m_ancestor->toXLink( ) + xlink );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void Ancestry::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::cout << "Node '" << moniker( ) << "' needs toXMLList methods." << std::endl;
}

/* *********************************************************************************************************//**
 * Calls **toXMLList** and then writes the XML lines to the file "test.xml".
 ***********************************************************************************************************/

void Ancestry::printXML( ) const {

    WriteInfo writeInfo;

    toXMLList( writeInfo, "" );

    std::ofstream fileio;
    fileio.open( "test.xml" );
    for( std::list<std::string>::iterator iter = writeInfo.m_lines.begin( ); iter != writeInfo.m_lines.end( ); ++iter ) {
        fileio << *iter << std::endl;
    }
    fileio.close( );
}

}

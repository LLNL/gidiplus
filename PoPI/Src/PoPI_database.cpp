/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdio.h>
#include <stdexcept>

#include "PoPI.hpp"

#define PoPI_gaugeBosonsChars "gaugeBosons"
#define PoPI_leptonsChars "leptons"
#define PoPI_baryonsChars "baryons"
#define PoPI_unorthodoxesChars "unorthodoxes"
#define PoPI_chemicalElementsChars "chemicalElements"

#define MsgSize (8 * 1024)
#ifdef WIN32
#define __func__ __FUNCTION__
#endif

namespace PoPI {

static void parseAliases( pugi::xml_node const &a_node, Database *a_DB );

/*
=========================================================
*/
Database::Database( ) : 
    m_gaugeBosons( PoPI_gaugeBosonsChars ),
    m_leptons( PoPI_leptonsChars ),
    m_baryons( PoPI_baryonsChars ),
    m_chemicalElements( PoPI_chemicalElementsChars ),
    m_unorthodoxes( PoPI_unorthodoxesChars ) {

}
/*
=========================================================
*/
Database::Database( std::string const &a_fileName ) : 
    m_gaugeBosons( PoPI_gaugeBosonsChars ),
    m_leptons( PoPI_leptonsChars ),
    m_baryons( PoPI_baryonsChars ),
    m_chemicalElements( PoPI_chemicalElementsChars ),
    m_unorthodoxes( PoPI_unorthodoxesChars ) {

    addFile( a_fileName, false );
}
/*
=========================================================
*/
Database::Database( pugi::xml_node const &a_database ) :
    m_gaugeBosons( PoPI_gaugeBosonsChars ),
    m_leptons( PoPI_leptonsChars ),
    m_baryons( PoPI_baryonsChars ),
    m_chemicalElements( PoPI_chemicalElementsChars ),
    m_unorthodoxes( PoPI_unorthodoxesChars ) {

    addDatabase( a_database, false );
}
/*
=========================================================
*/
void Database::addFile( std::string const &a_fileName, bool a_warnIfDuplicate ) {

    addFile( a_fileName.c_str( ), a_warnIfDuplicate );
}
/*
=========================================================
*/
void Database::addFile( char const *a_fileName, bool a_warnIfDuplicate ) {
    pugi::xml_document doc;

    pugi::xml_parse_result result = doc.load_file( a_fileName );
    if( result.status != pugi::status_ok ) {
        char Msg[MsgSize+1];

        snprintf( Msg, MsgSize, "ERROR: in file '%s' in method '%s': %s. Input file is '%s'.", __FILE__, __func__, result.description( ), a_fileName );
        throw Exception( Msg );
    }

    pugi::xml_node database = doc.first_child( );
    addDatabase( database, a_warnIfDuplicate );
}
/*
=========================================================
*/
void Database::addDatabase( std::string const &a_string, bool a_warnIfDuplicate ) {

    pugi::xml_document doc;

    pugi::xml_parse_result result = doc.load_string( a_string.c_str( ) );
    if( result.status != pugi::status_ok ) {
        char Msg[MsgSize+1];

        snprintf( Msg, MsgSize, "ERROR: in file '%s' in method '%s': %s.", __FILE__, __func__, result.description( ) );
        throw Exception( Msg );
    }

    pugi::xml_node database = doc.first_child( );
    addDatabase( database, a_warnIfDuplicate );
}
/*
=========================================================
*/
void Database::addDatabase( pugi::xml_node const &a_database, bool a_warnIfDuplicate ) {

    FormatVersion formatVersion( a_database.attribute( PoPI_formatChars ).value( ) );
    if( m_formatVersion.format( ) == "" ) m_formatVersion = formatVersion;

    if( m_name == "" ) m_name = a_database.attribute( PoPI_nameChars ).value( );
    if( m_version == "" ) m_version = a_database.attribute( PoPI_versionChars ).value( );

    for( pugi::xml_node child = a_database.first_child( ); child; child = child.next_sibling( ) ) {
        std::string s_name( child.name( ) );

        if(      s_name == PoPI_gaugeBosonsChars ) {
            m_gaugeBosons.appendFromParentNode( child, this, this ); }
        else if( s_name == PoPI_leptonsChars ) {
            m_leptons.appendFromParentNode( child, this, this ); }
        else if( s_name == PoPI_baryonsChars ) {
            m_baryons.appendFromParentNode( child, this, this ); }
        else if( s_name == PoPI_chemicalElementsChars ) {
            m_chemicalElements.appendFromParentNode( child, this, this ); }
        else if( s_name == PoPI_unorthodoxesChars ) {
            m_unorthodoxes.appendFromParentNode( child, this, this ); }
        else if( s_name == PoPI_aliasesChars ) {
            parseAliases( child, this ); }
        else {
        }
    }

    for( std::vector<Alias *>::iterator iter = m_unresolvedAliases.begin( ); iter != m_unresolvedAliases.end( ); ++iter ) {
        std::map<std::string, int>::const_iterator pidIter = m_map.find( (*iter)->pid( ) );            // Locate pid.

        if( pidIter == m_map.end( ) ) {
            std::string errorMessage( "Alias points to particle " + (*iter)->pid( ) + " that is not present in database -2." );
            throw Exception( errorMessage );
        }
        (*iter)->setPidIndex( pidIter->second );
    }
    m_unresolvedAliases.clear( );
}
/*
=========================================================
*/
static void parseAliases( pugi::xml_node const &a_node, Database *a_DB ) {

    for( pugi::xml_node child = a_node.first_child( ); child; child = child.next_sibling( ) ) {
        std::string name = child.name( );
        Alias *alias = nullptr;

        if( name == PoPI_particleChars ) {
            alias = new Alias( child, a_DB ); }
        else if( name == PoPI_metaStableChars ) {
            alias = new MetaStable( child, a_DB );
        }
        a_DB->addAlias( alias );
    }
}
/*
=========================================================
*/
Database::~Database( ) {

    for( std::vector<Alias *>::iterator iter = m_aliases.begin( ); iter != m_aliases.end( ); ++iter ) delete *iter;
}
/*
=========================================================
*/
int Database::operator[]( std::string const &a_id ) const {

    std::map<std::string, int>::const_iterator iter = m_map.find( a_id );
    if( iter == m_map.end( ) ) {
        std::string errorMessage( "particle " + a_id + " not in database -3." );
        throw Exception( errorMessage );
    }

    return( iter->second );
}
/*
=========================================================
*/
bool Database::exists( int a_index ) const {

    if( ( a_index < 0 ) || ( a_index >= (int) m_map.size( ) ) ) return( false );
    return( true );
}
/*
=========================================================
*/
bool Database::exists( std::string const &a_id ) const {

    std::map<std::string, int>::const_iterator iter = m_map.find( a_id );
    return( iter != m_map.end( ) );
}
/*
=========================================================
*/
std::string Database::final( std::string const &a_id, bool returnAtMetaStableAlias ) const {

    int index( final( (*this)[a_id], returnAtMetaStableAlias ) );

    return( m_list[index]->ID( ) );
}
/*
=========================================================
*/
int Database::final( int a_index, bool returnAtMetaStableAlias ) const {

    while( isAlias( a_index ) ) {
        if( returnAtMetaStableAlias && isMetaStableAlias( a_index ) ) return( a_index );
        a_index = ((Alias *) m_list[a_index])->pidIndex( );
    }
    return( a_index );
}
/*
=========================================================
*/
int Database::add( Base *a_item ) {

    int index = (int) m_list.size( );

    m_map[a_item->ID( )] = index;
    m_list.push_back( a_item );
    a_item->setIndex( index );

    if( a_item->isAlias( ) ) m_unresolvedAliases.push_back( (Alias *) a_item );
    return( index );
}
/*
=========================================================
*/
int Database::addSymbol( SymbolBase *a_item ) {

    if( a_item->Class( ) == Particle_class::chemicalElement ) return( this->add( a_item ) );

    int index = (int) m_symbolList.size( );

    m_symbolMap[a_item->symbol( )] = index;
    m_symbolList.push_back( a_item );
    a_item->setIndex( index );

    return( index );
}
/*
=========================================================
*/
void Database::calculateNuclideGammaBranchStateInfos( NuclideGammaBranchStateInfos &a_nuclideGammaBranchStateInfos ) const {

    for( std::size_t i1 = 0; i1 <  m_chemicalElements.size( ); ++i1 ) {
        ChemicalElement const &chemicalElement = m_chemicalElements[i1];

        chemicalElement.calculateNuclideGammaBranchStateInfos( *this, a_nuclideGammaBranchStateInfos );
    }

    std::vector<NuclideGammaBranchStateInfo *> &nuclideGammaBranchStateInfos = a_nuclideGammaBranchStateInfos.nuclideGammaBranchStateInfos( );
    for( std::size_t i1 = 0; i1 < nuclideGammaBranchStateInfos.size( ); ++i1 ) {
        NuclideGammaBranchStateInfo *nuclideGammaBranchStateInfo = nuclideGammaBranchStateInfos[i1];

        nuclideGammaBranchStateInfo->calculateDerivedData( a_nuclideGammaBranchStateInfos );
    }
}
/*
=========================================================
*/
void Database::saveAs( std::string const &a_fileName ) const {

    std::string indent1( "" );
    std::vector<std::string> XMLList;

    XMLList.push_back( "<?xml version=\"1.0\"?>" );
    toXMLList( XMLList, indent1 );

    std::ofstream fileio;
    fileio.open( a_fileName.c_str( ) );
    for( std::vector<std::string>::iterator iter = XMLList.begin( ); iter != XMLList.end( ); ++iter ) {
        fileio << *iter << std::endl;
    }
    fileio.close( );
}
/*
=========================================================
*/
void Database::toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const {

    std::string indent2 = a_indent1 + "  ";
    std::string indent3 = indent2 + "  ";

    std::string header1 = a_indent1 + "<PoPs name=\"" + m_name + "\" version=\"" + m_version + "\" format=\"" + m_formatVersion.format( ) + "\">";
    a_XMLList.push_back( header1 );

    if( m_aliases.size( ) > 0 ) {
        std::string header2 = indent2 + "<" + PoPI_aliasesChars + ">";
        a_XMLList.push_back( header2 );
        for( std::vector<Alias *>::const_iterator iter = m_aliases.begin( ); iter != m_aliases.end( ); ++iter )
            (*iter)->toXMLList( a_XMLList, indent3 );
        appendXMLEnd( a_XMLList, PoPI_aliasesChars );
    }
    m_gaugeBosons.toXMLList( a_XMLList, indent2 );
    m_leptons.toXMLList( a_XMLList, indent2 );
    m_baryons.toXMLList( a_XMLList, indent2 );
    m_unorthodoxes.toXMLList( a_XMLList, indent2 );
    m_chemicalElements.toXMLList( a_XMLList, indent2 );

    appendXMLEnd( a_XMLList, PoPI_PoPsChars );
}
/*
=========================================================
*/
void Database::print( void ) {

    for( std::map<std::string,int>::const_iterator iter = m_map.begin( ); iter != m_map.end( ); ++iter ) {
        std::string label( iter->first );
        int index = iter->second;
        Base *item = m_list[index];
        std::string is_alias( "" );
        std::string mass( "" );

        if( item->isAlias( ) ) {
            is_alias = " is an alias (final is label = '";
            int finalIndex = final( index );
            IDBase const &myfinal = get<IDBase>( finalIndex );
            is_alias += std::string( myfinal.ID( ) );
            is_alias += std::string( "')" ); }
        else if( item->isParticle( ) ) {
            Particle *particle = (Particle *) item;
        
            double dmass = particle->massValue( "amu" );
            char massString[64];

            sprintf( massString, "  mass = %e amu", dmass );
            mass = massString;
        }

        std::cout << iter->first << " (" << item->ID( ) << ") --> " << index << " (" << item->index( ) << ")" 
            << is_alias << mass << std::endl;
    }
}

}

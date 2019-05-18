/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "GIDI.hpp"

#ifndef PATH_MAX
#define PATH_MAX ( 4 * 4096 )
#endif

#ifdef WIN32
#include <windows.h>
char *realpath( char const *a_path, char *a_resolved ) {

    char resolvedPath[PATH_MAX+1], *p1 = NULL;

    DWORD length = GetFullPathName( a_path, PATH_MAX, resolvedPath, NULL );
    if( ( p1 = malloc( length + 1 ) ) == NULL ) return( NULL );
    strcpy( p1, resolvedPath );
    if( length == 0 ) return( NULL );
    return( p1 );
}
#endif

#define TNSLMapMoniker "TNSL"

static std::string GIDI_basePath( char const *a_path );
static std::string GIDI_basePath( std::string const a_path );
static std::string GIDI_addPaths( std::string const &a_base, std::string const &a_path );

namespace GIDI {

/* *********************************************************************************************************//**
 * User data passed to the Map::directory method. It stores the desired projectile, target and evalaute infomation
 * a the list of found matches. An empty string for the projectile's id matches all projectiles. 
 * A empty string for the target's id matches all projectiles. An empty evaluation string matches all evaluations.
 ***********************************************************************************************************/

class MapWalkDirectoryCallbackData {

    public:
        std::string const &m_projectileID;                      /**< The desired projectile's id. */
        std::string const &m_targetID;                          /**< The desired target's id. */
        std::string const &m_evaluation;                        /**< The desired evaluation's id. */

        std::vector<ProtareBaseEntry const *> m_protareEntries;     /**< list of matched protare entries. */

        /* *********************************************************************************************************//**
         *
         * @param a_projectileID        [in]    The projectile's id to match.
         * @param a_targetID            [in]    The target's id to match.
         * @param a_evaluation          [in]    The evaluation to match.
         ***********************************************************************************************************/

        MapWalkDirectoryCallbackData( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_evaluation ) :
                m_projectileID( a_projectileID ),
                m_targetID( a_targetID ),
                m_evaluation( a_evaluation ) {

        }
};

/* *********************************************************************************************************//**
 *
 * @param a_protareEntry        [in]    The protare entry to compare to the data protare parameters in *a_data*.
 * @param a_data                [in]    A MapWalkDirectoryCallbackData instance.
 * @param a_level               [in]    Nested level of *this* map file. For internal use.
 *
 * @return                              Always returns true.
 ***********************************************************************************************************/

bool MapWalkDirectoryCallback( GIDI::ProtareBaseEntry const *a_protareEntry, void *a_data, int a_level ) {

    MapWalkDirectoryCallbackData *mapWalkDirectoryCallbackData = static_cast<MapWalkDirectoryCallbackData *>( a_data );

    if( ( mapWalkDirectoryCallbackData->m_projectileID == "" ) || ( mapWalkDirectoryCallbackData->m_projectileID == a_protareEntry->projectileID( ) ) ) {
        if( ( mapWalkDirectoryCallbackData->m_targetID == "" ) || ( mapWalkDirectoryCallbackData->m_targetID == a_protareEntry->targetID( ) ) ) {
            if( ( mapWalkDirectoryCallbackData->m_evaluation == "" ) || ( mapWalkDirectoryCallbackData->m_evaluation == a_protareEntry->evaluation( ) ) ) {
                mapWalkDirectoryCallbackData->m_protareEntries.push_back( a_protareEntry );
            }
        }
    }
    return( true );
}

/*! \class MapBaseEntry
 * This is the virtual base class inherited by all map entry classes.
 */

/* *********************************************************************************************************//**
 *
 * @param a_node        [in]    The **pugi::xml_node** to be parsed.
 * @param a_basePath    [in]    A path prepended to this entry's path.
 * @param a_parent      [in]    Pointer to the *Map* containing *this*.
 ***********************************************************************************************************/

MapBaseEntry::MapBaseEntry( pugi::xml_node const &a_node, std::string const &a_basePath, Map const *a_parent ) :
        m_name( a_node.name( ) ),
        m_parent( a_parent ),
        m_path( a_node.attribute( "path" ).value( ) ),
        m_cumulativePath( GIDI_addPaths( a_basePath, m_path ) ),
        m_realPath( realPath( m_cumulativePath ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

MapBaseEntry::~MapBaseEntry( ) {

}

/* *********************************************************************************************************//**
 * Returns either the *entered*, *cumulative* or *realPath* path for the entry.
 *
 * @param a_form            [in]    The type of path to return.
 * @return                          The requested path.
 ***********************************************************************************************************/

std::string const &MapBaseEntry::path( pathForm a_form ) const {

    if( a_form == e_entered ) return( m_path );
    if( a_form == e_cumulative ) return( m_cumulativePath );
    return( m_realPath );
}

/* *********************************************************************************************************//**
 * Fills *a_libraries* with the name of all the libraries *this* is contained in. The first library in the list is the 
 * library *this* is defined in and the last is the starting library.
 *
 * @param a_libraries           [out]   The instances that is filled with the library names.
 ***********************************************************************************************************/

void MapBaseEntry::libraries( std::vector<std::string> &a_libraries ) const {

    parent( )->libraries( a_libraries );
}

/* *********************************************************************************************************//**
 *
 * @param a_node        [in]    The **pugi::xml_node** to be parsed to contruct a MapEntry instance.
 * @param a_pops        [in]    A PoPs::Database instance used to get particle indices and possibly other particle information.
 * @param a_basePath    [in]    A path prepended to this entry's path.
 * @param a_parent      [in]    Pointer to the *Map* containing *this*.
 ***********************************************************************************************************/

MapEntry::MapEntry( pugi::xml_node const &a_node, PoPs::Database const &a_pops, std::string const &a_basePath, Map const *a_parent ) :
        MapBaseEntry( a_node, a_basePath, a_parent ),
        m_map( NULL ) {

    m_map = new Map( path( MapBaseEntry::e_cumulative ), a_pops, a_parent );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

MapEntry::~MapEntry( ) {

    delete m_map;
}

/* *********************************************************************************************************//**
 * Returns the ProtareEntry to the first protare to match *a_projectileID*, *a_targetID* and *a_evaluation*. If
 * *a_evaluation* is an empty string, only *a_projectileID* and *a_targetID* are matched.
 *
 * @param a_projectileID        [in]    The projectile's id to match.
 * @param a_targetID            [in]    The target's id to match.
 * @param a_evaluation          [in]    The evaluation to match.
 *
 * @return                              The path to the matched protare.
 ***********************************************************************************************************/

ProtareBaseEntry const *MapEntry::findProtareEntry( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_evaluation ) const {

    return( m_map->findProtareEntry( a_projectileID, a_targetID, a_evaluation ) );
}

/* *********************************************************************************************************//**
 * Returns the path to the first protare to match *a_projectileID*, *a_targetID* and *a_evaluation*. If
 * *a_evaluation* is an empty string, only *a_projectileID* and *a_targetID* are matched.
 *
 * @param a_projectileID        [in]    The projectile's id to match.
 * @param a_targetID            [in]    The target's id to match.
 * @param a_evaluation          [in]    The evaluation to match.
 * @param a_form                [in]    Determines the form of the path returned.
 * @return                              The path to the matched protare.
 ***********************************************************************************************************/

std::string MapEntry::protareFilename( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_evaluation,
        pathForm a_form ) const {

    return( m_map->protareFilename( a_projectileID, a_targetID, a_evaluation, a_form ) );
}

/* *********************************************************************************************************//**
 * Returns a list of all evaluations with a match to *a_projectileID* and *a_targetID*.
 *
 * @param a_projectileID        [in]    The projectile's id to match.
 * @param a_targetID            [in]    The target's id to match.
 * @return                              List of evaluations.
 ***********************************************************************************************************/

std::vector<std::string> MapEntry::availableEvaluations( std::string const &a_projectileID, std::string const &a_targetID ) const {

    return( m_map->availableEvaluations( a_projectileID, a_targetID ) );
}

/* *********************************************************************************************************//**
 * @param a_node        [in]    The **pugi::xml_node** to be parsed.
 ***********************************************************************************************************/

IDBaseEntry::IDBaseEntry( pugi::xml_node const &a_node ) :
        m_projectileID( a_node.attribute( "projectile" ).value( ) ),
        m_targetID( a_node.attribute( "target" ).value( ) ),
        m_evaluation( a_node.attribute( "evaluation" ).value( ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

IDBaseEntry::~IDBaseEntry( ) {

}

/* *********************************************************************************************************//**
 * Compares *a_projectileID*, *a_targetID* and *a_evaluation* to *this* data and returns true if they match
 * and false otherwise. If *a_evaluation* is an empty string, only *a_projectileID* and *a_targetID* are compared.
 *
 * @param a_projectileID        [in]    The projectile's id to match.
 * @param a_targetID            [in]    The target's id to match.
 * @param a_evaluation          [in]    The evaluation to match.
 * @return                              true if match and false otherwise.
 ***********************************************************************************************************/

bool IDBaseEntry::isMatch( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_evaluation ) const {

    if( a_projectileID != m_projectileID ) return( false );
    if( a_targetID != m_targetID ) return( false );
    if( a_evaluation == "" ) return( true );
    return( a_evaluation == m_evaluation );
}

/* *********************************************************************************************************//**
 * @param a_node        [in]    The **pugi::xml_node** to be parsed.
 * @param a_basePath    [in]    A path prepended to this entry's path.
 * @param a_parent      [in]    Pointer to the *Map* containing *this*.
 ***********************************************************************************************************/

ProtareBaseEntry::ProtareBaseEntry( pugi::xml_node const &a_node, std::string const &a_basePath, Map const *const a_parent ) :
        MapBaseEntry( a_node, a_basePath, a_parent ),
        IDBaseEntry( a_node ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

ProtareBaseEntry::~ProtareBaseEntry( ) {

}

/* *********************************************************************************************************//**
 * Returns the library *this* is contained in.
 ***********************************************************************************************************/

std::string const &ProtareBaseEntry::library( ) const {

    return( parent( )->library( ) );
}

/* *********************************************************************************************************//**
 * Returns the resolved library *this* is contained in.
 ***********************************************************************************************************/

std::string const &ProtareBaseEntry::resolvedLibrary( ) const {

    return( parent( )->resolvedLibrary( ) );
}

/* *********************************************************************************************************//**
 * Compares *a_projectileID*, *a_targetID* and *a_evaluation* to *this* data and returns true if they match
 * and false otherwise. If *a_evaluation* is an empty string, only *a_projectileID* and *a_targetID* are compared.
 *
 * @param a_projectileID        [in]    The projectile's id to match.
 * @param a_targetID            [in]    The target's id to match.
 * @param a_evaluation          [in]    The evaluation to match.
 * @return                              true if match and false otherwise.
 ***********************************************************************************************************/

ProtareBaseEntry const *ProtareBaseEntry::findProtareEntry( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_evaluation ) const {

    if( a_projectileID != projectileID( ) ) return( NULL );
    if( a_targetID != targetID( ) ) return( NULL );
    if( ( a_evaluation == "" ) || ( a_evaluation == evaluation( ) ) ) return( this );
    return( NULL );
}

/* *********************************************************************************************************//**
 *
 * @param a_node        [in]    The **pugi::xml_node** to be parsed to contruct a ProtareEntry instance.
 * @param a_pops        [in]    A PoPs::Database instance used to get particle indices and possibly other particle information.
 * @param a_basePath    [in]    A path prepended to this entry's path.
 * @param a_parent      [in]    Pointer to the *Map* containing *this*.
 ***********************************************************************************************************/

ProtareEntry::ProtareEntry( pugi::xml_node const &a_node, PoPs::Database const &a_pops, std::string const &a_basePath, Map const *const a_parent ) :
        ProtareBaseEntry( a_node, a_basePath, a_parent ),
        m_isPhotoAtomic( false ) {

    if( PoPs::IDs::photon == projectileID( ) ) {
        PoPs::Base const &target( a_pops.get<PoPs::Base>( targetID( ) ) );
        m_isPhotoAtomic = target.isChemicalElement( );
    }
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

ProtareEntry::~ProtareEntry( ) {

}

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_pops            [in]    A PoPs::Database instance used to get particle indices and possibly other particle information.
 *
 * @return                          Returns the Protare matching the TNSL protare.
 ***********************************************************************************************************/

Protare *ProtareEntry::protare( Construction::Settings const &a_construction, PoPs::Database const &a_pops ) const {

    std::vector<std::string> libraries1;
    libraries( libraries1 );
    return( new ProtareSingleton( a_construction, path( ), XML, a_pops, libraries1, true ) );
}

/* *********************************************************************************************************//**
 *
 * @param a_node        [in]    The **pugi::xml_node** to be parsed to contruct a TNSLEntry instance.
 * @param a_pops        [in]    A PoPs::Database instance used to get particle indices and possibly other particle information.
 * @param a_basePath    [in]    A path prepended to this entry's path.
 * @param a_parent      [in]    Pointer to the *Map* containing *this*.
 ***********************************************************************************************************/

TNSLsProtare::TNSLsProtare( pugi::xml_node const &a_node, PoPs::Database const &a_pops, std::string const &a_basePath, Map const *const a_parent ) :
        IDBaseEntry( a_node ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

TNSLsProtare::~TNSLsProtare( ) {

}

/* *********************************************************************************************************//**
 *
 * @param a_node        [in]    The **pugi::xml_node** to be parsed to contruct a TNSLEntry instance.
 * @param a_pops        [in]    A PoPs::Database instance used to get particle indices and possibly other particle information.
 * @param a_basePath    [in]    A path prepended to this entry's path.
 * @param a_parent      [in]    Pointer to the *Map* containing *this*.
 ***********************************************************************************************************/

TNSLEntry::TNSLEntry( pugi::xml_node const &a_node, PoPs::Database const &a_pops, std::string const &a_basePath, Map const *const a_parent ) :
        ProtareBaseEntry( a_node, a_basePath, a_parent ),
        m_TNSLsProtare( a_node.child( GIDI_protareMoniker ), a_pops, a_basePath, NULL ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

TNSLEntry::~TNSLEntry( ) {

}

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_pops            [in]    A PoPs::Database instance used to get particle indices and possibly other particle information.
 *
 * @return                              Returns the Protare matching the TNSL protare.
 ***********************************************************************************************************/

Protare *TNSLEntry::protare( Construction::Settings const &a_construction, PoPs::Database const &a_pops ) const {

    Map const *map = parent( );

    std::vector<std::string> libraries1;
    libraries( libraries1 );
    ProtareSingleton *protare1 = new ProtareSingleton( a_construction, path( ), XML, a_pops, libraries1, false );

    while( true ) {
        Map const *parent = map->parent( );

        if( parent == NULL ) break;
        map = parent;
    }
    ProtareSingleton *protare2 = static_cast<ProtareSingleton *>( map->protare( a_construction, a_pops, m_TNSLsProtare.projectileID( ), 
            m_TNSLsProtare.targetID( ), m_TNSLsProtare.evaluation( ) ) );

    ProtareTNSL *protareTNSL = new ProtareTNSL( a_construction, protare2, protare1 );

    return( protareTNSL );
}

/* *********************************************************************************************************//**
 *
 * @param a_fileName    [in]    The path to the map file to parse to construct a Map instance.
 * @param a_pops        [in]    A PoPs::Database instance used to get particle indices and possibly other particle information.
 * @param a_parent      [in]    Pointer to the *Map* containing *this*.
 ***********************************************************************************************************/

Map::Map( std::string const &a_fileName, PoPs::Database const &a_pops, Map const *a_parent ) {

    initialize( a_fileName, a_pops, a_parent );
}

/* *********************************************************************************************************//**
 *
 * @param a_fileName    [in]    The path to the map file to parse to construct a Map instance.
 * @param a_pops        [in]    A PoPs::Database instance used to get particle indices and possibly other particle information.
 * @param a_parent      [in]    Pointer to the *Map* containing *this*.
 ***********************************************************************************************************/

Map::Map( char const *a_fileName, PoPs::Database const &a_pops, Map const *a_parent ) {

    std::string const fileName( a_fileName );

    initialize( fileName, a_pops, a_parent );
}

/* *********************************************************************************************************//**
 *
 * @param a_node        [in]    pugi::xml_node corresponding to the <map> element
 * @param a_fileName    [in]    std::string, the name of the file containing this map
 * @param a_pops        [in]    A PoPs::Database instance used to get particle indices and possibly other particle information.
 * @param a_parent      [in]    Pointer to the *Map* containing *this*.
 ***********************************************************************************************************/

Map::Map( pugi::xml_node const &a_node, std::string const &a_fileName, PoPs::Database const &a_pops, Map const *a_parent ) {

    initialize( a_node, a_fileName, a_pops, a_parent );
}

/* *********************************************************************************************************//**
 * This method is called by the fileName constructors, opens the document and calls the other initialize method
 *
 * @param a_fileName    [in]    The path to the map file to parse to construct a Map instance.
 * @param a_pops        [in]    A PoPs::Database instance used to get particle indices and possibly other particle information.
 * @param a_parent      [in]    Pointer to the *Map* containing *this*.
 ***********************************************************************************************************/

void Map::initialize( std::string const &a_fileName, PoPs::Database const &a_pops, Map const *a_parent ) {

    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file( a_fileName.c_str( ) );
    if( result.status != pugi::status_ok ) throw std::runtime_error( result.description( ) );

    pugi::xml_node map = doc.first_child( );

    if( strcmp( map.name( ), GIDI_mapMoniker ) != 0 ) throw std::runtime_error( "Invalid map file " + a_fileName );

    initialize( map, a_fileName, a_pops, a_parent );
}

/* *********************************************************************************************************//**
 * This method is called either by the constructor or by the other initialize method. Does most of the work of parsing
 *
 * @param a_node        [in]    pugi::xml_node corresponding to the <map> node.
 * @param a_fileName    [in]    std::string, the name of the file containing this map
 * @param a_pops        [in]    A PoPs::Database instance used to get particle indices and possibly other particle information.
 * @param a_parent      [in]    Pointer to the *Map* containing *this*.
 ***********************************************************************************************************/

void Map::initialize( pugi::xml_node const &a_node, std::string const &a_fileName, PoPs::Database const &a_pops, Map const *a_parent ) {

    m_parent = a_parent;
    m_fileName = a_fileName;
    m_realFileName = realPath( a_fileName );

    std::string basePath = GIDI_basePath( m_realFileName );

    std::string format = a_node.attribute( "format" ).value( );
    if( format != GIDI_mapFormat ) throw std::runtime_error( "Unsupported map format" );

    m_library = a_node.attribute( "library" ).value( );
    for( pugi::xml_node child = a_node.first_child( ); child; child = child.next_sibling( ) ) {
        if( strcmp( child.name( ), GIDI_importMoniker ) == 0 ) {
            m_entries.push_back( new MapEntry( child, a_pops, basePath, this ) ); }
        else if( strcmp( child.name( ), GIDI_protareMoniker ) == 0 ) {
            m_entries.push_back( new ProtareEntry( child, a_pops, basePath, this ) ); }
        else if( strcmp( child.name( ), GIDI_TNSLMoniker ) == 0 ) {
            m_entries.push_back( new TNSLEntry( child, a_pops, basePath, this ) ); }
        else {
            throw std::runtime_error( std::string( "Invalid entry '" ) + child.name( ) + std::string( "' in map file " ) + a_fileName );
        }
    }
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Map::~Map( ) {

    for( std::vector<MapBaseEntry *>::const_iterator iter = m_entries.begin( ); iter < m_entries.end( ); ++iter ) delete *iter;
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

std::string const &Map::resolvedLibrary( ) const {

    if( ( m_library == "" ) && ( m_parent != NULL ) ) return( m_parent->resolvedLibrary( ) );
    return( m_library );
}

/* *********************************************************************************************************//**
 * Fills *a_libraries* with the name of all the libraries *this* is contained in. The first library in the list is the 
 * library *this* is defined in and the last is the starting library.
 *
 * @param a_libraries           [out]   The instances that is filled with the library names.
 ***********************************************************************************************************/

void Map::libraries( std::vector<std::string> &a_libraries ) const {

    a_libraries.push_back( m_library );
    if( m_parent != NULL ) m_parent->libraries( a_libraries );
}

/* *********************************************************************************************************//**
 * Returns the ProtareEntry to the first protare to match *a_projectileID*, *a_targetID* and *a_evaluation*. If
 * *a_evaluation* is an empty string, only *a_projectileID* and *a_targetID* are matched.
 *
 * @param a_projectileID        [in]    The projectile's id to match.
 * @param a_targetID            [in]    The target's id to match.
 * @param a_evaluation          [in]    The evaluation to match.
 *
 * @return                              The path to the matched protare.
 ***********************************************************************************************************/

ProtareBaseEntry const *Map::findProtareEntry( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_evaluation ) const {

    ProtareBaseEntry const *protareEntry = NULL;

    for( std::vector<MapBaseEntry *>::const_iterator iter = m_entries.begin( ); iter != m_entries.end( ); ++iter ) {
        protareEntry = (*iter)->findProtareEntry( a_projectileID, a_targetID, a_evaluation );
        if( protareEntry != NULL ) break;
    }
    return( protareEntry );
}

/* *********************************************************************************************************//**
 * Returns the path to the first protare to match *a_projectileID*, *a_targetID* and *a_evaluation*. If
 * *a_evaluation* is an empty string, only *a_projectileID* and *a_targetID* are matched.
 *
 * @param a_projectileID        [in]    The projectile's id to match.
 * @param a_targetID            [in]    The target's id to match.
 * @param a_evaluation          [in]    The evaluation to match.
 * @param a_form                [in]    Determines the form of the path returned.
 * @return                              The path to the matched protare.
 ***********************************************************************************************************/

std::string Map::protareFilename( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_evaluation,
        MapBaseEntry::pathForm a_form ) const {

    ProtareBaseEntry const *protareEntry = findProtareEntry( a_projectileID, a_targetID, a_evaluation );

    if( protareEntry != NULL ) return( protareEntry->path( a_form ) );
    return( GIDI_emptyFileName );
}

/* *********************************************************************************************************//**
 * Returns a list of all evaluations with a match to *a_projectileID* and *a_targetID*.
 *
 * @param a_projectileID        [in]    The projectile's id to match.
 * @param a_targetID            [in]    The target's id to match.
 * @return                              List of evaluations.
 ***********************************************************************************************************/

std::vector<std::string> Map::availableEvaluations( std::string const &a_projectileID, std::string const &a_targetID ) const {

    std::vector<std::string> list;

    for( std::vector<MapBaseEntry *>::const_iterator iter1 = m_entries.begin( ); iter1 != m_entries.end( ); ++iter1 ) {
        if( (*iter1)->name( ) == GIDI_importMoniker ) {
            MapEntry *_mapEntry = dynamic_cast<MapEntry *> (*iter1);

            std::vector<std::string> sub_list = _mapEntry->availableEvaluations( a_projectileID, a_targetID );
            for( std::vector<std::string>::const_iterator iter2 = sub_list.begin( ); iter2 != sub_list.end( ); ++iter2 )
                list.push_back( *iter2 ); }
        else {
            ProtareBaseEntry *protareEntry = dynamic_cast<ProtareBaseEntry *> (*iter1);

            if( protareEntry->isMatch( a_projectileID, a_targetID ) ) list.push_back( protareEntry->evaluation( ) );
        }
    }
    return( list );
}

/* *********************************************************************************************************//**
 * If a protare matching *a_projectileID*, *a_targetID* and *a_evaluation* is found, the Protare constructor is called with
 * its fileName.
 *
 * @param a_construction                [in]    Pass to the Protare constructor.
 * @param a_pops                        [in]    A PoPs::Database instance used to get particle indices and possibly other particle information.
 * @param a_projectileID                [in]    The projectile's id to match.
 * @param a_targetID                    [in]    The target's id to match.
 * @param a_evaluation                  [in]    The evaluation to match.
 * @param a_targetRequiredInGlobalPoPs  [in]    If *true*, the target is required to be in **a_pops**.
 * @param a_ignorePoPs                  [in]    If *true*, no particle is required to be in **a_pops**.
 ***********************************************************************************************************/

Protare *Map::protare( Construction::Settings const &a_construction, PoPs::Database const &a_pops, std::string const &a_projectileID, 
                std::string const &a_targetID, std::string const &a_evaluation, bool a_targetRequiredInGlobalPoPs, bool a_ignorePoPs ) const {

    std::string targetID( a_targetID );
    std::string atomicTargetID;

    Protare *nuclear = NULL, *atomic = NULL, *protare;

    if( ( a_projectileID == PoPs::IDs::photon ) && ( a_construction.photoMode( ) != Construction::e_nuclearOnly ) ) {
        PoPs::Base const *popsBase = &a_pops.get<PoPs::Base>( targetID );
        if( popsBase->Class( ) == PoPs::class_nuclide ) {
            PoPs::Nuclide const *nuclide = static_cast<PoPs::Nuclide const *>( popsBase );
            PoPs::Isotope const *isotope = nuclide->isotope( );
            popsBase = isotope->chemicalElement( );
            atomicTargetID = popsBase->ID( );
        }
    }

    if( a_construction.photoMode( ) != Construction::e_atomicOnly ) {
        ProtareBaseEntry const *protareEntry = findProtareEntry( a_projectileID, a_targetID, a_evaluation );
        if( protareEntry != NULL ) nuclear = protareEntry->protare( a_construction, a_pops );
    }

    if( a_construction.photoMode( ) != Construction::e_nuclearOnly ) {
        ProtareBaseEntry const *protareEntry = findProtareEntry( a_projectileID, atomicTargetID, a_evaluation );
        if( protareEntry != NULL ) atomic = protareEntry->protare( a_construction, a_pops );
    }

    if( nuclear == NULL ) {
        protare = atomic; }
    else if( atomic == NULL ) {
        protare = nuclear; }
    else {
        ProtareComposite *protareComposite = new ProtareComposite( a_construction );

        protareComposite->projectile( nuclear->projectile( ) );
        protareComposite->target( nuclear->target( ) );
        protareComposite->append( nuclear );
        protareComposite->append( atomic );
        protare = protareComposite;
    }

    return( protare );
}

/* *********************************************************************************************************//**
 * Returns a list of all ProtareEntry's matching the input data.
 *
 * @param a_projectileID        [in]    The projectile's id to match.
 * @param a_targetID            [in]    The target's id to match.
 * @param a_evaluation          [in]    The evaluation to match.
 * @return                              List of all ProtareEntry's matching input parameters.
 ***********************************************************************************************************/

std::vector<ProtareBaseEntry const *> Map::directory( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_evaluation ) const {

    MapWalkDirectoryCallbackData mapWalkDirectoryCallbackData( a_projectileID, a_targetID, a_evaluation );

    walk( MapWalkDirectoryCallback, &mapWalkDirectoryCallbackData, 0 );
    return( mapWalkDirectoryCallbackData.m_protareEntries );
}

/* *********************************************************************************************************//**
 * A method to walk a map file. For each ProtareEntry found, the **a_mapWalkCallBack** function is called with
 * a pointer to the ProtareEntry and **a_userData** has its arguments.
 *
 * @param           a_mapWalkCallBack       [in]    The callback function.
 * @param           a_userData              [in]    Pointer to user data.
 * @param           a_level                 [in]    Nested level of *this* map file. For internal use.
 *
 * @return          true if no issue is found and false if an issue is found.
 ***********************************************************************************************************/

bool Map::walk( MapWalkCallBack a_mapWalkCallBack, void *a_userData, int a_level ) const {

    for( std::size_t i1 = 0; i1 < size( ); ++i1 ) {
        MapBaseEntry const *entry = (*this)[i1];

        std::string path = entry->path( GIDI::MapBaseEntry::e_cumulative );

        if( entry->name( ) == GIDI_importMoniker ) {
            MapEntry const *mapEntry = static_cast<MapEntry const *>( entry );
            if( !mapEntry->map( )->walk( a_mapWalkCallBack, a_userData, a_level + 1 ) ) return( true ); }
        else if( ( entry->name( ) == GIDI_protareMoniker ) || ( entry->name( ) == GIDI_TNSLMoniker ) ) {
            if( !a_mapWalkCallBack( static_cast<ProtareBaseEntry const *>( entry ), a_userData, a_level ) ) return( true ); }
        else {
            std::cerr << "    ERROR: unknown map entry name: " << entry->name( ) << std::endl;
        }
    }

    return( true );
}

}       // End of namespace GIDI.

/* *********************************************************************************************************//**
 * Splits the path at the last path separator (e.g., the '/' charactor on Unix systems) and returns the first (i.e.,
 * directory) part. Returns "." is no '/' is present.
 *
 * @param a_path            The path whose directory is to be returned.
 * @return                  The directory of file **a_path**
 ***********************************************************************************************************/

static std::string GIDI_basePath( char const *a_path ) {

    char *p1, realPath[PATH_MAX+1];

    strcpy( realPath, a_path );
    if( ( p1 = strrchr( realPath, '/' ) ) != NULL ) {
        *p1 = 0; }
    else {
        strcpy( realPath, "." );
    }
    std::string basePath( realPath );
    return( basePath );
}

/* *********************************************************************************************************//**
 * Calls GIDI_basePath( char const *a_path ).
 *
 * @param a_path
 * @return
 ***********************************************************************************************************/

static std::string GIDI_basePath( std::string const a_path ) {

    return( GIDI_basePath( a_path.c_str( ) ) );
}

/* *********************************************************************************************************//**
 * If **a_path** is not an absolute path, prepends **a_path** to it.
 *
 * @param a_base            [in]    Base path to prepend to **a_path**.
 * @param a_path            [in]    Path
 * @return                          Prepend path.
 ***********************************************************************************************************/

static std::string GIDI_addPaths( std::string const &a_base, std::string const &a_path ) {

    std::string path( a_path );

    if( ( a_base.size( ) > 0 ) && ( path[0] != FILE_SEPARATOR[0] ) ) path = a_base + FILE_SEPARATOR + path;
    return( path );
}

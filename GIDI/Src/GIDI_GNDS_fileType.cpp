/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <fcntl.h>
#include <unistd.h>

#ifdef GIDI_PLUS_INCLUDE_EXPAT
    #include <expat.h>
#else
    typedef unsigned char XML_Bool;
    typedef char XML_Char;
    typedef void * XML_Parser;
    typedef void (*XML_StartElementHandler)( void *a_userData, XML_Char const *a_name, XML_Char const **a_atts );
    typedef void (*XML_EndElementHandler)( void *a_userData, XML_Char const *a_name);

    enum XML_Status { XML_STATUS_ERROR = 0, XML_STATUS_OK = 1 };
    #define XML_TRUE 1

    XML_Parser XML_ParserCreate( void *a_dummy ) { return( nullptr ); }
    void XML_ParserFree( XML_Parser a_XML_Parser ) {}
    void XML_SetElementHandler( XML_Parser a_xmlParser, XML_StartElementHandler a_startElementHandler, XML_EndElementHandler a_endElementHandler ) { }
    void XML_SetUserData( XML_Parser a_xmlParser, void *a_userData ) {}
    enum XML_Status XML_Parse( XML_Parser a_xmlParser, char const *a_buffer, int a_count, int a_isFinal ) { return( XML_STATUS_ERROR ); }
//    XML_GetErrorCode
    void XML_StopParser( XML_Parser a_xmlParser, XML_Bool a_resumable ) {}
#endif

#include "GIDI.hpp"

namespace GIDI {

class GNDS_FileTypeInfoUserData {

    public:
        XML_Parser m_xmlParser;
        GNDS_FileTypeInfo &m_GNDS_fileTypeInfo;

        GNDS_FileTypeInfoUserData( XML_Parser a_xmlParser, GNDS_FileTypeInfo &a_GNDS_fileTypeInfo ) :
            m_xmlParser( a_xmlParser ),
            m_GNDS_fileTypeInfo( a_GNDS_fileTypeInfo ) {

        }
};

static void xml_startElementHandler( void *a_userData, XML_Char const *a_name, XML_Char const **a_attributes );
static void xml_endElementHandler( void *a_userData, XML_Char const *a_name );

/* *********************************************************************************************************//**
 * Constructor with no arguments.
 ***********************************************************************************************************/

GNDS_FileTypeInfo::GNDS_FileTypeInfo( ) :
        m_GNDS_fileType( GNDS_FileType::uninitialized ) {

}

/* *********************************************************************************************************//**
 * Constructor with all arguments.
 *
 * @param       a_GNDS_fileType         [in]        The enum describing the GNDS file's type.
 * @param       a_projectileID          [in]        The PoPs id for the protare's projectile.
 * @param       a_targetID              [in]        The PoPs id for the protare's target.
 * @param       a_evaluation            [in]        The protare's evaluation.
 * @param       a_interaction           [in]        The protare's interaction.
 ***********************************************************************************************************/

GNDS_FileTypeInfo::GNDS_FileTypeInfo( GNDS_FileType a_GNDS_fileType, std::string a_projectileID, std::string a_targetID, std::string a_evaluation,
                        std::string a_interaction ) :
        m_GNDS_fileType( a_GNDS_fileType ),
        m_projectileID( a_projectileID ),
        m_targetID( a_targetID ),
        m_evaluation( a_evaluation ),
        m_interaction( a_interaction ) {
        
}

/* *********************************************************************************************************//**
 * Copy constructor.
 *
 * @param       a_GNDS_fileTypeInfo     [in]        GNDS_FileTypeInfo instance to copy.
 ***********************************************************************************************************/

GNDS_FileTypeInfo::GNDS_FileTypeInfo( GNDS_FileTypeInfo const &a_GNDS_fileTypeInfo ) :
        m_GNDS_fileType( a_GNDS_fileTypeInfo.GNDS_fileType( ) ),
        m_projectileID( a_GNDS_fileTypeInfo.projectileID( ) ),
        m_targetID( a_GNDS_fileTypeInfo.targetID( ) ),
        m_evaluation( a_GNDS_fileTypeInfo.evaluation( ) ),
        m_interaction( a_GNDS_fileTypeInfo.interaction( ) ) {

}

/* *********************************************************************************************************//**
 * Opens the specified file and parses the first line to determine its GNDS type (i.e., protare (reactionSuite), map or PoPs file).
 * Returns the GNDS type via the GNDS_FileType enum. If the return value and the value of *a_GNDS_fileTypeInfo.GNDS_fileType( )* is 
 * *uninitialized* an error was detected opening the file or by the XML parser (expat). If it is *unknown* the parsed file is an 
 * XML file but not a valid GNDS file.
 *
 * @param       a_fileName          [in]    The path to the file whose GNDS type is to be determined.
 * @param       a_GNDS_fileTypeInfo [in]    The *GNDS_FileTypeInfo* instance containing the return information.
 *
 * @return                          enum indicating the GNDS type of file referened by *a_fileName*.
 ***********************************************************************************************************/

GNDS_FileType GNDS_fileType( std::string const &a_fileName, GNDS_FileTypeInfo &a_GNDS_fileTypeInfo ) {

    a_GNDS_fileTypeInfo.setGNDS_fileType( GNDS_FileType::uninitialized );

    char buffer[10 * 1024 + 1];
    ssize_t bufferSize = sizeof( buffer ) - 1;
    int fileDescriptor;

#ifdef GIDI_PLUS_NOEXPAT
    throw Exception( "\nGIDI::fileType failed as expat not included." );
#endif

    fileDescriptor = open( a_fileName.c_str( ), O_RDONLY );
    if( fileDescriptor < 0 ) throw Exception( "GIDI::fileType failed to open file '" + a_fileName + "'." );

    XML_Parser xmlParser = XML_ParserCreate( nullptr );
    if( xmlParser == nullptr ) {
        close( fileDescriptor );
        throw Exception( "XML_ParserCreate failed." );
    }

    XML_SetElementHandler( xmlParser, xml_startElementHandler, xml_endElementHandler );

    GNDS_FileTypeInfoUserData userData( xmlParser, a_GNDS_fileTypeInfo );
    XML_SetUserData( xmlParser, &userData );

    enum XML_Status status;
    ssize_t count, totalCount = 0;
    while( ( count = read( fileDescriptor, buffer, bufferSize ) ) > 0 ) {
        totalCount += count;
        status = XML_Parse( xmlParser, buffer, count, 0 );
        if( status != XML_STATUS_OK ) break;
    }

//    enum XML_Error error = XML_GetErrorCode( xmlParser );
    XML_ParserFree( xmlParser );
    close( fileDescriptor );

    if( status == XML_STATUS_ERROR ) throw Exception( "GIDI::fileType expat parsing error." );
//      What about other status values like XML_STATUS_SUSPENDED.
    return( a_GNDS_fileTypeInfo.GNDS_fileType( ) );
}

/* *********************************************************************************************************//**
 * For internal use only.
 *
 * @param       a_userData          [in]    The user data.
 * @param       a_name              [in]    The name of the start element.
 * @param       a_attributes        [in]    The list of attributes for the start element.
 *
 * @return                          enum indicating the GNDS type of file referened by *a_fileName*.
 ***********************************************************************************************************/

static void xml_startElementHandler( void *a_userData, XML_Char const *a_name, XML_Char const **a_attributes ) {

    GNDS_FileTypeInfoUserData *userData = static_cast<GNDS_FileTypeInfoUserData *>( a_userData );

    if( strcmp( a_name, PoPI_PoPsChars ) == 0 ) {
        userData->m_GNDS_fileTypeInfo.setGNDS_fileType( GNDS_FileType::pops ); }
    else if( strcmp( a_name, GIDI_topLevelChars ) == 0 ) {
        XML_Char const **attributes = a_attributes;
        std::string projectileID;
        std::string targetID;
        std::string evaluation;
        std::string interaction;

        while( *attributes != nullptr ) {
            std::string attributeName( *attributes );
            ++attributes;
            if( attributeName == GIDI_projectileChars ) {
                projectileID = *attributes; }
            if( attributeName == GIDI_targetChars ) {
                targetID = *attributes; }
            if( attributeName == GIDI_evaluationChars ) {
                evaluation = *attributes; }
            if( attributeName == GIDI_interactionChars ) {
                interaction = *attributes;
            }
            ++attributes;
        }
        GNDS_FileTypeInfo GNDS_fileTypeInfo( GNDS_FileType::protare, projectileID, targetID, evaluation, interaction );
        userData->m_GNDS_fileTypeInfo = GNDS_fileTypeInfo; }
    else if( strcmp( a_name, GIDI_covarianceSuiteChars ) == 0 ) {
        userData->m_GNDS_fileTypeInfo.setGNDS_fileType( GNDS_FileType::covarianceSuite ); }
    else if( strcmp( a_name, GIDI_mapChars ) == 0 ) {
        userData->m_GNDS_fileTypeInfo.setGNDS_fileType( GNDS_FileType::map ); }
    else {
        userData->m_GNDS_fileTypeInfo.setGNDS_fileType( GNDS_FileType::unknown );
    }

    XML_StopParser( userData->m_xmlParser, XML_TRUE );
}

/* *********************************************************************************************************//**
 * For internal use only.
 *
 * @param       a_userData          [in]    The user data.
 * @param       a_name              [in]    The name of the start element.
 *
 * @return                          enum indicating the GNDS type of file referened by *a_fileName*.
 ***********************************************************************************************************/

static void xml_endElementHandler( void *a_userData, XML_Char const *a_name ) {

}

}               // End namespace GIDI.

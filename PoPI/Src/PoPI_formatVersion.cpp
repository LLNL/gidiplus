/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <limits.h>

#include "PoPI.hpp"

namespace PoPI {

/*! \class FormatVersion
 * Class to store GNDS format.
 */

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

FormatVersion::FormatVersion( ) :
        m_format( "" ),
        m_major( -1 ),
        m_minor( -1 ),
        m_patch( "" ) {

}

/* *********************************************************************************************************//**
 * @param a_formatVersion       [in]    The GNDS format.
 ***********************************************************************************************************/

FormatVersion::FormatVersion( std::string const &a_formatVersion ) :
        m_format( a_formatVersion ),
        m_major( -1 ),
        m_minor( -1 ),
        m_patch( "" ) {

    setFormat( a_formatVersion );
}

/* *********************************************************************************************************//**
 * @param a_formatVersion       [in]    The GNDS format.
 ***********************************************************************************************************/

FormatVersion::FormatVersion( FormatVersion const &a_formatVersion ) :
        m_format( a_formatVersion.format( ) ),
        m_major( a_formatVersion.major( ) ),
        m_minor( a_formatVersion.minor( ) ),
        m_patch( a_formatVersion.patch( ) ) {

}

/* *********************************************************************************************************//**
 * Set the format to *a_formatVersion* and parse its components.
 *
 * @param   a_formatVersion       [in]      The GNDS format.
 *
 * @return                                  true if format of the form "MAJOR.MINOR[.PATCH]" where MAJOR and MINOR are integers. Otherwise returns false.
 ***********************************************************************************************************/

bool FormatVersion::setFormat( std::string const &a_formatVersion ) {

    m_format = a_formatVersion;

    std::vector<std::string> formatItems = splitString( a_formatVersion, '.' );

    if( ( formatItems.size( ) < 2 ) || ( formatItems.size( ) > 3 ) ) goto err;

    if( !stringToInt( formatItems[0], m_major ) ) goto err;
    if( !stringToInt( formatItems[1], m_minor ) ) goto err;

    if( formatItems.size( ) == 3 ) m_patch = formatItems[2];

    return( true );

err:
    m_major = -1;
    m_minor = -1;
    m_patch = "";
    return( false );
}

/* *********************************************************************************************************//**
 * Returns true if m_format is a supported format and false otherwise;
 *  
 * @return                                  true if format is supported and false otherwise.
 ***********************************************************************************************************/

bool FormatVersion::supported( ) {

    if( m_format == PoPI_formatVersion_0_1_Chars ) return( true );
    if( m_format == PoPI_formatVersion_1_10_Chars ) return( true );
    if( m_format == PoPI_formatVersion_2_0_LLNL_3_Chars ) return( true );

    return( false );
}

}               // End namespace PoPI.

/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>

#include <LUPI.hpp>

namespace LUPI {

namespace FileInfo {

/* *********************************************************************************************************//**
 * Returns the base name of a path.
 *
 * @param a_path            [in]    The path whose base name is returned.
 ***********************************************************************************************************/

std::string _basename( std::string const &a_path ) {

    char *path = new char[a_path.size( ) + 1];
    strcpy( path, a_path.c_str( ) );
    std::string basename1( basename( path ) );

    delete[] path;

    return( basename1 );
}

/* *********************************************************************************************************//**
 * Same as **basename** but removes, if a *period* (i.e., ".") exists in the string, the last "." and all following characters.
 *
 * @param a_path            [in]    The path whose base name is returned without its extension.
 ***********************************************************************************************************/

std::string basenameWithoutExtension( std::string const &a_path ) {

    std::size_t found = a_path.rfind( '.' );

    return( a_path.substr( 0, found ) );
}

/* *********************************************************************************************************//**
 * Returns the directory name of a path. This is, it removes the base name.
 *
 * @param a_path            [in]    The path whose base name is returned.
 ***********************************************************************************************************/

std::string _dirname( std::string const &a_path ) {

    char *path = new char[a_path.size( ) + 1];
    strcpy( path, a_path.c_str( ) );
    std::string dirname1( dirname( (char *) path ) );

    delete[] path;

    return( dirname1 );
}

/* *********************************************************************************************************//**
 * Returns *true* if the path exists and *false* otherwise.
 *
 * @param a_path            [in]    The path that is checked for existence.
 ***********************************************************************************************************/

bool exists( std::string const &a_path ) {

        return( access( a_path.c_str( ), F_OK ) == 0 );
}

/* *********************************************************************************************************//**
 * Returns *true* if path is a direction that exists and *false* otherwise.
 *
 * @param a_path            [in]    The path that is checked for existence and is it a directory.
 *
 * @return                          Returns *true* if the path exists (e.g., created if it does not exists) and *false* otherwise.
 ***********************************************************************************************************/

bool isDirectory( std::string const &a_path ) {

    try {
        FileStat fileStat( a_path );
        return( fileStat.isDirectory( ) ); }
    catch (...) {
    }

    return( false );
}

/* *********************************************************************************************************//**
 * Adds all needed directories to complete *a_path*.
 *
 * @param a_path            [in]    The path that is checked for existence.
 *
 * @return                          Returns *true* if the path exists (e.g., created if it does not exists) and *false* otherwise.
 ***********************************************************************************************************/

bool createDirectories( std::string const &a_path ) {

    if( isDirectory( a_path ) ) return( true );
    if( ( a_path == LUPI_FILE_SEPARATOR ) || ( a_path == "." ) || ( a_path == "" ) ) return( true );

    std::string dirname1( _dirname( a_path ) );
    if( createDirectories( dirname1 ) ) {
        int status = mkdir( a_path.c_str( ), S_IRWXU | S_IRWXG | S_IRWXG );
        if( status == 0 ) return( true );
        switch( errno ) {
        case EEXIST :
            return( true );
        default :
            return( false );
        }
    }

    return( false );
}

/* *********************************************************************************************************//**
 * Calls the C stat function and stores its information.
 *
 * @param a_path            [in]    The path (e.g., file, directory) whose stat is determined.
 ***********************************************************************************************************/

FileStat::FileStat( std::string const &a_path ) :
        m_path( a_path ) {

    int error = stat( a_path.c_str( ), &m_stat );

    if( error != 0 ) {
        switch( error ) {
        case EACCES :
            throw Exception( "FileStat::FileStat: Permission denied for file '" + a_path + "'.." );
        case EIO :
            throw Exception( "FileStat::FileStat: An error occurred while stat-ing file '" + a_path + "'.." );
        case ELOOP :
            throw Exception( "FileStat::FileStat: A loop exists in symbolic links for file '" + a_path + "'.." );
        case ENAMETOOLONG :
            throw Exception( "FileStat::FileStat: Path name too long '" + a_path + "'." );
        case ENOENT :
            throw Exception( "FileStat::FileStat: No such path '" + a_path + "'." );
        case ENOTDIR :
            throw Exception( "FileStat::FileStat: A component of the path prefix is not a directory '" + a_path + "'." );
        case EOVERFLOW :
            throw Exception( "FileStat::FileStat: File too big: '" + a_path + "'." );
        default :
            throw Exception( "FileStat::FileStat: Unknown error from C function 'stat' for file '" + a_path + "'." );
        }
    }
}

}               // End of namespace FileInfo.

}               // End of namespace LUPI.

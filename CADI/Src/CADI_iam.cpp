/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>
#include <algorithm>

#include <HAPI.hpp>
#include <CADI.hpp>

namespace CADI {

namespace Map {

static LUPI_HOST EntryBase *serializeEntry( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode, EntryBase *a_entry, IAM *a_ancestor );

/*! \class Base
 * The base class for IAM, Import and IsotopicAbundancesByChemicalElement classes.
 */

/* *********************************************************************************************************//**
 * Constructor.
 *
 * @param a_moniker                     [in]    The value of the *m_moniker* member for **GUPI::Ancestry**.
 * @param a_sourcePath                  [in]    If an **IAM** instance then this is its source path; otherwise, it is the source path of the parent **IAM** instance.
 ***********************************************************************************************************/

Base::Base( std::string const &a_moniker, std::string const &a_sourcePath  ) :
        GUPI::Ancestry( a_moniker ),
        m_sourcePath( a_sourcePath ),
        m_checksum( "" ),
        m_algorithm( "" ) {

}

/* *********************************************************************************************************//**
 * Base constructor.
 *
 * @param a_node                        [in]    HAPI node to be parsed and used to construct an *this* **Isotope**.
 ***********************************************************************************************************/

Base::Base( HAPI::Node const &a_node, std::string const &a_sourcePath  ) :
        GUPI::Ancestry( a_node.name( ) ),
        m_sourcePath( a_sourcePath ),
        m_checksum( a_node.attribute_as_string( CADI_checksumChars ) ),
        m_algorithm( a_node.attribute_as_string( CADI_algorithmChars ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Base::~Base( ) {

}

/* *********************************************************************************************************//**
 * Returns the value of *this* algorithm, or its parent's if *this* is an empty string.
 ***********************************************************************************************************/

std::string const Base::algorithm( ) const {

    if( m_algorithm != "" ) return( m_algorithm );

    if( ancestor( ) == nullptr ) return( "" );

    Base const *parent = static_cast<Base const *>( ancestor( ) );
    return( parent->algorithm( ) );
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

LUPI_HOST void Base::serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode ) {

    GUPI::Ancestry::serialize( a_buffer, a_mode );

    DATA_MEMBER_STD_STRING( m_sourcePath, a_buffer, a_mode );
    DATA_MEMBER_STD_STRING( m_checksum, a_buffer, a_mode );
    DATA_MEMBER_STD_STRING( m_algorithm, a_buffer, a_mode );
}

/* *********************************************************************************************************//**
 * Returns the checksum and algorithm attributes as an XML attribute string if they are present.
 *
 * @param a_checkAncestor   If true, do not include algorithm if it is the same as *this* parent's algorithm.
 ***********************************************************************************************************/

std::string Base::standardXML_attributes( bool a_checkAncestor ) const {

    std::string attributes;

    if( m_checksum != "" ) attributes += " checksum=\"" + m_checksum + "\"";

    std::string algorithm = m_algorithm;
    if( algorithm != "" ) {
        if( a_checkAncestor ) {
            Base const *parent = static_cast<Base const *>( ancestor( ) );
            if( algorithm == parent->algorithm( ) ) algorithm = "";
        }
    }
    if( algorithm != "" ) attributes += " algorithm=\"" + algorithm + "\"";

    return( attributes );
}

/*! \class EntryBase
 * The base class for Import and IsotopicAbundancesByChemicalElement classes.
 */

/* *********************************************************************************************************//**
 * Constructor.
 *
 * @param a_moniker                     [in]    The value of the *m_moniker* member for **GUPI::Ancestry**.
 ***********************************************************************************************************/

EntryBase::EntryBase( std::string const &a_moniker ) :
        Base( a_moniker, "" ) {

}

/* *********************************************************************************************************//**
 * Base constructor.
 * 
 * @param a_node                        [in]    HAPI node to be parsed and used to construct an *this* **Isotope**.
 ***********************************************************************************************************/

EntryBase::EntryBase( HAPI::Node const &a_node, std::string const &a_sourcePath ) :
        Base( a_node, a_sourcePath ),
        m_path( a_node.attribute_as_string( CADI_pathChars ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

EntryBase::~EntryBase( ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

std::string EntryBase::fileName( ) const { 

    if( m_path[0] == '/' ) return( m_path );

    return( LUPI::FileInfo::_dirname( sourcePath( ) ) + "/" + m_path );
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

LUPI_HOST void EntryBase::serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode ) {

    Base::serialize( a_buffer, a_mode );
    DATA_MEMBER_STD_STRING( m_path, a_buffer, a_mode );
}

/* *********************************************************************************************************//**
 * Returns the checksum and algorithm attributes as an XML attribute string if they are present.
 *
 * @param a_checkAncestor   If true, do not include algorithm if it is the same as *this* parent's algorithm.
 ***********************************************************************************************************/

std::string EntryBase::standardXML_attributes( bool a_checkAncestor ) const {

    return( " path=\"" + m_path + "\"" + Base::standardXML_attributes( a_checkAncestor ) );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void EntryBase::toXMLList( GUPI::WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    a_writeInfo.addNodeStarterEnder( a_indent, moniker( ), standardXML_attributes( ) );
}

/*! \class Import
 * Class for the **import** child node inside an **iam** node.
 */

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Import::Import( ) :
        EntryBase( CADI_importChars ),
        m_IAM( nullptr ) {

}

/* *********************************************************************************************************//**
 * Import constructor.
 *
 * @param a_node                        [in]    HAPI node to be parsed and used to construct an *this* **Isotope**.
 ***********************************************************************************************************/

Import::Import( HAPI::Node const &a_node, std::string const &a_sourcePath ) :
        EntryBase( a_node, a_sourcePath ),
        m_IAM( nullptr ) {

        m_IAM = new IAM( fileName( ) );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Import::~Import( ) {

    delete m_IAM;
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param a_symbol      [in]    The symbol for the chemical element whose isotopic abundance data are being requested.
 * @param a_evaluation  [in]    The name of the evaluation whose isotopic abundance data are being requested (optional).
 ***********************************************************************************************************/

ChemicalElement const *Import::find( std::string const &a_symbol, std::string const &a_evaluation ) const {

    return( m_IAM->find( a_symbol, a_evaluation ) );
}

/* *********************************************************************************************************//**
 * Return a *const* pointer to the **CADI::IsotopicAbundancesByChemicalElement** instance with evaluation *a_evaluation*, or a *nullptr*
 * if it does not exist.
 *
 * @param a_evaluation  [in]    The name of the evaluation whose isotopic abundance data are being requested.
 *
 * @return                      An **CADI::IsotopicAbundancesByChemicalElement** instance or *nullptr*.
 ***********************************************************************************************************/

CADI::IsotopicAbundancesByChemicalElement const *Import::findEvaluation( std::string const &a_evaluation ) const {

    return( m_IAM->findEvaluation( a_evaluation ) );
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

LUPI_HOST void Import::serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode ) {

    EntryBase::serialize( a_buffer, a_mode );
    if( a_mode == LUPI::DataBuffer::Mode::Unpack ) m_IAM = new IAM;
    m_IAM->serialize( a_buffer, a_mode );
}

/*! \class IsotopicAbundancesByChemicalElement
 * Class for the **isotopicAbundancesByChemicalElement** child node inside an **iam** node.
 */

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

IsotopicAbundancesByChemicalElement::IsotopicAbundancesByChemicalElement( ) :
        EntryBase( CADI_isotopicAbundancesByChemicalElementChars ),
        m_isotopicAbundancesByChemicalElement( nullptr ) {

}

/* *********************************************************************************************************//**
 * IsotopicAbundancesByChemicalElement constructor.
 *
 * @param a_node                        [in]    HAPI node to be parsed and used to construct an *this* **Isotope**.
 ***********************************************************************************************************/

IsotopicAbundancesByChemicalElement::IsotopicAbundancesByChemicalElement( HAPI::Node const &a_node, std::string const &a_sourcePath ) :
        EntryBase( a_node, a_sourcePath ),
        m_evaluation( a_node.attribute_as_string( CADI_evaluationChars ) ),
        m_isotopicAbundancesByChemicalElement( nullptr ) {

    m_isotopicAbundancesByChemicalElement = new CADI::IsotopicAbundancesByChemicalElement( fileName( ) );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

IsotopicAbundancesByChemicalElement::~IsotopicAbundancesByChemicalElement( ) {

    delete m_isotopicAbundancesByChemicalElement;
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param a_symbol      [in]    The symbol for the chemical element whose isotopic abundance data are being requested.
 * @param a_evaluation  [in]    The name of the evaluation whose isotopic abundance data are being requested (optional).
 ***********************************************************************************************************/

ChemicalElement const *IsotopicAbundancesByChemicalElement::find( std::string const &a_symbol, std::string const &a_evaluation ) const {

    return( m_isotopicAbundancesByChemicalElement->find( a_symbol, a_evaluation ) );
}

/* *********************************************************************************************************//**
 * Return a *const* pointer to the **CADI::IsotopicAbundancesByChemicalElement** instance with evaluation *a_evaluation*, or a *nullptr*
 * if it does not exist.
 *
 * @param a_evaluation  [in]    The name of the evaluation whose isotopic abundance data are being requested.
 *
 * @return                      An **CADI::IsotopicAbundancesByChemicalElement** instance or *nullptr*.
 ***********************************************************************************************************/

CADI::IsotopicAbundancesByChemicalElement const *IsotopicAbundancesByChemicalElement::findEvaluation( std::string const &a_evaluation ) const {

    if( m_evaluation == a_evaluation ) return( m_isotopicAbundancesByChemicalElement );
    return( nullptr );
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

LUPI_HOST void IsotopicAbundancesByChemicalElement::serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode ) {

    EntryBase::serialize( a_buffer, a_mode );
    DATA_MEMBER_STD_STRING( m_evaluation, a_buffer, a_mode );

    if( a_mode == LUPI::DataBuffer::Mode::Unpack ) {
        m_isotopicAbundancesByChemicalElement = new CADI::IsotopicAbundancesByChemicalElement;
    }
    m_isotopicAbundancesByChemicalElement->serialize( a_buffer, a_mode );
}

/* *********************************************************************************************************//**
 * Returns the evaluation, checksum and algorithm attributes as an XML attribute string.
 *
 * @param a_checkAncestor   If true, do not include algorithm if it is the same as *this* parent's algorithm.
 ***********************************************************************************************************/

std::string IsotopicAbundancesByChemicalElement::standardXML_attributes( bool a_checkAncestor ) const {

    return( " evaluation=\"" + m_evaluation + "\"" + EntryBase::standardXML_attributes( a_checkAncestor ) );
}

/*! \class IAm
 * Class for the **iam** node.
 */

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

IAM::IAM( ) :
        Base( CADI_iamChars, "" ) {

}

/* *********************************************************************************************************//**
 * Constucts a new IAM from the contents of *a_sourcePath*.
 *
 * @param a_sourcePath  [in]    The path to the file to read in.
 ***********************************************************************************************************/

IAM::IAM( std::string const &a_sourcePath ):
        Base( CADI_iamChars, "" ) {

    addFile( a_sourcePath );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

IAM::~IAM( ) {

    for( auto entryIter = m_entries.begin( ); entryIter != m_entries.end( ); ++entryIter ) delete *entryIter;
}

/* *********************************************************************************************************//**
 * Adds the contents of *a_sourcePath* to *this*. This can only be called once and only for *this* created with no *a_sourcePath* specified.
 *
 * @param a_sourcePath  [in]    The path to the file to read in.
 ***********************************************************************************************************/

void IAM::addFile( std::string const &a_sourcePath ) {

    if( sourcePath( ) != "" ) throw LUPI::Exception( "IAM::addFile: cannot read in more than one file to *this*." );
    setSourcePath( a_sourcePath );

    HAPI::File *doc = new HAPI::PugiXMLFile( a_sourcePath.c_str( ), "IAM::addFile" );
//    if( result.status != pugi::status_ok ) throw LUPI::Exception( "ERROR reading file '" + a_sourcePath + "': " + result.description( ) );

    HAPI::Node iam = doc->first_child( );

    if( iam.name( ) != CADI_iamChars ) throw LUPI::Exception( "Invalid iam file " + a_sourcePath );

    m_library = iam.attribute_as_string( CADI_libraryChars );
    m_format = iam.attribute_as_string( CADI_formatChars );
    setChecksum( iam.attribute_as_string( CADI_checksumChars ) );
    setAlgorithm( iam.attribute_as_string( CADI_algorithmChars ) );

    EntryBase *entry = nullptr;
    for( auto child = iam.first_child( ); !child.empty( ); child.to_next_sibling( ) ) {
        if( child.name( ) == CADI_importChars ) {
            entry = new Import( child, a_sourcePath ); }
        else if( child.name( ) == CADI_isotopicAbundancesByChemicalElementChars ) {
            entry = new IsotopicAbundancesByChemicalElement( child, a_sourcePath ); }
        else {
            throw LUPI::Exception( std::string( "Invalid entry '" ) + child.name( ) + std::string( "' in iam file " ) + a_sourcePath );
        }
        entry->setAncestor( this );
        m_entries.push_back( entry );
    }

    delete doc;
}

/* *********************************************************************************************************//**
 * Return a *const* pointer to the **ChemicalElement** matching *a_symbol* and *a_evaluation*, or a nullptr
 * if one is not found. If *a_evaluation* is an empty string, then it matches every evalutation.
 *
 * @param a_symbol      [in]    The symbol for the chemical element whose isotopic abundance data are being requested.
 * @param a_evaluation  [in]    The name of the evaluation whose isotopic abundance data are being requested (optional).
 ***********************************************************************************************************/

ChemicalElement const *IAM::find( std::string const &a_symbol, std::string const &a_evaluation ) const {

    for( auto entryIter = m_entries.begin( ); entryIter != m_entries.end( ); ++entryIter ) {
        ChemicalElement const *chemicalElement = (*entryIter)->find( a_symbol, a_evaluation );

        if( chemicalElement != nullptr ) return( chemicalElement );
    }

    return( nullptr );
}

/* *********************************************************************************************************//**
 * Return a *const* pointer to the **CADI::IsotopicAbundancesByChemicalElement** instance with evaluation *a_evaluation*, or a *nullptr*
 * if it does not exist.
 *
 * @param a_evaluation  [in]    The name of the evaluation whose isotopic abundance data are being requested.
 *
 * @return                      An **CADI::IsotopicAbundancesByChemicalElement** instance or *nullptr*.
 ***********************************************************************************************************/

CADI::IsotopicAbundancesByChemicalElement const *IAM::findEvaluation( std::string const &a_evaluation ) const {

    CADI::IsotopicAbundancesByChemicalElement const *isotopicAbundancesByChemicalElement = nullptr;

    for( auto entryIter = m_entries.begin( ); entryIter != m_entries.end( ); ++entryIter ) {
        isotopicAbundancesByChemicalElement = (*entryIter)->findEvaluation( a_evaluation );
        if( isotopicAbundancesByChemicalElement != nullptr ) break;
    }

    return( isotopicAbundancesByChemicalElement );
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

LUPI_HOST void IAM::serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode ) {

    Base::serialize( a_buffer, a_mode );

    DATA_MEMBER_STD_STRING( m_library, a_buffer, a_mode );
    DATA_MEMBER_STD_STRING( m_format, a_buffer, a_mode );

    std::size_t vectorSize = m_entries.size( );
    int vectorSizeInt = (int) vectorSize;
    DATA_MEMBER_INT( vectorSizeInt, a_buffer, a_mode );
    vectorSize = (std::size_t) vectorSizeInt;

    if( a_mode == LUPI::DataBuffer::Mode::Unpack ) m_entries.resize( vectorSize, nullptr );

    for( std::size_t vectorIndex = 0; vectorIndex < vectorSize; ++vectorIndex ) {
        m_entries[vectorIndex] = serializeEntry( a_buffer, a_mode, m_entries[vectorIndex], this );
    }
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void IAM::toXMLList( GUPI::WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string header = LUPI_XML_verionEncoding;
    std::string attributes;

    a_writeInfo.push_back( header );

    attributes  = a_writeInfo.addAttribute( CADI_libraryChars, m_library );
    attributes += a_writeInfo.addAttribute( CADI_formatChars, m_format );
    attributes += standardXML_attributes( false );

    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );
    for( auto iter = m_entries.begin( ); iter != m_entries.end( ); ++iter ) (*iter)->toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/* *********************************************************************************************************//**
 * This function returns a unique integer for the **Distributions::Type**. For internal use when broadcasting a
 * distribution for MPI and GPUs needs.
 *               
 * @param a_type                [in]    The distribution's type.
 *
 * @return                              Returns a unique integer for the distribution type.
 ***********************************************************************************************************/

static LUPI_HOST EntryBase *serializeEntry( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode, EntryBase *a_entry, IAM *a_ancestor ) {

    int entryType = 0;

    if( a_mode != LUPI::DataBuffer::Mode::Unpack ) {
        if( a_entry->moniker( ) == CADI_importChars ) {
            entryType = 1; }
        else if( a_entry->moniker( ) == CADI_isotopicAbundancesByChemicalElementChars ) {
            entryType = 2;
        }
    }
    DATA_MEMBER_INT( entryType, a_buffer, a_mode );

    if( a_mode == LUPI::DataBuffer::Mode::Unpack ) {
        if( entryType == 1 ) {
            a_entry = new Import; }
        else {
            a_entry = new IsotopicAbundancesByChemicalElement;
        }
        a_entry->setAncestor( a_ancestor );
    }
    a_entry->serialize( a_buffer, a_mode );

    return( a_entry );
}

}               // End of namespace MAP.

}               // End of namespace CADI.

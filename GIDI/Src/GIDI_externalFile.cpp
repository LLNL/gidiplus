/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>
#include "GIDI.hpp"

namespace GIDI {

/*! \class ExternalFile
 * This class represents the **GNDS** <**externalFile**> node.
 */

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

ExternalFile::ExternalFile( std::string const &a_label, std::string const &a_path ) :
        Form( externalFileMoniker, FormType::externalFile, a_label ),
        m_path( a_path ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

ExternalFile::ExternalFile( pugi::xml_node const &a_node, GIDI::Suite *a_parent ) :
        Form( a_node, FormType::externalFile ),
        m_path( a_node.attribute( "path" ).value( ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

ExternalFile::~ExternalFile( ) {

}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML line that represent *this*.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void ExternalFile::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string attributes;

    attributes  = a_writeInfo.addAttribute( "label", label( ) );
    attributes += a_writeInfo.addAttribute( "path", path( ) );

    a_writeInfo.addNodeStarterEnder( a_indent, moniker( ), attributes );
}

}               // End of namespace GIDI.

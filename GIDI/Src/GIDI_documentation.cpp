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

namespace Documentation_1_10 {

#define GIDI_nameChars "name"

/*! \class Suite
 * This is the GIDI::Suite class but with a different parse function.
 */

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Suite::Suite( ) :
        GIDI::Suite( GIDI_documentations_1_10_Chars ) {

}

/* *********************************************************************************************************//**
 * Parse a GNDS 1.10 documentations node.
 *
 * @param a_node        [in]    The **pugi::xml_node** to be parsed.
 * @param a_setupInfo   [in]    Information create my the Protare constructor to help in parsing.
 ***********************************************************************************************************/

void Suite::parse( pugi::xml_node const &a_node, SetupInfo &a_setupInfo ) {

    for( pugi::xml_node child = a_node.first_child( ); child; child = child.next_sibling( ) ) {
        add( new Documentation( child, a_setupInfo, this ) );
    }
}

/*! \class Documentation
 */

/* *********************************************************************************************************//**
 *
 * @param a_node        [in]    The **pugi::xml_node** to be parsed.
 * @param a_setupInfo   [in]    Information create my the Protare constructor to help in parsing.
 * @param a_parent      [in]    The parent GIDI::Suite.
 * @return
 ***********************************************************************************************************/

Documentation::Documentation( pugi::xml_node const &a_node, SetupInfo &a_setupInfo, GIDI::Suite *a_parent ) :
        Form( GIDI_documentation_1_10_Chars, FormType::generic, a_node.attribute( GIDI_nameChars ).value() ) {

    m_label = a_node.attribute( GIDI_nameChars ).value();
    m_text = std::string( a_node.text().get() );
}

}

}

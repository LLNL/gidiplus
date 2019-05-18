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

namespace Documentation {

/*! \class Suite
 * This is essentially the GIDI::Suite class with the addition of the **findLabelInLineage** method.
 */

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Suite::Suite( ) :
        GIDI::Suite( documentationsMoniker ) {

}

/*!
 *  parse a <documentations> node into a Documentation::Suite
 */
void Suite::parse(pugi::xml_node const &a_node) {
    for( pugi::xml_node child = a_node.first_child( ); child; child = child.next_sibling( ) ) {
        add( new Documentation( child, this ) );
    }
}

/*! \class Documentation
 */

/* *********************************************************************************************************//**
 *
 * @param a_node        [in]    The **pugi::xml_node** to be parsed.
 * @param a_parent      [in]    The parent GIDI::Suite.
 * @return
 ***********************************************************************************************************/

Documentation::Documentation( pugi::xml_node const &a_node, GIDI::Suite *a_parent ) :
        Form( GIDI::f_generic, a_node.attribute( "name" ).value() ) {
    m_label = a_node.attribute( "name" ).value();
    m_text = std::string( a_node.text().get() );
}

}

}

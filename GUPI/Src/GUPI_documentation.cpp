/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "GUPI.hpp"

namespace GUPI {

/*! \class Documentation
 *
 */

Documentation::Documentation(HAPI::Node const &a_node) :
        Ancestry(a_node.name()),
        m_doi(a_node.attribute_as_string(GUPI_doiChars)),
        m_publicationDate(a_node.attribute_as_string(GUPI_publicationDateChars)),
        m_version(a_node.attribute_as_string(GUPI_versionChars)),
        m_title(a_node.child(GUPI_titleChars)),
        m_abstract(a_node.child(GUPI_abstractChars)),
        m_body(a_node.child(GUPI_bodyChars)) {

    m_title.setAncestor(this);
    m_abstract.setAncestor(this);
    m_body.setAncestor(this);
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Documentation::~Documentation( ) {

}

}

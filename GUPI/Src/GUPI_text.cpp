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

/*! \class Text
 * 
 */

Text::Text(HAPI::Node const &a_node):
        Ancestry(a_node.name()), 
        m_body(a_node.text().get()), 
        m_encoding(Encoding::ascii), 
        m_markup(Markup::none), 
        m_label(a_node.attribute_as_string("label")) {
}

Text::~Text(){}
}

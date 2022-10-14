/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "HAPI.hpp"

namespace HAPI {

Node_internal::Node_internal( NodeInteralType a_type ) :
        m_type( a_type ) {
}

Node_internal::Node_internal( Node_internal const &a_node ) :
        m_type( a_node.type( ) ) {
}

/*
============================================================
======================= destructor =========================
============================================================
 *
 * @return
 */
Node_internal::~Node_internal( ) {

}

}

/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "PoPs.hpp"

namespace PoPs {

/*
=========================================================
*/
Unorthodox::Unorthodox( pugi::xml_node const &a_node, Database *a_DB, Database *a_parent ) :
        Particle( a_node, class_unorthodox, family_unorthodox ) {

    addToDatabase( a_DB );
}
/*
=========================================================
*/
Unorthodox::~Unorthodox( ) {

}

}

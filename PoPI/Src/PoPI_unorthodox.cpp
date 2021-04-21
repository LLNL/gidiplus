/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "PoPI.hpp"

namespace PoPI {

/*
=========================================================
*/
Unorthodox::Unorthodox( pugi::xml_node const &a_node, Database *a_DB, Database *a_parent ) :
        Particle( a_node, Particle_class::unorthodox, PoPI_unorthodoxChars ) {

    addToDatabase( a_DB );
}
/*
=========================================================
*/
Unorthodox::~Unorthodox( ) {

}

}

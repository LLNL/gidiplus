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
Baryon::Baryon( pugi::xml_node const &a_node, Database *a_DB, Database *a_parent ) :
        Particle( a_node, Particle_class::baryon, PoPI_baryonChars ) {

    addToDatabase( a_DB );
}
/*
=========================================================
*/
Baryon::~Baryon( ) {

}

}

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
GaugeBoson::GaugeBoson( pugi::xml_node const &a_node, Database *a_DB, Database *a_parent ) :
        Particle( a_node, Particle_class::gaugeBoson, PoPI_gaugeBosonChars ) {

    addToDatabase( a_DB );
}
/*
=========================================================
*/
GaugeBoson::~GaugeBoson( ) {

}

}

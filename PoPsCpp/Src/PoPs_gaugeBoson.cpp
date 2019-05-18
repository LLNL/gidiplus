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
GaugeBoson::GaugeBoson( pugi::xml_node const &a_node, Database *a_DB, Database *a_parent ) :
        Particle( a_node, class_gaugeBoson, family_gaugeBoson ) {

    addToDatabase( a_DB );
}
/*
=========================================================
*/
GaugeBoson::~GaugeBoson( ) {

}

}

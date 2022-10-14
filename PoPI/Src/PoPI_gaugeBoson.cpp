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

/*! \class GaugeBoson
 * This class represents **PoPs** gaugeBoson instance.
 */

/* *********************************************************************************************************//**
 * Constructor that parses an **HAPI** instance to create a **PoPs** gaugeBoson node.
 *
 * @param a_node            [in]    The **HAPI::Node** to be parsed.
 * @param a_DB              [in]    The **PoPI::Database:: instance to add the constructed **GaugeBoson** to.
 * @param a_parent          [in]    The parent suite that will contain *this*.
 ***********************************************************************************************************/

GaugeBoson::GaugeBoson( HAPI::Node const &a_node, Database *a_DB, Database *a_parent ) :
        Particle( a_node, Particle_class::gaugeBoson, PoPI_gaugeBosonChars ) {

    addToDatabase( a_DB );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

GaugeBoson::~GaugeBoson( ) {

}

}

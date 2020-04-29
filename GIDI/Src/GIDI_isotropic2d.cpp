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

namespace Functions {

/*! \class Isotropic2d
 * Class for the GNDS <**isotropic2d**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction    [in]    Used to pass user options to the constructor.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed and used to construct the XYs2d.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

Isotropic2d::Isotropic2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Function2dForm( a_construction, a_node, FormType::isotropic2d, a_parent ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Isotropic2d::~Isotropic2d( ) {

}

/* *********************************************************************************************************//**
 * Returns the domain minimum for the instance.
 *
 * @return          The domain minimum for the instance.
 ***********************************************************************************************************/

double Isotropic2d::domainMin( ) const {

// BRB FIXME.
    return( 0. );
}

/* *********************************************************************************************************//**
 * Returns the domain maximum for the instance.
 *
 * @return              The domain maximum for the instance.
 ***********************************************************************************************************/

double Isotropic2d::domainMax( ) const {

// BRB FIXME.
    return( 0. );
}

/* *********************************************************************************************************//**
 * Returns the value 0.5.
 *
 * @param a_x2              [in]    The is ignored.
 * @param a_x1              [in]    The is ignored.
 * @return                          The value 0.5.
 ***********************************************************************************************************/

double Isotropic2d::evaluate( double a_x2, double a_x1 ) const {

    return( 0.5 );
}

}               // End namespace Functions.

}               // End namespace GIDI.

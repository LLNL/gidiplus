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

/*! \class Reference1d
 * Class for the GNDS <**reference**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction    [in]     Used to pass user options to the constructor.
 * @param a_node            [in]     The **pugi::xml_node** to be parsed and used to construct the XYs2d.
 * @param a_parent          [in]     The parent GIDI::Suite.
 ***********************************************************************************************************/

Reference1d::Reference1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Function1dForm( a_construction, a_node, f_reference1d, a_parent ),
        m_xlink( a_node.attribute( "href" ).value( ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Reference1d::~Reference1d( ) {

}

/* *********************************************************************************************************//**
 * Returns the domain minimum for the instance.
 *
 * @return          The domain minimum for the instance.
 ***********************************************************************************************************/

double Reference1d::domainMin( ) const {

    throw std::runtime_error( "Reference1d::domainMin: not implemented" );
}

/* *********************************************************************************************************//**
 * Returns the domain maximum for the instance.
 *
 * @return              The domain maximum for the instance.
 ***********************************************************************************************************/

double Reference1d::domainMax( ) const {

    throw std::runtime_error( "Reference1d::domainMax: not implemented" );
}

/* *********************************************************************************************************//**
 * The **y** value of this at the domain value **a_x1**.
 * Currently not implemented.
 *
 * @param a_x1          [in]    Domain value to evaluate this at.
 * @return                      The value of this at the domain value **a_x1**.
 ***********************************************************************************************************/


double Reference1d::evaluate( double a_x1 ) const {

    throw std::runtime_error( "Reference1d::evaluate: not implemented" );
}

}

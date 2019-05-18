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

/*! \class Recoil2d
 * Class for the GNDS <**recoil**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_label           [in]     The GNDS **label** for *this*.
 * @param a_href            [in]     The GNDS **href** for *this*.
 ***********************************************************************************************************/

Recoil2d::Recoil2d( std::string const &a_label, std::string const &a_href ) :
        Function2dForm( recoilMoniker, f_recoil2d, a_label ),
        m_xlink( a_href ) {

}

/* *********************************************************************************************************//**
 *
 * @param a_construction    [in]     Used to pass user options to the constructor.
 * @param a_node            [in]     The **pugi::xml_node** to be parsed and used to construct the XYs2d.
 * @param a_parent          [in]     The parent GIDI::Suite.
 ***********************************************************************************************************/

Recoil2d::Recoil2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Function2dForm( a_construction, a_node, f_recoil2d, a_parent ),
        m_xlink( a_node.attribute( "href" ).value( ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Recoil2d::~Recoil2d( ) {

}

/* *********************************************************************************************************//**
 * Returns the domain minimum for the instance.
 *
 * @return          The domain minimum for the instance.
 ***********************************************************************************************************/

double Recoil2d::domainMin( ) const {

    return( 0.0 );              // FIXME
}

/* *********************************************************************************************************//**
 * Returns the domain maximum for the instance.
 *
 * @return              The domain maximum for the instance.
 ***********************************************************************************************************/

double Recoil2d::domainMax( ) const {

    return( 1.0 );              // FIXME
}

/* *********************************************************************************************************//**
 * The angular probability *P(mu|E)* at the point *E* = *a_x2* and *mu* = *a_x1*.
 * Currently not implemented.
 *
 * @param a_x2          [in]    The projectile's energy.
 * @param a_x1          [in]    The product's mu value.
 * @return                      The value of *P(mu|E)*.
 ***********************************************************************************************************/

double Recoil2d::evaluate( double a_x2, double a_x1 ) const {

    throw std::runtime_error( "Recoil2d::evaluate: not implemented" );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 * @param       a_embedded          [in]        If *true*, *this* function is embedded in a higher dimensional function.
 * @param       a_inRegions         [in]        This is not used in this method.
 ***********************************************************************************************************/

void Recoil2d::toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const {

    a_writeInfo.addNodeStarterEnder( a_indent, moniker( ), a_writeInfo.addAttribute( "href", m_xlink ) );
}

}

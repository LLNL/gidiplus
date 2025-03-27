/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>
#include <algorithm>

#include <HAPI.hpp>
#include <CADI.hpp>

namespace CADI {

/*! \class Isotope
 * The class that stores the atom fraction and its uncertainty for a chemical element's isotope.
 */

/* *********************************************************************************************************//**
 * Isotope constructor.
 *
 * @param a_id                  The **PoPs** id for the isotope.
 * @param a_atomFraction        The atom fraction for the isotope.
 * @param a_uncertainty         The uncertainty in the atom fraction.
 ***********************************************************************************************************/

Isotope::Isotope( std::string a_id, double a_atomFraction, double a_uncertainty ) :
        GUPI::Entry( CADI_isotopeChars, CADI_idChars, a_id ),
        m_atomFraction( a_atomFraction ),
        m_uncertainty( a_uncertainty ) {

}

/* *********************************************************************************************************//**
 * Isotope constructor.
 *
 * @param a_node                        [in]    HAPI node to be parsed and used to construct an *this* **Isotope**.
 ***********************************************************************************************************/

Isotope::Isotope( HAPI::Node const &a_node ) :
        GUPI::Entry( a_node, CADI_idChars ),
        m_atomFraction( a_node.attribute( CADI_atomFractionChars ).as_double( ) ),
        m_uncertainty( a_node.attribute( CADI_uncertaintyChars ).as_double( ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Isotope::~Isotope( ) {

}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

LUPI_HOST void Isotope::serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode ) {

    GUPI::Entry::serialize( a_buffer, a_mode );
    DATA_MEMBER_DOUBLE( m_atomFraction, a_buffer, a_mode );
    DATA_MEMBER_DOUBLE( m_uncertainty, a_buffer, a_mode );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void Isotope::toXMLList( GUPI::WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string attributes;

    attributes  = a_writeInfo.addAttribute( CADI_idChars, id( ) );
    attributes += a_writeInfo.addAttribute( CADI_atomFractionChars, LUPI::Misc::doubleToShortestString( atomFraction( ) ) );
    attributes += a_writeInfo.addAttribute( CADI_uncertaintyChars, LUPI::Misc::doubleToShortestString( uncertainty( ) ) );
    a_writeInfo.addNodeStarterEnder( a_indent, moniker( ), attributes );
}

}               // End of namespace CADI.

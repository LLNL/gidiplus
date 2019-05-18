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

namespace Settings {

/*! \class Settings
 * This class is used to instruct deterministic methods and the Monte Carlo API MCGIDI on what data are being requested.
*/

/* *********************************************************************************************************//**
 * @param a_projectileID            [in]    The PoPs id for the projectile.
 * @param a_label                   [in]    The label of the data being requested.
 * @param a_delayedNeutrons         [in]    Flag indicating whether or not delayed neutron data are to be include in the requested data.
 ***********************************************************************************************************/

Settings::Settings( std::string const &a_projectileID, std::string const &a_label, bool a_delayedNeutrons ) :
    m_projectileID( a_projectileID ),
    m_label( a_label ),
    m_delayedNeutrons( a_delayedNeutrons ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Settings::~Settings( ) {

}

/* *********************************************************************************************************//**
 * Returns a Vector of 0.0's of the proper length for the projectile's multi-group data.
 *
 * @param a_particles   [in]    The list of particles to be transported.
 * @param a_collapse    [in]    If true, the length of the returned vector is for the collapsed multi-group, otherwise, it is for the uncollapsed multi-group.
 *
 * @return                      The Vector of 0.0.
 ***********************************************************************************************************/

Vector Settings::multiGroupZeroVector( Particles const &a_particles, bool a_collapse ) const {

    Particle const *projectile( a_particles.particle( projectileID( ) ) );

    int n1 = projectile->fineMultiGroup( ).numberOfGroups( );
    if( a_collapse ) n1 = projectile->numberOfGroups( );

    Vector vector( n1 );
    return( vector );
}

/* *********************************************************************************************************//**
 * Returns a Matrix of 0.0's of the proper length for the projectile's and product's multi-group data.
 *
 * @param a_particles               [in]    The list of particles to be transported.
 * @param a_pid                     [in]    The PoPs index for the product.
 * @param a_collapse                [in]    If true, the length of the returned vector is for the collapsed multi-group, otherwise, it is for the uncollapsed multi-group.
 *
 * @return                                  The Matrix of 0.0.
 ***********************************************************************************************************/

Matrix Settings::multiGroupZeroMatrix( Particles const &a_particles, std::string const &a_pid, bool a_collapse ) const {

    Particle const *projectile( a_particles.particle( projectileID( ) ) );
    Particle const *product( a_particles.particle( a_pid ) );

    int n1 = projectile->fineMultiGroup( ).numberOfGroups( );
    int n2 = product->fineMultiGroup( ).numberOfGroups( );
    if( a_collapse ) {
        n1 = projectile->numberOfGroups( );
        n2 = product->numberOfGroups( );
    }

    Matrix matrix( n1, n2 );
    return( matrix );
}

/* *********************************************************************************************************//**
 * Prints the contents of *this* to std::cout. Mainly used for debugging.
 ***********************************************************************************************************/

void Settings::print( ) const {

    std::cout << "setting info:" << std::endl;
    std::cout << "  style label = " << m_label << std::endl;

    std::cout << "  delayed neutrons ";
    if( m_delayedNeutrons ) {
        std::cout << "on" << std::endl; }
    else {
        std::cout << "off" << std::endl;
    }
}

/*! \class MG
 * This class is used to instruct deterministic methods on what data are being requested.
*/

/* *********************************************************************************************************//**
 * @param a_projectileID            [in]    The PoPs index for the projectile.
 * @param a_label                   [in]    The label of the data being requested.
 * @param a_delayedNeutrons         [in]    Flag indicating whether or not delayed neutron data are to be include in the requested data.
 ***********************************************************************************************************/

MG::MG( std::string const &a_projectileID, std::string const &a_label, bool a_delayedNeutrons ) :
        Settings( a_projectileID, a_label, a_delayedNeutrons ) {

}

}

}

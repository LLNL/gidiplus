/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <climits>
#include <PoPI.hpp>

namespace PoPI {

/* *********************************************************************************************************//**
 * Returns an integer representing the particle's famuly *a_family*.
 *
 * @param a_isAnti          [in]    If **true** particle is an anti-particle and otherwise its a particle.
 ***********************************************************************************************************/

int family2Integer( Particle_class a_family ) {

    if( a_family == Particle_class::nucleus ) return( -1 );
    if( a_family == Particle_class::nuclide ) return( -2 );
    if( a_family == Particle_class::gaugeBoson ) return( 0 );
    if( a_family == Particle_class::lepton ) return( 1 );
    if( a_family == Particle_class::baryon ) return( 2 );
    if( a_family == Particle_class::ENDL_fissionProduct ) return( 98 );

// Still need ENDL_fissionProduct and TNSL.

    return( -3 );
}

/* *********************************************************************************************************//**
 * This function is for internal use.
 * Returns the intid for the particle of family *a_family* with family indentifier *a_SSSSSSS*.
 *
 * @param a_isAnti          [in]    If **true** particle is an anti-particle and otherwise its a particle.
 * @param a_family          [in]    The particle's family.
 * @param a_SSSSSSS         [in]    The particle's indentifier within its family.
 ***********************************************************************************************************/

int intidHelper( bool a_isAnti, Particle_class a_family, int a_SSSSSSS ) {

    int sign = a_isAnti ? -1 : 1;

    int intid = family2Integer( a_family );
    if( intid < 0 ) return( -1 );
    intid += 100;
    intid *= 10000000;

    return( sign * ( intid + a_SSSSSSS ) );
}

/*! \class ParseIntidInfo
 * This class represents **PoPs** nucleus instance.
 */

/* *********************************************************************************************************//**
 * Constructor that parses *a_intid* into its components and sets members per *a_intid*. If *m_III*, *m_ZZZ*, *m_AAA* and
 * *m_metaStableIndex* are positive (greater than or equal to 0), then the particle is a nuclear particle and 
 * *m_isNuclear* is *true*, otherwise the particle is not a nuclear particle and *m_isNuclear* is *false*.
 * Note, even if *m_metaStableIndex* > 0 (i.e., particle is a nuclear meta-stable), *m_III* is as expected. For
 * example, for intid = 481095242, *m_metaStableIndex* is 1 and *m_III* is 481.
 *
 * If *m_family* is **Particle_class::unknown** then all other members are undefined.
 *
 * @param a_intid           [in]    The intid for the particle to parse.
 ***********************************************************************************************************/

ParseIntidInfo::ParseIntidInfo( int a_intid ) :
        m_intid( a_intid ),
        m_family( Particle_class::unknown ),
        m_isAnti( a_intid < 0 ),
        m_isNuclear( false ),
        m_AAA( -1 ),
        m_ZZZ( -1 ),
        m_III( -1 ),
        m_nuclearLevelIndex( -1 ),
        m_metaStableIndex( -1 ),
        m_generation( -1 ),
        m_isNeutrino( false ),
        m_baryonGroup( -1 ),
        m_baryonId( -1 ),
        m_familyId( -1 ) {

    int intidAbs = std::abs( a_intid );

    int nuclearLike = intidAbs / 1000000000;
    int family = (intidAbs / 10000000) % 100;
    int SSSSSSS = intidAbs % 10000000;

    if(      nuclearLike == 0 ) {
        m_AAA = intidAbs % 1000;
        m_ZZZ = intidAbs % 1000000 / 1000;
        m_III = intidAbs % 1000000000 / 1000000;

        int III = m_III;
        if( m_III < 500 ) {
            m_family = Particle_class::nuclide; }
        else {
            III -= 500;
            m_family =  Particle_class::nucleus;
        }
        if( III <= 480 ) {
            m_nuclearLevelIndex = III; }
        else {
            m_metaStableIndex = III - 480;
        } }
    else if( nuclearLike == 1 ) {
        m_familyId = SSSSSSS;
        if(      family == 0 ) {
            m_family =  Particle_class::gaugeBoson; }
        else if( family == 1 ) {
            int neutronoFlag = ( SSSSSSS % 100 ) / 10;
            if( neutronoFlag > 1 ) return;                      // Invalid particle.

            m_family =  Particle_class::lepton;
            m_generation = SSSSSSS % 10;
            m_isNeutrino = neutronoFlag != 0; }
        else if( family == 2 ) {
            m_family =  Particle_class::baryon;
            m_baryonGroup = SSSSSSS / 1000000;
            m_baryonId = SSSSSSS % 1000000; }
        else if( family == 98 ) {
            m_family =  Particle_class::ENDL_fissionProduct; }
        else if( family == 99 ) {
            m_family =  Particle_class::TNSL;
        }
    }
}

/* *********************************************************************************************************//**
 * Returns the GNDS PoPs id for *this*. If particles is unknown, an empty string is returned.
 *
 ***********************************************************************************************************/

std::string ParseIntidInfo::id( ) {

    std::string pid;

    if( ( m_family == Particle_class::nuclide ) || ( m_family == Particle_class::nucleus ) ) {
        pid = chemicalElementInfoFromZ( m_ZZZ, true, m_family == Particle_class::nucleus );
        if( pid != "" ) {
            pid += LUPI::Misc::argumentsToString( "%d", m_AAA );

            int III = m_III;
            if( m_family == Particle_class::nucleus ) III -= 500;
            if( III > 0 ) {
                if( m_metaStableIndex > 0 ) {
                    pid += LUPI::Misc::argumentsToString( "_m%d", m_metaStableIndex ); }
                else {
                    pid += LUPI::Misc::argumentsToString( "_e%d", III );
                }
            }
        } }
    else if( m_family == Particle_class::gaugeBoson ) {
        if( m_familyId == 0 ) pid = IDs::photon; }
    else if( m_family == Particle_class::lepton ) {
        if( m_generation == 0 ) {
            if( !m_isNeutrino ) pid = IDs::electron;
        } }
    else if( m_family == Particle_class::baryon ) {
        if( m_baryonGroup == 0 ) {
            switch( m_baryonId ) {
            case 0:
                pid = IDs::neutron;
                break;
            case 1:
                pid = IDs::proton;
                break;
            default:
                break;
            }
        } }
    else if( m_family == Particle_class::ENDL_fissionProduct ) {
        if( m_familyId == 99120 ) {
            pid = IDs::FissionProductENDL99120; }
        else if( m_familyId == 99125 ) {
            pid = IDs::FissionProductENDL99125;
        }
    }

    if( ( pid.size( ) > 0 ) &&  m_isAnti ) pid += IDs::anti;

    return( pid );
}

}

/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "MCGIDI.hpp"

namespace MCGIDI {

namespace Sampling {

/*
=========================================================
*/
LUPI_HOST_DEVICE ClientRandomNumberGenerator::ClientRandomNumberGenerator( double (*a_generator)( void * ), void *a_state ) :
        m_generator( a_generator ),
        m_state( a_state ) {
}

/*
=========================================================
*/
LUPI_HOST_DEVICE ClientCodeRNGData::ClientCodeRNGData( double (*a_generator)( void * ), void *a_state ) :
        ClientRandomNumberGenerator( a_generator, a_state ) {
}

/*
=========================================================
*/
LUPI_HOST_DEVICE Input::Input( bool a_wantVelocity, Upscatter::Model a_upscatterModel ) :
        m_wantVelocity( a_wantVelocity ),
        m_upscatterModel( a_upscatterModel ),

        m_sampledType( SampledType::uncorrelatedBody ),
        m_reaction( nullptr ),

        m_frame( GIDI::Frame::lab ),
        m_mu( 0.0 ),
        m_phi( 0.0 ),

        m_energyOut1( 0.0 ),
        m_px_vx1( 0.0 ),
        m_py_vy1( 0.0 ),
        m_pz_vz1( 0.0 ),

        m_energyOut2( 0.0 ),
        m_px_vx2( 0.0 ),
        m_py_vy2( 0.0 ),
        m_pz_vz2( 0.0 ),

        m_delayedNeutronIndex( -1 ),
        m_delayedNeutronDecayRate( 0.0 ) {

}

/*
============================================================
========================= ProductHandler ===================
============================================================
*/
LUPI_HOST_DEVICE void ProductHandler::add( double a_projectileEnergy, int a_productIndex, int a_userProductIndex, double a_productMass, Input &a_input, 
                double (*a_userrng)( void * ), void *a_rngState, bool a_isPhoton ) {

    Product product;

    if( a_isPhoton && ( a_input.m_sampledType != SampledType::unspecified ) ) a_input.m_sampledType = SampledType::photon;

    product.m_sampledType = a_input.m_sampledType;
    product.m_isVelocity = a_input.wantVelocity( );
    product.m_productIndex = a_productIndex;
    product.m_userProductIndex = a_userProductIndex;
    product.m_productMass = a_productMass;

    product.m_delayedNeutronIndex = a_input.m_delayedNeutronIndex;
    product.m_delayedNeutronDecayRate = a_input.m_delayedNeutronDecayRate;
    product.m_birthTimeSec = 0.;
    if( product.m_delayedNeutronDecayRate > 0. ) {
        product.m_birthTimeSec = -log( a_userrng( a_rngState ) ) / product.m_delayedNeutronDecayRate;
    }

    if( a_input.m_sampledType == SampledType::unspecified ) {
        product.m_kineticEnergy = 0.0;
        product.m_px_vx = 0.0;
        product.m_py_vy = 0.0;
        product.m_pz_vz = 0.0; }
    else if( a_input.m_sampledType == SampledType::uncorrelatedBody ) {
        if( a_input.m_frame == GIDI::Frame::centerOfMass ) {
            a_input.m_frame = GIDI::Frame::lab;

            double massRatio = a_input.m_projectileMass + a_input.m_targetMass;
            massRatio = a_input.m_projectileMass * a_productMass / ( massRatio * massRatio );
            double modifiedProjectileEnergy = massRatio * a_projectileEnergy;

            double sqrtModifiedProjectileEnergy = sqrt( modifiedProjectileEnergy );
            double sqrtEnergyOut_com = a_input.m_mu * sqrt( a_input.m_energyOut1 );

            a_input.m_energyOut1 += modifiedProjectileEnergy + 2. * sqrtModifiedProjectileEnergy * sqrtEnergyOut_com;
            if( a_input.m_energyOut1 != 0 ) a_input.m_mu = ( sqrtModifiedProjectileEnergy + sqrtEnergyOut_com ) / sqrt( a_input.m_energyOut1 );
        }

        product.m_kineticEnergy = a_input.m_energyOut1;

        double p_v = sqrt( a_input.m_energyOut1 * ( a_input.m_energyOut1 + 2. * a_productMass ) );
        if( product.m_isVelocity ) p_v *= MCGIDI_speedOfLight_cm_sec / ( a_input.m_energyOut1 + a_productMass );

        product.m_pz_vz = p_v * a_input.m_mu;
        p_v *= sqrt( 1. - a_input.m_mu * a_input.m_mu );
        product.m_px_vx = p_v * sin( a_input.m_phi );
        product.m_py_vy = p_v * cos( a_input.m_phi ); }
    else if( a_input.m_sampledType == SampledType::firstTwoBody ) {
        product.m_kineticEnergy = a_input.m_energyOut1;
        product.m_px_vx = a_input.m_px_vx1;
        product.m_py_vy = a_input.m_py_vy1;
        product.m_pz_vz = a_input.m_pz_vz1;
        a_input.m_sampledType = SampledType::secondTwoBody; }
    else if( a_input.m_sampledType == SampledType::secondTwoBody ) {
        product.m_kineticEnergy = a_input.m_energyOut2;
        product.m_px_vx = a_input.m_px_vx2;
        product.m_py_vy = a_input.m_py_vy2;
        product.m_pz_vz = a_input.m_pz_vz2; }
    else if( a_input.m_sampledType == SampledType::photon ) {
        product.m_kineticEnergy = a_input.m_energyOut1;

        double pz_vz_factor = a_input.m_energyOut1;
        if( product.m_isVelocity ) pz_vz_factor = MCGIDI_speedOfLight_cm_sec;
        product.m_pz_vz = a_input.m_mu * pz_vz_factor;

        double v_perp = sqrt( 1.0 - a_input.m_mu * a_input.m_mu ) * pz_vz_factor;
        product.m_px_vx = cos( a_input.m_phi ) * v_perp;
        product.m_py_vy = sin( a_input.m_phi ) * v_perp; }
    else {
        product.m_kineticEnergy = a_input.m_energyOut2;
        product.m_px_vx = a_input.m_px_vx2;
        product.m_py_vy = a_input.m_py_vy2;
        product.m_pz_vz = a_input.m_pz_vz2;
    }

    if( a_input.m_dataInTargetFrame && ( a_input.m_sampledType != SampledType::photon ) ) upScatterModelABoostParticle( a_input, a_userrng, a_rngState, product );

    push_back( product );
}

}           // End of namespace Sampling.

}           // End of namespace MCGIDI.

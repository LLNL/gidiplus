/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <math.h>

#include "MCGIDI.hpp"
#include "MCGIDI_declareMacro.hpp"

static const double C0 = 1.0410423479, C1 = 3.9626339162e-4, C2 =-1.8654539193e-3, C3 = 1.0264818153e-4;

static const double Two_sqrtPi = 1.1283791670955125739;                                                       // 2 / Sqrt( Pi ).

namespace MCGIDI {

namespace Distributions {

HOST_DEVICE static void kinetics_COMKineticEnergy2LabEnergyAndMomentum( double a_beta, double a_kinetic_com, 
        double a_m3cc, double a_m4cc, Sampling::Input &a_input );
HOST_DEVICE static double coherentPhotoAtomicScatteringIntegrateSub( int a_n, double a_a, double a_logX, double a_energy1, double a_y1, double a_energy2, double a_y2 );

/*! \class Distribution
 * This class is the base class for all distribution forms.
 */

/* *********************************************************************************************************//**
 * Default constructor used when broadcasting a Protare as needed by MPI or GPUs.
 ***********************************************************************************************************/

HOST_DEVICE Distribution::Distribution( ) :
        m_type( Type::none ),
        m_productFrame( GIDI::Frame::lab ),
        m_projectileMass( 0.0 ),
        m_targetMass( 0.0 ),
        m_productMass( 0.0 ) {

}

/* *********************************************************************************************************//**
 * @param a_type                [in]    The Type of the distribution.
 * @param a_distribution        [in]    The GIDI::Distributions::Distribution instance whose data is to be used to construct *this*.
 * @param a_setupInfo           in]    Used internally when constructing a Protare to pass information to other constructors.
 ***********************************************************************************************************/

HOST Distribution::Distribution( Type a_type, GIDI::Distributions::Distribution const &a_distribution, SetupInfo &a_setupInfo ) :
        m_type( a_type ),
        m_productFrame( a_distribution.productFrame( ) ),
        m_projectileMass( a_setupInfo.m_protare.projectileMass( ) ),
        m_targetMass( a_setupInfo.m_protare.targetMass( ) ),
        m_productMass( a_setupInfo.m_product1Mass ) {                           // Includes nuclear excitation energy.

}

/* *********************************************************************************************************//**
 * @param a_type                [in]    The Type of the distribution.
 * @param a_productFrame        [in]    The frame of the product's data for distribution.
 * @param a_setupInfo               [in]    Used internally when constructing a Protare to pass information to other constructors.
 ***********************************************************************************************************/

HOST Distribution::Distribution( Type a_type, GIDI::Frame a_productFrame, SetupInfo &a_setupInfo ) :
        m_type( a_type ),
        m_productFrame( a_productFrame ),
        m_projectileMass( a_setupInfo.m_protare.projectileMass( ) ),
        m_targetMass( a_setupInfo.m_protare.targetMass( ) ),
        m_productMass( a_setupInfo.m_product1Mass ) {                           // Includes nuclear excitation energy.

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

HOST_DEVICE Distribution::~Distribution( ) {

}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

HOST_DEVICE void Distribution::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    int distributionType = 0;
    switch( m_type ) {
    case Distributions::Type::none :
        break;
    case Distributions::Type::unspecified :
        distributionType = 1;
        break;
    case Distributions::Type::angularTwoBody :
        distributionType = 2;
        break;
    case Distributions::Type::KalbachMann :
        distributionType = 3;
        break;
    case Distributions::Type::uncorrelated :
        distributionType = 4;
        break;
    case Distributions::Type::energyAngularMC :
        distributionType = 5;
        break;
    case Distributions::Type::angularEnergyMC :
        distributionType = 6;
        break;
    case Distributions::Type::coherentPhotoAtomicScattering :
        distributionType = 7;
        break;
    case Distributions::Type::incoherentPhotoAtomicScattering :
        distributionType = 8;
        break;
    case Distributions::Type::pairProductionGamma :
        distributionType = 9;
        break;
    }
    DATA_MEMBER_INT( distributionType, a_buffer, a_mode );
    if( a_mode == DataBuffer::Mode::Unpack ) {
        switch( distributionType ) {
        case 0 :
            m_type = Distributions::Type::none;
            break;
        case 1 :
            m_type = Distributions::Type::unspecified;
            break;
        case 2 :
            m_type = Distributions::Type::angularTwoBody;
            break;
        case 3 :
            m_type = Distributions::Type::KalbachMann;
            break;
        case 4 :
            m_type = Distributions::Type::uncorrelated;
            break;
        case 5 :
            m_type = Distributions::Type::energyAngularMC;
            break;
        case 6 :
            m_type = Distributions::Type::angularEnergyMC;
            break;
        case 7 :
            m_type = Distributions::Type::coherentPhotoAtomicScattering;
            break;
        case 8 :
            m_type = Distributions::Type::incoherentPhotoAtomicScattering;
            break;
        case 9 :
            m_type = Distributions::Type::pairProductionGamma;
            break;
        }
    }

    int frame = 0;
    if( m_productFrame == GIDI::Frame::centerOfMass ) frame = 1;
    DATA_MEMBER_INT( frame, a_buffer, a_mode );
    m_productFrame = GIDI::Frame::lab;
    if( frame == 1 ) m_productFrame = GIDI::Frame::centerOfMass;

    DATA_MEMBER_FLOAT( m_projectileMass, a_buffer, a_mode );
    DATA_MEMBER_FLOAT( m_targetMass, a_buffer, a_mode );
    DATA_MEMBER_FLOAT( m_productMass, a_buffer, a_mode );
}

/*! \class AngularTwoBody
 * This class represents the distribution for an outgoing product for a two-body interaction.
 */

HOST_DEVICE AngularTwoBody::AngularTwoBody( ) :
        m_residualMass( 0.0 ),
        m_Q( 0.0 ),
        m_crossSectionThreshold( 0.0 ),
        m_Upscatter( false ),
        m_angular( nullptr ) {

}

/* *********************************************************************************************************//**
 * @param a_angularTwoBody          [in]    The GIDI::Distributions::AngularTwoBody instance whose data is to be used to construct *this*.
 * @param a_setupInfo               [in]    Used internally when constructing a Protare to pass information to other constructors.
 ***********************************************************************************************************/

HOST AngularTwoBody::AngularTwoBody( GIDI::Distributions::AngularTwoBody const &a_angularTwoBody, SetupInfo &a_setupInfo ) :
        Distribution( Type::angularTwoBody, a_angularTwoBody, a_setupInfo ),
        m_residualMass( a_setupInfo.m_product2Mass ),                           // Includes nuclear excitation energy.
        m_Q( a_setupInfo.m_Q ),
        m_crossSectionThreshold( a_setupInfo.m_reaction->crossSectionThreshold( ) ),
        m_Upscatter( false ),
        m_angular( Probabilities::parseProbability2d( a_angularTwoBody.angular( ), &a_setupInfo ) ) {

    if( a_setupInfo.m_protare.projectileIndex( ) == a_setupInfo.m_protare.neutronIndex( ) ) {
        m_Upscatter = a_setupInfo.m_reaction->ENDF_MT( ) == 2;
    }
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

HOST_DEVICE AngularTwoBody::~AngularTwoBody( ) {

    delete m_angular;
}

/* *********************************************************************************************************//**
 * This method samples the outgoing product data for the two outgoing particles in a two-body outgoing channel. 
 * First, is samples *mu*, the cosine of the product's outgoing angle, since this is for two-body interactions, *mu*
 * is in the center-of-mass frame. It then calls kinetics_COMKineticEnergy2LabEnergyAndMomentum.
 *
 * @param a_X                       [in]    The energy of the projectile in the lab frame.
 * @param a_input                   [in]    Sample options requested by user.
 * @param a_userrng                 [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState                [in]    The current state for the random number generator.
 ***********************************************************************************************************/

HOST_DEVICE void AngularTwoBody::sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const {

    double initialMass = projectileMass( ) + targetMass( ), finalMass = productMass( ) + m_residualMass;
    double beta = sqrt( a_X * ( a_X + 2. * projectileMass( ) ) ) / ( a_X + initialMass );      // beta = v/c.
    double _x = targetMass( ) * ( a_X - m_crossSectionThreshold ) / ( finalMass * finalMass );
    double Kp;                          // Kp is the total kinetic energy for m3 and m4 in the COM frame.

    a_input.m_sampledType = Sampling::SampledType::firstTwoBody;

    if( m_Upscatter ) {
        if( ( a_input.m_upscatterModel == Sampling::Upscatter::Model::B ) || ( a_input.m_upscatterModel == Sampling::Upscatter::Model::BSnLimits ) ) {
            if( upscatterModelB( a_X, a_input, a_userrng, a_rngState ) ) return;
        }
    }

    if( _x < 2e-5 ) {
        Kp = finalMass * _x * ( 1 - 0.5 * _x * ( 1 - _x ) ); }
    else {          // This is the relativistic formula derived from E^2 - (pc)^2 is frame independent.
        Kp = sqrt( finalMass * finalMass + 2 * targetMass( ) * ( a_X - m_crossSectionThreshold ) ) - finalMass;
    }
    if( Kp < 0 ) Kp = 0.;           // FIXME There needs to be a better test here.

    a_input.m_mu = m_angular->sample( a_X, a_userrng( a_rngState ), a_userrng, a_rngState );
    a_input.m_phi = 2. * M_PI * a_userrng( a_rngState );
    kinetics_COMKineticEnergy2LabEnergyAndMomentum( beta, Kp, productMass( ), m_residualMass, a_input );
}

/* *********************************************************************************************************//**
 * Returns the probability for a projectile with energy *a_energy_in* to cause a particle to be emitted 
 * at angle *a_mu_lab* as seen in the lab frame. *a_energy_out* is the sampled outgoing energy.
 *
 * @param a_reaction                [in]    The reaction containing the particle which this distribution describes.
 * @param a_energy_in               [in]    The energy of the incident particle.
 * @param a_mu_lab                  [in]    The desired mu in the lab frame for the emitted particle.
 * @param a_userrng                 [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState                [in]    The current state for the random number generator.
 * @param a_energy_out              [out]   The energy of the emitted outgoing particle.
 *
 * @return                                  The probability of emitting outgoing particle into lab angle *a_mu_lab*.
 ***********************************************************************************************************/

HOST_DEVICE double AngularTwoBody::angleBiasing( Reaction const *a_reaction, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const {

    a_energy_out = 0.0;

    double initialMass = projectileMass( ) + targetMass( ), finalMass = productMass( ) + m_residualMass;
    double _x = targetMass( ) * ( a_energy_in - m_crossSectionThreshold ) / ( finalMass * finalMass );
    double Kp;                      // Total kinetic energy of products in the center-of-mass.

    if( _x < 2e-5 ) {
        Kp = finalMass * _x * ( 1 - 0.5 * _x * ( 1 - _x ) ); }
    else {          // This is the relativistic formula derived from E^2 - (pc)^2 which is frame independent (i.e., an invariant).
        Kp = sqrt( finalMass * finalMass + 2.0 * targetMass( ) * ( a_energy_in - m_crossSectionThreshold ) ) - finalMass;
    }
    if( Kp < 0 ) Kp = 0.;           // FIXME There needs to be a better test here.

    double energy_product_com = 0.5 * Kp * ( Kp + 2.0 * m_residualMass ) / ( Kp + productMass( ) + m_residualMass );
    double productBeta = MCGIDI_particleBeta( productMass( ), energy_product_com );
    double boostBeta = sqrt( a_energy_in * ( a_energy_in + 2. * projectileMass( ) ) ) / ( a_energy_in + initialMass );      // beta = v/c.
    double muPlus, JacobianPlus, muMinus, JacobianMinus;

    int numberOfMus = muCOM_From_muLab( a_mu_lab, boostBeta, productBeta, muPlus, JacobianPlus, muMinus, JacobianMinus );

    if( numberOfMus == 0 ) return( 0.0 );

    double probability = JacobianPlus * m_angular->evaluate( a_energy_in, muPlus );

    if( numberOfMus == 2 ) {
        double probabilityMinus = JacobianMinus * m_angular->evaluate( a_energy_in, muMinus );
        probability += probabilityMinus;
        if( probabilityMinus > a_userrng( a_rngState ) * probability ) {
            muPlus = muMinus;
        }
    }

    double productBeta2 = productBeta * productBeta;
    double productBetaLab2 = productBeta2 + boostBeta * boostBeta * ( 1.0 - productBeta2 * ( 1.0 - muPlus * muPlus ) ) + 2.0 * muPlus * productBeta * boostBeta;
    productBetaLab2 /= 1.0 - muPlus * productBeta * boostBeta;
    a_energy_out = particleKineticEnergyFromBeta2( productMass( ), productBetaLab2 );

    return( probability );
}

/* *********************************************************************************************************//**
 * This function calculates the products outgoing data (i.e., energy, velocity/momentum) for the two products of
 * a two-body interaction give the cosine of the first product's outgoing angle.
 *
 * @param a_beta                    [in]    The velocity/speedOflight of the com frame relative to the lab frame.
 * @param a_kinetic_com             [in]    Total kinetic energy (K1 + K2) in the COM frame.
 * @param a_m3cc                    [in]    The mass of the first product.
 * @param a_m4cc                    [in]    The mass of the second product.
 * @param a_input                   [in]    Sample options requested by user and where the products' outgoing data are returned.
 ***********************************************************************************************************/

HOST_DEVICE static void kinetics_COMKineticEnergy2LabEnergyAndMomentum( double a_beta, double a_kinetic_com, 
        double a_m3cc, double a_m4cc, Sampling::Input &a_input ) {
/*
    Relativity:
        E = K + m, E^2 = K^2 + 2 K m + m^2, E^2 - m^2 = p^2 = K^2 + 2 K m
    
         pc          p     v
        ---- = v,   --- = --- = beta = b
         E           E     c

           K ( K + 2 m )
    b^2 = ---------------
            ( K + m )^2
*/
    double x, v_p, p, pp3, pp4, px3, py3, pz3, pz4, pz, p_perp2, E3, E4, gamma, m3cc2 = a_m3cc * a_m3cc, m4cc2 = a_m4cc * a_m4cc;

    p = sqrt( a_kinetic_com * ( a_kinetic_com + 2. * a_m3cc ) * ( a_kinetic_com + 2. * a_m4cc )  * 
            ( a_kinetic_com + 2. * ( a_m3cc + a_m4cc ) ) ) / ( 2. * ( a_kinetic_com + a_m3cc + a_m4cc ) );
    py3 = p * sqrt( 1 - a_input.m_mu * a_input.m_mu );
    px3 = py3 * cos( a_input.m_phi );
    py3 *= sin( a_input.m_phi );
    pz = p * a_input.m_mu;
    if( 1 ) {                           // FIXME Assuming the answer is wanted in the lab frame for now.
        a_input.m_frame = GIDI::Frame::lab;
        E3 = sqrt( p * p + m3cc2 );
        E4 = sqrt( p * p + m4cc2 );
        gamma = sqrt( 1. / ( 1. - a_beta * a_beta ) );
        pz3 = gamma * (  pz + a_beta * E3 );
        pz4 = gamma * ( -pz + a_beta * E4 ); }
    else {                              // COM frame.
        a_input.m_frame = GIDI::Frame::centerOfMass;
        pz3 = pz;
        pz4 = -pz;
    }

    p_perp2 = px3 * px3 + py3 * py3;

    a_input.m_px_vx1 = px3;
    a_input.m_py_vy1 = py3;
    a_input.m_pz_vz1 = pz3;
    pp3 = p_perp2 + pz3 * pz3;
    x = ( a_m3cc > 0 ) ? pp3 / ( 2 * m3cc2 ) : 1.;
    if( x < 1e-5 ) {
        a_input.m_energyOut1 = a_m3cc * x  * ( 1 - 0.5 * x * ( 1 - x ) ); }
    else {
        a_input.m_energyOut1 = sqrt( m3cc2 + pp3 ) - a_m3cc;
    }
    a_input.m_px_vx2 = -px3;
    a_input.m_py_vy2 = -py3;
    a_input.m_pz_vz2 = pz4;
    pp4 = p_perp2 + pz4 * pz4;
    x = ( a_m4cc > 0 ) ? pp4 / ( 2 * m4cc2 ) : 1.;
    if( x < 1e-5 ) {
        a_input.m_energyOut2 = a_m4cc * x  * ( 1 - 0.5 * x * ( 1 - x ) ); }
    else {
        a_input.m_energyOut2 = sqrt( m4cc2 + pp4 ) - a_m4cc;
    }

    if( a_input.wantVelocity( ) ) {
        v_p = MCGIDI_speedOfLight_cm_sec / sqrt( pp3 + m3cc2 );
        a_input.m_px_vx1 *= v_p;
        a_input.m_py_vy1 *= v_p;
        a_input.m_pz_vz1 *= v_p;

        v_p = MCGIDI_speedOfLight_cm_sec / sqrt( pp4 + m4cc2 );
        a_input.m_px_vx2 *= v_p;
        a_input.m_py_vy2 *= v_p;
        a_input.m_pz_vz2 *= v_p;
    }
}

/* *********************************************************************************************************//**
 * This method samples a targets velocity for elastic upscattering for upscatter model B and then calculates the outgoing
 * product data for the projectile and target.
 *
 * @param a_kineticLab              [in]    The kinetic energy of the projectile in the lab frame.
 * @param a_input                   [in]    Sample options requested by user.
 * @param a_userrng                 [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState                [in]    The current state for the random number generator.
 ***********************************************************************************************************/

HOST_DEVICE bool AngularTwoBody::upscatterModelB( double a_kineticLab, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const {

    double neutronMass = projectileMass( );                             // Mass are in incident energy unit / c**2.
    double _targetMass = targetMass( );
    double temperature = 1e-3 * a_input.m_temperature;                  // Assumes m_temperature is in keV/K.
    double kineticLabMax = 1e4 * temperature;

    if( a_input.m_upscatterModel == Sampling::Upscatter::Model::BSnLimits ) {
        double kineticLabMax200 = 200.0 * temperature;

        kineticLabMax = 1e3 * temperature * neutronMass / _targetMass;
        if( kineticLabMax < kineticLabMax200 ) kineticLabMax = kineticLabMax200;
        if( a_kineticLab >= 0.1 ) kineticLabMax = 0.9 * a_kineticLab; }
    else {
        if( kineticLabMax > 1e-2 ) {
            kineticLabMax = 1e-2;
            if( kineticLabMax < 100.0 * temperature ) {
                kineticLabMax = 100.0 * temperature;
                if( kineticLabMax > 10.0 ) kineticLabMax = 10.0;        // Assumes energy is in MeV.
            }
        }
    }

    if( a_kineticLab > kineticLabMax ) return( false );                   // Only for low neutron energy.

    a_input.m_frame = GIDI::Frame::lab;

    double muProjectileTarget, relativeBeta, targetBeta;
    double targetThermalBeta = MCGIDI_particleBeta( _targetMass, temperature );
    double neutronBeta = MCGIDI_particleBeta( neutronMass, a_kineticLab );

    do {
        int MethodP1orP2 = 0;         /* Assume P2 */
        if( a_userrng( a_rngState ) * ( neutronBeta + Two_sqrtPi * targetThermalBeta ) < neutronBeta ) MethodP1orP2 = 1;
        muProjectileTarget = 1.0 - 2.0 * a_userrng( a_rngState );
        if( MethodP1orP2 == 0 ) {                                       // x Exp( -x ) term.
            targetBeta = targetThermalBeta * sqrt( -log( ( 1.0 - a_userrng( a_rngState ) ) * ( 1.0 - a_userrng( a_rngState ) ) ) ); }
        else {                                                          // x^2 Exp( -x^2 ) term.
            double x1;
            do {
                x1 = a_userrng( a_rngState );
                x1 = sqrt( -log( ( 1.0 - a_userrng( a_rngState ) ) * ( 1.0 - x1 * x1 ) ) );
                x1 = x1 / ( ( ( C3 * x1 + C2 ) * x1 + C1 ) * x1 + C0 );
            } while( x1 > 4.0 );
            targetBeta = targetThermalBeta * x1;
        }
        relativeBeta = sqrt( targetBeta * targetBeta + neutronBeta * neutronBeta - 2 * muProjectileTarget * targetBeta * neutronBeta );
    } while( relativeBeta < ( targetBeta + neutronBeta ) * a_userrng( a_rngState ) );

    double m1_12 = neutronMass / ( neutronMass + _targetMass );
    double m2_12 = _targetMass / ( neutronMass + _targetMass );

    double cosRelative = 0.0;                                           // Cosine of angle between projectile velocity and relative velocity.
    if( relativeBeta != 0.0 ) cosRelative = ( neutronBeta - muProjectileTarget * targetBeta ) / relativeBeta;
    if( cosRelative > 1.0 ) {
            cosRelative = 1.0; }
    else if( cosRelative < -1.0 ) {
            cosRelative = -1.0;
    }
    double sinRelative = sqrt( 1.0 - cosRelative * cosRelative );       // Sine of angle between projectile velocity and relative velocity.

    double betaNeutronOut = m2_12 * relativeBeta; 
    double kineticEnergyRelative = particleKineticEnergy( neutronMass, betaNeutronOut );
    double muCOM = m_angular->sample( kineticEnergyRelative, a_userrng( a_rngState ), a_userrng, a_rngState );
    double phiCOM = 2.0 * M_PI * a_userrng( a_rngState );
    double SCcom = sqrt( 1.0 - muCOM * muCOM );
    double SScom = SCcom * sin( phiCOM );
    SCcom *= cos( phiCOM );

    a_input.m_pz_vz1 = betaNeutronOut * ( muCOM * cosRelative - SCcom * sinRelative );
    a_input.m_px_vx1 = betaNeutronOut * ( muCOM * sinRelative + SCcom * cosRelative );
    a_input.m_py_vy1 = betaNeutronOut * SScom;

    double massRatio = -neutronMass / _targetMass;
    a_input.m_pz_vz2 = massRatio * a_input.m_pz_vz1;
    a_input.m_px_vx2 = massRatio * a_input.m_px_vx1;
    a_input.m_py_vy2 = massRatio * a_input.m_py_vy1;

    double vCOMz = m1_12 * neutronBeta + m2_12 * muProjectileTarget * targetBeta;                   // Boost from center-of-mass to lab frame.
    double vCOMx = m2_12 * sqrt( 1.0 - muProjectileTarget * muProjectileTarget ) * targetBeta;
    a_input.m_pz_vz1 += vCOMz;
    a_input.m_px_vx1 += vCOMx;
    a_input.m_pz_vz2 += vCOMz;
    a_input.m_px_vx2 += vCOMx;

    double vx2_vy2 = a_input.m_px_vx1 * a_input.m_px_vx1 + a_input.m_py_vy1 * a_input.m_py_vy1;
    double v2 = a_input.m_pz_vz1 * a_input.m_pz_vz1 + vx2_vy2;
    a_input.m_mu = 0.0;
    if( v2 != 0.0 ) a_input.m_mu = a_input.m_pz_vz1 / sqrt( v2 );
    a_input.m_phi = atan2( a_input.m_py_vy1, a_input.m_px_vx1 );

    a_input.m_energyOut1 = particleKineticEnergyFromBeta2( neutronMass, v2 );
    a_input.m_energyOut2 = particleKineticEnergyFromBeta2( _targetMass, a_input.m_px_vx2 * a_input.m_px_vx2 + a_input.m_py_vy2 * a_input.m_py_vy2 + a_input.m_pz_vz2 * a_input.m_pz_vz2 );

    a_input.m_px_vx1 *= MCGIDI_speedOfLight_cm_sec;
    a_input.m_py_vy1 *= MCGIDI_speedOfLight_cm_sec;
    a_input.m_pz_vz1 *= MCGIDI_speedOfLight_cm_sec;

    a_input.m_px_vx2 *= MCGIDI_speedOfLight_cm_sec;
    a_input.m_py_vy2 *= MCGIDI_speedOfLight_cm_sec;
    a_input.m_pz_vz2 *= MCGIDI_speedOfLight_cm_sec;

    if( !a_input.wantVelocity( ) ) {                // Return momenta.
        a_input.m_px_vx1 *= neutronMass;            // Non-relativistic.
        a_input.m_py_vy1 *= neutronMass;
        a_input.m_pz_vz1 *= neutronMass;

        a_input.m_px_vx2 *= _targetMass;
        a_input.m_py_vy2 *= _targetMass;
        a_input.m_pz_vz2 *= _targetMass;
    }

    double phi = 2.0 * M_PI * a_userrng( a_rngState );
    double sine = sin( phi );
    double cosine = cos( phi );

    double saved = a_input.m_px_vx1;
    a_input.m_px_vx1 = cosine * a_input.m_px_vx1 - sine   * a_input.m_py_vy1;
    a_input.m_py_vy1 = sine   * saved            + cosine * a_input.m_py_vy1;

    return( true );
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

HOST_DEVICE void AngularTwoBody::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    Distribution::serialize( a_buffer, a_mode );
    DATA_MEMBER_FLOAT( m_residualMass, a_buffer, a_mode );
    DATA_MEMBER_FLOAT( m_Q, a_buffer, a_mode );
    DATA_MEMBER_FLOAT( m_crossSectionThreshold, a_buffer, a_mode );
    DATA_MEMBER_INT( m_Upscatter, a_buffer, a_mode );

    m_angular = serializeProbability2d( a_buffer, a_mode, m_angular );
}

/*! \class Uncorrelated
 * This class represents the distribution for an outgoing product for which the distribution is the product of uncorrelated
 * angular (i.e., P(mu|E)) and energy (i.e., P(E'|E)) distributions.
 */

/* *********************************************************************************************************//**
 * Default constructor used when broadcasting a Protare as needed by MPI or GPUs.
 ***********************************************************************************************************/

HOST_DEVICE Uncorrelated::Uncorrelated( ) :
        m_angular( nullptr ),
        m_energy( nullptr ) {

}

/* *********************************************************************************************************//**
 * @param a_uncorrelated            [in]    The GIDI::Distributions::Uncorrelated instance whose data is to be used to construct *this*.
 * @param a_setupInfo               [in]    Used internally when constructing a Protare to pass information to other constructors.
 ***********************************************************************************************************/

HOST Uncorrelated::Uncorrelated( GIDI::Distributions::Uncorrelated const &a_uncorrelated, SetupInfo &a_setupInfo ) :
        Distribution( Type::uncorrelated, a_uncorrelated, a_setupInfo ),
        m_angular( Probabilities::parseProbability2d( a_uncorrelated.angular( ), nullptr ) ),
        m_energy( Probabilities::parseProbability2d( a_uncorrelated.energy( ), &a_setupInfo ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

HOST_DEVICE Uncorrelated::~Uncorrelated( ) {

    delete m_angular;
    delete m_energy;
}

/* *********************************************************************************************************//**
 * This method samples the outgoing product data by sampling the outgoing energy E' and mu from the uncorrelated
 * E and mu probabilities. It also samples the outgoing phi uniformly between 0 and 2 pi.
 *
 * @param a_X                       [in]    The energy of the projectile.
 * @param a_input                   [in]    Sample options requested by user.
 * @param a_userrng                 [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState                [in]    The current state for the random number generator.
 ***********************************************************************************************************/

HOST_DEVICE void Uncorrelated::sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const {

    a_input.m_sampledType = Sampling::SampledType::uncorrelatedBody;
    a_input.m_mu = m_angular->sample( a_X, a_userrng( a_rngState ), a_userrng, a_rngState );
    a_input.m_energyOut1 = m_energy->sample( a_X, a_userrng( a_rngState ), a_userrng, a_rngState );
    a_input.m_phi = 2. * M_PI * a_userrng( a_rngState );
    a_input.m_frame = productFrame( );
}

/* *********************************************************************************************************//**
 * Returns the probability for a projectile with energy *a_energy_in* to cause a particle to be emitted 
 * at angle *a_mu_lab* as seen in the lab frame. *a_energy_out* is the sampled outgoing energy.
 *
 * @param a_reaction                [in]    The reaction containing the particle which this distribution describes.
 * @param a_energy_in               [in]    The energy of the incident particle.
 * @param a_mu_lab                  [in]    The desired mu in the lab frame for the emitted particle.
 * @param a_userrng                 [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState                [in]    The current state for the random number generator.
 * @param a_energy_out              [in]    The energy of the emitted outgoing particle.
 *
 * @return                                  The probability of emitting outgoing particle into lab angle *a_mu_lab*.
 ***********************************************************************************************************/

HOST_DEVICE double Uncorrelated::angleBiasing( Reaction const *a_reaction, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const {

    if( productFrame( ) != GIDI::Frame::lab ) THROW( "Uncorrelated::angleBiasing: center-of-mass not supported." );

    a_energy_out = m_energy->sample( a_energy_in, a_userrng( a_rngState ), a_userrng, a_rngState );
    return( m_angular->evaluate( a_energy_in, a_mu_lab ) );
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

HOST_DEVICE void Uncorrelated::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    Distribution::serialize( a_buffer, a_mode );
    m_angular = serializeProbability2d( a_buffer, a_mode, m_angular );
    m_energy = serializeProbability2d( a_buffer, a_mode, m_energy );
}

/*! \class EnergyAngularMC
 * This class represents the distribution for an outgoing particle where the distribution is give as 
 * P(E'|E) * P(mu|E,E') where E is the projectile's energy, E' is the product's outgoing energy, mu is the 
 * cosine of the product's outgoing angle relative to the projectile's velocity, P(E'|E) is the probability for E' given E
 * and (P(mu|E,E') is the probability for mu given E and E'.
 */

/* *********************************************************************************************************//**
 * Default constructor used when broadcasting a Protare as needed by MPI or GPUs.
 ***********************************************************************************************************/

HOST_DEVICE EnergyAngularMC::EnergyAngularMC( ) :
        m_energy( nullptr ),
        m_angularGivenEnergy( nullptr ) {

}

/* *********************************************************************************************************//**
 * @param a_energyAngularMC         [in]    The GIDI::Distributions::EnergyAngularMC instance whose data is to be used to construct *this*.
 * @param a_setupInfo               [in]    Used internally when constructing a Protare to pass information to other constructors.
 ***********************************************************************************************************/

HOST EnergyAngularMC::EnergyAngularMC( GIDI::Distributions::EnergyAngularMC const &a_energyAngularMC, SetupInfo &a_setupInfo ) :
        Distribution( Type::energyAngularMC, a_energyAngularMC, a_setupInfo ),
        m_energy( Probabilities::parseProbability2d( a_energyAngularMC.energy( ), nullptr ) ),
        m_angularGivenEnergy( Probabilities::parseProbability3d( a_energyAngularMC.energyAngular( ) ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

HOST_DEVICE EnergyAngularMC::~EnergyAngularMC( ) {

    delete m_energy;
    delete m_angularGivenEnergy;
}

/* *********************************************************************************************************//**
 * This method samples the outgoing product data by sampling the outgoing energy E' from the probability P(E'|E) and then samples mu from 
 * the probability P(mu|E,E'). It also samples the outgoing phi uniformly between 0 and 2 pi.
 *
 * @param a_X                       [in]    The energy of the projectile.
 * @param a_input                   [in]    Sample options requested by user.
 * @param a_userrng                 [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState                [in]    The current state for the random number generator.
 ***********************************************************************************************************/

HOST_DEVICE void EnergyAngularMC::sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const {

    a_input.m_sampledType = Sampling::SampledType::uncorrelatedBody;
    a_input.m_energyOut1 = m_energy->sample( a_X, a_userrng( a_rngState ), a_userrng, a_rngState );
    a_input.m_mu = m_angularGivenEnergy->sample( a_X, a_input.m_energyOut1, a_userrng( a_rngState ), a_userrng, a_rngState );
    a_input.m_phi = 2. * M_PI * a_userrng( a_rngState );
    a_input.m_frame = productFrame( );
}

/* *********************************************************************************************************//**
 * Returns the probability for a projectile with energy *a_energy_in* to cause a particle to be emitted 
 * at angle *a_mu_lab* as seen in the lab frame. *a_energy_out* is the sampled outgoing energy.
 *
 * @param a_reaction                [in]    The reaction containing the particle which this distribution describes.
 * @param a_energy_in               [in]    The energy of the incident particle.
 * @param a_mu_lab                  [in]    The desired mu in the lab frame for the emitted particle.
 * @param a_userrng                 [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState                [in]    The current state for the random number generator.
 * @param a_energy_out              [in]    The energy of the emitted outgoing particle.
 *
 * @return                                  The probability of emitting outgoing particle into lab angle *a_mu_lab*.
 ***********************************************************************************************************/

HOST_DEVICE double EnergyAngularMC::angleBiasing( Reaction const *a_reaction, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const {

    double probability = 0.0;

    if( productFrame( ) == GIDI::Frame::centerOfMass ) {
        a_energy_out = m_energy->sample( a_energy_in, a_userrng( a_rngState ), a_userrng, a_rngState );

        double initialMass = projectileMass( ) + targetMass( );
        double energy_out_com = m_energy->sample( a_energy_in, a_userrng( a_rngState ), a_userrng, a_rngState );
        double productBeta = MCGIDI_particleBeta( productMass( ), energy_out_com );
        double boostBeta = sqrt( a_energy_in * ( a_energy_in + 2. * projectileMass( ) ) ) / ( a_energy_in + initialMass );      // beta = v/c.

        double muPlus, JacobianPlus, muMinus, JacobianMinus;

        int numberOfMus = muCOM_From_muLab( a_mu_lab, boostBeta, productBeta, muPlus, JacobianPlus, muMinus, JacobianMinus );

        if( numberOfMus == 0 ) return( 0.0 );

        probability = JacobianPlus * m_angularGivenEnergy->evaluate( a_energy_in, energy_out_com, muPlus );

        if( numberOfMus == 2 ) {
            double probabilityMinus = JacobianMinus * m_angularGivenEnergy->evaluate( a_energy_in, energy_out_com, muMinus );

            probability += probabilityMinus;
            if( probabilityMinus > a_userrng( a_rngState ) * probability ) muPlus = muMinus;
        }

        double productBeta2 = productBeta * productBeta;
        double productBetaLab2 = productBeta2 + boostBeta * boostBeta * ( 1.0 - productBeta2 * ( 1.0 - muPlus * muPlus ) ) + 2.0 * muPlus * productBeta * boostBeta;
        productBetaLab2 /= 1.0 - muPlus * productBeta * boostBeta;
        a_energy_out = particleKineticEnergyFromBeta2( productMass( ), productBetaLab2 ); }
    else {
        a_energy_out = m_energy->sample( a_energy_in, a_userrng( a_rngState ), a_userrng, a_rngState );
        probability =  m_angularGivenEnergy->evaluate( a_energy_in, a_energy_out, a_mu_lab );
    }

    return( probability );
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

HOST_DEVICE void EnergyAngularMC::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    Distribution::serialize( a_buffer, a_mode );
    m_energy = serializeProbability2d( a_buffer, a_mode, m_energy );
    m_angularGivenEnergy = serializeProbability3d( a_buffer, a_mode, m_angularGivenEnergy );
}

/*! \class AngularEnergyMC
 * This class represents the distribution for an outgoing particle where the distribution is give as 
 * P(mu|E) * P(E'|E,mu) where E is the projectile's energy, E' is the product's outgoing energy, mu is the 
 * cosine of the product's outgoing angle relative to the projectile's velocity, P(mu|E) is the probability for mu given E
 * and (P(E'|E,mu) is the probability for E' given E and mu.
 */

/* *********************************************************************************************************//**
 * Default constructor used when broadcasting a Protare as needed by MPI or GPUs.
 ***********************************************************************************************************/

HOST_DEVICE AngularEnergyMC::AngularEnergyMC( ) :
        m_angular( nullptr ),
        m_energyGivenAngular( nullptr ) {

}

/* *********************************************************************************************************//**
 * @param a_angularEnergyMC         [in]    The GIDI::Distributions::AngularEnergyMC instance whose data is to be used to construct *this*.
 * @param a_setupInfo               [in]    Used internally when constructing a Protare to pass information to other constructors.
 ***********************************************************************************************************/

HOST AngularEnergyMC::AngularEnergyMC( GIDI::Distributions::AngularEnergyMC const &a_angularEnergyMC, SetupInfo &a_setupInfo ) :
        Distribution( Type::angularEnergyMC, a_angularEnergyMC, a_setupInfo ),
        m_angular( Probabilities::parseProbability2d( a_angularEnergyMC.angular( ), nullptr ) ),
        m_energyGivenAngular( Probabilities::parseProbability3d( a_angularEnergyMC.angularEnergy( ) ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

HOST_DEVICE AngularEnergyMC::~AngularEnergyMC( ) {

    delete m_angular;
    delete m_energyGivenAngular;
}

/* *********************************************************************************************************//**
 * This method samples the outgoing product data by sampling the outgoing mu from the probability P(mu|E) and then samples E' from 
 * the probability P(E'|E,mu). It also samples the outgoing phi uniformly between 0 and 2 pi.
 *
 * @param a_X                       [in]    The energy of the projectile.
 * @param a_input                   [in]    Sample options requested by user.
 * @param a_userrng                 [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState                [in]    The current state for the random number generator.
 ***********************************************************************************************************/

HOST_DEVICE void AngularEnergyMC::sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const {

    a_input.m_sampledType = Sampling::SampledType::uncorrelatedBody;
    a_input.m_mu = m_angular->sample( a_X, a_userrng( a_rngState ), a_userrng, a_rngState );
    a_input.m_energyOut1 = m_energyGivenAngular->sample( a_X, a_input.m_mu, a_userrng( a_rngState ), a_userrng, a_rngState );
    a_input.m_phi = 2. * M_PI * a_userrng( a_rngState );
    a_input.m_frame = productFrame( );
}

/* *********************************************************************************************************//**
 * Returns the probability for a projectile with energy *a_energy_in* to cause a particle to be emitted 
 * at angle *a_mu_lab* as seen in the lab frame. *a_energy_out* is the sampled outgoing energy.
 *
 * @param a_reaction                [in]    The reaction containing the particle which this distribution describes.
 * @param a_energy_in               [in]    The energy of the incident particle.
 * @param a_mu_lab                  [in]    The desired mu in the lab frame for the emitted particle.
 * @param a_userrng                 [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState                [in]    The current state for the random number generator.
 * @param a_energy_out              [in]    The energy of the emitted outgoing particle.
 *
 * @return                                  The probability of emitting outgoing particle into lab angle *a_mu_lab*.
 ***********************************************************************************************************/

HOST_DEVICE double AngularEnergyMC::angleBiasing( Reaction const *a_reaction, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const {

    if( productFrame( ) != GIDI::Frame::lab ) THROW( "AngularEnergyMC::angleBiasing: center-of-mass not supported." );

    a_energy_out = m_energyGivenAngular->sample( a_energy_in, a_mu_lab, a_userrng( a_rngState), a_userrng, a_rngState );
    return( m_angular->evaluate( a_energy_in, a_mu_lab ) );
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

HOST_DEVICE void AngularEnergyMC::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    Distribution::serialize( a_buffer, a_mode );
    m_angular = serializeProbability2d( a_buffer, a_mode, m_angular );
    m_energyGivenAngular = serializeProbability3d( a_buffer, a_mode, m_energyGivenAngular );
}

/*! \class KalbachMann
 * This class represents the distribution for an outgoing product whose distribution is represented by Kalbach-Mann systematics.
 */

/* *********************************************************************************************************//**
 * Default constructor used when broadcasting a Protare as needed by MPI or GPUs.
 ***********************************************************************************************************/

HOST_DEVICE KalbachMann::KalbachMann( ) :
        m_energyToMeVFactor( 0.0 ),
        m_eb_massFactor( 0.0 ),
        m_f( nullptr ),
        m_r( nullptr ),
        m_a( nullptr ) {

}

/* *********************************************************************************************************//**
 * @param a_KalbachMann             [in]    The GIDI::Distributions::KalbachMann instance whose data is to be used to construct *this*.
 * @param a_setupInfo               [in]    Used internally when constructing a Protare to pass information to other constructors.
 ***********************************************************************************************************/

HOST KalbachMann::KalbachMann( GIDI::Distributions::KalbachMann const &a_KalbachMann, SetupInfo &a_setupInfo ) :
        Distribution( Type::KalbachMann, a_KalbachMann, a_setupInfo ),
        m_energyToMeVFactor( 1 ),                                           // FIXME.
        m_eb_massFactor( 1 ),                                               // FIXME.
        m_f( Probabilities::parseProbability2d( a_KalbachMann.f( ), nullptr ) ),
        m_r( Functions::parseFunction2d( a_KalbachMann.r( ) ) ),
        m_a( Functions::parseFunction2d( a_KalbachMann.a( ) ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

HOST_DEVICE KalbachMann::~KalbachMann( ) {

    delete m_f;
    delete m_r;
    delete m_a;
}

/* *********************************************************************************************************//**
 * This method samples the outgoing product data using the Kalbach-Mann formalism.
 *
 * @param a_X                       [in]    The energy of the projectile.
 * @param a_input                   [in]    Sample options requested by user.
 * @param a_userrng                 [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState                [in]    The current state for the random number generator.
 ***********************************************************************************************************/

HOST_DEVICE void KalbachMann::sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const {

    a_input.m_sampledType = Sampling::SampledType::uncorrelatedBody;
    a_input.m_energyOut1 = m_f->sample( a_X, a_userrng( a_rngState ), a_userrng, a_rngState );
    double rValue = m_r->evaluate( a_X, a_input.m_energyOut1 );
    double aValue = m_a->evaluate( a_X, a_input.m_energyOut1 );

        // In the following: Cosh[ a mu ] + r Sinh[ a mu ] = ( 1 - r ) Cosh[ a mu ] + r ( Cosh[ a mu ] + Sinh[ a mu ] ).
    if( a_userrng( a_rngState ) >= rValue ) { // Sample the '( 1 - r ) Cosh[ a mu ]' term.
        double T = ( 2. * a_userrng( a_rngState ) - 1. ) * sinh( aValue );

        a_input.m_mu = log( T + sqrt( T * T + 1. ) ) / aValue; }
    else {                                                                  // Sample the 'r ( Cosh[ a mu ] + Sinh[ a mu ] )' term.
        double rng1 = a_userrng( a_rngState ), exp_a = exp( aValue );

        a_input.m_mu = log( rng1 * exp_a + ( 1. - rng1 ) / exp_a ) / aValue;
    }
    if( a_input.m_mu < -1 ) a_input.m_mu = -1;
    if( a_input.m_mu >  1 ) a_input.m_mu = 1;

    a_input.m_phi = 2. * M_PI * a_userrng( a_rngState );
    a_input.m_frame = productFrame( );
}

/* *********************************************************************************************************//**
 * Returns the probability for a projectile with energy *a_energy_in* to cause a particle to be emitted 
 * at angle *a_mu_lab* as seen in the lab frame. *a_energy_out* is the sampled outgoing energy.
 *
 * @param a_reaction                [in]    The reaction containing the particle which this distribution describes.
 * @param a_energy_in               [in]    The energy of the incident particle.
 * @param a_mu_lab                  [in]    The desired mu in the lab frame for the emitted particle.
 * @param a_userrng                 [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState                [in]    The current state for the random number generator.
 * @param a_energy_out              [in]    The energy of the emitted outgoing particle.
 *
 * @return                                  The probability of emitting outgoing particle into lab angle *a_mu_lab*.
 ***********************************************************************************************************/

HOST_DEVICE double KalbachMann::angleBiasing( Reaction const *a_reaction, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const {

    a_energy_out = 0.0;

    double initialMass = projectileMass( ) + targetMass( );
    double energy_out_com = m_f->sample( a_energy_in, a_userrng( a_rngState ), a_userrng, a_rngState );
    double productBeta = MCGIDI_particleBeta( productMass( ), energy_out_com );
    double boostBeta = sqrt( a_energy_in * ( a_energy_in + 2. * projectileMass( ) ) ) / ( a_energy_in + initialMass );      // beta = v/c.

    double muPlus, JacobianPlus, muMinus, JacobianMinus;

    int numberOfMus = muCOM_From_muLab( a_mu_lab, boostBeta, productBeta, muPlus, JacobianPlus, muMinus, JacobianMinus );

    if( numberOfMus == 0 ) return( 0.0 );

    double rAtEnergyEnergyPrime = m_r->evaluate( a_energy_in, energy_out_com );
    double aAtEnergyEnergyPrime = m_a->evaluate( a_energy_in, energy_out_com );
    double aMu = aAtEnergyEnergyPrime * muPlus;

    double probability = 0.5 * JacobianPlus;
    if( productMass( ) == 0.0 ) {
        probability *= 1.0 - rAtEnergyEnergyPrime + rAtEnergyEnergyPrime * aAtEnergyEnergyPrime * exp( aMu ) / sinh( aAtEnergyEnergyPrime ); }
    else {
        probability *= aAtEnergyEnergyPrime * ( cosh( aMu ) + rAtEnergyEnergyPrime * cosh( aMu ) ) / sinh( aAtEnergyEnergyPrime );
    }

    if( numberOfMus == 2 ) {
        aMu = aAtEnergyEnergyPrime * muMinus;

        double probabilityMinus = 0.5 * JacobianMinus;
        if( productMass( ) == 0.0 ) {
            probabilityMinus *= 1.0 - rAtEnergyEnergyPrime + rAtEnergyEnergyPrime * aAtEnergyEnergyPrime * exp( aMu ) / sinh( aAtEnergyEnergyPrime ); }
        else {
            probabilityMinus *= aAtEnergyEnergyPrime * ( cosh( aMu ) + rAtEnergyEnergyPrime * cosh( aMu ) ) / sinh( aAtEnergyEnergyPrime );
        }
        probability += probabilityMinus;

        if( probabilityMinus > a_userrng( a_rngState ) * probability ) muPlus = muMinus;
    }

    double productBeta2 = productBeta * productBeta;
    double productBetaLab2 = productBeta2 + boostBeta * boostBeta * ( 1.0 - productBeta2 * ( 1.0 - muPlus * muPlus ) ) + 2.0 * muPlus * productBeta * boostBeta;
    productBetaLab2 /= 1.0 - muPlus * productBeta * boostBeta;
    a_energy_out = particleKineticEnergyFromBeta2( productMass( ), productBetaLab2 );

    return( probability );
}

/* *********************************************************************************************************//**
 * This method evaluates the Kalbach-Mann formalism at the projectile energy a_energy, and outgoing product energy a_energyOut and a_mu.
 *
 * @param a_energy                  [in]    The energy of the projectile in the lab frame.
 * @param a_energyOut               [in]    The energy of the product in the center-of-mass frame.
 * @param a_mu                      [in]    The mu of the product in the center-of-mass frame.
 ***********************************************************************************************************/

HOST_DEVICE double KalbachMann::evaluate( double a_energy, double a_energyOut, double a_mu ) {

//    double f_0 = m_f->evaluate( a_energy, a_energyOut );
    double rValue = m_r->evaluate( a_energy, a_energyOut );
    double aValue = m_a->evaluate( a_energy, a_energyOut );
//    double pdf_val = aValue * f_0 / 2.0 / sinh( aValue ) * (cosh(aValue * a_mu) + rValue * sinh( aValue * a_mu ) ); // double-differential PDF for a_energyOut and a_mu (Eq. 6.4 in ENDF-102, 2012)
    double pdf_val = aValue / ( 2.0 * sinh( aValue ) ) * ( cosh( aValue * a_mu ) + rValue * sinh( aValue * a_mu ) ); // double-differential PDF for a_energyOut and mu (Eq. 6.4 in ENDF-102, 2012)
    return pdf_val;
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

HOST_DEVICE void KalbachMann::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    Distribution::serialize( a_buffer, a_mode );
    DATA_MEMBER_FLOAT( m_energyToMeVFactor, a_buffer, a_mode );
    DATA_MEMBER_FLOAT( m_eb_massFactor, a_buffer, a_mode );

    m_f = serializeProbability2d( a_buffer, a_mode, m_f );
    m_r = serializeFunction2d( a_buffer, a_mode, m_r );
    m_a = serializeFunction2d( a_buffer, a_mode, m_a );
}

/*! \class CoherentPhotoAtomicScattering
 * This class represents the distribution for an outgoing photon via coherent photo-atomic elastic scattering.
 */

/* *********************************************************************************************************//**
 * Default constructor used when broadcasting a Protare as needed by MPI or GPUs.
 ***********************************************************************************************************/

HOST_DEVICE CoherentPhotoAtomicScattering::CoherentPhotoAtomicScattering( ) :
        m_realAnomalousFactor( nullptr ),
        m_imaginaryAnomalousFactor( nullptr ) {

}

/* *********************************************************************************************************//**
 * @param a_coherentPhotoAtomicScattering   [in]    GIDI::Distributions::CoherentPhotoAtomicScattering instance whose data is to be used to construct *this*.
 * @param a_setupInfo                       [in]    Used internally when constructing a Protare to pass information to other constructors.
 ***********************************************************************************************************/

HOST CoherentPhotoAtomicScattering::CoherentPhotoAtomicScattering( GIDI::Distributions::CoherentPhotoAtomicScattering const &a_coherentPhotoAtomicScattering, SetupInfo &a_setupInfo ) :
        Distribution( Type::coherentPhotoAtomicScattering, a_coherentPhotoAtomicScattering, a_setupInfo ),
        m_anomalousDataPresent( false ),
        m_realAnomalousFactor( nullptr ),
        m_imaginaryAnomalousFactor( nullptr ) {

    GIDI::Ancestry const *link = a_coherentPhotoAtomicScattering.findInAncestry( a_coherentPhotoAtomicScattering.href( ) );
    GIDI::DoubleDifferentialCrossSection::CoherentPhotoAtomicScattering const &coherentPhotoAtomicScattering = 
            *static_cast<GIDI::DoubleDifferentialCrossSection::CoherentPhotoAtomicScattering const *>( link );

    std::string domainUnit;
    GIDI::Functions::XYs1d const *xys1d0, *xys1d1;
    std::size_t dataSize = 0, offset = 0;

    GIDI::Functions::Function1dForm const *formFactor = coherentPhotoAtomicScattering.formFactor( );
    if( formFactor->type( ) == GIDI::FormType::XYs1d ) {
        xys1d0 = static_cast<GIDI::Functions::XYs1d const *>( formFactor );
        xys1d1 = xys1d0;

        domainUnit = xys1d0->axes( )[0]->unit( );

        dataSize = xys1d1->size( );
        offset = 1; }
    else if( formFactor->type( ) == GIDI::FormType::regions1d ) {
        GIDI::Functions::Regions1d const *regions1d = static_cast<GIDI::Functions::Regions1d const *>( formFactor );
        if( regions1d->size( ) != 2 ) THROW( "MCGIDI::CoherentPhotoAtomicScattering::CoherentPhotoAtomicScattering: unsupported form factor size." );

        domainUnit = regions1d->axes( )[0]->unit( );

        GIDI::Functions::Function1dForm const *region0 = (*regions1d)[0];
        if( region0->type( ) != GIDI::FormType::XYs1d ) THROW( "MCGIDI::CoherentPhotoAtomicScattering::CoherentPhotoAtomicScattering: unsupported form factor for region 0." );
        xys1d0 = static_cast<GIDI::Functions::XYs1d const *>( region0 );
        if( xys1d0->size( ) != 2 ) THROW( "MCGIDI::CoherentPhotoAtomicScattering::CoherentPhotoAtomicScattering: unsupported size of region 1 of form factor." );

        GIDI::Functions::Function1dForm const *region1 = (*regions1d)[1];
        if( region1->type( ) != GIDI::FormType::XYs1d ) THROW( "MCGIDI::CoherentPhotoAtomicScattering::CoherentPhotoAtomicScattering: unsupported form factor for region 1." );
        xys1d1 = static_cast<GIDI::Functions::XYs1d const *>( region1 );

        dataSize = xys1d1->size( ) + 1; }
    else {
        THROW( "MCGIDI::CoherentPhotoAtomicScattering::CoherentPhotoAtomicScattering: unsupported form factor. Must be XYs1d or regions1d." );
    }

    double domainFactor = 1.0;
    if( domainUnit == "1/Ang" ) {
        domainFactor = 0.012398419739640716; }                      // Converts 'h * c /Ang' to MeV.
    else if( domainUnit == "1/cm" ) {
        domainFactor = 0.012398419739640716 * 1e-8; }               // Converts 'h * c /cm' to MeV.
    else {
        THROW( "MCGIDI::CoherentPhotoAtomicScattering::CoherentPhotoAtomicScattering: unsupported domain unit" );
    }

    m_energies.resize( dataSize );
    m_formFactor.resize( dataSize );
    m_a.resize( dataSize );
    m_integratedFormFactor.resize( dataSize );
    m_integratedFormFactorSquared.resize( dataSize );
    m_probabilityNorm1_1.resize( dataSize );
    m_probabilityNorm1_3.resize( dataSize );
    m_probabilityNorm1_5.resize( dataSize );
    m_probabilityNorm2_1.resize( dataSize );
    m_probabilityNorm2_3.resize( dataSize );
    m_probabilityNorm2_5.resize( dataSize );

    std::pair<double, double> xy = (*xys1d0)[0];
    m_energies[0] = 0.0;
    m_formFactor[0] = xy.second;
    m_a[0] = 0.0;
    m_integratedFormFactor[0] = 0.0;
    m_integratedFormFactorSquared[0] = 0.0;

    xy = (*xys1d1)[offset];
    double energy1 = domainFactor * xy.first;
    double y1 = xy.second;
    m_energies[1] = energy1;
    m_formFactor[1] = y1;
    m_integratedFormFactor[1] = 0.5 * energy1 * energy1 * y1;
    m_integratedFormFactorSquared[1] = 0.5 * energy1 * energy1 * y1 * y1;

    double sum1 = m_integratedFormFactor[1];
    double sum2 = m_integratedFormFactorSquared[1];
    for( std::size_t i1 = 1 + offset; i1 < xys1d1->size( ); ++i1 ) {
        xy = (*xys1d1)[i1];
        double energy2 = domainFactor * xy.first;
        double y2 = xy.second;

        double logEs = log( energy2 / energy1 );
        double _a = log( y2 / y1 ) / logEs;

        m_energies[i1+1-offset] = energy2;
        m_formFactor[i1+1-offset] = y2;
        m_a[i1-offset] = _a;

        sum1 += coherentPhotoAtomicScatteringIntegrateSub( 1,       _a, logEs, energy1,      y1, energy2,      y2 );
        m_integratedFormFactor[i1+1-offset] = sum1;

        sum2 += coherentPhotoAtomicScatteringIntegrateSub( 1, 2.0 * _a, logEs, energy1, y1 * y1, energy2, y2 * y2 );
        m_integratedFormFactorSquared[i1+1-offset] = sum2;

        energy1 = energy2;
        y1 = y2;
    }

    m_a[m_a.size()-1] = 0.0;

    if( coherentPhotoAtomicScattering.realAnomalousFactor( ) != nullptr ) {
        m_anomalousDataPresent = true;
        m_realAnomalousFactor = Functions::parseFunction1d( coherentPhotoAtomicScattering.realAnomalousFactor( ) );
        m_imaginaryAnomalousFactor = Functions::parseFunction1d( coherentPhotoAtomicScattering.imaginaryAnomalousFactor( ) );
    }

    m_probabilityNorm1_1[0] = 0.0;
    m_probabilityNorm1_3[0] = 0.0;
    m_probabilityNorm1_5[0] = 0.0;
    m_probabilityNorm2_1[0] = 0.0;
    m_probabilityNorm2_3[0] = 0.0;
    m_probabilityNorm2_5[0] = 0.0;
    energy1 = m_energies[1];
    y1 = m_formFactor[0];
    for( MCGIDI_VectorSizeType i1 = 1; i1 < m_probabilityNorm1_1.size( ); ++i1 ) {
        double energy2 = m_energies[i1];
        double y2 = m_formFactor[i1];
        double logEs = log( energy2 / energy1 );

        m_probabilityNorm1_1[i1] = m_probabilityNorm1_1[i1-1] + coherentPhotoAtomicScatteringIntegrateSub( 1,       m_a[i1-1], logEs, energy1,      y1, energy2,      y2 );
        m_probabilityNorm1_3[i1] = m_probabilityNorm1_3[i1-1] + coherentPhotoAtomicScatteringIntegrateSub( 3,       m_a[i1-1], logEs, energy1,      y1, energy2,      y2 );
        m_probabilityNorm1_5[i1] = m_probabilityNorm1_5[i1-1] + coherentPhotoAtomicScatteringIntegrateSub( 5,       m_a[i1-1], logEs, energy1,      y1, energy2,      y2 );

        m_probabilityNorm2_1[i1] = m_probabilityNorm2_1[i1-1] + coherentPhotoAtomicScatteringIntegrateSub( 1, 2.0 * m_a[i1-1], logEs, energy1, y1 * y1, energy2, y2 * y2 );
        m_probabilityNorm2_3[i1] = m_probabilityNorm2_3[i1-1] + coherentPhotoAtomicScatteringIntegrateSub( 3, 2.0 * m_a[i1-1], logEs, energy1, y1 * y1, energy2, y2 * y2 );
        m_probabilityNorm2_5[i1] = m_probabilityNorm2_5[i1-1] + coherentPhotoAtomicScatteringIntegrateSub( 5, 2.0 * m_a[i1-1], logEs, energy1, y1 * y1, energy2, y2 * y2 );

        energy1 = energy2;
        y1 = y2;
    }
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

HOST_DEVICE CoherentPhotoAtomicScattering::~CoherentPhotoAtomicScattering( ) {

    delete m_realAnomalousFactor;
    delete m_imaginaryAnomalousFactor;
}

/* *********************************************************************************************************//**
 * This method evaluates the coherent photo-atomic scattering double differentil at the projectile energy a_energy and product cosine of angle a_mu.
 *
 * @param a_energyIn                [in]    The energy of the projectile in the lab frame.
 * @param a_mu                      [in]    The mu of the product in the center-of-mass frame.
 ***********************************************************************************************************/

HOST_DEVICE double CoherentPhotoAtomicScattering::evaluate( double a_energyIn, double a_mu ) const {

    double probability;
    MCGIDI_VectorSizeType lowerIndexEnergy = binarySearchVector( a_energyIn, m_energies );
    double _a = m_a[lowerIndexEnergy];
    double _a_2 = _a * _a;
    double X1 = m_energies[lowerIndexEnergy];
    double logEs = log( a_energyIn / X1 );
    double formFactor_1 = m_formFactor[lowerIndexEnergy];
    double formFactor_2 = formFactor_1 * formFactor_1;
    double formFactorEnergyIn_1 = formFactor_1 * pow( a_energyIn / X1, _a );
    double formFactorEnergyIn_2 = formFactorEnergyIn_1 * formFactorEnergyIn_1;
    double inverseEnergyIn_1 = 1.0 / a_energyIn;
    double inverseEnergyIn_2 = inverseEnergyIn_1 * inverseEnergyIn_1;
    double inverseEnergyIn_3 = inverseEnergyIn_1 * inverseEnergyIn_2;
    double inverseEnergyIn_4 = inverseEnergyIn_2 * inverseEnergyIn_2;
    double inverseEnergyIn_5 = inverseEnergyIn_1 * inverseEnergyIn_4;
    double inverseEnergyIn_6 = inverseEnergyIn_2 * inverseEnergyIn_4;

    double norm = 0.5 * inverseEnergyIn_2 * ( m_probabilityNorm2_1[lowerIndexEnergy] + coherentPhotoAtomicScatteringIntegrateSub( 1, _a_2, logEs, X1, formFactor_2, a_energyIn, formFactorEnergyIn_2 ) )
                      - inverseEnergyIn_4 * ( m_probabilityNorm2_3[lowerIndexEnergy] + coherentPhotoAtomicScatteringIntegrateSub( 3, _a_2, logEs, X1, formFactor_2, a_energyIn, formFactorEnergyIn_2 ) )
                      + inverseEnergyIn_6 * ( m_probabilityNorm2_5[lowerIndexEnergy] + coherentPhotoAtomicScatteringIntegrateSub( 5, _a_2, logEs, X1, formFactor_2, a_energyIn, formFactorEnergyIn_2 ) );

    double realAnomalousFactor = 0.0;
    double imaginaryAnomalousFactor = 0.0;
    if( m_anomalousDataPresent ) {
        realAnomalousFactor = m_realAnomalousFactor->evaluate( a_energyIn );
        imaginaryAnomalousFactor = m_imaginaryAnomalousFactor->evaluate( a_energyIn );
        norm += realAnomalousFactor * (         inverseEnergyIn_1 * ( m_probabilityNorm1_1[lowerIndexEnergy] + coherentPhotoAtomicScatteringIntegrateSub( 1, _a, logEs, X1, formFactor_1, a_energyIn, formFactorEnergyIn_1 ) )
                                        - 2.0 * inverseEnergyIn_3 * ( m_probabilityNorm1_3[lowerIndexEnergy] + coherentPhotoAtomicScatteringIntegrateSub( 3, _a, logEs, X1, formFactor_1, a_energyIn, formFactorEnergyIn_1 ) )
                                        + 2.0 * inverseEnergyIn_5 * ( m_probabilityNorm1_5[lowerIndexEnergy] + coherentPhotoAtomicScatteringIntegrateSub( 5, _a, logEs, X1, formFactor_1, a_energyIn, formFactorEnergyIn_1 ) ) );
    }
    norm *= 16.0;
    norm += 8.0 / 3.0 * ( realAnomalousFactor * realAnomalousFactor + imaginaryAnomalousFactor * imaginaryAnomalousFactor );

    double _formFactor = evaluateFormFactor( a_energyIn, a_mu );
    probability = ( 1.0 + a_mu * a_mu ) * ( ( _formFactor + realAnomalousFactor ) * ( _formFactor + realAnomalousFactor ) + imaginaryAnomalousFactor * imaginaryAnomalousFactor ) / norm;

    return( probability );
}

/* *********************************************************************************************************//**
 * This method evaluates the coherent photo-atomic form factor at the projectile energy a_energy and product cosine of angle a_mu.
 *
 * @param a_energyIn                [in]    The energy of the projectile in the lab frame.
 * @param a_mu                      [in]    The mu of the product in the center-of-mass frame.
 ***********************************************************************************************************/

HOST_DEVICE double CoherentPhotoAtomicScattering::evaluateFormFactor( double a_energyIn, double a_mu ) const {

    double X = a_energyIn * sqrt( 0.5 * ( 1 - a_mu ) );
    MCGIDI_VectorSizeType lowerIndex = binarySearchVector( X, m_energies );

    if( lowerIndex < 0 ) {
        if( lowerIndex == -2 ) return( m_formFactor[0] );               // This should never happend for proper a_energyIn and a_mu.
        return( m_formFactor.back( ) );
    }

    return( m_formFactor[lowerIndex] * pow( X / m_energies[lowerIndex] , m_a[lowerIndex] ) );
}

/* *********************************************************************************************************//**
 * This method samples the outgoing product data from the coherent photo-atomic scattering law.
 * It also samples the outgoing phi uniformly between 0 and 2 pi.
 *
 * @param a_X                       [in]    The energy of the projectile in the lab frame.
 * @param a_input                   [in]    Sample options requested by user.
 * @param a_userrng                 [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState                [in]    The current state for the random number generator.
 ***********************************************************************************************************/

HOST_DEVICE void CoherentPhotoAtomicScattering::sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const {

    a_input.m_energyOut1 = a_X;

    MCGIDI_VectorSizeType lowerIndex = binarySearchVector( a_X, m_energies );

    if( lowerIndex < 1 ) {
        do {
            a_input.m_mu = 1.0 - 2.0 * a_userrng( a_rngState );
        } while( ( 1.0 + a_input.m_mu * a_input.m_mu ) < 2.0 * a_userrng( a_rngState ) ); }
    else {
        double _a = m_a[lowerIndex];
        double X_i = m_energies[lowerIndex];
        double formFactor_i = m_formFactor[lowerIndex];
        double formFactor_X_i = formFactor_i * X_i;
        double Z = a_X / X_i;
        double realAnomalousFactor = 0.0;
        double imaginaryAnomalousFactor = 0.0;

        if( m_anomalousDataPresent ) {
            realAnomalousFactor = m_realAnomalousFactor->evaluate( a_X );
            imaginaryAnomalousFactor = m_imaginaryAnomalousFactor->evaluate( a_X );
        }

        double anomalousFactorSquared = realAnomalousFactor * realAnomalousFactor + imaginaryAnomalousFactor + imaginaryAnomalousFactor;
        double normalization = m_integratedFormFactorSquared[lowerIndex] + formFactor_X_i * formFactor_X_i * Z_a( Z, 2.0 * _a + 2.0 );

        anomalousFactorSquared = 0.0;
        if( anomalousFactorSquared != 0.0 ) {
            double integratedFormFactor_i = m_integratedFormFactor[lowerIndex] + formFactor_X_i * Z_a( Z, _a + 2.0 );

            normalization += 2.0  * integratedFormFactor_i * realAnomalousFactor + 0.5 * anomalousFactorSquared * a_X * a_X;
        }

        do {
            double partialIntegral = a_userrng( a_rngState ) * normalization;
            double X;
            if( anomalousFactorSquared == 0.0 ) {
                lowerIndex = binarySearchVector( partialIntegral, m_integratedFormFactorSquared );

                if( lowerIndex == 0 ) {
                    X = sqrt( 2.0 * partialIntegral ) / m_formFactor[0]; }
                else {
                    double remainer = partialIntegral - m_integratedFormFactorSquared[lowerIndex];
                    double epsilon = 2.0 * m_a[lowerIndex] + 2.0;

                    X_i = m_energies[lowerIndex];
                    formFactor_i = m_formFactor[lowerIndex];
                    formFactor_X_i = formFactor_i * X_i;

                    remainer /= formFactor_X_i * formFactor_X_i;
                    if( fabs( epsilon ) < 1e-6 ) {
                        X = X_i * exp( remainer ); }
                    else {
                        X = X_i * pow( 1.0 + epsilon * remainer, 1.0 / epsilon );
                    }
                } }
            else {                                  // Currently not implemented.
                X = 0.5 * a_X;
            }
            double X_E = X / a_X;
            a_input.m_mu = 1.0 - 2.0 * X_E * X_E;
        } while( ( 1.0 + a_input.m_mu * a_input.m_mu ) < 2.0 * a_userrng( a_rngState ) );
    }

    a_input.m_phi = 2.0 * M_PI * a_userrng( a_rngState );
    a_input.m_frame = productFrame( );
}

/* *********************************************************************************************************//**
 * Returns the probability for a projectile with energy *a_energy_in* to cause a particle to be emitted 
 * at angle *a_mu_lab* as seen in the lab frame. *a_energy_out* is the sampled outgoing energy.
 *
 * @param a_reaction                [in]    The reaction containing the particle which this distribution describes.
 * @param a_energy_in               [in]    The energy of the incident particle.
 * @param a_mu_lab                  [in]    The desired mu in the lab frame for the emitted particle.
 * @param a_userrng                 [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState                [in]    The current state for the random number generator.
 * @param a_energy_out              [in]    The energy of the emitted outgoing particle.
 *
 * @return                                  The probability of emitting outgoing particle into lab angle *a_mu_lab*.
 ***********************************************************************************************************/

HOST_DEVICE double CoherentPhotoAtomicScattering::angleBiasing( Reaction const *a_reaction, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const {

    a_energy_out = a_energy_in;

    URR_protareInfos URR_protareInfos1;
    double sigma = a_reaction->protareSingle( )->reactionCrossSection( a_reaction->reactionIndex( ), URR_protareInfos1, 0.0, a_energy_in );
    double formFactor = evaluateFormFactor( a_energy_in, a_mu_lab );
    double imaginaryAnomalousFactor = 0.0;

    if( m_anomalousDataPresent ) {
        formFactor += m_realAnomalousFactor->evaluate( a_energy_in );
        imaginaryAnomalousFactor = m_imaginaryAnomalousFactor->evaluate( a_energy_in );
    }

    double probability = M_PI * MCGIDI_classicalElectronRadius * MCGIDI_classicalElectronRadius * ( 1.0 + a_mu_lab * a_mu_lab )
                        * ( formFactor * formFactor + imaginaryAnomalousFactor * imaginaryAnomalousFactor ) / sigma;

    return( probability );
}

/* *********************************************************************************************************//**
 * FIX ME.
 *
 * @param a_Z                       [in]    
 * @param a_a                       [in]    
 ***********************************************************************************************************/

HOST_DEVICE double CoherentPhotoAtomicScattering::Z_a( double a_Z, double a_a ) const {

    if( fabs( a_a ) < 1e-3 ) {
        double logZ = log( a_Z );
        double a_logZ = a_a * logZ;
        return( logZ * ( 1.0 + 0.5 * a_logZ * ( 1.0 + a_logZ / 3.0 * ( 1.0 + 0.25 * a_logZ ) ) ) );
    }
    return( ( pow( a_Z, a_a ) - 1.0 ) / a_a );
}

/* *********************************************************************************************************//**
 * FIX ME.
 *
 * @param a_n                       [in]    
 * @param a_a                       [in]    
 * @param a_logX                    [in]    
 * @param a_energy1                 [in]    
 * @param a_y1                      [in]    
 * @param a_energy2                 [in]    
 * @param a_y2                      [in]    
 ***********************************************************************************************************/

HOST_DEVICE static double coherentPhotoAtomicScatteringIntegrateSub( int a_n, double a_a, double a_logX, double a_energy1, double a_y1, double a_energy2, double a_y2 ) {

    double epsilon = a_a + a_n + 1.0;
    double integral = 0.0;

    if( fabs( epsilon ) < 1e-3 ) {
        double epsilon_logX = epsilon * a_logX;
        integral = a_y1 * pow( a_energy1, a_n + 1.0 ) * a_logX * ( 1.0 + 0.5 * epsilon_logX * ( 1.0 + epsilon_logX / 3.0 * ( 1.0 + 0.25 * epsilon_logX ) ) ); }
    else {
        integral = ( a_y2 * pow( a_energy2, a_n + 1.0 ) - a_y1 * pow( a_energy1, a_n + 1.0 ) ) / epsilon;
    }

    return( integral );
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

HOST_DEVICE void CoherentPhotoAtomicScattering::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    Distribution::serialize( a_buffer, a_mode );

    DATA_MEMBER_INT( m_anomalousDataPresent, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_energies, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_formFactor, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_a, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_integratedFormFactor, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_integratedFormFactorSquared, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_probabilityNorm1_1, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_probabilityNorm1_3, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_probabilityNorm1_5, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_probabilityNorm2_1, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_probabilityNorm2_3, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_probabilityNorm2_5, a_buffer, a_mode );

    if( m_anomalousDataPresent ) {
        m_realAnomalousFactor->serialize( a_buffer, a_mode );
        m_imaginaryAnomalousFactor->serialize( a_buffer, a_mode );
    }
}

/* *********************************************************************************************************//**
 * This method counts the number of bytes of memory allocated by *this*. That is the member needed by *this* that is greater than
 * sizeof( *this );
 ***********************************************************************************************************/

HOST_DEVICE long CoherentPhotoAtomicScattering::internalSize( ) const {

    long size = (long) ( m_energies.internalSize( ) + m_formFactor.internalSize( ) + m_a.internalSize( ) 
            + m_integratedFormFactor.internalSize( ) + m_integratedFormFactorSquared.internalSize( )
            + m_probabilityNorm1_1.internalSize( )  + m_probabilityNorm1_3.internalSize( )  + m_probabilityNorm1_5.internalSize( )
            + m_probabilityNorm2_1.internalSize( ) + m_probabilityNorm2_3.internalSize( ) + m_probabilityNorm2_5.internalSize( ) );

    if( m_anomalousDataPresent ) size += m_realAnomalousFactor->sizeOf( ) + m_imaginaryAnomalousFactor->sizeOf( );

    return( size );
}

/* *********************************************************************************************************//**
 * This method counts the number of bytes of memory allocated by *this*. That is the member needed by *this* that is greater than
 * sizeof( *this );
 ***********************************************************************************************************/

HOST_DEVICE long CoherentPhotoAtomicScattering::sizeOf( ) const {

    return( sizeof( *this ) );
}

/*! \class IncoherentPhotoAtomicScattering
 * This class represents the distribution for an outgoing photon via incoherent photo-atomic elastic scattering.
 */

/* *********************************************************************************************************//**
 * Default constructor used when broadcasting a Protare as needed by MPI or GPUs.
 ***********************************************************************************************************/

HOST_DEVICE IncoherentPhotoAtomicScattering::IncoherentPhotoAtomicScattering( ) {

}

/* *********************************************************************************************************//**
 * @param a_incoherentPhotoAtomicScattering     [in]    The GIDI::Distributions::IncoherentPhotoAtomicScattering instance whose data is to be used to construct *this*.
 * @param a_setupInfo                           [in]    Used internally when constructing a Protare to pass information to other constructors.
 ***********************************************************************************************************/

HOST IncoherentPhotoAtomicScattering::IncoherentPhotoAtomicScattering( GIDI::Distributions::IncoherentPhotoAtomicScattering const &a_incoherentPhotoAtomicScattering, SetupInfo &a_setupInfo ) :
        Distribution( Type::incoherentPhotoAtomicScattering, a_incoherentPhotoAtomicScattering, a_setupInfo ) {

    GIDI::Ancestry const *link = a_incoherentPhotoAtomicScattering.findInAncestry( a_incoherentPhotoAtomicScattering.href( ) );
    GIDI::DoubleDifferentialCrossSection::IncoherentPhotoAtomicScattering const &incoherentPhotoAtomicScattering = 
            *static_cast<GIDI::DoubleDifferentialCrossSection::IncoherentPhotoAtomicScattering const *>( link );

    std::string domainUnit;
    GIDI::Functions::XYs1d const *xys1d0, *xys1d1;
    std::size_t dataSize = 0, offset = 0;

    GIDI::Functions::Function1dForm const *scatteringFunction = incoherentPhotoAtomicScattering.scatteringFunction( );
    if( scatteringFunction->type( ) == GIDI::FormType::XYs1d ) {
        xys1d0 = static_cast<GIDI::Functions::XYs1d const *>( scatteringFunction );
        xys1d1 = xys1d0;

        domainUnit = xys1d0->axes( )[0]->unit( );

        dataSize = xys1d1->size( );
        offset = 1; }
    else if( scatteringFunction->type( ) == GIDI::FormType::regions1d ) {
        GIDI::Functions::Regions1d const *regions1d = static_cast<GIDI::Functions::Regions1d const *>( scatteringFunction );
        if( regions1d->size( ) != 2 ) THROW( "MCGIDI::CoherentPhotoAtomicScattering::CoherentPhotoAtomicScattering: unsupported form factor size." );

        domainUnit = regions1d->axes( )[0]->unit( );

        GIDI::Functions::Function1dForm const *region0 = (*regions1d)[0];
        if( region0->type( ) != GIDI::FormType::XYs1d ) THROW( "MCGIDI::CoherentPhotoAtomicScattering::CoherentPhotoAtomicScattering: unsupported form factor for region 0." );
        xys1d0 = static_cast<GIDI::Functions::XYs1d const *>( region0 );
        if( xys1d0->size( ) != 2 ) THROW( "MCGIDI::CoherentPhotoAtomicScattering::CoherentPhotoAtomicScattering: unsupported size of region 1 of form factor." );

        GIDI::Functions::Function1dForm const *region1 = (*regions1d)[1];
        if( region1->type( ) != GIDI::FormType::XYs1d ) THROW( "MCGIDI::CoherentPhotoAtomicScattering::CoherentPhotoAtomicScattering: unsupported form factor for region 1." );
        xys1d1 = static_cast<GIDI::Functions::XYs1d const *>( region1 );

        dataSize = xys1d1->size( ) + 1; }
    else {
        THROW( "MCGIDI::CoherentPhotoAtomicScattering::CoherentPhotoAtomicScattering: unsupported form factor. Must be XYs1d or regions1d." );
    }

    double domainFactor = 1.0;
    if( domainUnit == "1/Ang" ) {
        domainFactor = 0.012398419739640716; }                      // Converts 'h * c /Ang' to MeV.
    else if( domainUnit == "1/cm" ) {
        domainFactor = 0.012398419739640716 * 1e-8; }               // Converts 'h * c /cm' to MeV.
    else {
        THROW( "MCGIDI::IncoherentPhotoAtomicScattering::IncoherentPhotoAtomicScattering: unsupported domain unit" );
    }

    m_energies.resize( dataSize );
    m_scatteringFunction.resize( dataSize );
    m_a.resize( dataSize );

    std::pair<double, double> xy = (*xys1d0)[0];
    m_energies[0] = domainFactor * xy.first;
    m_scatteringFunction[0] = xy.second;
    m_a[0] = 1.0;

    xy = (*xys1d1)[offset];
    double energy1 = domainFactor * xy.first;
    double y1 = xy.second;

    m_energies[1] = energy1;
    m_scatteringFunction[1] = y1;

    for( std::size_t i1 = 1 + offset; i1 < xys1d1->size( ); ++i1 ) {
        xy = (*xys1d1)[i1];
        double energy2 = domainFactor * xy.first;
        double y2 = xy.second;

        m_energies[i1+1-offset] = energy2;
        m_scatteringFunction[i1+1-offset] = y2;

        double _a = log( y2 / y1 ) / log( energy2 / energy1 );
        m_a[i1-offset] = _a;

        energy1 = energy2;
        y1 = y2;
    }
    m_a[m_a.size()-1] = 0.0;
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

HOST_DEVICE IncoherentPhotoAtomicScattering::~IncoherentPhotoAtomicScattering( ) {

}

/* *********************************************************************************************************//**
 * FIX ME.
 *
 * @param a_energyIn                [in]    
 * @param a_mu                      [in]    
 ***********************************************************************************************************/

HOST_DEVICE double IncoherentPhotoAtomicScattering::energyRatio( double a_energyIn, double a_mu ) const {

    double relativeEnergy = a_energyIn / MCGIDI_electronMass_c2;

    return( 1.0 / ( 1.0 + relativeEnergy * ( 1.0 - a_mu ) ) );
}

/* *********************************************************************************************************//**
 * This method evaluates the Klein-Nishina. FIX ME. This should be a function as it does not use member data.
 *
 * @param a_energyIn                [in]    
 * @param a_mu                      [in]    
 ***********************************************************************************************************/

HOST_DEVICE double IncoherentPhotoAtomicScattering::evaluateKleinNishina( double a_energyIn, double a_mu ) const {

    double relativeEnergy = a_energyIn / MCGIDI_electronMass_c2;
    double _energyRatio = energyRatio( a_energyIn, a_mu );
    double one_minus_mu = 1.0 - a_mu;

    double norm = ( 1.0 + 2.0 * relativeEnergy );
    norm = 2.0 * relativeEnergy * ( 2.0 + relativeEnergy * ( 1.0 + relativeEnergy ) * ( 8.0 + relativeEnergy ) ) / ( norm * norm );
    norm += ( ( relativeEnergy - 2.0 ) * relativeEnergy - 2.0 ) * log( 1.0 + 2.0 * relativeEnergy );
    norm /= relativeEnergy * relativeEnergy * relativeEnergy;

    return( _energyRatio * _energyRatio * ( _energyRatio + a_mu * a_mu + relativeEnergy * one_minus_mu * one_minus_mu ) / norm );
}

/* *********************************************************************************************************//**
 * This method evaluates the Klein-Nishina.
 *
 * @param a_energyIn                [in]    The energy of the projectile.
 ***********************************************************************************************************/

HOST_DEVICE double IncoherentPhotoAtomicScattering::evaluateScatteringFunction( double a_energyIn ) const {

    MCGIDI_VectorSizeType lowerIndex = binarySearchVector( a_energyIn, m_energies );

    if( lowerIndex < 1 ) {
        if( lowerIndex == -1 ) return( m_scatteringFunction.back( ) );
        return( m_scatteringFunction[1] * a_energyIn / m_energies[1] );
    }

    return( m_scatteringFunction[lowerIndex] * pow( a_energyIn / m_energies[lowerIndex], m_a[lowerIndex] ) );
}

/* *********************************************************************************************************//**
 * This method samples the outgoing product data by sampling the outgoing energy E' from the probability P(E'|E) and then samples mu from 
 * the probability P(mu|E,E'). It also samples the outgoing phi uniformly between 0 and 2 pi.
 *
 * @param a_X                       [in]    The energy of the projectile.
 * @param a_input                   [in]    Sample options requested by user.
 * @param a_userrng                 [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState                [in]    The current state for the random number generator.
 ***********************************************************************************************************/

HOST_DEVICE void IncoherentPhotoAtomicScattering::sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const {

    double k1 = a_X / MCGIDI_electronMass_c2;
    double energyOut, mu, scatteringFunction;

    if( a_X >= m_energies.back( ) ) {
        MCGIDI_sampleKleinNishina( k1, a_userrng, a_rngState, &energyOut, &mu ); }
    else {
        double scatteringFunctionMax = evaluateScatteringFunction( a_X );
        do {
            MCGIDI_sampleKleinNishina( k1, a_userrng, a_rngState, &energyOut, &mu );
            scatteringFunction = evaluateScatteringFunction( a_X * sqrt( 0.5 * ( 1.0 - mu ) ) );
        } while( scatteringFunction < a_userrng( a_rngState ) * scatteringFunctionMax );
    }

    a_input.m_sampledType = Sampling::SampledType::uncorrelatedBody;
    a_input.m_energyOut1 = energyOut * MCGIDI_electronMass_c2;
    a_input.m_mu = mu;
    a_input.m_phi = 2.0 * M_PI * a_userrng( a_rngState );
    a_input.m_frame = productFrame( );
}

/* *********************************************************************************************************//**
 * Returns the probability for a projectile with energy *a_energy_in* to cause a particle to be emitted 
 * at angle *a_mu_lab* as seen in the lab frame. *a_energy_out* is the sampled outgoing energy.
 *
 * @param a_reaction                [in]    The reaction containing the particle which this distribution describes.
 * @param a_energy_in               [in]    The energy of the incident particle.
 * @param a_mu_lab                  [in]    The desired mu in the lab frame for the emitted particle.
 * @param a_userrng                 [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState                [in]    The current state for the random number generator.
 * @param a_energy_out              [in]    The energy of the emitted outgoing particle.
 *
 * @return                                  The probability of emitting outgoing particle into lab angle *a_mu_lab*.
 ***********************************************************************************************************/

HOST_DEVICE double IncoherentPhotoAtomicScattering::angleBiasing( Reaction const *a_reaction, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const {

    URR_protareInfos URR_protareInfos1;
    double sigma = a_reaction->protareSingle( )->reactionCrossSection( a_reaction->reactionIndex( ), URR_protareInfos1, 0.0, a_energy_in );

    double norm = M_PI * MCGIDI_classicalElectronRadius * MCGIDI_classicalElectronRadius / sigma;

    double one_minus_mu = 1.0 - a_mu_lab;
    double k_in = a_energy_in / MCGIDI_electronMass_c2;
    a_energy_out = a_energy_in / ( 1.0 + k_in * one_minus_mu );
    double k_out = a_energy_out / MCGIDI_electronMass_c2;

    double k_ratio = k_out / k_in;
    double probability = evaluateScatteringFunction( a_energy_in * sqrt( 0.5 * one_minus_mu ) );
    probability *= k_ratio * k_ratio * ( 1.0 + a_mu_lab * a_mu_lab + k_in * k_out * one_minus_mu * one_minus_mu ) * norm;

    return( probability );
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

HOST_DEVICE void IncoherentPhotoAtomicScattering::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    Distribution::serialize( a_buffer, a_mode );

    DATA_MEMBER_VECTOR_DOUBLE( m_energies, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_scatteringFunction, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_a, a_buffer, a_mode );

}

/* *********************************************************************************************************//**
 * This method counts the number of bytes of memory allocated by *this*. That is the member needed by *this* that is greater than
 * sizeof( *this );
 ***********************************************************************************************************/

HOST_DEVICE long IncoherentPhotoAtomicScattering::internalSize( ) const {

    long size = (long) ( m_energies.internalSize( ) + m_scatteringFunction.internalSize( ) + m_a.internalSize( ) );

    return( size );
}

/* *********************************************************************************************************//**
 * This method counts the number of bytes of memory allocated by *this*. That is the member needed by *this* that is greater than
 * sizeof( *this );
 ***********************************************************************************************************/

HOST_DEVICE long IncoherentPhotoAtomicScattering::sizeOf( ) const {

    return( sizeof( *this ) );
}

/*! \class PairProductionGamma
 * This class represents the distribution for an outgoing photon the is the result of an electron annihilating with a positron.
 */

HOST_DEVICE PairProductionGamma::PairProductionGamma( ) {

}

/* *********************************************************************************************************//**
 * @param a_setupInfo               [in]    Used internally when constructing a Protare to pass information to other constructors.
 * @param a_firstSampled            [in]    FIX ME
 ***********************************************************************************************************/

HOST PairProductionGamma::PairProductionGamma( SetupInfo &a_setupInfo, bool a_firstSampled ) :
        Distribution( Type::pairProductionGamma, GIDI::Frame::lab, a_setupInfo ),
        m_firstSampled( a_firstSampled ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

HOST PairProductionGamma::~PairProductionGamma( ) {

}

/* *********************************************************************************************************//**
 * This method samples the outgoing photon by assigning the electron rest mass energy as the photon's energy and,
 * if m_firstSampled is true, randomly picking mu and phi. If m_firstSampled is false, the previous sampled particle
 * that filled in a_input must be the other sampled photon, then, the mu and phi for the second-sampled photon is such that 
 * it is back-to-back with the other photon.
 *
 * @param a_X                       [in]    The energy of the projectile.
 * @param a_input                   [in]    Sample options requested by user.
 * @param a_userrng                 [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState                [in]    The current state for the random number generator.
 ***********************************************************************************************************/

HOST_DEVICE void PairProductionGamma::sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const {

    if( m_firstSampled ) {
        a_input.m_mu = 1.0 - 2.0 * a_userrng( a_rngState );
        a_input.m_phi = M_PI * a_userrng( a_rngState ); }
    else {
        a_input.m_mu *= -1.0;
        a_input.m_phi += M_PI;
    }
    a_input.m_sampledType = Sampling::SampledType::uncorrelatedBody;
    a_input.m_energyOut1 = MCGIDI_electronMass_c2;
    a_input.m_frame = productFrame( );
}

/* *********************************************************************************************************//**
 * Returns the probability for a projectile with energy *a_energy_in* to cause a particle to be emitted 
 * at angle *a_mu_lab* as seen in the lab frame. *a_energy_out* is the sampled outgoing energy.
 *
 * @param a_reaction                [in]    The reaction containing the particle which this distribution describes.
 * @param a_energy_in               [in]    The energy of the incident particle.
 * @param a_mu_lab                  [in]    The desired mu in the lab frame for the emitted particle.
 * @param a_userrng                 [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState                [in]    The current state for the random number generator.
 * @param a_energy_out              [in]    The energy of the emitted outgoing particle.
 *
 * @return                                  The probability of emitting outgoing particle into lab angle *a_mu_lab*.
 ***********************************************************************************************************/

HOST_DEVICE double PairProductionGamma::angleBiasing( Reaction const *a_reaction, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const {

    a_energy_out = MCGIDI_electronMass_c2;
    return( 1.0 );                          // 1.0 as there are two photons.
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

HOST_DEVICE void PairProductionGamma::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    Distribution::serialize( a_buffer, a_mode );

    DATA_MEMBER_INT( m_firstSampled, a_buffer, a_mode );
}

/*! \class Unspecified
 * This class represents the distribution for an outgoing product whose distribution is not specified.
 */

HOST_DEVICE Unspecified::Unspecified( ) {

}

/* *********************************************************************************************************//**
 * @param a_distribution            [in]    The GIDI::Distributions::Unspecified instance whose data is to be used to construct *this*.
 * @param a_setupInfo               [in]    Used internally when constructing a Protare to pass information to other constructors.
 ***********************************************************************************************************/

HOST Unspecified::Unspecified( GIDI::Distributions::Distribution const &a_distribution, SetupInfo &a_setupInfo ) :
        Distribution( Type::unspecified, a_distribution, a_setupInfo ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

HOST_DEVICE Unspecified::~Unspecified( ) {

}

/* *********************************************************************************************************//**
 * The method sets all outgoing product data to 0.0 and set the sampledType to Sampling::unspecified.
 *
 * @param a_X                       [in]    The energy of the projectile.
 * @param a_input                   [in]    Sample options requested by user.
 * @param a_userrng                 [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState                [in]    The current state for the random number generator.
 ***********************************************************************************************************/

HOST_DEVICE void Unspecified::sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const {

    a_input.m_sampledType = Sampling::SampledType::unspecified;
    a_input.m_energyOut1 = 0.;
    a_input.m_mu = 0.;
    a_input.m_phi = 0.;
    a_input.m_frame = productFrame( );
}

/* *********************************************************************************************************//**
 * Returns the probability for a projectile with energy *a_energy_in* to cause a particle to be emitted 
 * at angle *a_mu_lab* as seen in the lab frame. *a_energy_out* is the sampled outgoing energy. This one should never
 * be called. If called, returns 0.0 for a probability.
 *
 * @param a_reaction                [in]    The reaction containing the particle which this distribution describes.
 * @param a_energy_in               [in]    The energy of the incident particle.
 * @param a_mu_lab                  [in]    The desired mu in the lab frame for the emitted particle.
 * @param a_userrng                 [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState                [in]    The current state for the random number generator.
 * @param a_energy_out              [in]    The energy of the emitted outgoing particle.
 *
 * @return                                  The probability of emitting outgoing particle into lab angle *a_mu_lab*.
 ***********************************************************************************************************/

HOST_DEVICE double Unspecified::angleBiasing( Reaction const *a_reaction, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const {

    a_energy_out = 0.0;
    return( 0.0 );
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

HOST_DEVICE void Unspecified::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    Distribution::serialize( a_buffer, a_mode );
}


/* *********************************************************************************************************//**
 * This function is used to call the proper distribution constructor for *a_distribution*.
 *
 * @param a_distribution        [in]    The GIDI::Protare whose data is to be used to construct *this*.
 * @param a_setupInfo           [in]    Used internally when constructing a Protare to pass information to other constructors.
 * @param a_settings            [in]    Used to pass user options to the *this* to instruct it which data are desired.
 ***********************************************************************************************************/

HOST Distribution *parseGIDI( GIDI::Suite const &a_distribution, SetupInfo &a_setupInfo, Transporting::MC const &a_settings ) {

// BRB6
    std::string const *label = a_settings.styles( )->findLabelInLineage( a_distribution, a_setupInfo.m_distributionLabel );
    GIDI::Distributions::Distribution const &GIDI_distribution = *a_distribution.get<GIDI::Distributions::Distribution>( *label );

    Distribution *distribution = nullptr;

    GIDI::FormType type = GIDI_distribution.type( );

    switch( type ) {
    case GIDI::FormType::angularTwoBody :
        distribution = new AngularTwoBody( static_cast<GIDI::Distributions::AngularTwoBody const &>( GIDI_distribution ), a_setupInfo );
        break;
    case GIDI::FormType::uncorrelated :
        distribution = new Uncorrelated( static_cast<GIDI::Distributions::Uncorrelated const &>( GIDI_distribution ), a_setupInfo );
        break;
    case GIDI::FormType::KalbachMann :
        distribution = new KalbachMann( static_cast<GIDI::Distributions::KalbachMann const &>( GIDI_distribution ), a_setupInfo );
        break;
    case GIDI::FormType::energyAngularMC :
        distribution = new EnergyAngularMC( static_cast<GIDI::Distributions::EnergyAngularMC const &>( GIDI_distribution ), a_setupInfo );
        break;
    case GIDI::FormType::angularEnergyMC :
        distribution = new AngularEnergyMC( static_cast<GIDI::Distributions::AngularEnergyMC const &>( GIDI_distribution ), a_setupInfo );
        break;
    case GIDI::FormType::coherentPhotonScattering :
        distribution = new CoherentPhotoAtomicScattering( static_cast<GIDI::Distributions::CoherentPhotoAtomicScattering const &>( GIDI_distribution ), a_setupInfo );
        break;
    case GIDI::FormType::incoherentPhotonScattering :
        distribution = new IncoherentPhotoAtomicScattering( static_cast<GIDI::Distributions::IncoherentPhotoAtomicScattering const &>( GIDI_distribution ), a_setupInfo );
        break;
    case GIDI::FormType::branching3d :      // FIXME
    case GIDI::FormType::unspecified :
        distribution = new Unspecified( GIDI_distribution, a_setupInfo );
        break;
    default :
        std::cout << "distribution = moniker = " << GIDI_distribution.moniker( ) << " label = " << GIDI_distribution.label( ) << std::endl;
        THROW( "MCGIDI::Distributions::parseGIDI: unsupported distribution" );
    }

    return( distribution );
}

/* *********************************************************************************************************//**
 * @param a_distribution        [in]    The GIDI::Protare whose data is to be used to construct *this*.
 *
 * @return                              The type of the distribution or Distributions::Type::none if *a_distribution* is a *nullptr* pointer.
 ***********************************************************************************************************/

HOST_DEVICE Type DistributionType( Distribution const *a_distribution ) {

    if( a_distribution == nullptr ) return( Type::none );
    return( a_distribution->type( ) );
}

}

}

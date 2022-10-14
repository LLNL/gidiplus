/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "math.h"

#include "MCGIDI.hpp"

namespace MCGIDI {

/*! \class SetupInfo
 * This class is used internally when constructing a Protare to pass internal information to other constructors.
 */

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

MCGIDI_HOST SetupInfo::SetupInfo( ProtareSingle &a_protare ) :
        m_protare( a_protare ),
        m_initialStateIndex( -1 ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

MCGIDI_HOST SetupInfo::~SetupInfo( ) {

    for( auto iter = m_ACE_URR_protabilityTablesFromGIDI.begin( ); iter != m_ACE_URR_protabilityTablesFromGIDI.end( ); ++iter ) delete (*iter).second;
}


/* *********************************************************************************************************//**
 * @return                          The *index*.
 ***********************************************************************************************************/

MCGIDI_HOST int MCGIDI_popsIndex( PoPI::Database const &a_pops, std::string const &a_ID ) {

    int index = -1;

    try {
        index = a_pops[a_ID]; }
    catch (...) {
        index = -1;
    }

    return( index );
}

/* *********************************************************************************************************//**
 * @param           a_vector    [in]    The GIDI::Vector whose contents are coped to a MCGIGI::Vector.
 *
 * @return                              The MCGIGI::Vector.
 ***********************************************************************************************************/


MCGIDI_HOST Vector<double> GIDI_VectorDoublesToMCGIDI_VectorDoubles( GIDI::Vector a_vector ) {

    Vector<double> vector( static_cast<MCGIDI_VectorSizeType>( a_vector.size( ) ) );

    for( std::size_t i1 = 0; i1 < a_vector.size( ); ++i1 ) vector[i1] = a_vector[i1];

    return( vector );
}

/* *********************************************************************************************************//**
 * Adds the items in *a_productIndicesFrom* to the set *a_productIndicesTo*.
 *
 * @param a_productIndicesTo            [in]    The list of ints to add to the set.
 * @param a_productIndicesFrom          [in]    The set to add the ints to.
 ***********************************************************************************************************/

MCGIDI_HOST void addVectorItemsToSet( Vector<int> const &a_productIndicesFrom, std::set<int> &a_productIndicesTo ) {

    for( Vector<int>::const_iterator iter = a_productIndicesFrom.begin( ); iter != a_productIndicesFrom.end( ); ++iter ) a_productIndicesTo.insert( *iter );
}

/* *********************************************************************************************************//**
 * This function returns a particle kinetic energy from its mass and beta (i.e., v/c) using a relativistic formula.
 *
 * @param a_mass_unitOfEnergy   [in]    The particle's mass in units of energy.
 * @param a_particleBeta        [in]    The particle's velocity divided by the speed of light (i.e., beta = v/c).
 *
 * @return                              The relativistic kinetic energy of the particle.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE double particleKineticEnergy( double a_mass_unitOfEnergy, double a_particleBeta ) {

    if( a_particleBeta < 1e-4 ) return( 0.5 * a_mass_unitOfEnergy * a_particleBeta * a_particleBeta );

    return( a_mass_unitOfEnergy * ( 1.0 / sqrt( 1.0 - a_particleBeta * a_particleBeta ) - 1.0 ) );
}


/* *********************************************************************************************************//**
 * This function is like particleKineticEnergy except that *a_particleBeta2* is beta squared (i.e., (v/c)^2).
 *
 * @param a_mass_unitOfEnergy   [in]    The particle's mass in units of energy.
 * @param a_particleBeta2       [in]    The square of beta (i.e., beta^2 where beta = v/c).
 *
 * @return                              The relativistic kinetic energy of the particle.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE double particleKineticEnergyFromBeta2( double a_mass_unitOfEnergy, double a_particleBeta2 ) {

    if( a_particleBeta2 < 1e-8 ) return( 0.5 * a_mass_unitOfEnergy * a_particleBeta2 );

    return( a_mass_unitOfEnergy * ( 1.0 / sqrt( 1.0 - a_particleBeta2 ) - 1.0 ) );
}

/* *********************************************************************************************************//**
 * This function returns the boost speed required to boost to the center-of-mass for a projectile hitting a target.
 *
 * @param a_massProjectile              [in]    The mass of the projectile in energy units.
 * @param a_kineticEnergyProjectile     [in]    The kinetic energy of the projectile.
 * @param a_massTarget                  [in]    The mass of the target in energy units.
 *
 * @return                              The relativistic kinetic energy of the particle.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE double boostSpeed( double a_massProjectile, double a_kineticEnergyProjectile, double a_massTarget ) {

    double betaProjectile = MCGIDI_particleBeta( a_massProjectile, a_kineticEnergyProjectile );

    return( betaProjectile / ( 1.0 + a_massTarget / ( a_massProjectile + a_kineticEnergyProjectile ) ) );
}

/* *********************************************************************************************************//**
 * This function determines the mu value(s) in the center-of-mass frame for a specified mu value in the lab frame for a
 * product with speed *a_productBeta* and a boost speed of *a_productBeta*. The returned value is the number of mu values
 * in the center-of-mass frame. The return value can be 0, 1 or 2. *a_muMinus* and *a_JacobianMinus* are undefined when the
 * returned value is less than 2. *a_muPlus* and *a_JacobianPlus* are undefined when the returned value is 0.
 *
 * @param a_muLab                       [in]    The mu specified mu value in the lab frame.
 * @param a_boostBeta                   [in]    The boost speed from the lab from to the center-of-mass frame in units of the speed-of-light.
 * @param a_productBeta                 [in]    The speed of the product in the center-of-mass frame in units of the speed-of-light.
 * @param a_muPlus                      [in]    The first mu value if the returned is greater than 0.
 * @param a_JacobianPlus                [in]    The partial derivative of mu_com with respect to mu_lab at a_muPlus.
 * @param a_muMinus                     [in]    The second mu value if the returned value is 2.
 * @param a_JacobianMinus               [in]    The partial derivative of mu_com with respect to mu_lab at a_muMinus.
 *
 * @return                                      The number of returned center-of-mass frame mu values. Can be 0, 1 or 2.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE int muCOM_From_muLab( double a_muLab, double a_boostBeta, double a_productBeta, double &a_muPlus, double &a_JacobianPlus, 
                double &a_muMinus, double &a_JacobianMinus ) {

    int numberOfSolutions = 0;
    double boostBeta2 = a_boostBeta * a_boostBeta;
    double productBeta2 = a_productBeta * a_productBeta;
    double muLab2 = a_muLab * a_muLab;
    double oneMinusMuLab2 = 1.0 - muLab2;
    double oneMinusBoostBeta2 = 1.0 - boostBeta2;
    double oneMinusBoostBeta2MuLab2 = 1.0 - boostBeta2 * muLab2;

    a_muPlus = a_muLab;                                 // Handles case when a_productBeta is 0.0 or a_muLab is +/-1.0. Actually, when a_productBeta is 0.0 it is not defined.
    a_muMinus = 0.0;

    if( ( a_productBeta == 0.0 ) || ( a_muLab == 1.0 ) ) return( 1 );

    if( a_productBeta >= a_boostBeta ) {                // Intentionally treating case where a_productBeta == a_boostBeta as one solution even though is it
        numberOfSolutions = 1; }
    else {
        if( a_muLab > 0.0 ) {                           // Only have solutions for positive mu. The next expression only test mu^2 and therefore treats negative mu like positive mu.
            if( productBeta2 * oneMinusBoostBeta2MuLab2 > boostBeta2 * oneMinusMuLab2 ) numberOfSolutions = 2;      // This ignores the case for numberOfSolutions = 1 as it probabily will never happen.
        }
    }

    if( numberOfSolutions == 0 ) return( 0 );

    double sqrt_b2minus4ac = sqrt( oneMinusBoostBeta2 * ( productBeta2 * oneMinusBoostBeta2MuLab2 - boostBeta2 * oneMinusMuLab2 ) );
    double minusbTerm = a_boostBeta * oneMinusMuLab2;
    double inv2a = 1.0 / ( a_productBeta * oneMinusBoostBeta2MuLab2 );

    a_muPlus =  ( a_muLab * sqrt_b2minus4ac - minusbTerm ) * inv2a;
    a_muMinus = ( -a_muLab * sqrt_b2minus4ac - minusbTerm ) * inv2a;      // This is meaningless when numberOfSolutions is not 1, but why add an if test.

    double JacobianTerm1 = 2.0 * boostBeta2 * a_muLab / oneMinusBoostBeta2MuLab2;
    double JacobianTerm2 = 2.0 * a_muLab * a_boostBeta / ( a_productBeta * oneMinusBoostBeta2MuLab2 );
    double JacobianTerm3 = productBeta2 * ( 1.0 - 2.0 * boostBeta2 * muLab2 ) - boostBeta2 * ( 1.0 - 2.0 * muLab2 );
    JacobianTerm3 *= oneMinusBoostBeta2 / ( a_productBeta * oneMinusBoostBeta2MuLab2 * sqrt_b2minus4ac );

    a_JacobianPlus  = fabs( a_muPlus  * JacobianTerm1 + JacobianTerm2 + JacobianTerm3 );
    a_JacobianMinus = fabs( a_muMinus * JacobianTerm1 + JacobianTerm2 - JacobianTerm3 );

    return( numberOfSolutions );
}

/* *********************************************************************************************************//**
 * The function returns a normalized Maxwellian speed (i.e., v = |velocity|) in 3d (i.e., v^2 Exp( -v^2 )).
 * Using formula in https://link.springer.com/content/pdf/10.1007%2Fs10955-011-0364-y.pdf.
 * Author Nader M.A. Mohamed, title "Efficient Algorithm for Generating Maxwell Random Variables".
 *
 * @param a_userrng         [in]    The random number generator function to use.
 * @param a_rngState        [in]    The value of the random number generator state to use.
 *
 * @return                              The sampled normalized Maxwellian speed.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE double sampleBetaFromMaxwellian( double (*a_userrng)( void * ), void *a_rngState ) {

    double _g = 2.0 / ( 1.37 * 0.5 * 1.772453850905516 );      // 1.772453850905516 = sqrt( pi ).
    double beta, r1;

    do {
        r1 = a_userrng( a_rngState );
        beta = sqrt( -2.0 * log( r1 ) );
    } while( _g * r1 * beta < a_userrng( a_rngState ) );

    return( beta );
}

/* *********************************************************************************************************//**
 * This function is used internally to sample a target's velocity (speed and cosine of angle relative to projectile)
 * for a heated target using zero temperature, multi-grouped cross sections.
 *
 * @param a_protare             [in]    The Protare instance for the projectile and target.
 * @param a_projectileEnergy    [in]    The energy of the projectile in the lab frame of the target.
 * @param a_input               [in]    Contains needed input like the targets temperature. Also will have the target sampled velocity on return if return value is *true*.
 * @param a_userrng             [in]    The random number generator function to use.
 * @param a_rngState            [in]    The value of the random number generator state to use.
 *
 * @return                              Returns *true* if target velocity is sampled and false otherwise.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE bool sampleTargetBetaForUpscatterModelA( Protare const *a_protare, double a_projectileEnergy, Sampling::Input &a_input,
                double (*a_userrng)( void * ), void *a_rngState ) {

    double projectileBeta = MCGIDI_particleBeta( a_protare->projectileMass( ), a_projectileEnergy );

    double temperature = a_input.m_temperature * 1e-3;                   // FIXME Assumes m_temperature is in keV/k for now.
    double targetThermalBeta = MCGIDI_particleBeta( a_protare->targetMass( ), temperature );

    if( targetThermalBeta < 1e-4 * projectileBeta ) return( false );

    double relativeBetaMin = projectileBeta - 2.0 * targetThermalBeta;
    double relativeBetaMax = projectileBeta + 2.0 * targetThermalBeta;

    Vector<double> const &upscatterModelAGroupVelocities = a_protare->upscatterModelAGroupVelocities( );
    MCGIDI_VectorSizeType maxIndex = upscatterModelAGroupVelocities.size( ) - 2;
    MCGIDI_VectorSizeType relativeBetaMinIndex = binarySearchVector( relativeBetaMin, upscatterModelAGroupVelocities, true );
    MCGIDI_VectorSizeType relativeBetaMaxIndex = binarySearchVector( relativeBetaMax, upscatterModelAGroupVelocities, true );
    double targetBeta, relativeBeta, mu;

    if( relativeBetaMinIndex >= maxIndex ) relativeBetaMinIndex = maxIndex;
    if( relativeBetaMaxIndex >= maxIndex ) relativeBetaMaxIndex = maxIndex;

    if( relativeBetaMinIndex == relativeBetaMaxIndex ) {
        targetBeta = targetThermalBeta * sampleBetaFromMaxwellian( a_userrng, a_rngState );
        mu = 1.0 - 2.0 * a_userrng( a_rngState );
        relativeBeta = sqrt( targetBeta * targetBeta + projectileBeta * projectileBeta - 2.0 * mu * targetBeta * projectileBeta ); }
    else {

        Vector<double> const &upscatterModelACrossSection = a_input.m_reaction->upscatterModelACrossSection( );
        double reactionRate;
        double reactionRateMax = 0;
        for( MCGIDI_VectorSizeType i1 = relativeBetaMinIndex; i1 <= relativeBetaMaxIndex; ++i1 ) {
            reactionRate = upscatterModelACrossSection[i1] * upscatterModelAGroupVelocities[i1+1];
            if( reactionRate > reactionRateMax ) reactionRateMax = reactionRate;
        }

        do {
            targetBeta = targetThermalBeta * sampleBetaFromMaxwellian( a_userrng, a_rngState );
            mu = 1.0 - 2.0 * a_userrng( a_rngState );
            relativeBeta = sqrt( targetBeta * targetBeta + projectileBeta * projectileBeta - 2.0 * mu * targetBeta * projectileBeta );

            MCGIDI_VectorSizeType index = binarySearchVector( relativeBeta, upscatterModelAGroupVelocities, true );
            if( index > maxIndex ) index = maxIndex;
            reactionRate = upscatterModelACrossSection[index] * relativeBeta;
        } while( reactionRate <  a_userrng( a_rngState ) * reactionRateMax );
    }

    a_input.m_projectileBeta = projectileBeta;
    a_input.m_relativeMu = mu;
    a_input.m_targetBeta = targetBeta;
    a_input.m_relativeBeta = relativeBeta;
    a_input.m_projectileEnergy = particleKineticEnergy( a_protare->projectileMass( ), relativeBeta );

    return( true );
}

/* *********************************************************************************************************//**
 * This function boost a particle from one frame to another frame. The frames have a relative speed *a_boostSpeed*
 * and cosine of angle *a_boostMu* between their z-axes. BRB FIXME, currently it is the x-axis.
 *
 * @param a_input                   [in]    Instance containing a random number generator that returns a double in the range [0, 1).
 * @param a_userrng                 [in]    The random number generator function to use.
 * @param a_rngState                [in]    The value of the random number generator state to use.
 * @param a_product                 [in]    The particle to boost.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE void upScatterModelABoostParticle( Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState, Sampling::Product &a_product ) {

    double C_rel = 1.0;
    if( a_input.m_relativeBeta != 0.0 ) C_rel = ( a_input.m_projectileBeta - a_input.m_relativeMu * a_input.m_targetBeta ) / a_input.m_relativeBeta;
    double S_rel = sqrt( 1.0 - C_rel * C_rel );

    double pz_vz = a_product.m_pz_vz;
    a_product.m_pz_vz =  C_rel * a_product.m_pz_vz + S_rel * a_product.m_px_vx;
    a_product.m_px_vx = -S_rel * pz_vz             + C_rel * a_product.m_px_vx;

    double targetSpeed = MCGIDI_speedOfLight_cm_sec * a_input.m_targetBeta;
    a_product.m_pz_vz += a_input.m_relativeMu * targetSpeed;
    a_product.m_px_vx += sqrt( 1.0 - a_input.m_relativeMu * a_input.m_relativeMu ) * targetSpeed;

    double phi = 2.0 * M_PI * a_userrng( a_rngState );
    double sine = sin( phi );
    double cosine = cos( phi );
    double px_vx = a_product.m_px_vx;
    a_product.m_px_vx = cosine * a_product.m_px_vx - sine   * a_product.m_py_vy;
    a_product.m_py_vy = sine   * px_vx             + cosine * a_product.m_py_vy;

    double speed2 = a_product.m_px_vx * a_product.m_px_vx + a_product.m_py_vy * a_product.m_py_vy + a_product.m_pz_vz * a_product.m_pz_vz;
    speed2 /= MCGIDI_speedOfLight_cm_sec * MCGIDI_speedOfLight_cm_sec;

    a_product.m_kineticEnergy = particleKineticEnergyFromBeta2( a_product.m_productMass, speed2 );
}

/* *********************************************************************************************************//**
 * This function samples an energy and cosine of the angle for a photon for Klein Nishina scattering (i.e, incoherent photo-atomic scattering).
 *
 * @param a_energyIn            [in]    The energy of the incoming photon.
 * @param a_userrng             [in]    The random number generator function to use.
 * @param a_rngState            [in]    The value of the random number generator state to use.
 * @param a_energyOut           [in]    The energy of the scattered photon.
 * @param a_mu                  [in]    The cosine of the angle of the scattered photon's z-axis and the incoming photon's z-axis.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE void MCGIDI_sampleKleinNishina( double a_energyIn, double (*a_userrng)( void * ), void *a_rngState, double *a_energyOut, double *a_mu ) {
/*
  Description
    Sample the Klein-Nishina distribution.
      The unit of energy is the rest mass of the electron.
      Reference: R. N. Blomquist and E. N. Gelbard, Nuclear Science
      and Engineering, 83, 380-384 (1983)

   This routine was taken from MCAPM which was from MCNP with only cosmetic changes.

   Input
     a_energyIn       - incident photon energy ( in electron rest mass units )
     *userrng       - user supplied random number generator
     *rngstate      - random number generator state
   Output
     *a_energyOut   - exiting photon energy ( in electron rest mass units )
     *a_mu          - exiting photon cosine
*/

    double a1, b1, t1, s1, r1, mu, energyOut;

    a1 = 1.0 / a_energyIn;
    b1 = 1.0 / ( 1.0 + 2.0 * a_energyIn );

    if( a_energyIn < 3.0 ) {                      // Kahn''s method ( e < 1.5 MeV ) AECU-3259.
        bool reject = true;

        t1 = 1.0 / ( 1.0 + 8.0 * b1 );
        do {
            if( a_userrng( a_rngState ) <= t1 ) {
                r1 = 2.0 * a_userrng( a_rngState );
                s1 = 1.0 / ( 1.0 + a_energyIn * r1 );
                mu = 1.0 - r1;
                reject = a_userrng( a_rngState ) > 4.0 * s1 * ( 1.0 - s1 ); }
            else {
                s1 = ( 1.0 + 2.0 * a_energyIn * a_userrng( a_rngState ) ) * b1;
                mu = 1.0 + a1 * ( 1.0 - 1.0 / s1 );
                reject = a_userrng( a_rngState ) > 0.5 * ( mu * mu + s1 );
            }
        } while( reject );
        energyOut = a_energyIn / ( 1 + a_energyIn * ( 1 - mu ) ); }
    else {                                        // Koblinger''s method ( e > 1.5 MeV ) NSE 56, 218 ( 1975 ).
        t1 = a_userrng( a_rngState ) * ( 4.0 * a1 + 0.5 * ( 1.0 - b1 * b1 ) - ( 1.0 - 2.0 * ( 1.0 + a_energyIn ) * ( a1 * a1 ) ) * log( b1 ) );
        if( t1 > 2.0 * a1 ) {
            if( t1 > 4.0 * a1 ) {
                if( t1 > 4.0 * a1 + 0.5 * ( 1.0 - b1 * b1 ) ) {
                    energyOut = a_energyIn * pow( b1, a_userrng( a_rngState ) );
                    mu = 1.0 + a1 - 1.0 / energyOut; }
                else {
                    energyOut = a_energyIn * sqrt( 1.0 - a_userrng( a_rngState ) * ( 1.0 - b1 * b1 ) );
                    mu = 1.0 + a1 - 1.0 / energyOut;
                  } }
            else {
                energyOut = a_energyIn * ( 1.0 + a_userrng( a_rngState ) * ( b1 - 1.0 ) );
                mu =  1.0 + a1 - 1.0 / energyOut; } }
        else {
            r1 = 2.0 * a_userrng( a_rngState );
            mu = 1.0 - r1;
            energyOut = 1.0 / ( a1 + r1 );
          }
    }

    *a_mu = mu;
    *a_energyOut = energyOut;

    return;
}

/* *********************************************************************************************************//**
 * This function returns a unique integer for the **Distributions::Type**. For internal use when broadcasting a
 * distribution for MPI and GPUs needs.
 *              
 * @param a_type                [in]    The distribution's type.
 *
 * @return                              Returns a unique integer for the distribution type.
 ***********************************************************************************************************/
            
MCGIDI_HOST_DEVICE int distributionTypeToInt( Distributions::Type a_type ) {

    int distributionType = 0;

    switch( a_type ) {
    case Distributions::Type::none :
        distributionType = 0;
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
    case Distributions::Type::coherentElasticTNSL :
        distributionType = 10;
        break;
    case Distributions::Type::incoherentElasticTNSL :
        distributionType = 11;
        break;
    case Distributions::Type::incoherentPhotoAtomicScatteringElectron :
        distributionType = 12;
        break;
    }

    return( distributionType );
}

/* *********************************************************************************************************//**
 * This function returns the **Distributions::Type** corresponding to the integer returned by **distributionTypeToInt**.
 *
 * @param a_type                [in]    The value returned by **distributionTypeToInt**.
 *
 * @return                              The **Distributions::Type** corresponding to *a_type*.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE Distributions::Type intToDistributionType( int a_type ) {

    Distributions::Type type = Distributions::Type::none;

    switch( a_type ) {
    case 0 :
        type = Distributions::Type::none;
        break;
    case 1 :
        type = Distributions::Type::unspecified;
        break;
    case 2 :
        type = Distributions::Type::angularTwoBody;
        break;
    case 3 :
        type = Distributions::Type::KalbachMann;
        break;
    case 4 :
        type = Distributions::Type::uncorrelated;
        break;
    case 5 :
        type = Distributions::Type::energyAngularMC;
        break;
    case 6 :
        type = Distributions::Type::angularEnergyMC;
        break;
    case 7 :
        type = Distributions::Type::coherentPhotoAtomicScattering;
        break;
    case 8 :
        type = Distributions::Type::incoherentPhotoAtomicScattering;
        break;
    case 9 :
        type = Distributions::Type::pairProductionGamma;
        break;
    case 10 :
        type = Distributions::Type::coherentElasticTNSL;
        break;
    case 11 :
        type = Distributions::Type::incoherentElasticTNSL;
        break;
    case 12 :
        type = Distributions::Type::incoherentPhotoAtomicScatteringElectron;
        break;
    default:
        MCGIDI_THROW( "intToDistributionType: unsupported distribution type." );
    }

    return( type );
}

}

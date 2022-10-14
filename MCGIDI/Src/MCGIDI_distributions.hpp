/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#ifndef MCGIDI_distributions_hpp_included
#define MCGIDI_distributions_hpp_included 1

#include "MCGIDI_declareMacro.hpp"

namespace MCGIDI {

namespace Distributions {

enum class Type { none, unspecified, angularTwoBody, KalbachMann, uncorrelated, energyAngularMC, angularEnergyMC, coherentPhotoAtomicScattering,
        incoherentPhotoAtomicScattering, incoherentPhotoAtomicScatteringElectron, pairProductionGamma, coherentElasticTNSL, incoherentElasticTNSL };

/*
============================================================
======================= Distribution =======================
============================================================
*/
class Distribution {

    private:
        Type m_type;                                    /**< Specifies the Type of the distribution. */
        GIDI::Frame m_productFrame;                     /**< Specifies the frame the product data are given in. */
        double m_projectileMass;                        /**< The mass of the projectile. */
        double m_targetMass;                            /**< The mass of the target. */
        double m_productMass;                           /**< The mass of the first product. */

    public:
        MCGIDI_HOST_DEVICE Distribution( );
        MCGIDI_HOST Distribution( Type a_type, GIDI::Distributions::Distribution const &a_distribution, SetupInfo &a_setupInfo );
        MCGIDI_HOST Distribution( Type a_type, GIDI::Frame a_productFrame, SetupInfo &a_setupInfo );
        MCGIDI_HOST_DEVICE virtual ~Distribution( ) = 0;

        MCGIDI_HOST_DEVICE Type type( ) const { return( m_type ); }                            /**< Returns the value of the **m_type**. */
        MCGIDI_HOST_DEVICE GIDI::Frame productFrame( ) const { return( m_productFrame ); }     /**< Returns the value of the **m_productFrame**. */

        MCGIDI_HOST_DEVICE double projectileMass( ) const { return( m_projectileMass ); }                  /**< Returns the value of the **m_projectileMass**. */
        MCGIDI_HOST_DEVICE double targetMass( ) const { return( m_targetMass ); }                          /**< Returns the value of the **m_targetMass**. */
        MCGIDI_HOST_DEVICE double productMass( ) const { return( m_productMass ); }                        /**< Returns the value of the **m_productMass**. */

        MCGIDI_HOST_DEVICE virtual void sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const = 0;
        MCGIDI_HOST_DEVICE virtual double angleBiasing( Reaction const *a_reaction, double a_temperature, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const = 0;
        MCGIDI_HOST_DEVICE virtual void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
====================== AngularTwoBody ======================
============================================================
*/
class AngularTwoBody : public Distribution {

    private:
        double m_residualMass;                                  /**< The mass of the second product (often the  residual). */
        double m_Q;                                             /**< FIX ME. */
        double m_twoBodyThreshold;                              /**< This is the T_1 value needed to do two-body kinematics (i.e., in the equation (K_{com,3_4} = m_2 * (K_1 - T_1) / (m_1 + m_2)). */
        bool m_Upscatter;                                       /**< Set to true if reaction is elastic which is the only reaction upscatter Model B is applied to. */
        Probabilities::ProbabilityBase2d *m_angular;

        MCGIDI_HOST_DEVICE bool upscatterModelB( double a_kineticLab, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;

    public:
        MCGIDI_HOST_DEVICE AngularTwoBody( );
        MCGIDI_HOST AngularTwoBody( GIDI::Distributions::AngularTwoBody const &a_angularTwoBody, SetupInfo &a_setupInfo );
        MCGIDI_HOST_DEVICE ~AngularTwoBody( );

        MCGIDI_HOST_DEVICE double residualMass( ) const { return( m_residualMass ); }                      /**< Returns the value of the **m_residualMass**. */
        MCGIDI_HOST_DEVICE double Q( ) const { return( m_Q ); }                                            /**< Returns the value of the **m_Q**. */
        MCGIDI_HOST_DEVICE Probabilities::ProbabilityBase2d *angular( ) const { return( m_angular ); }     /**< Returns the value of the **m_angular**. */
        MCGIDI_HOST_DEVICE void sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE double angleBiasing( Reaction const *a_reaction, double a_temperature, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        MCGIDI_HOST_DEVICE bool Upscatter( ) const { return( m_Upscatter ); }                              /**< Returns the value of the **m_Upscatter**. */
};

/*
============================================================
======================= Uncorrelated =======================
============================================================
*/
class Uncorrelated : public Distribution {

    private:
        Probabilities::ProbabilityBase2d *m_angular;            /**< The angular probability P(mu|E). */
        Probabilities::ProbabilityBase2d *m_energy;             /**< The energy probability P(E'|E). */
        
    public:
        MCGIDI_HOST_DEVICE Uncorrelated( );
        MCGIDI_HOST Uncorrelated( GIDI::Distributions::Uncorrelated const &a_uncorrelated, SetupInfo &a_setupInfo );
        MCGIDI_HOST_DEVICE ~Uncorrelated( );

        MCGIDI_HOST_DEVICE Probabilities::ProbabilityBase2d *angular( ) const { return( m_angular ); }     /**< Returns the value of the **m_angular**. */
        MCGIDI_HOST_DEVICE Probabilities::ProbabilityBase2d *energy( ) const { return( m_energy ); }       /**< Returns the value of the **m_energy**. */
        MCGIDI_HOST_DEVICE void sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE double angleBiasing( Reaction const *a_reaction, double a_temperature, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
====================== EnergyAngularMC =====================
============================================================
*/
class EnergyAngularMC : public Distribution {

    private:
        Probabilities::ProbabilityBase2d *m_energy;                 /**< The energy probability P(E'|E). */
        Probabilities::ProbabilityBase3d *m_angularGivenEnergy;     /**< The angular probability given E', P(mu|E,E'). */
        
    public:
        MCGIDI_HOST_DEVICE EnergyAngularMC( );
        MCGIDI_HOST EnergyAngularMC( GIDI::Distributions::EnergyAngularMC const &a_energyAngularMC, SetupInfo &a_setupInfo );
        MCGIDI_HOST_DEVICE ~EnergyAngularMC( );

        MCGIDI_HOST_DEVICE Probabilities::ProbabilityBase2d *energy( ) const { return( m_energy ); }       /**< Returns the value of the **m_energy**. */
        MCGIDI_HOST_DEVICE Probabilities::ProbabilityBase3d *angularGivenEnergy( ) const { return( m_angularGivenEnergy ); }   /**< Returns the value of the **m_angularGivenEnergy**. */
        MCGIDI_HOST_DEVICE void sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE double angleBiasing( Reaction const *a_reaction, double a_temperature, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
====================== AngularEnergyMC =====================
============================================================
*/
class AngularEnergyMC : public Distribution {

    private:
        Probabilities::ProbabilityBase2d *m_angular;                /**< The angular probability P(mu|E). */
        Probabilities::ProbabilityBase3d *m_energyGivenAngular;     /**< The energy probability P(E'|E,mu). */
        
    public:
        MCGIDI_HOST_DEVICE AngularEnergyMC( );
        MCGIDI_HOST AngularEnergyMC( GIDI::Distributions::AngularEnergyMC const &a_angularEnergyMC, SetupInfo &a_setupInfo );
        MCGIDI_HOST_DEVICE ~AngularEnergyMC( );

        MCGIDI_HOST_DEVICE Probabilities::ProbabilityBase2d *angular( ) const { return( m_angular ); }     /**< Returns the value of the **m_angular**. */
        MCGIDI_HOST_DEVICE Probabilities::ProbabilityBase3d *energyGivenAngular( ) const { return( m_energyGivenAngular ); }   /**< Returns the value of the **m_energyGivenAngular**. */
        MCGIDI_HOST_DEVICE void sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE double angleBiasing( Reaction const *a_reaction, double a_temperature, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
======================== KalbachMann =======================
============================================================
*/
class KalbachMann : public Distribution {

    private:
        double m_energyToMeVFactor;                                 /**< The factor that converts energies to MeV. */
        double m_eb_massFactor;                                     /**< FIX ME */
        Probabilities::ProbabilityBase2d *m_f;                      /**< The energy probability P(E'|E). */
        Functions::Function2d *m_r;                                 /**< The Kalbach-Mann r(E,E') function. */
        Functions::Function2d *m_a;                                 /**< The Kalbach-Mann a(E,E') function. */

    public:
        MCGIDI_HOST_DEVICE KalbachMann( );
        MCGIDI_HOST KalbachMann( GIDI::Distributions::KalbachMann const &a_KalbachMann, SetupInfo &a_setupInfo );
        MCGIDI_HOST_DEVICE ~KalbachMann( );

        MCGIDI_HOST_DEVICE double energyToMeVFactor( ) const { return( m_energyToMeVFactor ); }    /**< Returns the value of the **m_energyToMeVFactor**. */
        MCGIDI_HOST_DEVICE double eb_massFactor( ) const { return( m_eb_massFactor ); }            /**< Returns the value of the **m_eb_massFactor**. */
        MCGIDI_HOST_DEVICE Probabilities::ProbabilityBase2d *f( ) const { return( m_f ); }         /**< Returns the value of the **m_f**. */
        MCGIDI_HOST_DEVICE Functions::Function2d *r( ) const { return( m_r ); }                    /**< Returns the value of the **m_r**. */
        MCGIDI_HOST_DEVICE Functions::Function2d *a( ) const { return( m_a ); }                    /**< Returns the value of the **m_a**. */
        MCGIDI_HOST_DEVICE void sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE double angleBiasing( Reaction const *a_reaction, double a_temperature, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );

        MCGIDI_HOST_DEVICE double evaluate( double E_in_lab, double E_out, double mu );
};

/*
============================================================
=============== CoherentPhotoAtomicScattering ==============
============================================================
*/
class CoherentPhotoAtomicScattering : public Distribution {

    private:
        bool m_anomalousDataPresent;                                /**< FIX ME */
        Vector<double> m_energies;                                  /**< FIX ME */
        Vector<double> m_formFactor;                                /**< FIX ME */
        Vector<double> m_a;                                         /**< FIX ME */
        Vector<double> m_integratedFormFactor;                      /**< FIX ME */
        Vector<double> m_integratedFormFactorSquared;               /**< FIX ME */
        Vector<double> m_probabilityNorm1_1;                        /**< FIX ME */
        Vector<double> m_probabilityNorm1_3;                        /**< FIX ME */
        Vector<double> m_probabilityNorm1_5;                        /**< FIX ME */
        Vector<double> m_probabilityNorm2_1;                        /**< FIX ME */
        Vector<double> m_probabilityNorm2_3;                        /**< FIX ME */
        Vector<double> m_probabilityNorm2_5;                        /**< FIX ME */
        Functions::Function1d *m_realAnomalousFactor;               /**< The real part of the anomalous scattering factor. */
        Functions::Function1d *m_imaginaryAnomalousFactor;          /**< The imaginary part of the anomalous scattering factor. */

        MCGIDI_HOST_DEVICE double Z_a( double a_Z, double a_a ) const ;

    public:
        MCGIDI_HOST_DEVICE CoherentPhotoAtomicScattering( );
        MCGIDI_HOST CoherentPhotoAtomicScattering( GIDI::Distributions::CoherentPhotoAtomicScattering const &a_coherentPhotoAtomicScattering, SetupInfo &a_setupInfo );
        MCGIDI_HOST_DEVICE ~CoherentPhotoAtomicScattering( );

        MCGIDI_HOST_DEVICE double evaluate( double a_energyIn, double a_mu ) const ;
        MCGIDI_HOST_DEVICE double evaluateFormFactor( double a_energyIn, double a_mu ) const ;
        MCGIDI_HOST_DEVICE void sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE double angleBiasing( Reaction const *a_reaction, double a_temperature, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const ;

        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
============== IncoherentPhotoAtomicScattering =============
============================================================
*/
class IncoherentPhotoAtomicScattering : public Distribution {

    private:
        Vector<double> m_energies;                                  /**< FIX ME */
        Vector<double> m_scatteringFactor;                          /**< FIX ME */
        Vector<double> m_a;                                         /**< FIX ME */

    public:
        MCGIDI_HOST_DEVICE IncoherentPhotoAtomicScattering( );
        MCGIDI_HOST IncoherentPhotoAtomicScattering( GIDI::Distributions::IncoherentPhotoAtomicScattering const &a_incoherentPhotoAtomicScattering, SetupInfo &a_setupInfo );
        MCGIDI_HOST_DEVICE ~IncoherentPhotoAtomicScattering( );

        MCGIDI_HOST_DEVICE double energyRatio( double a_energyIn, double a_mu ) const ;
        MCGIDI_HOST_DEVICE double evaluateKleinNishina( double a_energyIn, double a_mu ) const ;
        MCGIDI_HOST_DEVICE double evaluateScatteringFactor( double a_X ) const ;
        MCGIDI_HOST_DEVICE void sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE double angleBiasing( Reaction const *a_reaction, double a_temperature, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
/*
        MCGIDI_HOST_DEVICE double evaluate( double E_in_lab, double mu );
*/
};

/*
============================================================
========== IncoherentPhotoAtomicScatteringElectron =========
============================================================
*/
class IncoherentPhotoAtomicScatteringElectron : public Distribution {

    public:
        MCGIDI_HOST_DEVICE IncoherentPhotoAtomicScatteringElectron( );
        MCGIDI_HOST IncoherentPhotoAtomicScatteringElectron( SetupInfo &a_setupInfo );
        MCGIDI_HOST_DEVICE ~IncoherentPhotoAtomicScatteringElectron( );

        MCGIDI_HOST_DEVICE void sample( double a_energy, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE double angleBiasing( Reaction const *a_reaction, double a_temperature, double a_energy_in, double a_mu_lab,
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
==================== PairProductionGamma ===================
============================================================
*/
class PairProductionGamma : public Distribution {

    private:
        bool m_firstSampled;                                    /**< When sampling photons for pair production, the photons must be emitted back-to-back. The flag help do this. */

    public:
        MCGIDI_HOST_DEVICE PairProductionGamma( );
        MCGIDI_HOST PairProductionGamma( SetupInfo &a_setupInfo, bool a_firstSampled );
        MCGIDI_HOST_DEVICE ~PairProductionGamma( );

        MCGIDI_HOST_DEVICE void sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE double angleBiasing( Reaction const *a_reaction, double a_temperature, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
==================== CoherentElasticTNSL ===================
============================================================
*/
class CoherentElasticTNSL : public Distribution {

    private:
        Interpolation m_temperatureInterpolation;
        Vector<double> m_temperatures;
        Vector<double> m_energies;
        Vector<double> m_S_table;

    public:
        MCGIDI_HOST_DEVICE CoherentElasticTNSL( );
        MCGIDI_HOST CoherentElasticTNSL( GIDI::DoubleDifferentialCrossSection::n_ThermalNeutronScatteringLaw::CoherentElastic const *a_coherentElasticTNSL, 
                SetupInfo &a_setupInfo );
        MCGIDI_HOST_DEVICE ~CoherentElasticTNSL( ) {}

        MCGIDI_HOST_DEVICE void sample( double a_energy, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE double angleBiasing( Reaction const *a_reaction, double a_temperature, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
==================== IncoherentElasticTNSL ===================
============================================================
*/
class IncoherentElasticTNSL : public Distribution {

    private:
        double m_temperatureToMeV_K;
        Functions::Function1d *m_DebyeWallerIntegral;

    public:
        MCGIDI_HOST_DEVICE IncoherentElasticTNSL( );
        MCGIDI_HOST IncoherentElasticTNSL( GIDI::DoubleDifferentialCrossSection::n_ThermalNeutronScatteringLaw::IncoherentElastic const *a_incoherentElasticTNSL, 
                SetupInfo &a_setupInfo );
        MCGIDI_HOST_DEVICE ~IncoherentElasticTNSL( ) {}

        MCGIDI_HOST_DEVICE void sample( double a_energy, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE double angleBiasing( Reaction const *a_reaction, double a_temperature, double a_energy_in, double a_mu_lab,
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );

        Functions::Function1d       *DebyeWallerIntegral( )       { return( m_DebyeWallerIntegral ); }
        Functions::Function1d const *DebyeWallerIntegral( ) const { return( m_DebyeWallerIntegral ); }
};

/*
============================================================
======================= Unspecified ========================
============================================================
*/
class Unspecified : public Distribution {

    public:
        MCGIDI_HOST_DEVICE Unspecified( );
        MCGIDI_HOST Unspecified( GIDI::Distributions::Distribution const &a_distribution, SetupInfo &a_setupInfo );
        MCGIDI_HOST_DEVICE ~Unspecified( );

        MCGIDI_HOST_DEVICE void sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE double angleBiasing( Reaction const *a_reaction, double a_temperature, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
========================== Others ==========================
============================================================
*/
MCGIDI_HOST Distribution *parseGIDI( GIDI::Suite const &a_distribution, SetupInfo &a_setupInfo, Transporting::MC const &a_settings );
MCGIDI_HOST_DEVICE Type DistributionType( Distribution const *a_distribution );

}

}

#endif      // End of MCGIDI_distributions_hpp_included

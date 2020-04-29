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

#define MCGIDI_electronMass_c2 0.510998946269       // electron mass * c^2 in MeV.

namespace Distributions {

enum class Type { none, unspecified, angularTwoBody, KalbachMann, uncorrelated, energyAngularMC, angularEnergyMC, coherentPhotoAtomicScattering,
        incoherentPhotoAtomicScattering, pairProductionGamma };

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
        double m_productMass;                                   /**< The mass of the first product. */

    public:
        HOST_DEVICE Distribution( );
        HOST Distribution( Type a_type, GIDI::Distributions::Distribution const &a_distribution, SetupInfo &a_setupInfo );
        HOST Distribution( Type a_type, GIDI::Frame a_productFrame, SetupInfo &a_setupInfo );
        HOST_DEVICE virtual ~Distribution( ) = 0;

        HOST_DEVICE Type type( ) const { return( m_type ); }                            /**< Returns the value of the **m_type**. */
        HOST_DEVICE GIDI::Frame productFrame( ) const { return( m_productFrame ); }     /**< Returns the value of the **m_productFrame**. */

        HOST_DEVICE double projectileMass( ) const { return( m_projectileMass ); }                  /**< Returns the value of the **m_projectileMass**. */
        HOST_DEVICE double targetMass( ) const { return( m_targetMass ); }                          /**< Returns the value of the **m_targetMass**. */
        HOST_DEVICE double productMass( ) const { return( m_productMass ); }                        /**< Returns the value of the **m_productMass**. */

        HOST_DEVICE virtual void sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const = 0;
        HOST_DEVICE virtual double angleBiasing( Reaction const *a_reaction, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const = 0;
        HOST_DEVICE virtual void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE virtual long internalSize( ) const { return( 0 ); }                 /**< Returns the total member in bytes allocated for *this*. */
        HOST_DEVICE virtual long sizeOf( ) const { return( sizeof( *this ) ); }         /**< Returns sizeof( *this* ) + internalSize( ). */

        HOST_DEVICE virtual void evaluate_pdf(double E_in_lab, double mu,    double &pdf_val) 
            { printf("MCGIDI Programmer Error. Distribution base class evaluate_pdf should never be used"); }

        HOST_DEVICE virtual void evaluate_pdf(double E_in_lab, double E_out, double mu, double &pdf_val) 
            { printf("MCGIDI Programmer Error. Distribution base class evaluate_pdf should never be used"); }

        HOST_DEVICE virtual void sample_pdf( double E_in_lab, double mu, double &E_out, double random_num, Sampling::Input &a_input, 
                double (*a_userrng)( void * ), void *a_rngState  ) const {}
};

/*
============================================================
====================== AngularTwoBody ======================
============================================================
*/
class AngularTwoBody : public Distribution {

    // Suppress warnings about overloaded virtual classes hidden by their definitions in this class
    // NOTE: these hidden functions should never be implemented
    using Distribution::evaluate_pdf;

    private:
        double m_residualMass;                                  /**< The mass of the second product (often the  residual). */
        double m_Q;                                             /**< FIX ME. */
        double m_crossSectionThreshold;                         /**< Threshold value derived from cross section data via *evaluated* or *griddedCrossSection*. */
        bool m_Upscatter;                                       /**< Set to true if reaction is elastic which is the only reaction upscatter Model B is applied to. */
        Probabilities::ProbabilityBase2d *m_angular;

        HOST_DEVICE bool upscatterModelB( double a_kineticLab, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;

    public:
        HOST_DEVICE AngularTwoBody( );
        HOST AngularTwoBody( GIDI::Distributions::AngularTwoBody const &a_angularTwoBody, SetupInfo &a_setupInfo );
        HOST_DEVICE ~AngularTwoBody( );

        HOST_DEVICE double residualMass( ) const { return( m_residualMass ); }                      /**< Returns the value of the **m_residualMass**. */
        HOST_DEVICE double Q( ) const { return( m_Q ); }                                            /**< Returns the value of the **m_Q**. */
        HOST_DEVICE Probabilities::ProbabilityBase2d *angular( ) const { return( m_angular ); }     /**< Returns the value of the **m_angular**. */
        HOST_DEVICE void sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;
        HOST_DEVICE double angleBiasing( Reaction const *a_reaction, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const ;
        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long internalSize( ) const { return( m_angular == nullptr ? 0 : m_angular->sizeOf( ) + m_angular->internalSize( ) ); }
        HOST_DEVICE long sizeOf( ) const { return( sizeof( *this ) ); }                             /**< Returns the total member in bytes allocated for *this*. */
        HOST_DEVICE bool Upscatter( ) const { return( m_Upscatter ); }                              /**< Returns the value of the **m_Upscatter**. */
        HOST_DEVICE void evaluate_pdf( double E_in_lab, double mu, double &pdf_val ) {
            pdf_val = m_angular->evaluate( E_in_lab, mu );
        }
};

/*
============================================================
======================= Uncorrelated =======================
============================================================
*/
class Uncorrelated : public Distribution {

    // Suppress warnings about overloaded virtual classes hidden by their definitions in this class
    // NOTE: these hidden functions should never be implemented
    using Distribution::evaluate_pdf;

    private:
        Probabilities::ProbabilityBase2d *m_angular;            /**< The angular probability P(mu|E). */
        Probabilities::ProbabilityBase2d *m_energy;             /**< The energy probability P(E'|E). */
        
    public:
        HOST_DEVICE Uncorrelated( );
        HOST Uncorrelated( GIDI::Distributions::Uncorrelated const &a_uncorrelated, SetupInfo &a_setupInfo );
        HOST_DEVICE ~Uncorrelated( );

        HOST_DEVICE Probabilities::ProbabilityBase2d *angular( ) const { return( m_angular ); }     /**< Returns the value of the **m_angular**. */
        HOST_DEVICE Probabilities::ProbabilityBase2d *energy( ) const { return( m_energy ); }       /**< Returns the value of the **m_energy**. */
        HOST_DEVICE void sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;
        HOST_DEVICE double angleBiasing( Reaction const *a_reaction, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const ;
        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long internalSize( ) const { 
                return( ( m_angular == nullptr ? 0 : m_angular->sizeOf( ) + m_angular->internalSize( ) ) + 
                        ( m_energy == nullptr ? 0 : m_energy->sizeOf( ) + m_energy->internalSize( ) ) ); } /**< Returns the total member in bytes allocated for *this*. */
        HOST_DEVICE long sizeOf( ) const { return( sizeof( *this ) ); }                                 /**< Returns sizeof( *this* ) + internalSize( ). */

        HOST_DEVICE void evaluate_pdf(double E_in_lab, double mu, double &pdf_val)
        {
            pdf_val = m_angular->evaluate(E_in_lab, mu);
        }
};

/*
============================================================
====================== EnergyAngularMC =====================
============================================================
*/
class EnergyAngularMC : public Distribution {

    // Suppress warnings about overloaded virtual classes hidden by their definitions in this class
    // NOTE: these hidden functions should never be implemented
    using Distribution::evaluate_pdf;

    private:
        Probabilities::ProbabilityBase2d *m_energy;                 /**< The energy probability P(E'|E). */
        Probabilities::ProbabilityBase3d *m_angularGivenEnergy;     /**< The angular probability given E', P(mu|E,E'). */
        
    public:
        HOST_DEVICE EnergyAngularMC( );
        HOST EnergyAngularMC( GIDI::Distributions::EnergyAngularMC const &a_energyAngularMC, SetupInfo &a_setupInfo );
        HOST_DEVICE ~EnergyAngularMC( );

        HOST_DEVICE Probabilities::ProbabilityBase2d *energy( ) const { return( m_energy ); }       /**< Returns the value of the **m_energy**. */
        HOST_DEVICE Probabilities::ProbabilityBase3d *angularGivenEnergy( ) const { return( m_angularGivenEnergy ); }   /**< Returns the value of the **m_angularGivenEnergy**. */
        HOST_DEVICE void sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;
        HOST_DEVICE double angleBiasing( Reaction const *a_reaction, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const ;
        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long internalSize( ) const {
                return( ( m_energy == nullptr ? 0 : m_energy->sizeOf( ) + m_energy->internalSize( ) ) + 
                        ( m_angularGivenEnergy == nullptr ? 0 : m_angularGivenEnergy->sizeOf( ) + m_angularGivenEnergy->internalSize( ) ) ); }
                                                                                                    /**< Returns the total member in bytes allocated for *this*. */
        HOST_DEVICE long sizeOf( ) const { return( sizeof( *this ) ); }                             /**< Returns sizeof( *this* ) + internalSize( ). */

        HOST_DEVICE void evaluate_pdf(double E_in_lab, double E_out, double mu, double &pdf_val)
        {
            pdf_val = m_angularGivenEnergy->evaluate(E_in_lab, E_out, mu);
        }
};

/*
============================================================
====================== AngularEnergyMC =====================
============================================================
*/
class AngularEnergyMC : public Distribution {

    // Suppress warnings about overloaded virtual classes hidden by their definitions in this class
    // NOTE: these hidden functions should never be implemented
    using Distribution::evaluate_pdf;

    private:
        Probabilities::ProbabilityBase2d *m_angular;                /**< The angular probability P(mu|E). */
        Probabilities::ProbabilityBase3d *m_energyGivenAngular;     /**< The energy probability P(E'|E,mu). */
        
    public:
        HOST_DEVICE AngularEnergyMC( );
        HOST AngularEnergyMC( GIDI::Distributions::AngularEnergyMC const &a_angularEnergyMC, SetupInfo &a_setupInfo );
        HOST_DEVICE ~AngularEnergyMC( );

        HOST_DEVICE Probabilities::ProbabilityBase2d *angular( ) const { return( m_angular ); }     /**< Returns the value of the **m_angular**. */
        HOST_DEVICE Probabilities::ProbabilityBase3d *energyGivenAngular( ) const { return( m_energyGivenAngular ); }   /**< Returns the value of the **m_energyGivenAngular**. */
        HOST_DEVICE void sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;
        HOST_DEVICE double angleBiasing( Reaction const *a_reaction, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const ;
        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long internalSize( ) const { 
                return( ( m_angular == nullptr ? 0 : m_angular->sizeOf( ) + m_angular->internalSize( ) ) + 
                        ( m_energyGivenAngular == nullptr ? 0 : m_energyGivenAngular->sizeOf( ) + m_energyGivenAngular->internalSize( ) ) ); }
                                                                                                    /**< Returns the total member in bytes allocated for *this*. */
        HOST_DEVICE long sizeOf( ) const { return( sizeof( *this ) ); }                             /**< Returns sizeof( *this* ) + internalSize( ). */

        HOST_DEVICE void evaluate_pdf(double E_in_lab, double mu, double &pdf_val)
        {
            pdf_val = m_angular->evaluate(E_in_lab, mu);
        }
        HOST_DEVICE void sample_pdf( double E_in_lab, double mu, double &E_out, double random_num, Sampling::Input &a_input,
                double (*a_userrng)( void * ), void *a_rngState ) const {
            E_out = m_energyGivenAngular->sample( E_in_lab, mu, random_num, a_userrng, a_rngState );
        }
};

/*
============================================================
======================== KalbachMann =======================
============================================================
*/
class KalbachMann : public Distribution {

    // Suppress warnings about overloaded virtual classes hidden by their definitions in this class
    // NOTE: these hidden functions should never be implemented
    using Distribution::evaluate_pdf;

    private:
        double m_energyToMeVFactor;                                 /**< The factor that converts energies to MeV. */
        double m_eb_massFactor;                                     /**< FIX ME */
        Probabilities::ProbabilityBase2d *m_f;                      /**< The energy probability P(E'|E). */
        Functions::Function2d *m_r;                                 /**< The Kalbach-Mann r(E,E') function. */
        Functions::Function2d *m_a;                                 /**< The Kalbach-Mann a(E,E') function. */

    public:
        HOST_DEVICE KalbachMann( );
        HOST KalbachMann( GIDI::Distributions::KalbachMann const &a_KalbachMann, SetupInfo &a_setupInfo );
        HOST_DEVICE ~KalbachMann( );

        HOST_DEVICE double energyToMeVFactor( ) const { return( m_energyToMeVFactor ); }    /**< Returns the value of the **m_energyToMeVFactor**. */
        HOST_DEVICE double eb_massFactor( ) const { return( m_eb_massFactor ); }            /**< Returns the value of the **m_eb_massFactor**. */
        HOST_DEVICE Probabilities::ProbabilityBase2d *f( ) const { return( m_f ); }         /**< Returns the value of the **m_f**. */
        HOST_DEVICE Functions::Function2d *r( ) const { return( m_r ); }                    /**< Returns the value of the **m_r**. */
        HOST_DEVICE Functions::Function2d *a( ) const { return( m_a ); }                    /**< Returns the value of the **m_a**. */
        HOST_DEVICE void sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;
        HOST_DEVICE double angleBiasing( Reaction const *a_reaction, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const ;
        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long internalSize( ) const {
            return( ( m_f == nullptr ? 0 : m_f->sizeOf( ) + m_f->internalSize( ) ) + 
                    ( m_r == nullptr ? 0 : m_r->sizeOf( ) + m_r->internalSize( ) ) + 
                    ( m_a == nullptr ? 0 : m_a->sizeOf( ) + m_a->internalSize( ) ) ); }        /**< Returns the total member in bytes allocated for *this*. */
        HOST_DEVICE long sizeOf( ) const { return( sizeof( *this ) ); }                     /**< Returns sizeof( *this* ) + internalSize( ). */

        HOST_DEVICE double evaluate( double E_in_lab, double E_out, double mu );
        HOST_DEVICE void evaluate_pdf(double E_in_lab, double E_out, double mu, double &pdf_val)
        {
            pdf_val = evaluate(E_in_lab, E_out, mu);
        }
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

        HOST_DEVICE double Z_a( double a_Z, double a_a ) const ;

    public:
        HOST_DEVICE CoherentPhotoAtomicScattering( );
        HOST CoherentPhotoAtomicScattering( GIDI::Distributions::CoherentPhotoAtomicScattering const &a_coherentPhotoAtomicScattering, SetupInfo &a_setupInfo );
        HOST_DEVICE ~CoherentPhotoAtomicScattering( );

        HOST_DEVICE double evaluate( double a_energyIn, double a_mu ) const ;
        HOST_DEVICE double evaluateFormFactor( double a_energyIn, double a_mu ) const ;
        HOST_DEVICE void sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;
        HOST_DEVICE double angleBiasing( Reaction const *a_reaction, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const ;

        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long internalSize( ) const ;
        HOST_DEVICE long sizeOf( ) const ;
};

/*
============================================================
=============== IncoherentPhotoAtomicScattering ==============
============================================================
*/
class IncoherentPhotoAtomicScattering : public Distribution {

    private:
        Vector<double> m_energies;                                  /**< FIX ME */
        Vector<double> m_scatteringFunction;                        /**< FIX ME */
        Vector<double> m_a;                                         /**< FIX ME */

    public:
        HOST_DEVICE IncoherentPhotoAtomicScattering( );
        HOST IncoherentPhotoAtomicScattering( GIDI::Distributions::IncoherentPhotoAtomicScattering const &a_incoherentPhotoAtomicScattering, SetupInfo &a_setupInfo );
        HOST_DEVICE ~IncoherentPhotoAtomicScattering( );

        HOST_DEVICE double energyRatio( double a_energyIn, double a_mu ) const ;
        HOST_DEVICE double evaluateKleinNishina( double a_energyIn, double a_mu ) const ;
        HOST_DEVICE double evaluateScatteringFunction( double a_X ) const ;
        HOST_DEVICE void sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;
        HOST_DEVICE double angleBiasing( Reaction const *a_reaction, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const ;
        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long internalSize( ) const ;
        HOST_DEVICE long sizeOf( ) const ;
/*
        HOST_DEVICE double evaluate( double E_in_lab, double mu );
*/
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
        HOST_DEVICE PairProductionGamma( );
        HOST PairProductionGamma( SetupInfo &a_setupInfo, bool a_firstSampled );
        HOST_DEVICE ~PairProductionGamma( );

        HOST_DEVICE void sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;
        HOST_DEVICE double angleBiasing( Reaction const *a_reaction, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const ;
        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE virtual long internalSize( ) const { return( 0 ); }                 /**< Returns the total member in bytes allocated for *this*. */
        HOST_DEVICE long sizeOf( ) const { return( sizeof( *this ) ); }                 /**< Returns sizeof( *this* ) + internalSize( ). */
};

/*
============================================================
======================= Unspecified ========================
============================================================
*/
class Unspecified : public Distribution {

    // Suppress warnings about overloaded virtual classes hidden by their definitions in this class
    // NOTE: these hidden functions should never be implemented
    using Distribution::evaluate_pdf;

    public:
        HOST_DEVICE Unspecified( );
        HOST Unspecified( GIDI::Distributions::Distribution const &a_distribution, SetupInfo &a_setupInfo );
        HOST_DEVICE ~Unspecified( );

        HOST_DEVICE void sample( double a_X, Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState ) const ;
        HOST_DEVICE double angleBiasing( Reaction const *a_reaction, double a_energy_in, double a_mu_lab, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_energy_out ) const ;
        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long sizeOf( ) const { return( sizeof( *this ) ); }             /**< Returns sizeof( *this* ) + internalSize( ). */
};

/*
============================================================
========================== Others ==========================
============================================================
*/
HOST Distribution *parseGIDI( GIDI::Suite const &a_distribution, SetupInfo &a_setupInfo, Transporting::MC const &a_settings );
HOST_DEVICE Type DistributionType( Distribution const *a_distribution );

}

}

#endif      // End of MCGIDI_distributions_hpp_included

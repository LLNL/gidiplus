/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#ifndef MCGIDI_hpp_included
#define MCGIDI_hpp_included 1

#include "math.h"

#include <PoPI.hpp>
#include <GIDI.hpp>

namespace MCGIDI {

class Protare;
class ProtareSingle;
class Reaction;
class OutputChannel;

}           // End of namespace MCGIDI.

#include "MCGIDI_sampling.hpp"
#include "MCGIDI_dataBuffer.hpp"
#include "MCGIDI_vector.hpp"
#include "MCGIDI_string.hpp"
#include "MCGIDI_declareMacro.hpp"

namespace MCGIDI {

#define MCGIDI_nullReaction -10001

// FIXME, this should not be used once physicalQuantity can handle changing units.
#define MCGIDI_speedOfLight_cm_sh 299.792458
#define MCGIDI_speedOfLight_cm_sec ( MCGIDI_speedOfLight_cm_sh * 1e8 )
#define MCGIDI_classicalElectronRadius 0.2817940322010228 // Classical electron radius in unit of sqrt( b ).

#define MCGIDI_particleBeta( a_mass_unitOfEnergy, a_kineticEnergy ) ( sqrt( (a_kineticEnergy) * ( (a_kineticEnergy) + 2.0 * (a_mass_unitOfEnergy) ) ) / ( (a_kineticEnergy) + (a_mass_unitOfEnergy) ) )

HOST_DEVICE double particleKineticEnergy( double a_mass_unitOfEnergy, double a_particleBeta );
HOST_DEVICE double particleKineticEnergyFromBeta2( double a_mass_unitOfEnergy, double a_particleBeta2 );    // a_particleBeta2 = a_particleBeta^2.
HOST_DEVICE double boostSpeed( double a_massProjectile, double a_kineticEnergyProjectile, double a_massTarget );
HOST_DEVICE int muCOM_From_muLab( double a_muLab, double a_boostBeta, double a_productBeta, double &a_muPlus, double &a_JacobianPlus, double &a_muMinus, double &a_JacobianMinus );

enum class ProtareType { single, composite, TNSL };

namespace Transporting {

namespace LookupMode {

    enum class Data1d { continuousEnergy, multiGroup };
    enum class Distribution { pdf_cdf, epbs };

}           // End of namespace LookupMode.

namespace Reaction {

    enum class Type { Reactions, OrphanProducts };

}           // End of namespace Reaction.

/*
============================================================
============================ MC ============================
============================================================
*/
class MC : public GIDI::Transporting::Settings {

    private:
        PoPI::Database const &m_pops;
        int m_neutronIndex;
        int m_photonIndex;
        GIDI::Styles::Suite const *m_styles;
        std::string m_label;
        double m_energyDomainMax;
        bool m_ignoreENDF_MT5;
        bool m_sampleNonTransportingParticles;                                             /**< If true, all products are sampled, otherwise only transporting particles are sampled. */
        LookupMode::Data1d m_crossSectionLookupMode;
        LookupMode::Data1d m_other1dDataLookupMode;
        LookupMode::Distribution m_distributionLookupMode;
        Sampling::Upscatter::Model m_upscatterModel;
        std::string m_upscatterModelALabel;
        bool m_want_URR_probabilityTables;
        bool m_wantTerrellPromptNeutronDistribution;
        std::vector<double> m_fixedGridPoints;

    public:
        MC( PoPI::Database const &a_pops, std::string const &a_projectileID, GIDI::Styles::Suite const *a_styles, std::string const &a_label, GIDI::Transporting::DelayedNeutrons a_delayedNeutrons, double energyDomainMax );

        GIDI::Styles::Suite const *styles( ) const { return( m_styles ); }               /**< Returns the value of the **m_styles**. */
        void styles( GIDI::Styles::Suite const *a_styles ) { m_styles = a_styles; }      /**< This is needed for ProtareTNSL, but should be avoided otherwise. FIXME, need to have a better way. */

        std::string label( ) const { return( m_label ); }

// FIXME (1) should this not be something like
// GIDI::Styles::Suite const &suite( ) const { return( *m_styles ); }                   /**< Returns a reference to **m_styles**. */
        double energyDomainMax( ) const { return( m_energyDomainMax ); }                /**< Returns the value of the **m_energyDomainMax**. */

        bool ignoreENDF_MT5( ) const { return( m_ignoreENDF_MT5 ); }                    /**< Returns the value of the **m_ignoreENDF_MT5**. */
        void ignoreENDF_MT5( bool a_ignoreENDF_MT5 ) { m_ignoreENDF_MT5 = a_ignoreENDF_MT5; }

        bool sampleNonTransportingParticles( ) const { return( m_sampleNonTransportingParticles); }
        void sampleNonTransportingParticles( bool a_sampleNonTransportingParticles ) { m_sampleNonTransportingParticles = a_sampleNonTransportingParticles; }

        LookupMode::Data1d crossSectionLookupMode( ) const { return( m_crossSectionLookupMode ); }    /**< Returns the value of the **m_crossSectionLookupMode**. */
        void crossSectionLookupMode( LookupMode::Data1d a_crossSectionLookupMode );
        LookupMode::Data1d other1dDataLookupMode( ) const { return( m_other1dDataLookupMode ); }      /**< Returns the value of the **m_other1dDataLookupMode**. */
        void other1dDataLookupMode( LookupMode::Data1d a_other1dDataLookupMode );
        LookupMode::Distribution distributionLookupMode( ) const { return( m_distributionLookupMode ); }  /**< Returns the value of the **m_distributionLookupMode**. */
        void distributionLookupMode( LookupMode::Distribution a_distributionLookupMode );

        Sampling::Upscatter::Model upscatterModel( ) const { return( m_upscatterModel ); }             /**< Returns the value of the **m_upscatterModel**. */
        void setUpscatterModelA( std::string const &a_upscatterModelALabel );
        std::string upscatterModelALabel( ) const { return( m_upscatterModelALabel ); }                 /**< Returns the value of the **m_upscatterModelALabel**. */

        bool want_URR_probabilityTables( ) const { return( m_want_URR_probabilityTables ); }
        void want_URR_probabilityTables( bool a_want_URR_probabilityTables ) { m_want_URR_probabilityTables = a_want_URR_probabilityTables; }

        bool wantTerrellPromptNeutronDistribution( ) const { return( m_wantTerrellPromptNeutronDistribution ); }
        void wantTerrellPromptNeutronDistribution( bool a_wantTerrellPromptNeutronDistribution ) { m_wantTerrellPromptNeutronDistribution = a_wantTerrellPromptNeutronDistribution; }

        std::vector<double> fixedGridPoints( ) const { return( m_fixedGridPoints ); }
        void fixedGridPoints( std::vector<double> a_fixedGridPoints ) { m_fixedGridPoints = a_fixedGridPoints; }

        PoPI::Database const &pops( ) const { return( m_pops ); }                       /**< Returns a reference to **m_styles**. */
        int neutronIndex( ) const { return( m_neutronIndex ); }
        int photonIndex( ) const { return( m_photonIndex ); }

        void process( GIDI::Protare const &a_protare );
};

}           // End of namespace Transporting.

enum class TwoBodyOrder { notApplicable, firstParticle, secondParticle };

/*
============================================================
========================= SetupInfo ========================
============================================================
*/
class SetupInfo {

    public:
        Protare &m_protare;
        double m_Q;
        double m_product1Mass;
        double m_product2Mass;
        double m_domainMin;
        double m_domainMax;
        TwoBodyOrder m_twoBodyOrder;
        bool m_isPairProduction;
        std::string m_distributionLabel;                        /**< Set by the ProtareSingle constructor to the distribution label to use for all products. */
        std::map<std::string, int> m_particleIndices;
        GIDI::Reaction const *m_reaction;
        Transporting::Reaction::Type m_reactionType;
        GIDI::OutputChannel const *m_outputChannel;
        std::map<std::string, int> m_stateNamesToIndices;

        HOST SetupInfo( Protare &a_protare );
};

/*
============================================================
=========================== Others =========================
============================================================
*/
HOST int MCGIDI_popsIndex( PoPI::Database const &a_pops, std::string const &a_ID );

#if 0
/* *********************************************************************************************************//**
 * This function does a binary search of *a_Xs* for the *index* for which *a_Xs*[*index*] <= *a_x* < *a_Xs*[*index*+1].
 * The values of *a_Xs* must be ascending (i.e., *a_Xs*[i] < *a_Xs*[i+1]).
 *
 *
 *   Returns -2 if a_x < a_Xs[0] or 0 if a_boundIndex is true,
 *           -1 if a_x > last point of a_Xs or a_Xs.size( ) - 1 if a_boundIndex is true, or
 *           the lower index of a_Xs which bound a_x otherwise.
 *
 * Note, when *a_boundIndex* is false the returned *index* can be negative and when it is true
 * the return value will be a valid index of *a_Xs*, including its last point. The index of the last
 * point is only returned when *a_boundIndex* is true and *a_x* is great than the last point of *a_Xs*.
 *
 * @param a_x               [in]    The values whose bounding index within *a_Xs* is to be determined.
 * @param a_Xs              [in]    The list of ascending values.
 * @param a_boundIndex      [in]    If true, out-of-bounds values a treated as end points.
 *
 * @return                          The *index*.
 ***********************************************************************************************************/
#endif

HOST_DEVICE inline MCGIDI_VectorSizeType binarySearchVector( double a_x, Vector<double> const &a_Xs, bool a_boundIndex = false ) {

    MCGIDI_VectorSizeType lower = 0, middle, upper = (MCGIDI_VectorSizeType) a_Xs.size( ) - 1;

    if( a_x < a_Xs[0] ) {
        if( a_boundIndex ) return( 0 );
        return( -2 );
    }

    if( a_x > a_Xs[upper] ) {
        if( a_boundIndex ) return( upper );
        return( -1 );
    }

    while( 1 ) {
        middle = ( lower + upper ) >> 1;
        if( middle == lower ) break;
        if( a_x < a_Xs[middle] ) {
            upper = middle; }
        else {
            lower = middle;
        }
    }
    return( lower );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

HOST_DEVICE inline MCGIDI_VectorSizeType binarySearchVectorBounded( double a_x, Vector<double> const &a_Xs, MCGIDI_VectorSizeType a_lower, 
                MCGIDI_VectorSizeType a_upper, bool a_boundIndex ) {

    MCGIDI_VectorSizeType middle;

    if( a_x < a_Xs[a_lower] ) {
        if( a_boundIndex ) return( 0 );
        return( -2 );
    }

    if( a_x > a_Xs[a_upper] ) {
        if( a_boundIndex ) return( a_upper );
        return( -1 );
    }

    while( 1 ) {
        middle = ( a_lower + a_upper ) >> 1;
        if( middle == a_lower ) break;
        if( a_x < a_Xs[middle] ) {
            a_upper = middle; }
        else {
            a_lower = middle;
        }
    }
    return( a_lower );
}

}           // End of namespace MCGIDI.

#include "MCGIDI_functions.hpp"
#include "MCGIDI_distributions.hpp"

namespace MCGIDI {

enum class ChannelType { none, twoBody, uncorrelatedBodies };

/*
============================================================
======================= DomainHash =========================
============================================================
*/
class DomainHash {

    private:
        int m_bins;                                                         /**< The number of bins for the hash. */
        double m_domainMin;                                                 /**< The minimum domain value for the hash. */
        double m_domainMax;                                                 /**< The maximum domain value for the hash. */
        double m_u_domainMin;                                               /**< The log of m_domainMin ). */
        double m_u_domainMax;                                               /**< The log of m_domainMax ). */
        double m_inverse_du;                                                /**< The value *m_bins* / ( *m_u_domainMax* - *m_u_domainMin* ). */

    public:
        HOST_DEVICE DomainHash( );
        HOST_DEVICE DomainHash( int a_bins, double a_domainMin, double a_domainMax );
        HOST_DEVICE DomainHash( DomainHash const &a_domainHash );

        HOST_DEVICE int bins( ) const { return( m_bins ); }                     /**< Returns the value of the **m_bins**. */
        HOST_DEVICE double domainMin( ) const { return( m_domainMin ); }        /**< Returns the value of the **m_domainMax**. */
        HOST_DEVICE double domainMax( ) const { return( m_domainMax ); }        /**< Returns the value of the **m_domainMax**. */
        HOST_DEVICE double u_domainMin( ) const { return( m_u_domainMin ); }    /**< Returns the value of the **m_u_domainMin**. */
        HOST_DEVICE double u_domainMax( ) const { return( m_u_domainMax ); }    /**< Returns the value of the **m_u_domainMax**. */
        HOST_DEVICE double inverse_du( ) const { return( m_inverse_du ); }      /**< Returns the value of the **m_inverse_du**. */

        HOST_DEVICE int index( double a_domain ) const ;
        HOST_DEVICE Vector<int> map( Vector<double> &a_domainValues ) const ;

        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE void print( bool a_printValues ) const ;
};

/*
============================================================
===================== MultiGroupHash =======================
============================================================
*/
class MultiGroupHash {

    private:
        Vector<double> m_boundaries;                                    /**< The list of multi-group boundaries. */

        void initialize( GIDI::Protare const &a_protare, std::string const &a_label, std::string a_particleID );

    public:
        MultiGroupHash( std::vector<double> a_boundaries );
        MultiGroupHash( GIDI::Protare const &a_protare, std::string const &a_label, std::string const &a_particleID = "" );
        MultiGroupHash( GIDI::Protare const &a_protare, std::string const &a_particleID = "" );

        HOST_DEVICE Vector<double> const &boundaries( ) const { return( m_boundaries ); }   /**< Returns a reference to **m_styles**. */
        HOST_DEVICE int index( double a_domain ) const {
            MCGIDI_VectorSizeType _index = binarySearchVector( a_domain, m_boundaries );

            if( _index == -2 ) return( 0 );
            if( _index == -1 ) return( m_boundaries.size( ) - 2 );
            return( _index );
        }
        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
====================== URR_protareInfo =====================
============================================================
*/
class URR_protareInfo {

    public:
        bool m_inURR;
        double m_rng_Value;

        HOST_DEVICE URR_protareInfo( ) : m_inURR( false ), m_rng_Value( 0.0 ) { }
        HOST_DEVICE URR_protareInfo( URR_protareInfo const &a_URR_protareInfo ) { m_inURR = a_URR_protareInfo.m_inURR; m_rng_Value = a_URR_protareInfo.m_rng_Value; }

        HOST_DEVICE bool inURR( ) const { return( m_inURR ); }
        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
===================== URR_protareInfos =====================
============================================================
*/
class URR_protareInfos {

    private:
        Vector<URR_protareInfo> m_URR_protareInfos;

    public:
        HOST_DEVICE URR_protareInfos( ) : m_URR_protareInfos( ) { }
        HOST URR_protareInfos( Vector<Protare *> &a_protares );

        HOST void setup( Vector<Protare *> &a_protares );

        HOST_DEVICE MCGIDI_VectorSizeType size( ) const { return( m_URR_protareInfos.size( ) ); }
        HOST_DEVICE URR_protareInfo const &operator[]( MCGIDI_VectorSizeType a_index ) const { return( m_URR_protareInfos[a_index] ); }  /**< Returns the instance of *m_URR_protareInfos* at index *a_index*. */
        HOST_DEVICE void updateProtare( MCGIDI::Protare const *a_protare, double a_energy, double (*a_userrng)( void * ), void *a_rngState );

        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long internalSize( ) const { return m_URR_protareInfos.internalSize( ); }
};

/*
============================================================
======== HeatedReactionCrossSectionContinuousEnergy ========
============================================================
*/
class HeatedReactionCrossSectionContinuousEnergy {

    private:
        int m_offset;
        double m_threshold;
        Vector<double> m_crossSection;                          // Reaction cross section
        Probabilities::ProbabilityBase2d *m_URR_probabilityTables;

    public:
        HOST_DEVICE HeatedReactionCrossSectionContinuousEnergy( );
        HOST HeatedReactionCrossSectionContinuousEnergy( int a_offset, double a_threshold, Vector<double> &a_crossSection );
        HOST HeatedReactionCrossSectionContinuousEnergy( double a_threshold, GIDI::Functions::Ys1d const &a_crossSection, Probabilities::ProbabilityBase2d *a_URR_probabilityTables );

        HOST_DEVICE double threshold( ) const { return( m_threshold ); }                            /**< Returns the value of the **m_threshold**. */
        HOST_DEVICE int offset( ) const { return( m_offset ); }                                     /**< Returns the value of the **m_offset**. */
        HOST_DEVICE bool hasURR_probabilityTables( ) const { return( m_URR_probabilityTables != nullptr ); }   /**< Returns true if URR probability tables data present and false otherwise. */
        HOST_DEVICE double URR_domainMin( ) const ;
        HOST_DEVICE double URR_domainMax( ) const ;
        HOST_DEVICE Probabilities::ProbabilityBase2d *URR_probabilityTables( ) const { return( m_URR_probabilityTables ); }             /**< Returns the value of the **m_URR_probabilityTables**. */
        HOST_DEVICE double crossSection( std::size_t a_index ) const {
            int index = static_cast<int>( a_index ) - m_offset;
            if( index < 0 ) return( 0 );
            if( index >= static_cast<int>( m_crossSection.size( ) ) ) return( 0.0 );

            return( m_crossSection[index] );
        }
        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long sizeOf( ) const ;
};

/*
============================================================
=================== ContinuousEnergyGain ===================
============================================================
*/
class ContinuousEnergyGain {

    private:
        int m_particleIndex;
        int m_userParticleIndex;
        Vector<double> m_gain;

    public:
        ContinuousEnergyGain( );
        ContinuousEnergyGain( int a_particleIndex, std::size_t a_size );

        ContinuousEnergyGain &operator=( ContinuousEnergyGain const &a_continuousEnergyGain );

        HOST_DEVICE int particleIndex( ) const { return( m_particleIndex ); }
        HOST_DEVICE int userParticleIndex( ) const { return( m_userParticleIndex ); }
        HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex ) { if( a_particleIndex == m_particleIndex ) m_userParticleIndex = a_userParticleIndex; }
        HOST_DEVICE Vector<double> const &gain( ) const { return( m_gain ); }
        HOST void adjustGain( int a_energy_index, double a_gain ) { m_gain[a_energy_index] += a_gain; }
        HOST_DEVICE double gain( int a_energy_index, double a_energy_fraction ) const ;

        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long sizeOf( ) const ;
};

/*
============================================================
=========== HeatedCrossSectionContinuousEnergy =============
============================================================
*/
class HeatedCrossSectionContinuousEnergy {

    private:
        Vector<int> m_hashIndices;
        Vector<double> m_energies;                              /**< Energy grid for cross sections. */
        Vector<double> m_totalCrossSection;                     /**< The total cross section. */
        Vector<double> m_depositionEnergy;                      /**< The total continuous energy, deposition-energy cross section (related to the kinetic energy of the untracked outgoing particles). */
        Vector<double> m_depositionMomentum;                    /**< The total continuous energy, deposition-momentum cross section. */
        Vector<double> m_productionEnergy;                      /**< The total continuous energy, Q-value cross section. */
        Vector<ContinuousEnergyGain> m_gains;                   /**< The total continuous energy, gain cross section for each tracked particle. */
        Vector<int> m_reactionsInURR_region;                    /**< A list of reactions with in or below the upper URR regions. Empty unless URR probability tables present and used. */
        Vector<HeatedReactionCrossSectionContinuousEnergy *> m_reactionCrossSections;

    public:
        HOST_DEVICE HeatedCrossSectionContinuousEnergy( );
        HOST HeatedCrossSectionContinuousEnergy( SetupInfo &a_setupInfo, Transporting::MC const &a_settings, GIDI::Transporting::Particles const &a_particles,
                DomainHash const &a_domainHash, GIDI::Styles::TemperatureInfo const &a_temperatureInfo, std::vector<GIDI::Reaction const *> const &a_reactions,
                bool a_fixedGrid );
        HOST_DEVICE ~HeatedCrossSectionContinuousEnergy( );

        HOST_DEVICE int evaluationInfo( int a_hashIndex, double a_energy, double *a_energyFraction ) const ;

        HOST_DEVICE double minimumEnergy( ) const { return( m_energies[0] ); }          /**< Returns the minimum cross section domain. */
        HOST_DEVICE double maximumEnergy( ) const { return( m_energies.back( ) ); }     /**< Returns the maximum cross section domain. */
        HOST_DEVICE int numberOfReactions( ) const { return( (int) m_reactionCrossSections.size( ) ); } 
                                                                /**< Returns the number of reaction cross section. */

        HOST_DEVICE int thresholdOffset( int a_reactionIndex ) const { return( m_reactionCrossSections[a_reactionIndex]->offset( ) ); }
                                                                /**< Returns the offset for the cross section for the reaction with index *a_reactionIndex*. */
        HOST_DEVICE double threshold( int a_reactionIndex ) const { return( m_reactionCrossSections[a_reactionIndex]->threshold( ) ); }
                                                                /**< Returns the threshold for the reaction with index *a_reactionIndex*. */
        HOST_DEVICE bool hasURR_probabilityTables( ) const ;
        HOST_DEVICE double URR_domainMin( ) const ;
        HOST_DEVICE double URR_domainMax( ) const ;
        HOST_DEVICE bool reactionHasURR_probabilityTables( int a_index ) const { return( m_reactionCrossSections[a_index]->hasURR_probabilityTables( ) ); }

        HOST_DEVICE double crossSection(                               URR_protareInfos const &a_URR_protareInfos, int a_URR_index, int a_hashIndex, double a_energy, bool a_sampling = false ) const ;
        HOST_DEVICE double reactionCrossSection(  int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_URR_index, int a_hashIndex, double a_energy, bool a_sampling = false ) const ;
        HOST_DEVICE double reactionCrossSection2( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_URR_index, double a_energy, int a_energyIndex, double a_energyFraction, bool a_sampling = false ) const ;
        HOST_DEVICE double reactionCrossSection(  int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_URR_index, double a_energy ) const ;

        HOST_DEVICE double depositionEnergy(   int a_hashIndex, double a_energy ) const ;
        HOST_DEVICE double depositionMomentum( int a_hashIndex, double a_energy ) const ;
        HOST_DEVICE double productionEnergy(   int a_hashIndex, double a_energy ) const ;
        HOST_DEVICE double gain(               int a_hashIndex, double a_energy, int a_particleIndex ) const ;

        HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex );
        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long sizeOf( ) const ;

        HOST_DEVICE Vector<double> const &energies( ) const { return( m_energies ); }       /**< Returns a reference to **m_styles**. */
};

/*
============================================================
============ HeatedCrossSectionsContinuousEnergy ===========
============================================================
*/
class HeatedCrossSectionsContinuousEnergy {

    private:
        Vector<double> m_temperatures;
        Vector<double> m_thresholds;
        Vector<HeatedCrossSectionContinuousEnergy *> m_heatedCrossSections;

    public:
        HOST_DEVICE HeatedCrossSectionsContinuousEnergy( );
        HOST_DEVICE ~HeatedCrossSectionsContinuousEnergy( );

        HOST void update( SetupInfo &a_setupInfo, Transporting::MC const &a_settings, GIDI::Transporting::Particles const &a_particles, DomainHash const &a_domainHash, 
                GIDI::Styles::TemperatureInfos const &a_temperatureInfos, std::vector<GIDI::Reaction const *> const &a_reactions, 
                std::vector<GIDI::Reaction const *> const &a_orphanProducts, bool a_fixedGrid );

        HOST_DEVICE double minimumEnergy( ) const { return( m_heatedCrossSections[0]->minimumEnergy( ) ); }
                                                                    /**< Returns the minimum cross section domain. */
        HOST_DEVICE double maximumEnergy( ) const { return( m_heatedCrossSections[0]->maximumEnergy( ) ); }
                                                                    /**< Returns the maximum cross section domain. */
        HOST_DEVICE Vector<double> const &temperatures( ) const { return( m_temperatures ); }   /**< Returns the value of the **m_temperatures**. */

        HOST_DEVICE double threshold( MCGIDI_VectorSizeType a_index ) const { return( m_thresholds[a_index] ); }     /**< Returns the threshold for the reaction at index *a_index*. */
        HOST_DEVICE bool hasURR_probabilityTables( ) const { return( m_heatedCrossSections[0]->hasURR_probabilityTables( ) ); }
        HOST_DEVICE double URR_domainMin( ) const { return( m_heatedCrossSections[0]->URR_domainMin( ) ); }
        HOST_DEVICE double URR_domainMax( ) const { return( m_heatedCrossSections[0]->URR_domainMax( ) ); }
        HOST_DEVICE bool reactionHasURR_probabilityTables( int a_index ) const { return( m_heatedCrossSections[0]->reactionHasURR_probabilityTables( a_index ) ); }

        HOST_DEVICE double crossSection(                              URR_protareInfos const &a_URR_protareInfos, int a_URR_index, int a_hashIndex, 
                double a_temperature, double a_energy, bool a_sampling = false ) const ;
        HOST_DEVICE double reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_URR_index, int a_hashIndex, 
                double a_temperature, double a_energy, bool a_sampling = false ) const ;
        HOST_DEVICE double reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_URR_index, double a_temperature, double a_energy_in ) const ;
        HOST_DEVICE int sampleReaction(                               URR_protareInfos const &a_URR_protareInfos, int a_URR_index, int a_hashIndex, 
                double a_temperature, double a_energy, double a_crossSection, double (*userrng)( void * ), void *rngState ) const ;

        HOST_DEVICE double depositionEnergy(   int a_hashIndex, double a_temperature, double a_energy ) const ;
        HOST_DEVICE double depositionMomentum( int a_hashIndex, double a_temperature, double a_energy ) const ;
        HOST_DEVICE double productionEnergy(   int a_hashIndex, double a_temperature, double a_energy ) const ;
        HOST_DEVICE double gain(               int a_hashIndex, double a_temperature, double a_energy, int a_particleIndex ) const ;

        HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex );
        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long internalSize( ) const;
};

/*
============================================================
====================== MultiGroupGain ======================
============================================================
*/
class MultiGroupGain {

    private:
        int m_particleIndex;
        int m_userParticleIndex;
        Vector<double> m_gain;

    public:
        HOST_DEVICE MultiGroupGain( );
        HOST MultiGroupGain( int a_particleIndex, GIDI::Vector const &a_gain );

        HOST MultiGroupGain &operator=( MultiGroupGain const &a_multiGroupGain );

        HOST_DEVICE int particleIndex( ) const { return( m_particleIndex ); }
        HOST_DEVICE int userParticleIndex( ) const { return( m_userParticleIndex ); }
        HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex ) { if( a_particleIndex == m_particleIndex ) m_userParticleIndex = a_userParticleIndex; }
        HOST_DEVICE Vector<double> const &gain( ) const { return( m_gain ); }
        HOST_DEVICE double gain( int a_hashIndex ) const { return( m_gain[a_hashIndex] ); }
        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long sizeOf( ) const { return( sizeof( *this ) + m_gain.internalSize( ) ); }        /**< Returns the total member in bytes allocated for *this*. */
};

/*
============================================================
=========== HeatedReactionCrossSectionMultiGroup ===========
============================================================
*/
class HeatedReactionCrossSectionMultiGroup {

    private:
        double m_threshold;
        int m_offset;
        Vector<double> m_crossSection;              // Multi-group reaction cross section
        double m_augmentedThresholdCrossSection;    // Augmented cross section at m_offset for rejecting when projectile energy is below m_threshold.
                                                    // This value is added to m_crossSection[m_offset] when sampling an isotope or reaction.

    public:
        HOST_DEVICE HeatedReactionCrossSectionMultiGroup( );
        HOST HeatedReactionCrossSectionMultiGroup( SetupInfo &a_setupInfo, Transporting::MC const &a_settings, int a_offset, 
                std::vector<double> const &a_crossSection, double a_threshold );

        HOST_DEVICE double operator[]( MCGIDI_VectorSizeType a_index ) const { return( m_crossSection[a_index] ); }  /**< Returns the value of the cross section at multi-group index *a_index*. */
        HOST_DEVICE double threshold( ) const { return( m_threshold ); }        /**< Returns the value of the **m_threshold**. */
        HOST_DEVICE int offset( ) const { return( m_offset ); }                 /**< Returns the value of the **m_offset**. */
        HOST_DEVICE double crossSection( std::size_t a_index, bool a_sampling = false ) const {
            int index = (int)a_index - m_offset;
            if( index < 0 ) return( 0 );
            if( index >= (int)m_crossSection.size( ) ) return( 0 );

            double _crossSection( m_crossSection[index] );
            if( a_sampling && ( index == 0 ) ) {
                _crossSection += m_augmentedThresholdCrossSection;
            }
            return( _crossSection );
        }
        HOST_DEVICE double augmentedThresholdCrossSection( ) const { return( m_augmentedThresholdCrossSection ); }  /**< Returns the value of the **m_augmentedThresholdCrossSection**. */
        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long sizeOf( ) const { return( sizeof( *this ) + m_crossSection.internalSize( ) ); }            /**< Returns the total member in bytes allocated for *this*. */
};

/*
============================================================
============== HeatedCrossSectionMultiGroup  ==============
============================================================
*/
class HeatedCrossSectionMultiGroup {

    private:
        Vector<double> m_totalCrossSection;                 /**< The total multi-group cross section. */
        Vector<double> m_augmentedCrossSection;             /**< The total multi-group cross section used for sampling with rejection (i.e., null-reactions). */
        Vector<double> m_depositionEnergy;                  /**< The total multi-group, deposition-energy cross section (related to the kinetic energy of the untracked outgoing particles). */
        Vector<double> m_depositionMomentum;                /**< The total multi-group, deposition-momentum cross section. */
        Vector<double> m_productionEnergy;                  /**< The total multi-group, Q-value cross section. */
        Vector<MultiGroupGain> m_gains;                     /**< The total multi-group, gain cross section for each tracked particle. */
        Vector<HeatedReactionCrossSectionMultiGroup *> m_reactionCrossSections;

    public:
        HOST_DEVICE HeatedCrossSectionMultiGroup( );
        HOST HeatedCrossSectionMultiGroup( GIDI::ProtareSingle const &a_protare, SetupInfo &a_setupInfo, 
                Transporting::MC const &a_settings, GIDI::Styles::TemperatureInfo const &a_temperatureInfo,
                GIDI::Transporting::Particles const &a_particles, std::vector<GIDI::Reaction const *> const &a_reactions, std::string const &a_label );
        HOST_DEVICE ~HeatedCrossSectionMultiGroup( );

        HOST_DEVICE HeatedReactionCrossSectionMultiGroup *operator[]( MCGIDI_VectorSizeType a_index ) const { return( m_reactionCrossSections[a_index] ); }
                                                                                /**< Returns the HeatedReactionCrossSectionMultiGroup for the reaction at index *a_index *a_index*. */
        HOST_DEVICE int numberOfReactions( ) const { return( (int) m_reactionCrossSections.size( ) ); }
                                                                                /**< Returns the number of reactions stored in *this*. */

        HOST_DEVICE int thresholdOffset(                              int a_index ) const { return( m_reactionCrossSections[a_index]->offset( ) ); }
                                                                                /**< Returns the offset for the cross section for the reaction with index *a_index*. */
        HOST_DEVICE double threshold(                                 int a_index ) const { return( m_reactionCrossSections[a_index]->threshold( ) ); }

        HOST_DEVICE double crossSection(                              int a_hashIndex, bool a_sampling = false ) const ;
        HOST_DEVICE double augmentedCrossSection(                     int a_hashIndex ) const { return( m_augmentedCrossSection[a_hashIndex] ); }
                                                                                /**< Returns the value of the of the augmented cross section the reaction at index *a_index*. */
        HOST_DEVICE double reactionCrossSection( int a_reactionIndex, int a_hashIndex, bool a_sampling = false ) const {
                return( m_reactionCrossSections[a_reactionIndex]->crossSection( a_hashIndex, a_sampling ) ); }
                /**< Returns the reaction's cross section for the reaction at index *a_reactionIndex* and multi-group index *a_hashIndex*. */

        HOST_DEVICE double depositionEnergy(   int a_hashIndex ) const { return( m_depositionEnergy[a_hashIndex] ); }
        HOST_DEVICE double depositionMomentum( int a_hashIndex ) const { return( m_depositionMomentum[a_hashIndex] ); }
        HOST_DEVICE double productionEnergy(   int a_hashIndex ) const { return( m_productionEnergy[a_hashIndex] ); }
        HOST_DEVICE double gain(               int a_hashIndex, int a_particleIndex ) const ;

        HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex );

        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long sizeOf( ) const ;
};

/*
============================================================
============== HeatedCrossSectionsMultiGroup ==============
============================================================
*/
class HeatedCrossSectionsMultiGroup {

    private:
        Vector<double> m_temperatures;
        Vector<double> m_thresholds;
        Vector<int> m_multiGroupThresholdIndex;                         /**< This is the group where threshold starts, -1 otherwise. */
        Vector<double> m_projectileMultiGroupBoundaries;
        Vector<HeatedCrossSectionMultiGroup *> m_heatedCrossSections;

    public:
        HOST_DEVICE HeatedCrossSectionsMultiGroup( );
        HOST_DEVICE ~HeatedCrossSectionsMultiGroup( );

        HOST_DEVICE double minimumEnergy( ) const { return( m_projectileMultiGroupBoundaries[0] ); }
        HOST_DEVICE double maximumEnergy( ) const { return( m_projectileMultiGroupBoundaries.back( ) ); }
        HOST_DEVICE Vector<double> const &temperatures( ) const { return( m_temperatures ); }   /**< Returns the value of the **m_temperatures**. */

        HOST void update( GIDI::ProtareSingle const &a_protare, SetupInfo &a_setupInfo, Transporting::MC const &a_settings, GIDI::Transporting::Particles const &a_particles, 
                GIDI::Styles::TemperatureInfos const &a_temperatureInfos, std::vector<GIDI::Reaction const *> const &a_reactions, 
                std::vector<GIDI::Reaction const *> const &a_orphanProducts );

        HOST_DEVICE int multiGroupThresholdIndex( MCGIDI_VectorSizeType a_index ) const { return( m_multiGroupThresholdIndex[a_index] ); }
                                                                                                    /**< Returns the threshold for the reaction at index *a_index*. */
        HOST_DEVICE Vector<double> const &projectileMultiGroupBoundaries( ) const { return( m_projectileMultiGroupBoundaries ); }
                                                                                                    /**< Returns the value of the **m_projectileMultiGroupBoundaries**. */
        HOST_DEVICE Vector<HeatedCrossSectionMultiGroup *> const &heatedCrossSections( ) const { return( m_heatedCrossSections ); }

        HOST_DEVICE double threshold( MCGIDI_VectorSizeType a_index ) const { return( m_thresholds[a_index] ); }     /**< Returns the threshold for the reaction at index *a_index*. */

        HOST_DEVICE double crossSection(                              int a_hashIndex, double a_temperature, bool a_sampling = false ) const ;
        HOST_DEVICE double reactionCrossSection( int a_reactionIndex, int a_hashIndex, double a_temperature, bool a_sampling = false ) const ;
        HOST_DEVICE double reactionCrossSection( int a_reactionIndex, double a_temperature, double a_energy_in ) const ;
        HOST_DEVICE int sampleReaction(                               int a_hashIndex, double a_temperature, double a_energy_in, double a_crossSection, 
                        double (*userrng)( void * ), void *rngState ) const ;

        HOST_DEVICE double depositionEnergy(   int a_hashIndex, double a_temperature ) const ;
        HOST_DEVICE double depositionMomentum( int a_hashIndex, double a_temperature ) const ;
        HOST_DEVICE double productionEnergy(   int a_hashIndex, double a_temperature ) const ;
        HOST_DEVICE double gain(               int a_hashIndex, double a_temperature, int a_particleIndex ) const ;

        HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex );

        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long internalSize( ) const;
};

/*
============================================================
================== NuclideGammaBranchInfo ==================
============================================================
*/
class NuclideGammaBranchInfo {

    private:
        double m_probability;
        double m_photonEmissionProbability;
        double m_gammaEnergy;
        int m_residualStateIndex;

    public:
        HOST_DEVICE NuclideGammaBranchInfo( );
        HOST NuclideGammaBranchInfo( PoPI::NuclideGammaBranchInfo const &a_nuclideGammaBranchInfo, std::map<std::string, int> &a_stateNamesToIndices );

        HOST_DEVICE double probability( ) const { return( m_probability ); }                                /**< Returns the value of the **m_probability**. */
        HOST_DEVICE double photonEmissionProbability( ) const { return( m_photonEmissionProbability ); }    /**< Returns the value of the **m_photonEmissionProbability**. */
        HOST_DEVICE double gammaEnergy( ) const { return( m_gammaEnergy ); }                                /**< Returns the value of the **m_gammaEnergy**. */
        HOST_DEVICE int residualStateIndex( ) const { return( m_residualStateIndex ); }                     /**< Returns the value of the **m_residualStateIndex**. */

        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long internalSize( ) const { return( 0 ); }
};

/*
============================================================
============== NuclideGammaBranchStateInfo =================
============================================================
*/
class NuclideGammaBranchStateInfo {

    private:
        char m_state[16];
        double m_multiplicity;
        double m_averageGammaEnergy;
        Vector<int> m_branches;

    public:
        HOST_DEVICE NuclideGammaBranchStateInfo( );
        HOST NuclideGammaBranchStateInfo( PoPI::NuclideGammaBranchStateInfo const &a_nuclideGammaBranchingInfo, 
                std::vector<NuclideGammaBranchInfo *> &a_nuclideGammaBranchInfos, std::map<std::string, int> &a_stateNamesToIndices );

        HOST_DEVICE Vector<int> const &branches( ) const { return( m_branches ); }                      /**< Returns the value of the **m_branches**. */

        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long internalSize( ) const { return( m_branches.internalSize( ) ); }
};

/*
============================================================
========================= Product ==========================
============================================================
*/
class Product {

    private:
        String m_ID;
        int m_index;
        int m_userParticleIndex;
        String m_label;
        double m_mass;
        double m_excitationEnergy;
        TwoBodyOrder m_twoBodyOrder;
        int m_neutronIndex;
        Functions::Function1d *m_multiplicity;
        Distributions::Distribution *m_distribution;
// still need *m_averageEnergy *m_averageMomentum;
        double m_angleBiasingEnergy;
        double m_angleBiasingSpeed;
        double m_angleBiasingPDFValue;

        OutputChannel *m_outputChannel;

    public:
        HOST_DEVICE Product( );
        HOST Product( GIDI::Product const *a_product, SetupInfo &a_setupInfo, Transporting::MC const &a_settings, GIDI::Transporting::Particles const &a_particles,
                bool a_isFission );
        HOST Product( PoPI::Database const &a_pop, std::string const &a_ID, std::string const &a_label );
        HOST_DEVICE ~Product( );

        HOST String const &ID( ) const { return( m_ID ); }                                  /**< Returns the value of the **m_ID**. */
        HOST_DEVICE int index( ) const { return( m_index ); }                               /**< Returns the value of the **m_index**. */
        HOST_DEVICE int userParticleIndex( ) const { return( m_userParticleIndex ); }       /**< Returns the value of the **m_userParticleIndex**. */
        HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex );
        HOST_DEVICE String label( ) const { return( m_label ); }                            /**< Returns the value of the **m_label**. */
        HOST_DEVICE double mass( ) const { return( m_mass ); }                              /**< Returns the value of the **m_mass**. */
        HOST_DEVICE double excitationEnergy( ) const { return( m_excitationEnergy ); }      /**< Returns the value of the **m_excitationEnergy**. */
        HOST_DEVICE TwoBodyOrder twoBodyOrder( ) const { return( m_twoBodyOrder ); }      /**< Returns the value of the **m_twoBodyOrder**. */
        HOST_DEVICE int neutronIndex( ) const { return( m_neutronIndex ); }                 /**< Returns the value of the **m_neutronIndex**. */
        HOST_DEVICE double finalQ( double a_x1 ) const ;
        HOST_DEVICE bool hasFission( ) const ;

// FIXME (1) see FIXME (1) in MC class.
        HOST_DEVICE Functions::Function1d const *multiplicity( ) const { return( m_multiplicity ); }      /**< Returns the value of the **m_multiplicity**. */
        HOST void multiplicity( Functions::Function1d *a_multiplicity ) { m_multiplicity = a_multiplicity; }
// FIXME (1) see FIXME (1) in MC class.
        HOST_DEVICE Distributions::Distribution const *distribution( ) const { return( m_distribution ); }      /**< Returns the value of the **m_distribution**. */
        HOST void distribution( Distributions::Distribution *a_distribution ) { m_distribution = a_distribution; }
// FIXME (1) see FIXME (1) in MC class.
        HOST_DEVICE OutputChannel *outputChannel( ) { return( m_outputChannel ); }                  /**< Returns the value of the **m_outputChannel**. */

        HOST_DEVICE void sampleProducts( Protare const *a_protare, double a_projectileEnergy, Sampling::Input &a_input, 
                double (*a_userrng)( void * ), void *a_rngState, Sampling::ProductHandler &a_products ) const ;
        HOST_DEVICE void angleBiasing( Reaction const *a_reaction, int a_pid, double a_energy_in, double a_mu_lab, double &a_probability, double &a_energy_out,
                double (*a_userrng)( void * ), void *a_rngState, double &a_cumulative_weight ) const ;
        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long internalSize( ) const;

        HOST_DEVICE void evaluate_angular_pdf(double E_in_lab, double mu, double &pdf_val);
        HOST_DEVICE void evaluate_angular_pdf(double E_in_lab, double E_out, double mu, double &pdf_val);
        HOST_DEVICE void angleBiasingEnergy( double a_angleBiasingEnergy ) { m_angleBiasingEnergy = a_angleBiasingEnergy; };
        HOST_DEVICE double angleBiasingEnergy() { return( m_angleBiasingEnergy ); };        /**< Returns the value of the **m_angleBiasingEnergy**. */
        HOST_DEVICE void angleBiasingSpeed( double a_angleBiasingSpeed ) { m_angleBiasingSpeed = a_angleBiasingSpeed; };
        HOST_DEVICE double angleBiasingSpeed() { return( m_angleBiasingSpeed ); };          /**< Returns the value of the **m_angleBiasingSpeed**. */
        HOST_DEVICE void angleBiasingPDFValue( double a_angleBiasingPDFValue ) { m_angleBiasingPDFValue = a_angleBiasingPDFValue; };
        HOST_DEVICE double angleBiasingPDFValue() { return( m_angleBiasingPDFValue ); };    /**< Returns the value of the **m_angleBiasingPDFValue**. */
};

/*
============================================================
====================== DelayedNeutron ======================
============================================================
*/
class DelayedNeutron {

    private:
        int m_delayedNeutronIndex;                  /**< If this is a delayed fission neutron, this is its index. */
        double m_rate;                              /**< The GNDS rate for the delayed neutron. */
        Product m_product;                          /**< The GNDS <**product**> node. */

    public:
        HOST_DEVICE DelayedNeutron( );
        HOST DelayedNeutron( int a_index, GIDI::DelayedNeutron const *a_delayedNeutron, SetupInfo &a_setupInfo, Transporting::MC const &a_settings, GIDI::Transporting::Particles const &a_particles );
        HOST_DEVICE ~DelayedNeutron( );

        HOST_DEVICE int delayedNeutronIndex( ) const { return( m_delayedNeutronIndex ); };
        HOST_DEVICE double rate( ) const { return( m_rate ); }
        HOST_DEVICE Product const &product( ) const { return( m_product ); }
        HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex );

        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long internalSize( ) const;
};

/*
============================================================
======================= OutputChannel ======================
============================================================
*/
class OutputChannel {

    private:
        ChannelType m_channelType;
        bool m_isFission;
        int m_neutronIndex;

        Functions::Function1d *m_Q;
        Vector<Product *> m_products;

        Functions::Function1d *m_totalDelayedNeutronMultiplicity;
        Vector<DelayedNeutron *> m_delayedNeutrons;

    public:
        HOST_DEVICE OutputChannel( );
        HOST OutputChannel( GIDI::OutputChannel const *a_outputChannel, SetupInfo &a_setupInfo, Transporting::MC const &a_settings, GIDI::Transporting::Particles const &a_particles );
        HOST_DEVICE ~OutputChannel( );

        HOST_DEVICE Product *operator[]( MCGIDI_VectorSizeType a_index ) { return( m_products[a_index] ); }  /**< Returns a pointer to the product at index *a_index*. */

        HOST_DEVICE int neutronIndex( ) const { return( m_neutronIndex ); }                 /**< Returns the value of the **m_neutronIndex**. */
        HOST_DEVICE bool isTwoBody( ) const { return( m_channelType == ChannelType::twoBody ); }         /**< Returns true if output channel is two-body and false otherwise. */
        HOST_DEVICE double finalQ( double a_x1 ) const ;
        HOST_DEVICE bool isFission( ) const { return( m_isFission ); }                      /**< Returns the value of the **m_isFission**. */
        HOST_DEVICE bool hasFission( ) const ;
// FIXME (1) see FIXME (1) in MC class.
        HOST_DEVICE Functions::Function1d *Q( ) { return( m_Q ); }                          /**< Returns the value of the **m_Q**. */

        HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex );
// FIXME (1) see FIXME (1) in MC class.
        HOST_DEVICE Vector<Product *> const &products( ) const { return( m_products ); }    /**< Returns the value of the **m_products**. */

        Vector<DelayedNeutron *> delayedNeutrons( ) const { return( m_delayedNeutrons ); }
        HOST_DEVICE DelayedNeutron const *delayedNeutron( int a_index ) const { return( m_delayedNeutrons[a_index] ); }

        HOST_DEVICE void sampleProducts( Protare const *a_protare, double a_projectileEnergy, Sampling::Input &a_input, 
                double (*a_userrng)( void * ), void *a_rngState, Sampling::ProductHandler &a_products ) const ;
        HOST_DEVICE void angleBiasing( Reaction const *a_reaction, int a_pid, double a_energy_in, double a_mu_lab, double &a_probability, double &a_energy_out, 
                double (*a_userrng)( void * ), void *a_rngState, double &a_cumulative_weight ) const ;
        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long internalSize( ) const;
};

/*
============================================================
========================= Reaction =========================
============================================================
*/
class Reaction {

    private:
        ProtareSingle *m_protareSingle;                     /**< The ProtareSingle this reaction resides in. */
        int m_reactionIndex;                                /**< The index of the reaction in the ProtareSingle. */
        String m_label;                                     /**< The **GNDS** label for the reaction. */
        int m_ENDF_MT;                                      /**< The ENDF MT value for the reaction. */
        int m_ENDL_C;                                       /**< The ENDL C value for the reaction. */
        int m_ENDL_S;                                       /**< The ENDL S value for the reaction. */
        int m_neutronIndex;                                 /**< The neutron's PoPs index.*/
        bool m_hasFission;                                  /**< *true* if the reaction is a fission reaction and false otherwise. */
        double m_projectileMass;                            /**< The mass of the projectile. */
        double m_targetMass;                                /**< The mass of the target. */
        double m_crossSectionThreshold;                     /**< The threshold for the reaction. */
        bool m_upscatterModelASupported;
        Vector<double> m_upscatterModelACrossSection;       /**< The multi-group cross section to use for upscatter model A. */
        Vector<int> m_productIndices;                       /**< The list of all products *this* reaction can product. */
        Vector<int> m_userProductIndices;                   /**< The list of all products *this* reaction can product as user indices. */
        Vector<int> m_productMultiplicities;                /**< The list of all multiplicities for each product in *m_productIndices* . */
        Vector<int> m_productIndicesTransportable;          /**< The list of all transportabls products *this* reaction can product. */
        Vector<int> m_userProductIndicesTransportable;      /**< The list of all transportabls products *this* reaction can product as user indices. */

        OutputChannel m_outputChannel;                      /**< The output channel for this reaction. */
        int m_associatedOrphanProductIndex;                 /**< The index in the Protare's m_orphanProducts member for the orphanProduct associated with this reaction. */
        Reaction *m_associatedOrphanProduct;                /**< A pointer to the orphanProduct associated with this reaction. */
// Still need m_availableEnergy and m_availableMomentum.

    public:
        HOST_DEVICE Reaction( );
        HOST Reaction( GIDI::Reaction const &a_reaction, SetupInfo &a_setupInfo, Transporting::MC const &a_settings, GIDI::Transporting::Particles const &a_particles,
                GIDI::Styles::TemperatureInfos const &a_temperatureInfos );
        HOST_DEVICE ~Reaction( );

        inline HOST_DEVICE void updateProtareSingleInfo( ProtareSingle *a_protareSingle, int a_reactionIndex ) {
                m_protareSingle = a_protareSingle;
                m_reactionIndex = a_reactionIndex;
        }
        HOST_DEVICE ProtareSingle const *protareSingle( ) const { return( m_protareSingle ); }   /**< Returns the value of the **m_protareSingle**. */
        HOST_DEVICE int reactionIndex( ) const { return( m_reactionIndex ); }       /**< Returns the value of the **m_reactionIndex**. */
        HOST_DEVICE String const &label( ) const { return( m_label ); }             /**< Returns the value of the **m_label**. */
        HOST_DEVICE int ENDF_MT( ) const { return( m_ENDF_MT ); }                   /**< Returns the value of the **m_ENDF_MT**. */
        HOST_DEVICE int ENDL_C( ) const { return( m_ENDL_C ); }                     /**< Returns the value of the **m_ENDL_C**. */
        HOST_DEVICE int ENDL_S( ) const { return( m_ENDL_S ); }                     /**< Returns the value of the **m_ENDL_S**. */
        HOST_DEVICE int neutronIndex( ) const { return( m_neutronIndex ); }         /**< Returns the value of the **m_neutronIndex**. */
        HOST_DEVICE double finalQ( double a_x1 ) const { return( m_outputChannel.finalQ( a_x1 ) ); }    /**< Returns the Q-value for projectile energy *a_x1*. */
        HOST_DEVICE bool hasFission( ) const { return( m_hasFission ); }            /**< Returns the value of the **m_hasFission**. */
        HOST_DEVICE double projectileMass( ) const { return( m_projectileMass ); }  /**< Returns the value of the **m_projectileMass**. */
        HOST_DEVICE double targetMass( ) const { return( m_targetMass ); }          /**< Returns the value of the **m_targetMass**. */
        HOST_DEVICE double crossSectionThreshold( ) const { return( m_crossSectionThreshold ); }    /**< Returns the value of the **m_crossSectionThreshold**. */
        HOST_DEVICE double crossSection( URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy ) const ;
        HOST_DEVICE double crossSection( URR_protareInfos const &a_URR_protareInfos, double a_temperature, double a_energy ) const ;

        HOST Vector<int> const &productIndices( ) const { return( m_productIndices ); }
        HOST Vector<int> const &userProductIndices( ) const { return( m_userProductIndices ); }
        HOST int productMultiplicities( int a_index ) const ;
        HOST Vector<int> const &productIndicesTransportable( ) const { return( m_productIndicesTransportable ); }
        HOST Vector<int> const &userProductIndicesTransportable( ) const { return( m_userProductIndicesTransportable ); }

        HOST_DEVICE OutputChannel const &outputChannel( ) const { return( m_outputChannel ); }              /**< Returns the value of the **m_outputChannel**. */
        HOST_DEVICE int associatedOrphanProductIndex( ) const { return( m_associatedOrphanProductIndex ); } /**< Returns the value of the **m_associatedOrphanProductIndex**. */
        HOST_DEVICE void associatedOrphanProductIndex( int a_associatedOrphanProductIndex ) { m_associatedOrphanProductIndex = a_associatedOrphanProductIndex; }
        HOST_DEVICE Reaction *associatedOrphanProduct( ) const { return( m_associatedOrphanProduct ); }     /**< Returns the value of the **m_associatedOrphanProduct**. */
        HOST_DEVICE void associatedOrphanProduct( Reaction *a_associatedOrphanProduct ) { m_associatedOrphanProduct = a_associatedOrphanProduct; }
        HOST_DEVICE bool upscatterModelASupported( ) const { return( m_upscatterModelASupported ); }
        HOST_DEVICE Vector<double> const &upscatterModelACrossSection( ) const { return( m_upscatterModelACrossSection ); } 
                                                                                                            /**< Returns the value of the **m_upscatterModelACrossSection**. */

        HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex );

        HOST_DEVICE void sampleProducts( Protare const *a_protare, double a_projectileEnergy, Sampling::Input &a_input, 
                double (*a_userrng)( void * ), void *a_rngState, Sampling::ProductHandler &a_products ) const ;
        HOST_DEVICE static void sampleNullProducts( Protare const &a_protare, double a_projectileEnergy, Sampling::Input &a_input, 
                double (*a_userrng)( void * ), void *a_rngState, Sampling::ProductHandler &a_products );
        HOST_DEVICE double angleBiasing( int a_pid, double a_energy_in, double a_mu_lab, double &a_energy_out, 
                double (*a_userrng)( void * ), void *a_rngState, double *a_cumulative_weight = nullptr ) const ;
        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long internalSize( ) const;
};

/*
============================================================
========================== Protare =========================
============================================================
*/
class Protare {

    private:
        ProtareType m_protareType;                          /**< The type of protare *this* is. */

        String m_projectileID;                              /**< The PoPs id of the projectile. */
        int m_projectileIndex;                              /**< The PoPs index of the projectile. */
        int m_projectileUserIndex;                          /**< The projectile's index as specified by the user. */
        double m_projectileMass;                            /**< The mass of the projectile. */
        double m_projectileExcitationEnergy;                /**< The nuclear excitation of the projectile. */

        String m_targetID;                                  /**< The PoPs id of the target. */
        int m_targetIndex;                                  /**< The PoPs index of the target. */
        int m_targetUserIndex;                              /**< The target's index as specified by the user. */
        double m_targetMass;                                /**< The mass of the target. */
        double m_targetExcitationEnergy;                    /**< The nuclear excitation of the target. */

        int m_neutronIndex;                                 /**< The PoPs index of the neutron. */
        int m_userNeutronIndex;                             /**< The neutron particle index defined by the user. */
        int m_photonIndex;                                  /**< The PoPs index of the photon. */
        int m_userPhotonIndex;                              /**< The photon particle index defined by the user. */
        String m_evaluation;                                /**< The evaluation string for the Protare. */
        GIDI::Frame m_projectileFrame;                      /**< The frame the projectile data are given in. */

        Vector<int> m_productIndices;                       /**< The list of all products *this* protare can product. */
        Vector<int> m_userProductIndices;                   /**< The list of all products *this* reaction can product as user indices. */
        Vector<int> m_productIndicesTransportable;          /**< The list of all transportabls products *this* protare can product. */
        Vector<int> m_userProductIndicesTransportable;      /**< The list of all transportabls products *this* reaction can product as user indices. */

        bool m_isTNSL_ProtareSingle;                        /**< If *this* is a ProtareSingle instance with TNSL data *true* and otherwise *false*. */

    public:
        HOST_DEVICE Protare( ProtareType a_protareType );
        HOST Protare( ProtareType a_protareType, GIDI::Protare const &a_protare, PoPI::Database const &a_pops, Transporting::MC const &a_settings );
        virtual HOST_DEVICE ~Protare( );

        HOST_DEVICE String const &projectileID( ) const { return( m_projectileID ); }                       /**< Returns the value of the **m_projectileID** member. */
        HOST_DEVICE int projectileIndex( ) const { return( m_projectileIndex ); }                           /**< Returns the value of the **m_projectileIndex** member. */
        HOST_DEVICE int projectileUserIndex( ) const { return( m_projectileUserIndex ); }                   /**< Returns the value of the **m_projectileUserIndex** member. */
        HOST_DEVICE double projectileMass( ) const { return( m_projectileMass ); }                          /**< Returns the value of the **m_projectileMass** member. */
        HOST_DEVICE double projectileExcitationEnergy( ) const { return( m_projectileExcitationEnergy ); }  /**< Returns the value of the **m_projectileExcitationEnergy** member. */

        HOST_DEVICE String const &targetID( ) const { return( m_targetID ); }                               /**< Returns the value of the **m_targetID** member. */
        HOST_DEVICE int targetIndex( ) const { return( m_targetIndex ); }                                   /**< Returns the value of the **m_targetIndex** member. */
        HOST_DEVICE int targetUserIndex( ) const { return( m_targetUserIndex ); }                           /**< Returns the value of the **m_targetUserIndex** member. */
        HOST_DEVICE double targetMass( ) const { return( m_targetMass ); }                                  /**< Returns the value of the **m_targetMass** member. */
        HOST_DEVICE double targetExcitationEnergy( ) const { return( m_targetExcitationEnergy ); }          /**< Returns the value of the **m_targetExcitationEnergy** member. */

        HOST_DEVICE int neutronIndex( ) const { return( m_neutronIndex ); }                                 /**< Returns the value of the **m_neutronIndex** member. */
        HOST_DEVICE int userNeutronIndex( ) const { return( m_userNeutronIndex ); }                         /**< Returns the value of the **m_userNeutronIndex** member. */
        HOST_DEVICE int photonIndex( ) const { return( m_photonIndex ); }                                   /**< Returns the value of the **m_photonIndex** member. */
        HOST_DEVICE int userPhotonIndex( ) const { return( m_userPhotonIndex ); }                           /**< Returns the value of the **m_userPhotonIndex** member. */
        HOST_DEVICE String evaluation( ) const { return( m_evaluation ); }                                  /**< Returns the value of the **m_evaluation** member. */
        HOST GIDI::Frame projectileFrame( ) const { return( m_projectileFrame ); }                          /**< Returns the value of the **m_projectileFrame** member. */

        HOST Vector<int> const &productIndices( bool a_transportablesOnly ) const ;
        HOST void productIndices( std::set<int> const &a_indices, std::set<int> const &a_transportableIndices );
        HOST Vector<int> const &userProductIndices( bool a_transportablesOnly ) const ;
        virtual HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex );

        virtual ProtareType protareType( ) const { return( m_protareType ); }                               /**< Returns the value of the **m_protareType** member. */    
        HOST_DEVICE bool isTNSL_ProtareSingle( ) const { return( m_isTNSL_ProtareSingle ); }                /**< Returns the value of the **m_isTNSL_ProtareSingle** member. */
        virtual HOST_DEVICE MCGIDI_VectorSizeType numberOfProtares( ) const = 0;                            /**< Returns the number of protares contained in *this*. */
        virtual HOST_DEVICE ProtareSingle const *protare( MCGIDI_VectorSizeType a_index ) const = 0;        /**< Returns the **a_index** - 1 Protare contained in *this*. */
        virtual HOST_DEVICE ProtareSingle const *protareWithReaction( int a_index ) const = 0;              /**< Returns the *ProtareSingle* that contains the (*a_index* - 1) reaction. */

        virtual HOST_DEVICE double minimumEnergy( ) const = 0;                                              /**< Returns the minimum cross section domain. */
        virtual HOST_DEVICE double maximumEnergy( ) const = 0 ;                                             /**< Returns the maximum cross section domain. */
        virtual HOST_DEVICE Vector<double> temperatures( MCGIDI_VectorSizeType a_index = 0 ) const = 0 ;    /**< Returns the list of temperatures for the requested ProtareSingle. */

        virtual HOST Vector<double> const &projectileMultiGroupBoundaries( ) const = 0;
        virtual HOST Vector<double> const &projectileFixedGrid( ) const = 0;

        virtual HOST_DEVICE std::size_t numberOfReactions( ) const = 0;
        virtual HOST_DEVICE Reaction const *reaction( int a_index ) const = 0;
        virtual HOST_DEVICE std::size_t numberOfOrphanProducts( ) const = 0;
        virtual HOST_DEVICE Reaction const *orphanProduct( int a_index ) const = 0;

        virtual HOST_DEVICE bool hasFission( ) const = 0;

        virtual HOST_DEVICE int URR_index( ) const = 0;
        virtual HOST_DEVICE bool hasURR_probabilityTables( ) const = 0;
        virtual HOST_DEVICE double URR_domainMin( ) const = 0;
        virtual HOST_DEVICE double URR_domainMax( ) const = 0;
        virtual HOST_DEVICE bool reactionHasURR_probabilityTables( int a_index ) const = 0 ;

        virtual HOST_DEVICE double threshold( MCGIDI_VectorSizeType a_index ) const = 0;

        virtual HOST_DEVICE double crossSection(                              URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, bool a_sampling = false ) const = 0;
        virtual HOST_DEVICE double reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, bool a_sampling = false ) const = 0;
        virtual HOST_DEVICE double reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos,                  double a_temperature, double a_energy ) const = 0;
        virtual HOST_DEVICE int sampleReaction(                               URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, double a_crossSection, double (*a_userrng)( void * ), void *a_rngState ) const = 0;

        virtual HOST_DEVICE double depositionEnergy(   int a_hashIndex, double a_temperature, double a_energy ) const = 0;
        virtual HOST_DEVICE double depositionMomentum( int a_hashIndex, double a_temperature, double a_energy ) const = 0;
        virtual HOST_DEVICE double productionEnergy(   int a_hashIndex, double a_temperature, double a_energy ) const = 0;
        virtual HOST_DEVICE double gain(               int a_hashIndex, double a_temperature, double a_energy, int a_particleIndex ) const = 0;

        virtual HOST_DEVICE Vector<double> const &upscatterModelAGroupVelocities( ) const = 0;

        virtual HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        virtual HOST_DEVICE long internalSize( ) const;
};

/*
============================================================
===================== ProtareSingle =====================
============================================================
*/
class ProtareSingle : public Protare {

    private:
        int m_URR_index;                                                            /**< The index of the protare in the URR_protareInfos list. If negative, not in list. */
        bool m_hasURR_probabilityTables;                                            /**< *true* if URR probability tables present and *false* otherwise. */
        double m_URR_domainMin;                                                     /**< If URR probability tables present this is the minimum of the projectile energy domain for the tables. */
        double m_URR_domainMax;                                                     /**< If URR probability tables present this is the maximum of the projectile energy domain for the tables. */
        Vector<double> m_projectileMultiGroupBoundaries;                            /**< The multi-group boundaries for the projectile. Only used if m_crossSectionLookupMode and/or m_other1dDataLookupMode is multiGroup. */
        Vector<double> m_projectileFixedGrid;                                       /**< The fixed-grid points for the projectile. Only used if m_crossSectionLookupMode is fixedGrid. */
        Vector<double> m_upscatterModelAGroupVelocities;                            /**< The speed of the projectile at each multi-group boundary. Need by upscatter model A. */

        Vector<Reaction *> m_reactions;                                             /**< The list of reactions. */
        Vector<Reaction *> m_orphanProducts;                                        /**< The list of orphan products. */
        bool m_continuousEnergy;                                                    /**< If *true*, protare has continuous energy cross sections; otherwise, multi-group cross sections. */
        bool m_fixedGrid;                                                           /**< If *true*, continuous energy cross sections are fixed grid. */
        HeatedCrossSectionsContinuousEnergy m_heatedCrossSections;                  /**< Stores all cross section data for total and all reactions for all requested temperatures. */
        HeatedCrossSectionsMultiGroup m_heatedMultigroupCrossSections;              /**< Stores all multi-group cross section data for total and all reactions for all requested temperatures. */

        Vector<NuclideGammaBranchStateInfo *> m_nuclideGammaBranchStateInfos;       /**< List of all gamma branches for a nuclide. */
        Vector<NuclideGammaBranchInfo *> m_branches;                                /**< Condensed data on a nuclide's gamma branch including the gamma's energy, probability and the nuclide's residual state. */

        HOST void setupNuclideGammaBranchStateInfos( SetupInfo &a_setupInfo, GIDI::ProtareSingle const &a_protare );

    public:
        HOST_DEVICE ProtareSingle( );
        HOST ProtareSingle( GIDI::ProtareSingle const &a_protare, PoPI::Database const &a_pops, Transporting::MC &a_settings, 
                GIDI::Transporting::Particles const &a_particles, DomainHash const &a_domainHash, GIDI::Styles::TemperatureInfos const &a_temperatureInfos,
                std::set<int> const &a_reactionsToExclude, int a_reactionsToExcludeOffset = 0, bool a_allowFixedGrid = true );
        HOST_DEVICE ~ProtareSingle( );

        HOST_DEVICE bool continuousEnergy( ) const { return( m_continuousEnergy ); }
        HOST_DEVICE bool fixedGrid( ) const { return( m_fixedGrid ); }
        HOST_DEVICE HeatedCrossSectionsContinuousEnergy const &heatedCrossSections( ) const { return( m_heatedCrossSections ); }  /**< Returns a reference to the **m_heatedCrossSections** member. */
        HOST_DEVICE HeatedCrossSectionsMultiGroup const &heatedMultigroupCrossSections( ) const { return( m_heatedMultigroupCrossSections ); } /**< Returns a reference to the **m_heatedMultigroupCrossSections** member. */

// FIXME (1) see FIXME (1) in MC class.
        HOST_DEVICE Vector<Reaction *> const &reactions( ) const { return( m_reactions ); }                 /**< Returns the value of the **m_reactions** member. */
// FIXME (1) see FIXME (1) in MC class.
        HOST_DEVICE Vector<Reaction *> const &orphanProducts( ) const { return( m_orphanProducts ); }       /**< Returns the value of the **m_orphanProducts** member. */

        HOST_DEVICE void sampleBranchingGammas( Sampling::Input &a_input, double a_projectileEnergy, int initialStateIndex, 
                double (*a_userrng)( void * ), void *a_rngState, Sampling::ProductHandler &a_products ) const ;

// The rest are virtual methods defined in the Protare class.
        HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex );

        HOST_DEVICE MCGIDI_VectorSizeType numberOfProtares( ) const { return( 1 ); }                        /**< Returns the number of protares contained in *this*. */
        HOST_DEVICE ProtareSingle const *protare( MCGIDI_VectorSizeType a_index ) const ;
        HOST_DEVICE ProtareSingle const *protareWithReaction( int a_index ) const ;

        HOST_DEVICE double minimumEnergy( ) const { 
            if( m_continuousEnergy ) return( m_heatedCrossSections.minimumEnergy( ) );
            return( m_heatedMultigroupCrossSections.minimumEnergy( ) ); }                                   /**< Returns the minimum cross section domain. */
        HOST_DEVICE double maximumEnergy( ) const { 
            if( m_continuousEnergy ) return( m_heatedCrossSections.maximumEnergy( ) );
            return( m_heatedMultigroupCrossSections.maximumEnergy( ) ); }                                   /**< Returns the maximum cross section domain. */
        HOST_DEVICE Vector<double> temperatures( MCGIDI_VectorSizeType a_index = 0 ) const ;

        HOST Vector<double> const &projectileMultiGroupBoundaries( ) const { return( m_projectileMultiGroupBoundaries ); }    /**< Returns the value of the **m_projectileMultiGroupBoundaries** member. */
        HOST Vector<double> const &projectileFixedGrid( ) const { return( m_projectileFixedGrid ); }                /**< Returns the value of the **m_projectileFixedGrid** member. */

        HOST_DEVICE std::size_t numberOfReactions( ) const { return( m_reactions.size( ) ); }                       /**< Returns the number of reactions of *this*. */
        HOST_DEVICE Reaction const *reaction( int a_index ) const { return( m_reactions[a_index] ); }               /**< Returns the (a_index-1)^th reaction of *this*. */
        HOST_DEVICE std::size_t numberOfOrphanProducts( ) const { return( m_orphanProducts.size( ) ); }             /**< Returns the number of orphan products of *this*. */
        HOST_DEVICE Reaction const *orphanProduct( int a_index ) const { return( m_orphanProducts[a_index] ); }     /**< Returns the (a_index-1)^th orphan product of *this*. */

        HOST_DEVICE bool hasFission( ) const ;

        HOST_DEVICE int URR_index( ) const { return( m_URR_index ); }
        HOST_DEVICE void URR_index( int a_URR_index ) { m_URR_index = a_URR_index; }
        HOST_DEVICE bool inURR( double a_energy ) const ;
        HOST_DEVICE bool hasURR_probabilityTables( ) const { return( m_hasURR_probabilityTables ); }
        HOST_DEVICE double URR_domainMin( ) const { return( m_URR_domainMin ); }
        HOST_DEVICE double URR_domainMax( ) const { return( m_URR_domainMax ); }
        HOST_DEVICE bool reactionHasURR_probabilityTables( int a_index ) const { return( m_heatedCrossSections.reactionHasURR_probabilityTables( a_index ) ); }

        HOST_DEVICE double threshold( MCGIDI_VectorSizeType a_index ) const {
            if( m_continuousEnergy ) return( m_heatedCrossSections.threshold( a_index ) );
            return( m_heatedMultigroupCrossSections.threshold( a_index ) ); }                                       /**< Returns the threshold for the reaction at index *a_index*. */

        HOST_DEVICE double crossSection(                              URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, bool a_sampling = false ) const ;
        HOST_DEVICE double reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, bool a_sampling = false ) const ;
        HOST_DEVICE double reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos,                  double a_temperature, double a_energy ) const ;
        HOST_DEVICE int sampleReaction(                               URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, double a_crossSection, double (*a_userrng)( void * ), void *a_rngState ) const ;

        HOST_DEVICE double depositionEnergy(   int a_hashIndex, double a_temperature, double a_energy ) const ;
        HOST_DEVICE double depositionMomentum( int a_hashIndex, double a_temperature, double a_energy ) const ;
        HOST_DEVICE double productionEnergy(   int a_hashIndex, double a_temperature, double a_energy ) const ;
        HOST_DEVICE double gain(               int a_hashIndex, double a_temperature, double a_energy, int a_particleIndex ) const ;

        HOST_DEVICE Vector<double> const &upscatterModelAGroupVelocities( ) const { return( m_upscatterModelAGroupVelocities ); }   /**< Returns a reference to the **m_upscatterModelAGroupVelocities** member. */

        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long internalSize( ) const;
};

/*
============================================================
===================== ProtareComposite =====================
============================================================
*/
class ProtareComposite : public Protare {

    private:
        Vector<ProtareSingle *> m_protares;                              /**< List of protares added to *this* instance. */
        std::size_t m_numberOfReactions;                                    /**< The sum of the number of reaction for all stored protares. */
        std::size_t m_numberOfOrphanProducts;                               /**< The sum of the number of reaction for all stored protares. */
        double m_minimumEnergy;                                             /**< The maximum of the minimum cross section domains. */
        double m_maximumEnergy;                                             /**< The minimum of the maximum cross section domains. */

    public:
        HOST_DEVICE ProtareComposite( );
        HOST ProtareComposite( GIDI::ProtareComposite const &a_protare, PoPI::Database const &a_pops, Transporting::MC &a_settings, 
                GIDI::Transporting::Particles const &a_particles, DomainHash const &a_domainHash, GIDI::Styles::TemperatureInfos const &a_temperatureInfos,
                std::set<int> const &a_reactionsToExclude, int a_reactionsToExcludeOffset = 0, bool a_allowFixedGrid = true );
        HOST_DEVICE ~ProtareComposite( );

        Vector<ProtareSingle *> protares( ) const { return( m_protares ); }       /**< Returns the value of the **m_protares** member. */

// The rest are virtual methods defined in the Protare class.
        HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex );

        HOST_DEVICE MCGIDI_VectorSizeType numberOfProtares( ) const { return( m_protares.size( ) ); }     /**< Returns the number of protares contained in *this*. */
        HOST_DEVICE ProtareSingle const *protare( MCGIDI_VectorSizeType a_index ) const ;
        HOST_DEVICE ProtareSingle const *protareWithReaction( int a_index ) const ;

        HOST_DEVICE double minimumEnergy( ) const { return( m_minimumEnergy ); }     /**< Returns the value of the **m_minimumEnergy** member. */
        HOST_DEVICE double maximumEnergy( ) const { return( m_maximumEnergy ); }     /**< Returns the value of the **m_maximumEnergy** member. */
        HOST_DEVICE Vector<double> temperatures( MCGIDI_VectorSizeType a_index = 0 ) const ;

        HOST Vector<double> const &projectileMultiGroupBoundaries( ) const { return( m_protares[0]->projectileMultiGroupBoundaries( ) ); }    
                                                                            /**< Returns the value of the **m_projectileMultiGroupBoundaries** member. */
        HOST Vector<double> const &projectileFixedGrid( ) const { return( m_protares[0]->projectileFixedGrid( ) ); }
                                                                            /**< Returns the value of the **m_projectileFixedGrid** member. */

        HOST_DEVICE std::size_t numberOfReactions( ) const { return( m_numberOfReactions ); }
                                                                            /**< Returns the value of the **m_numberOfReactions** member. */
        HOST_DEVICE Reaction const *reaction( int a_index ) const ;
        HOST_DEVICE std::size_t numberOfOrphanProducts( ) const { return( m_numberOfOrphanProducts ); }
                                                                            /**< Returns the value of the **m_numberOfOrphanProducts** member. */
        HOST_DEVICE Reaction const *orphanProduct( int a_index ) const ;

        HOST_DEVICE bool hasFission( ) const ;

        HOST_DEVICE int URR_index( ) const { return( -1 ); }
        HOST_DEVICE bool hasURR_probabilityTables( ) const ;
        HOST_DEVICE double URR_domainMin( ) const ;
        HOST_DEVICE double URR_domainMax( ) const ;
        HOST_DEVICE bool reactionHasURR_probabilityTables( int a_index ) const ;

        HOST_DEVICE double threshold( MCGIDI_VectorSizeType a_index ) const ;

        HOST_DEVICE double crossSection(                              URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, bool a_sampling = false ) const ;
        HOST_DEVICE double reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, bool a_sampling = false ) const ;
        HOST_DEVICE double reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos,                  double a_temperature, double a_energy ) const ;
        HOST_DEVICE int sampleReaction(                               URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, double a_crossSection, double (*a_userrng)( void * ), void *a_rngState ) const ;

        HOST_DEVICE double depositionEnergy(   int a_hashIndex, double a_temperature, double a_energy ) const ;
        HOST_DEVICE double depositionMomentum( int a_hashIndex, double a_temperature, double a_energy ) const ;
        HOST_DEVICE double productionEnergy(   int a_hashIndex, double a_temperature, double a_energy ) const ;
        HOST_DEVICE double gain(               int a_hashIndex, double a_temperature, double a_energy, int a_particleIndex ) const ;

        HOST_DEVICE Vector<double> const &upscatterModelAGroupVelocities( ) const { return( m_protares[0]->upscatterModelAGroupVelocities( ) ); }
                                                                            /**< Returns a reference to the **m_upscatterModelAGroupVelocities** member. */

        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long internalSize( ) const ;
};

/*
============================================================
======================== ProtareTNSL =======================
============================================================
*/
class ProtareTNSL : public Protare {

    private:
        std::size_t m_numberOfTNSLReactions;                                /**< The number of reactions of the TNSL protare. */
        double m_TNSL_maximumEnergy;                                        /**< The maximum energy of the cross section domain for the TNSL protare. */
        double m_TNSL_maximumTemperature;                                   /**< The highest temperature for processed data for the TNSL protare. */
        ProtareSingle *m_protareWithElastic;                                /**< Protare with non thermal neutron scattering law data. */
        ProtareSingle *m_TNSL;                                              /**< Protare with thermal neutron scattering law data. */
        ProtareSingle *m_protareWithoutElastic;                             /**< Same as *m_protare* but without elastic. */

    public:
        HOST_DEVICE ProtareTNSL( );
        HOST ProtareTNSL( GIDI::ProtareTNSL const &a_protare, PoPI::Database const &a_pops, Transporting::MC &a_settings, 
                GIDI::Transporting::Particles const &a_particles, DomainHash const &a_domainHash, GIDI::Styles::TemperatureInfos const &a_temperatureInfos,
                std::set<int> const &a_reactionsToExclude, int a_reactionsToExcludeOffset = 0, bool a_allowFixedGrid = true );
        HOST_DEVICE ~ProtareTNSL( );

        HOST_DEVICE ProtareSingle const *protareWithElastic( ) const { return( m_protareWithElastic ); }        /**< Returns the **m_protareWithElastic** member. */
        HOST_DEVICE ProtareSingle const *TNSL( ) const { return( m_TNSL ); }                                    /**< Returns the **m_TNSL** member. */
        HOST_DEVICE ProtareSingle const *protareWithoutElastic( ) const { return( m_protareWithoutElastic ); }  /**< Returns the **m_protareWithoutElastic** member. */

        HOST_DEVICE double TNSL_maximumEnergy( ) const { return( m_TNSL_maximumEnergy ); }
        HOST_DEVICE double TNSL_maximumTemperature( ) const { return( m_TNSL_maximumTemperature ); }

// The rest are virtual methods defined in the Protare class.
        HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex );

        HOST_DEVICE MCGIDI_VectorSizeType numberOfProtares( ) const { return( 2 ); }  /**< Always Returns 2. */
        HOST_DEVICE ProtareSingle const *protare( MCGIDI_VectorSizeType a_index ) const ;
        HOST_DEVICE ProtareSingle const *protareWithReaction( int a_index ) const ;

        HOST_DEVICE double minimumEnergy( ) const { return( m_protareWithElastic->minimumEnergy( ) ); }   /**< Returns the minimum cross section domain. */
        HOST_DEVICE double maximumEnergy( ) const { return( m_protareWithElastic->maximumEnergy( ) ); }   /**< Returns the maximum cross section domain. */
        HOST_DEVICE Vector<double> temperatures( MCGIDI_VectorSizeType a_index = 0 ) const ;

        HOST Vector<double> const &projectileMultiGroupBoundaries( ) const { return( m_protareWithElastic->projectileMultiGroupBoundaries( ) ); }    
                                                                            /**< Returns the value of the **m_projectileMultiGroupBoundaries** member. */
        HOST Vector<double> const &projectileFixedGrid( ) const { return( m_protareWithElastic->projectileFixedGrid( ) ); }
                                                                            /**< Returns the value of the **m_projectileFixedGrid** member. */

        HOST_DEVICE std::size_t numberOfReactions( ) const { return( m_TNSL->numberOfReactions( ) + m_protareWithElastic->numberOfReactions( ) ); }
        HOST_DEVICE Reaction const *reaction( int a_index ) const ;
        HOST_DEVICE std::size_t numberOfOrphanProducts( ) const { return( m_protareWithElastic->numberOfOrphanProducts( ) ); }
                                                                            /**< Returns the number of orphan products in the normal ProtareSingle. */
        HOST_DEVICE Reaction const *orphanProduct( int a_index ) const { return( m_protareWithElastic->orphanProduct( a_index ) ); }
                                                                            /**< Returns the (a_index - 1 )^th orphan product in the normal ProtareSingle. */

        HOST_DEVICE bool hasFission( ) const { return( m_protareWithElastic->hasFission( ) ); }    /* Returns the normal ProtareSingle's hasFission value. */

        HOST_DEVICE int URR_index( ) const { return( -1 ); }
        HOST_DEVICE bool hasURR_probabilityTables( ) const { return( m_protareWithElastic->hasURR_probabilityTables( ) ); }
        HOST_DEVICE double URR_domainMin( ) const { return( m_protareWithElastic->URR_domainMin( ) ); }
        HOST_DEVICE double URR_domainMax( ) const { return( m_protareWithElastic->URR_domainMax( ) ); }
        HOST_DEVICE bool reactionHasURR_probabilityTables( int a_index ) const ;

        HOST_DEVICE double threshold( MCGIDI_VectorSizeType a_index ) const ;

        HOST_DEVICE double crossSection(                              URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, bool a_sampling = false ) const ;
        HOST_DEVICE double reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, bool a_sampling = false ) const ;
        HOST_DEVICE double reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos,                  double a_temperature, double a_energy ) const ;
        HOST_DEVICE int sampleReaction(                               URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, double a_crossSection, double (*a_userrng)( void * ), void *a_rngState ) const ;

        HOST_DEVICE double depositionEnergy(   int a_hashIndex, double a_temperature, double a_energy ) const ;
        HOST_DEVICE double depositionMomentum( int a_hashIndex, double a_temperature, double a_energy ) const ;
        HOST_DEVICE double productionEnergy(   int a_hashIndex, double a_temperature, double a_energy ) const ;
        HOST_DEVICE double gain(               int a_hashIndex, double a_temperature, double a_energy, int a_particleIndex ) const ;

        HOST_DEVICE Vector<double> const &upscatterModelAGroupVelocities( ) const { return( m_protareWithElastic->upscatterModelAGroupVelocities( ) ); }
                                                                            /**< Returns a reference to the **m_upscatterModelAGroupVelocities** member. */

        HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
        HOST_DEVICE long internalSize( ) const ;
};

/*
============================================================
=========================== Others =========================
============================================================
*/
HOST Protare *protareFromGIDIProtare( GIDI::Protare const &a_protare, PoPI::Database const &a_pops, Transporting::MC &a_settings, GIDI::Transporting::Particles const &a_particles,
                DomainHash const &a_domainHash, GIDI::Styles::TemperatureInfos const &a_temperatureInfos, std::set<int> const &a_reactionsToExclude,
                int a_reactionsToExcludeOffset = 0, bool a_allowFixedGrid = true );
HOST Vector<double> GIDI_VectorDoublesToMCGIDI_VectorDoubles( GIDI::Vector a_vector );
HOST void addVectorItemsToSet( Vector<int> const &a_productIndicesFrom, std::set<int> &a_productIndicesTo );

HOST_DEVICE double sampleBetaFromMaxwellian( double (*a_userrng)( void * ), void *a_rngState );
HOST_DEVICE bool sampleTargetBetaForUpscatterModelA( Protare const *a_protare, double a_projectileEnergy, Sampling::Input &a_input,
                double (*a_userrng)( void * ), void *a_rngState );
HOST_DEVICE void upScatterModelABoostParticle( Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState, Sampling::Product &a_product );
HOST_DEVICE void MCGIDI_sampleKleinNishina( double a_k1, double (*a_userrng)( void * ), void *a_rngState, double *a_energyOut, double *a_mu );

/*
============================================================
======================== Angle Biasing =====================
============================================================
*/
HOST_DEVICE void angleBiasing( int secondary_particle_pops_id,
                               double wgt_pre_bias,
                               double mu_p3_lab,
                               double KE_p1_lab,
                               const Protare *gidi_interaction_data,
                               const Reaction *gidi_reaction,
                               Sampling::ClientCodeRNGData rng_data,
                               double &wgt_p3,
                               double &KE_p3_lab,
                               double &speed );
}           // End of namespace MCGIDI.

#endif      // End of MCGIDI_hpp_included

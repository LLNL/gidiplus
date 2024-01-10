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

#include <LUPI.hpp>
#include <PoPI.hpp>
#include <GIDI.hpp>

#ifdef MCGIDI_USE_VIRTUAL_FUNCTIONS
    #define MCGIDI_VIRTUAL_FUNCTION virtual
    #define MCGIDI_TRUE_VIRTUAL = 0
#else
    #define MCGIDI_VIRTUAL_FUNCTION
    #define MCGIDI_TRUE_VIRTUAL
#endif

namespace MCGIDI {

class Protare;
class ProtareSingle;
class Reaction;
class OutputChannel;
class ACE_URR_probabilityTables;

}           // End of namespace MCGIDI.

#include <LUPI_dataBuffer.hpp>
#include "MCGIDI_sampling.hpp"
#include "MCGIDI_vector.hpp"
#include "MCGIDI_string.hpp"

namespace MCGIDI {

#define MCGIDI_nullReaction -10001

// FIXME, this should not be used once physicalQuantity can handle changing units.
#define MCGIDI_speedOfLight_cm_sh 299.792458
#define MCGIDI_speedOfLight_cm_sec ( MCGIDI_speedOfLight_cm_sh * 1e8 )
#define MCGIDI_classicalElectronRadius 0.2817940322010228 // Classical electron radius in unit of sqrt( b ).

#define MCGIDI_particleBeta( a_mass_unitOfEnergy, a_kineticEnergy ) ( sqrt( (a_kineticEnergy) * ( (a_kineticEnergy) + 2.0 * (a_mass_unitOfEnergy) ) ) / ( (a_kineticEnergy) + (a_mass_unitOfEnergy) ) )

LUPI_HOST_DEVICE double particleKineticEnergy( double a_mass_unitOfEnergy, double a_particleBeta );
LUPI_HOST_DEVICE double particleKineticEnergyFromBeta2( double a_mass_unitOfEnergy, double a_particleBeta2 );    // a_particleBeta2 = a_particleBeta^2.
LUPI_HOST_DEVICE double boostSpeed( double a_massProjectile, double a_kineticEnergyProjectile, double a_massTarget );
LUPI_HOST_DEVICE int muCOM_From_muLab( double a_muLab, double a_boostBeta, double a_productBeta, double &a_muPlus, double &a_JacobianPlus, double &a_muMinus, double &a_JacobianMinus );

enum class ProtareType { single, composite, TNSL };

namespace Transporting {

enum class URR_mode { none, pdfs, ACE_URR_protabilityTables };

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
        PoPI::Database const &m_pops;                               /**< The PoPs database to get particle information from.*/
        int m_neutronIndex;                                         /**< The PoPs index for the neutron. */
        int m_photonIndex;                                          /**< The PoPs index for the photon. */
        int m_electronIndex;                                        /**< The PoPs index for the electron. */
        GIDI::Styles::Suite const *m_styles;                        /**< FIXME. */
        std::string m_label;                                        /**< FIXME. */
        double m_energyDomainMax;                                   /**< All reactions with a threshold greater than or equal to this value are ignored. */
        bool m_ignoreENDF_MT5;                                      /**< If true, the ENDF MT 5 reaction is ignored. */
        bool m_sampleNonTransportingParticles;                      /**< If true, all products are sampled, otherwise only transporting particles are sampled. */
        bool m_useSlowerContinuousEnergyConversion;                   /**< If true, the old slower conversion of GIDI::ProtareSingle to MCGIDI::HeatedCrossSectionContinuousEnergy is used. */
        LookupMode::Data1d m_crossSectionLookupMode;                /**< Determines how cross sections are evaluated. */
        LookupMode::Data1d m_other1dDataLookupMode;                 /**< Determines how 1d data other than cross sections are evaluated. */
        LookupMode::Distribution m_distributionLookupMode;          /**< Determines how distributions are evaluated and sampled. Currently, only pdf_cdf is allowed. */
        Sampling::Upscatter::Model m_upscatterModel;                /**< FIXME. */
        std::string m_upscatterModelALabel;                         /**< FIXME. */
        URR_mode m_URR_mode;                                        /**< Selects if URR data are to be used, and it so, which type. */
        bool m_wantTerrellPromptNeutronDistribution;                /**< If true, prompt fission neutron distributions are sampled from the Terrell mode. */
        bool m_wantRawTNSL_distributionSampling;                    /**< If true, the TNSL neutron distributions for coherent and incoherent elastic scattering are sampled from the double differential data. Otherwise, they are sampled from the distribution data. */
        std::vector<double> m_fixedGridPoints;                      /**< FIXME. */

    public:
        LUPI_HOST MC( PoPI::Database const &a_pops, std::string const &a_projectileID, GIDI::Styles::Suite const *a_styles, std::string const &a_label, 
                GIDI::Transporting::DelayedNeutrons a_delayedNeutrons, double energyDomainMax );
        LUPI_HOST MC( PoPI::Database const &a_pops, GIDI::Protare const &a_protare, std::string const &a_label,
                GIDI::Transporting::DelayedNeutrons a_delayedNeutrons, double energyDomainMax );

        LUPI_HOST GIDI::Styles::Suite const *styles( ) const { return( m_styles ); }               /**< Returns the value of the **m_styles**. */
        LUPI_HOST void styles( GIDI::Styles::Suite const *a_styles ) { m_styles = a_styles; }      /**< This is needed for ProtareTNSL, but should be avoided otherwise. FIXME, need to have a better way. */

        LUPI_HOST std::string label( ) const { return( m_label ); }

// FIXME (1) should this not be something like
// GIDI::Styles::Suite const &suite( ) const { return( *m_styles ); }                   /**< Returns a reference to **m_styles**. */
        LUPI_HOST double energyDomainMax( ) const { return( m_energyDomainMax ); }                /**< Returns the value of the **m_energyDomainMax**. */

        LUPI_HOST bool ignoreENDF_MT5( ) const { return( m_ignoreENDF_MT5 ); }                    /**< Returns the value of the **m_ignoreENDF_MT5**. */
        LUPI_HOST void set_ignoreENDF_MT5( bool a_ignoreENDF_MT5 ) { m_ignoreENDF_MT5 = a_ignoreENDF_MT5; }
        LUPI_HOST void setIgnoreENDF_MT5( bool a_ignoreENDF_MT5 ) { set_ignoreENDF_MT5( a_ignoreENDF_MT5 ); }
                            /**< This function is deprecated. Use **set_ignoreENDF_MT5** instead. */

        LUPI_HOST bool sampleNonTransportingParticles( ) const { return( m_sampleNonTransportingParticles); }
        LUPI_HOST void sampleNonTransportingParticles( bool a_sampleNonTransportingParticles ) {
                LUPI::deprecatedFunction( "MCGIDI::Transporting::MC::sampleNonTransportingParticles", "MCGIDI::Transporting::MC::setSampleNonTransportingParticles", "" );
                setSampleNonTransportingParticles( a_sampleNonTransportingParticles ); }
        LUPI_HOST void setSampleNonTransportingParticles( bool a_sampleNonTransportingParticles ) { m_sampleNonTransportingParticles = a_sampleNonTransportingParticles; }

        LUPI_HOST bool useSlowerContinuousEnergyConversion( ) const { return( m_useSlowerContinuousEnergyConversion ); }
        LUPI_HOST void setUseSlowerContinuousEnergyConversion( bool a_useSlowerContinuousEnergyConversion ) {
                m_useSlowerContinuousEnergyConversion = a_useSlowerContinuousEnergyConversion; }

        LUPI_HOST LookupMode::Data1d crossSectionLookupMode( ) const { return( m_crossSectionLookupMode ); }      /**< Returns the value of the **m_crossSectionLookupMode**. */
        LUPI_HOST void setCrossSectionLookupMode( LookupMode::Data1d a_crossSectionLookupMode );
        LUPI_HOST void crossSectionLookupMode( LookupMode::Data1d a_crossSectionLookupMode ) {
                LUPI::deprecatedFunction( "MCGIDI::Transporting::MC::crossSectionLookupMode", "MCGIDI::Transporting::MC::setCrossSectionLookupMode", "" );
                setCrossSectionLookupMode( a_crossSectionLookupMode ); }                                            /**< See method **setCrossSectionLookupMode**. This method is deprecated. */

        LUPI_HOST LookupMode::Data1d other1dDataLookupMode( ) const { return( m_other1dDataLookupMode ); }        /**< Returns the value of the **m_other1dDataLookupMode**. */
        LUPI_HOST void setOther1dDataLookupMode( LookupMode::Data1d a_other1dDataLookupMode );
        LUPI_HOST void other1dDataLookupMode( LookupMode::Data1d a_other1dDataLookupMode ) {
                LUPI::deprecatedFunction( "MCGIDI::Transporting::MC::other1dDataLookupMode", "MCGIDI::Transporting::MC::setOther1dDataLookupMode", "" );
                setOther1dDataLookupMode( a_other1dDataLookupMode ); }                                              /**< See method **setOther1dDataLookupMode**. This method is deprecated. */

        LUPI_HOST LookupMode::Distribution distributionLookupMode( ) const { return( m_distributionLookupMode ); }  /**< Returns the value of the **m_distributionLookupMode**. */
        LUPI_HOST void setDistributionLookupMode( LookupMode::Distribution a_distributionLookupMode );
        LUPI_HOST void distributionLookupMode( LookupMode::Distribution a_distributionLookupMode ) { 
                LUPI::deprecatedFunction( "MCGIDI::Transporting::MC::distributionLookupMode", "MCGIDI::Transporting::MC::setDistributionLookupMode", "" );
                setDistributionLookupMode( a_distributionLookupMode ); }                                            /**< See method **setDistributionLookupMode**. This method is deprecated. */

        LUPI_HOST Sampling::Upscatter::Model upscatterModel( ) const { return( m_upscatterModel ); }              /**< Returns the value of the **m_upscatterModel**. */
        LUPI_HOST void set_upscatterModelA( std::string const &a_upscatterModelALabel );
        LUPI_HOST void setUpscatterModelA( std::string const &a_upscatterModelALabel ) { set_upscatterModelA( a_upscatterModelALabel ); }
                                                                                                                    /**< See method **set_upscatterModelA**. */
        LUPI_HOST std::string upscatterModelALabel( ) const { return( m_upscatterModelALabel ); }                 /**< Returns the value of the **m_upscatterModelALabel**. */
        LUPI_HOST void setUpscatterModelB( ) { m_upscatterModel = Sampling::Upscatter::Model::B; }                /**< Set member *m_upscatterModel* to Sampling::Upscatter::Model::B. */

        LUPI_HOST bool want_URR_probabilityTables( ) const {
                LUPI::deprecatedFunction( "MCGIDI::Transporting::MC::want_URR_probabilityTables", "MCGIDI::Transporting::MC::_URR_mode", "" );
                return( m_URR_mode != URR_mode::none ); }            /**< Returns *false* if *m_URR_mode* is **URR_mode::none** and *true* otherwise. This method is deprecated. Please use **_URR_mode** instead. */
        LUPI_HOST void want_URR_probabilityTables( bool a_want_URR_probabilityTables ) {
                LUPI::deprecatedFunction( "MCGIDI::Transporting::MC::want_URR_probabilityTables", "MCGIDI::Transporting::MC::setURR_mode", "" );
                m_URR_mode = URR_mode::none;
                if( a_want_URR_probabilityTables ) m_URR_mode = URR_mode::pdfs;
        }           /**< If *a_want_URR_probabilityTables* is *true* sets *m_URR_mode* to **URR_mode::pdfs**, otherwise set it to **URR_mode::none**. This method is deprecated. Please use **setURR_mode** instead. */

        LUPI_HOST URR_mode _URR_mode( ) const { return( m_URR_mode ); }                                           /**< Returns the value of the **m_URR_mode** member. */
        LUPI_HOST void setURR_mode( URR_mode a_URR_mode ) { m_URR_mode = a_URR_mode; }                            /**< This methods sets member *m_URR_mode* to *a_URR_mode*. */

        LUPI_HOST bool wantTerrellPromptNeutronDistribution( ) const { return( m_wantTerrellPromptNeutronDistribution ); }
                                    /**< Returns the value of the **m_wantTerrellPromptNeutronDistribution** member. */
        LUPI_HOST void setWantTerrellPromptNeutronDistribution( bool a_wantTerrellPromptNeutronDistribution ) {
                m_wantTerrellPromptNeutronDistribution = a_wantTerrellPromptNeutronDistribution;
        }       /**< Set the *m_wantTerrellPromptNeutronDistribution* member to *a_wantTerrellPromptNeutronDistribution*. */
        LUPI_HOST void wantTerrellPromptNeutronDistribution( bool a_wantTerrellPromptNeutronDistribution ) {
                LUPI::deprecatedFunction( "MCGIDI::Transporting::MC::wantTerrellPromptNeutronDistribution", "MCGIDI::Transporting::MC::setWantTerrellPromptNeutronDistribution", "" );
                setWantTerrellPromptNeutronDistribution( a_wantTerrellPromptNeutronDistribution );
        }       /**< See method *setWantTerrellPromptNeutronDistribution*. This method is deprecated. */

        LUPI_HOST bool wantRawTNSL_distributionSampling( ) const { return( m_wantRawTNSL_distributionSampling ); }
        LUPI_HOST bool wantRawTNSL_distributionSampling( ) { return( m_wantRawTNSL_distributionSampling ); }
        LUPI_HOST void set_wantRawTNSL_distributionSampling( bool a_wantRawTNSL_distributionSampling ) { 
                m_wantRawTNSL_distributionSampling = a_wantRawTNSL_distributionSampling; }

        LUPI_HOST std::vector<double> fixedGridPoints( ) const { return( m_fixedGridPoints ); }
        LUPI_HOST void fixedGridPoints( std::vector<double> a_fixedGridPoints ) { m_fixedGridPoints = a_fixedGridPoints; }

        LUPI_HOST PoPI::Database const &pops( ) const { return( m_pops ); }                       /**< Returns a reference to **m_styles**. */
        LUPI_HOST int neutronIndex( ) const { return( m_neutronIndex ); }                         /**< Returns the value of **m_neutronIndex**. */
        LUPI_HOST int photonIndex( ) const { return( m_photonIndex ); }                           /**< Returns the value of **m_photonIndex**. */
        LUPI_HOST int electronIndex( ) const { return( m_electronIndex ); }                       /**< Returns the value of **m_electronIndex**. */

        LUPI_HOST void process( GIDI::Protare const &a_protare );
};

}           // End of namespace Transporting.

enum class TwoBodyOrder { notApplicable, firstParticle, secondParticle };

/*
============================================================
============= ACE_URR_protabilityTablesFromGIDI ============
============================================================
*/

class ACE_URR_protabilityTablesFromGIDI {

    public:
        LUPI_HOST ACE_URR_protabilityTablesFromGIDI( );
        LUPI_HOST ~ACE_URR_protabilityTablesFromGIDI( );

        std::map<std::string, ACE_URR_probabilityTables *> m_ACE_URR_probabilityTables;     // The string is the reaction's label.
};

/*
============================================================
========================= SetupInfo ========================
============================================================
*/
class SetupInfo {

    public:
        ProtareSingle &m_protare;
        LUPI::FormatVersion m_formatVersion;
        double m_Q;
        double m_product1Mass;
        double m_product2Mass;
        double m_domainMin;
        double m_domainMax;
        TwoBodyOrder m_twoBodyOrder;
        bool m_isPairProduction;
        bool m_isPhotoAtomicIncoherentScattering;
        std::string m_distributionLabel;                        /**< Set by the ProtareSingle constructor to the distribution label to use for all products. */
        std::map<std::string, int> m_particleIndices;
        GIDI::Reaction const *m_reaction;
        Transporting::Reaction::Type m_reactionType;
        int m_initialStateIndex;                                /**< If not -1, then reaction contains a branching gamma data with this index for the data in m_nuclideGammaBranchStateInfos member of its **ProtareSingle** instance. */
        bool m_hasFinalStatePhotons;                            /**< If **true**, the reaction has a photon with finalState attribute. */
        std::map<std::string, int> m_initialStateIndices;       /**< If not -1, then reaction contains a branching gamma data with this index for the data in m_nuclideGammaBranchStateInfos member of its **ProtareSingle** instance. */
        std::map<std::string, int> m_stateNamesToIndices;
        std::map<std::string, ACE_URR_protabilityTablesFromGIDI *> m_ACE_URR_protabilityTablesFromGIDI;

        LUPI_HOST SetupInfo( ProtareSingle &a_protare );
        LUPI_HOST ~SetupInfo( );
};

/*
============================================================
=========================== Others =========================
============================================================
*/
LUPI_HOST int MCGIDI_popsIndex( PoPI::Database const &a_pops, std::string const &a_ID );

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

LUPI_HOST_DEVICE inline MCGIDI_VectorSizeType binarySearchVector( double a_x, Vector<double> const &a_Xs, bool a_boundIndex = false ) {

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

LUPI_HOST_DEVICE inline MCGIDI_VectorSizeType binarySearchVectorBounded( double a_x, Vector<double> const &a_Xs, MCGIDI_VectorSizeType a_lower, 
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
        LUPI_HOST_DEVICE DomainHash( );
        LUPI_HOST_DEVICE DomainHash( int a_bins, double a_domainMin, double a_domainMax );
        LUPI_HOST_DEVICE DomainHash( DomainHash const &a_domainHash );

        LUPI_HOST_DEVICE int bins( ) const { return( m_bins ); }                     /**< Returns the value of the **m_bins**. */
        LUPI_HOST_DEVICE double domainMin( ) const { return( m_domainMin ); }        /**< Returns the value of the **m_domainMax**. */
        LUPI_HOST_DEVICE double domainMax( ) const { return( m_domainMax ); }        /**< Returns the value of the **m_domainMax**. */
        LUPI_HOST_DEVICE double u_domainMin( ) const { return( m_u_domainMin ); }    /**< Returns the value of the **m_u_domainMin**. */
        LUPI_HOST_DEVICE double u_domainMax( ) const { return( m_u_domainMax ); }    /**< Returns the value of the **m_u_domainMax**. */
        LUPI_HOST_DEVICE double inverse_du( ) const { return( m_inverse_du ); }      /**< Returns the value of the **m_inverse_du**. */

        LUPI_HOST_DEVICE int index( double a_domain ) const ;
        LUPI_HOST_DEVICE Vector<int> map( Vector<double> &a_domainValues ) const ;

        LUPI_HOST_DEVICE void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );

        LUPI_HOST void print( bool a_printValues ) const ;
};

/*
============================================================
===================== MultiGroupHash =======================
============================================================
*/
class MultiGroupHash {

    private:
        Vector<double> m_boundaries;                                    /**< The list of multi-group boundaries. */

        LUPI_HOST void initialize( GIDI::Protare const &a_protare, GIDI::Styles::TemperatureInfo const &a_temperatureInfo, std::string a_particleID );

    public:
        LUPI_HOST MultiGroupHash( std::vector<double> a_boundaries );
        LUPI_HOST MultiGroupHash( GIDI::Protare const &a_protare, GIDI::Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_particleID = "" );
        LUPI_HOST MultiGroupHash( GIDI::Protare const &a_protare, GIDI::Transporting::Particles const &a_particles );

        LUPI_HOST_DEVICE Vector<double> const &boundaries( ) const { return( m_boundaries ); }   /**< Returns a reference to **m_styles**. */
        LUPI_HOST_DEVICE int index( double a_domain ) const {
            MCGIDI_VectorSizeType _index = binarySearchVector( a_domain, m_boundaries );

            if( _index == -2 ) return( 0 );
            if( _index == -1 ) return( m_boundaries.size( ) - 2 );
            return( _index );
        }
        LUPI_HOST_DEVICE void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
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

        LUPI_HOST_DEVICE URR_protareInfo( ) : m_inURR( false ), m_rng_Value( 0.0 ) { }
        LUPI_HOST_DEVICE URR_protareInfo( URR_protareInfo const &a_URR_protareInfo ) { m_inURR = a_URR_protareInfo.m_inURR; m_rng_Value = a_URR_protareInfo.m_rng_Value; }

        LUPI_HOST_DEVICE bool inURR( ) const { return( m_inURR ); }
        LUPI_HOST_DEVICE void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
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
        LUPI_HOST_DEVICE URR_protareInfos( ) : m_URR_protareInfos( ) { }
        LUPI_HOST URR_protareInfos( Vector<Protare *> &a_protares );

        LUPI_HOST void setup( Vector<Protare *> &a_protares );

        LUPI_HOST_DEVICE MCGIDI_VectorSizeType size( ) const { return( m_URR_protareInfos.size( ) ); }
        LUPI_HOST_DEVICE URR_protareInfo const &operator[]( MCGIDI_VectorSizeType a_index ) const { return( m_URR_protareInfos[a_index] ); }  /**< Returns the instance of *m_URR_protareInfos* at index *a_index*. */
        LUPI_HOST_DEVICE void updateProtare( MCGIDI::Protare const *a_protare, double a_energy, double (*a_userrng)( void * ), void *a_rngState );

        LUPI_HOST_DEVICE void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
        LUPI_HOST_DEVICE long internalSize( ) const { return m_URR_protareInfos.internalSize( ); }
};

/*
============================================================
================= ACE_URR_probabilityTable =================
============================================================
*/

class ACE_URR_probabilityTable {

    public:
        double m_energy;                                                /**< The projectile energy where the data are specified. */
        Vector<double> m_propabilities;                                 /**< The probability for each cross section. */
        Vector<double> m_crossSection;                                  /**< The cross section for each probability. */

        LUPI_HOST_DEVICE ACE_URR_probabilityTable( );
        LUPI_HOST ACE_URR_probabilityTable( double a_energy, std::vector<double> const &a_propabilities, std::vector<double> const &a_crossSection );
        LUPI_HOST_DEVICE ~ACE_URR_probabilityTable( );

        LUPI_HOST_DEVICE double energy( ) const { return( m_energy ); }
        LUPI_HOST_DEVICE double sample( double a_rng_Value );
        LUPI_HOST_DEVICE void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
};

/*
============================================================
================= ACE_URR_probabilityTables ================
============================================================
*/

class ACE_URR_probabilityTables {

    public:
        Vector<double> m_energies;                                          /**< List of energies where probabilities tables are given. */
        Vector<ACE_URR_probabilityTable *> m_ACE_URR_probabilityTables;     /**< List of probabilities tables. One for each energy in *m_energies*. */

        LUPI_HOST_DEVICE ACE_URR_probabilityTables( );
        LUPI_HOST_DEVICE ACE_URR_probabilityTables( std::size_t a_capacity );
        LUPI_HOST_DEVICE ~ACE_URR_probabilityTables( );

        LUPI_HOST_DEVICE MCGIDI_VectorSizeType capacity( ) const { return( m_energies.capacity( ) ); }
                                                                            /**< Returns the number of energies allocated to store probability tables. */
        LUPI_HOST_DEVICE MCGIDI_VectorSizeType size( ) const { return( m_energies.size( ) ); }
                                                                            /**< Returns the number of energies that have URR probability tables. */
        LUPI_HOST_DEVICE void reserve( MCGIDI_VectorSizeType a_capacity );
        LUPI_HOST_DEVICE void push_back( ACE_URR_probabilityTable *a_ACE_URR_probabilityTable );

        LUPI_HOST_DEVICE double domainMin( ) const { return( m_energies[0] ); }         /**< Returns the minimum energy where URR data are specified. */
        LUPI_HOST_DEVICE double domainMax( ) const { return( m_energies.back( ) ); }    /**< Returns the maximum energy where URR data are specified. */
        LUPI_HOST_DEVICE double sample( double a_energy, double a_rng_Value );

        LUPI_HOST_DEVICE void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
};

/*
============================================================
======== HeatedReactionCrossSectionContinuousEnergy ========
============================================================
*/

class HeatedReactionCrossSectionContinuousEnergy {

    private:
        int m_offset;                                               /**< The offset relative to the cross section grid of the first cross section value in *m_crossSection*. */
        double m_threshold;                                         /**< The threshold for the reaction. */
        Vector<double> m_crossSection;                              /**< The reaction's cross section. */
        Transporting::URR_mode m_URR_mode;                          /**< The URR data (i.e., mode) *this* has. */
        Probabilities::ProbabilityBase2d *m_URR_probabilityTables;  /**< Pointer to pdf URR probabilities if they were loaded. */
        ACE_URR_probabilityTables *m_ACE_URR_probabilityTables;     /**< The ACE URR probability tables for the reaction's cross section, if they were loaded. */

    public:
        LUPI_HOST_DEVICE HeatedReactionCrossSectionContinuousEnergy( );
        LUPI_HOST HeatedReactionCrossSectionContinuousEnergy( int a_offset, double a_threshold, Vector<double> &a_crossSection );
        LUPI_HOST HeatedReactionCrossSectionContinuousEnergy( double a_threshold, GIDI::Functions::Ys1d const &a_crossSection, 
                        Probabilities::ProbabilityBase2d *a_URR_probabilityTables, ACE_URR_probabilityTables *a_ACE_URR_probabilityTables );
        LUPI_HOST_DEVICE ~HeatedReactionCrossSectionContinuousEnergy( );

        LUPI_HOST_DEVICE double threshold( ) const { return( m_threshold ); }                            /**< Returns the value of the **m_threshold**. */
        LUPI_HOST_DEVICE int offset( ) const { return( m_offset ); }                                     /**< Returns the value of the **m_offset**. */
        LUPI_HOST_DEVICE bool hasURR_probabilityTables( ) const {
            return( ( m_URR_probabilityTables != nullptr ) || ( m_ACE_URR_probabilityTables != nullptr ) );
        }                                                           /**< Returns true if URR probability tables data present and false otherwise. */
        LUPI_HOST_DEVICE Transporting::URR_mode URR_mode( ) const { return( m_URR_mode ); }           /**< Returns the value of **m_URR_mode**. */
        LUPI_HOST_DEVICE double URR_domainMin( ) const ;
        LUPI_HOST_DEVICE double URR_domainMax( ) const ;
        LUPI_HOST_DEVICE Probabilities::ProbabilityBase2d *URR_probabilityTables( ) const { return( m_URR_probabilityTables ); }      /**< Returns the value of the *m_URR_probabilityTables*. */
        LUPI_HOST_DEVICE ACE_URR_probabilityTables *_ACE_URR_probabilityTables( ) const { return( m_ACE_URR_probabilityTables ); }    /**< Returns the value of the *m_ACE_URR_probabilityTables*. */
        LUPI_HOST_DEVICE double crossSection( std::size_t a_index ) const {
            int index = static_cast<int>( a_index ) - m_offset;
            if( index < 0 ) return( 0.0 );
            if( index >= static_cast<int>( m_crossSection.size( ) ) ) return( 0.0 );

            return( m_crossSection[index] );
        }
        LUPI_HOST_DEVICE void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );

        LUPI_HOST void print( ProtareSingle const *a_protareSingle, std::string const &a_indent, std::string const &a_iFormat, 
                std::string const &a_energyFormat, std::string const &a_dFormat ) const ;
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
        LUPI_HOST_DEVICE ContinuousEnergyGain( );
        ContinuousEnergyGain( int a_particleIndex, std::size_t a_size );

        ContinuousEnergyGain &operator=( ContinuousEnergyGain const &a_continuousEnergyGain );

        LUPI_HOST_DEVICE int particleIndex( ) const { return( m_particleIndex ); }
        LUPI_HOST_DEVICE int userParticleIndex( ) const { return( m_userParticleIndex ); }
        LUPI_HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex ) { if( a_particleIndex == m_particleIndex ) m_userParticleIndex = a_userParticleIndex; }
        LUPI_HOST_DEVICE Vector<double> const &gain( ) const { return( m_gain ); }
        LUPI_HOST void adjustGain( int a_energy_index, double a_gain ) { m_gain[a_energy_index] += a_gain; }
        LUPI_HOST_DEVICE double gain( int a_energy_index, double a_energy_fraction ) const ;

        LUPI_HOST_DEVICE void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
        LUPI_HOST void print( ProtareSingle const *a_protareSingle, std::string const &a_indent, std::string const &a_iFormat, 
                std::string const &a_energyFormat, std::string const &a_dFormat ) const ;
};

/*
============================================================
=========== HeatedCrossSectionContinuousEnergy =============
============================================================
*/
class HeatedCrossSectionContinuousEnergy {

    private:
        double m_temperature;                                   /**< The target temperature of the data. */
        Vector<int> m_hashIndices;                              /**< The indicies for the energy hash function. */
        Vector<double> m_energies;                              /**< Energy grid for cross sections. */
        Vector<double> m_totalCrossSection;                     /**< The total cross section. */
        Vector<double> m_depositionEnergy;                      /**< The total continuous energy, deposition-energy cross section (related to the kinetic energy of the untracked outgoing particles). */
        Vector<double> m_depositionMomentum;                    /**< The total continuous energy, deposition-momentum cross section. */
        Vector<double> m_productionEnergy;                      /**< The total continuous energy, Q-value cross section. */
        Vector<ContinuousEnergyGain> m_gains;                   /**< The total continuous energy, gain cross section for each tracked particle. */
        Transporting::URR_mode m_URR_mode;                      /**< The URR data (i.e., mode) *this* has. */
        Vector<int> m_reactionsInURR_region;                    /**< A list of reactions with in or below the upper URR regions. Empty unless URR probability tables present and used. */
        Vector<HeatedReactionCrossSectionContinuousEnergy *> m_reactionCrossSections;
                                                                /**< Reaction cross section data for each reaction. */
        ACE_URR_probabilityTables *m_ACE_URR_probabilityTables; /**< The ACE URR probability tables for the summed URR cross section, if they were loaded. */

    public:
        LUPI_HOST_DEVICE HeatedCrossSectionContinuousEnergy( );
        LUPI_HOST HeatedCrossSectionContinuousEnergy( SetupInfo &a_setupInfo, Transporting::MC const &a_settings, GIDI::Transporting::Particles const &a_particles,
                DomainHash const &a_domainHash, GIDI::Styles::TemperatureInfo const &a_temperatureInfo, std::vector<GIDI::Reaction const *> const &a_reactions,
                std::vector<GIDI::Reaction const *> const &a_orphanProducts, bool a_fixedGrid, bool a_zeroReactions );
        LUPI_HOST_DEVICE ~HeatedCrossSectionContinuousEnergy( );

        LUPI_HOST_DEVICE int evaluationInfo( int a_hashIndex, double a_energy, double *a_energyFraction ) const ;

        LUPI_HOST_DEVICE double minimumEnergy( ) const { return( m_energies[0] ); }          /**< Returns the minimum cross section domain. */
        LUPI_HOST_DEVICE double maximumEnergy( ) const { return( m_energies.back( ) ); }     /**< Returns the maximum cross section domain. */
        LUPI_HOST_DEVICE int numberOfReactions( ) const { return( (int) m_reactionCrossSections.size( ) ); } 
                                                                /**< Returns the number of reaction cross section. */

        LUPI_HOST_DEVICE int thresholdOffset( int a_reactionIndex ) const { return( m_reactionCrossSections[a_reactionIndex]->offset( ) ); }
                                                                /**< Returns the offset for the cross section for the reaction with index *a_reactionIndex*. */
        LUPI_HOST_DEVICE double threshold( int a_reactionIndex ) const { return( m_reactionCrossSections[a_reactionIndex]->threshold( ) ); }
                                                                /**< Returns the threshold for the reaction with index *a_reactionIndex*. */
        LUPI_HOST_DEVICE bool hasURR_probabilityTables( ) const ;
        LUPI_HOST_DEVICE double URR_domainMin( ) const ;
        LUPI_HOST_DEVICE double URR_domainMax( ) const ;
        LUPI_HOST_DEVICE bool reactionHasURR_probabilityTables( int a_index ) const { return( m_reactionCrossSections[a_index]->hasURR_probabilityTables( ) ); }

        LUPI_HOST_DEVICE Vector<double> &totalCrossSection( ) { return( m_totalCrossSection ); }     /**< Returns a reference to member *m_totalCrossSection*. */
        LUPI_HOST_DEVICE double crossSection(                               URR_protareInfos const &a_URR_protareInfos, int a_URR_index, int a_hashIndex, double a_energy, bool a_sampling = false ) const ;
        LUPI_HOST_DEVICE double reactionCrossSection(  int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_URR_index, int a_hashIndex, double a_energy, bool a_sampling = false ) const ;
        LUPI_HOST_DEVICE double reactionCrossSection2( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_URR_index, double a_energy, int a_energyIndex, double a_energyFraction, bool a_sampling = false ) const ;
        LUPI_HOST_DEVICE double reactionCrossSection(  int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_URR_index, double a_energy ) const ;

        LUPI_HOST_DEVICE double depositionEnergy(   int a_hashIndex, double a_energy ) const ;
        LUPI_HOST_DEVICE double depositionMomentum( int a_hashIndex, double a_energy ) const ;
        LUPI_HOST_DEVICE double productionEnergy(   int a_hashIndex, double a_energy ) const ;
        LUPI_HOST_DEVICE double gain(               int a_hashIndex, double a_energy, int a_particleIndex ) const ;

        LUPI_HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex );
        LUPI_HOST_DEVICE void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );

        LUPI_HOST_DEVICE Vector<double> const &energies( ) const { return( m_energies ); }       /**< Returns a reference to **m_styles**. */

        LUPI_HOST void print( ProtareSingle const *a_protareSingle, std::string const &a_indent, std::string const &a_iFormat, 
                std::string const &a_energyFormat, std::string const &a_dFormat ) const ;
};

/*
============================================================
============ HeatedCrossSectionsContinuousEnergy ===========
============================================================
*/
class HeatedCrossSectionsContinuousEnergy {

    private:
        Vector<double> m_temperatures;                                      /**< The list of temperatures that have **HeatedCrossSectionContinuousEnergy** data. */
        Vector<double> m_thresholds;                                        /**< The threshold for each reaction. */
        Vector<HeatedCrossSectionContinuousEnergy *> m_heatedCrossSections; /**< One **HeatedCrossSectionContinuousEnergy** instance for each temperature in *m_temperature*. */

    public:
        LUPI_HOST_DEVICE HeatedCrossSectionsContinuousEnergy( );
        LUPI_HOST_DEVICE ~HeatedCrossSectionsContinuousEnergy( );

        LUPI_HOST void update( LUPI::StatusMessageReporting &a_smr, SetupInfo &a_setupInfo, Transporting::MC const &a_settings, GIDI::Transporting::Particles const &a_particles, DomainHash const &a_domainHash, 
                GIDI::Styles::TemperatureInfos const &a_temperatureInfos, std::vector<GIDI::Reaction const *> const &a_reactions, 
                std::vector<GIDI::Reaction const *> const &a_orphanProducts, bool a_fixedGrid, bool a_zeroReactions );

        LUPI_HOST_DEVICE double minimumEnergy( ) const { return( m_heatedCrossSections[0]->minimumEnergy( ) ); }
                                                                    /**< Returns the minimum cross section domain. */
        LUPI_HOST_DEVICE double maximumEnergy( ) const { return( m_heatedCrossSections[0]->maximumEnergy( ) ); }
                                                                    /**< Returns the maximum cross section domain. */
        LUPI_HOST_DEVICE Vector<double> const &temperatures( ) const { return( m_temperatures ); }   /**< Returns the value of the **m_temperatures**. */
        Vector<HeatedCrossSectionContinuousEnergy *> &heatedCrossSections( ) { return( m_heatedCrossSections ); }

        LUPI_HOST_DEVICE double threshold( MCGIDI_VectorSizeType a_index ) const { return( m_thresholds[a_index] ); }     /**< Returns the threshold for the reaction at index *a_index*. */
        LUPI_HOST_DEVICE bool hasURR_probabilityTables( ) const { return( m_heatedCrossSections[0]->hasURR_probabilityTables( ) ); }
        LUPI_HOST_DEVICE double URR_domainMin( ) const { return( m_heatedCrossSections[0]->URR_domainMin( ) ); }
        LUPI_HOST_DEVICE double URR_domainMax( ) const { return( m_heatedCrossSections[0]->URR_domainMax( ) ); }
        LUPI_HOST_DEVICE bool reactionHasURR_probabilityTables( int a_index ) const { return( m_heatedCrossSections[0]->reactionHasURR_probabilityTables( a_index ) ); }

        LUPI_HOST_DEVICE double crossSection(                              URR_protareInfos const &a_URR_protareInfos, int a_URR_index, int a_hashIndex, 
                double a_temperature, double a_energy, bool a_sampling = false ) const ;
        LUPI_HOST_DEVICE void crossSectionVector( double a_temperature, double a_userFactor, int a_numberAllocated, double *a_crossSectionVector ) const ;
        LUPI_HOST_DEVICE double reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_URR_index, int a_hashIndex, 
                double a_temperature, double a_energy, bool a_sampling = false ) const ;
        LUPI_HOST_DEVICE double reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_URR_index, double a_temperature, double a_energy_in ) const ;
        LUPI_HOST_DEVICE int sampleReaction(                               URR_protareInfos const &a_URR_protareInfos, int a_URR_index, int a_hashIndex, 
                double a_temperature, double a_energy, double a_crossSection, double (*a_userrng)( void * ), void *a_rngState ) const ;

        LUPI_HOST_DEVICE double depositionEnergy(   int a_hashIndex, double a_temperature, double a_energy ) const ;
        LUPI_HOST_DEVICE double depositionMomentum( int a_hashIndex, double a_temperature, double a_energy ) const ;
        LUPI_HOST_DEVICE double productionEnergy(   int a_hashIndex, double a_temperature, double a_energy ) const ;
        LUPI_HOST_DEVICE double gain(               int a_hashIndex, double a_temperature, double a_energy, int a_particleIndex ) const ;

        LUPI_HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex );
        LUPI_HOST_DEVICE void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );

        LUPI_HOST void print( ProtareSingle const *a_protareSingle, std::string const &a_indent, std::string const &a_iFormat, 
                std::string const &a_energyFormat, std::string const &a_dFormat ) const ;
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
        LUPI_HOST_DEVICE MultiGroupGain( );
        LUPI_HOST MultiGroupGain( int a_particleIndex, GIDI::Vector const &a_gain );

        LUPI_HOST MultiGroupGain &operator=( MultiGroupGain const &a_multiGroupGain );

        LUPI_HOST_DEVICE int particleIndex( ) const { return( m_particleIndex ); }
        LUPI_HOST_DEVICE int userParticleIndex( ) const { return( m_userParticleIndex ); }
        LUPI_HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex ) { if( a_particleIndex == m_particleIndex ) m_userParticleIndex = a_userParticleIndex; }
        LUPI_HOST_DEVICE Vector<double> const &gain( ) const { return( m_gain ); }
        LUPI_HOST_DEVICE double gain( int a_hashIndex ) const { return( m_gain[a_hashIndex] ); }

        LUPI_HOST_DEVICE void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
        LUPI_HOST void write( FILE *a_file ) const ;
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
        LUPI_HOST_DEVICE HeatedReactionCrossSectionMultiGroup( );
        LUPI_HOST HeatedReactionCrossSectionMultiGroup( SetupInfo &a_setupInfo, Transporting::MC const &a_settings, int a_offset, 
                std::vector<double> const &a_crossSection, double a_threshold );

        LUPI_HOST_DEVICE double operator[]( MCGIDI_VectorSizeType a_index ) const { return( m_crossSection[a_index] ); }  /**< Returns the value of the cross section at multi-group index *a_index*. */
        LUPI_HOST_DEVICE double threshold( ) const { return( m_threshold ); }        /**< Returns the value of the **m_threshold**. */
        LUPI_HOST_DEVICE int offset( ) const { return( m_offset ); }                 /**< Returns the value of the **m_offset**. */
        LUPI_HOST_DEVICE double crossSection( std::size_t a_index, bool a_sampling = false ) const {
            int index = (int)a_index - m_offset;
            if( index < 0 ) return( 0 );
            if( index >= (int)m_crossSection.size( ) ) return( 0 );

            double _crossSection( m_crossSection[index] );
            if( a_sampling && ( index == 0 ) ) {
                _crossSection += m_augmentedThresholdCrossSection;
            }
            return( _crossSection );
        }
        LUPI_HOST_DEVICE double augmentedThresholdCrossSection( ) const { return( m_augmentedThresholdCrossSection ); }  /**< Returns the value of the **m_augmentedThresholdCrossSection**. */
        LUPI_HOST_DEVICE void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
        LUPI_HOST void write( FILE *a_file, int a_reactionIndex ) const ;
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
        LUPI_HOST_DEVICE HeatedCrossSectionMultiGroup( );
        LUPI_HOST HeatedCrossSectionMultiGroup( LUPI::StatusMessageReporting &a_smr, GIDI::ProtareSingle const &a_protare, SetupInfo &a_setupInfo, 
                Transporting::MC const &a_settings, GIDI::Styles::TemperatureInfo const &a_temperatureInfo,
                GIDI::Transporting::Particles const &a_particles, std::vector<GIDI::Reaction const *> const &a_reactions, std::string const &a_label,
                bool a_zeroReactions, GIDI::ExcludeReactionsSet const &a_reactionsToExclude );
        LUPI_HOST_DEVICE ~HeatedCrossSectionMultiGroup( );

        LUPI_HOST_DEVICE HeatedReactionCrossSectionMultiGroup *operator[]( MCGIDI_VectorSizeType a_index ) const { return( m_reactionCrossSections[a_index] ); }
                                                                                /**< Returns the HeatedReactionCrossSectionMultiGroup for the reaction at index *a_index *a_index*. */
        LUPI_HOST_DEVICE int numberOfReactions( ) const { return( (int) m_reactionCrossSections.size( ) ); }
                                                                                /**< Returns the number of reactions stored in *this*. */

        LUPI_HOST_DEVICE int thresholdOffset(                              int a_index ) const { return( m_reactionCrossSections[a_index]->offset( ) ); }
                                                                                /**< Returns the offset for the cross section for the reaction with index *a_index*. */
        LUPI_HOST_DEVICE double threshold(                                 int a_index ) const { return( m_reactionCrossSections[a_index]->threshold( ) ); }

        LUPI_HOST_DEVICE Vector<double> &totalCrossSection( ) { return( m_totalCrossSection ); }    /**< Returns a reference to member *m_totalCrossSection*. */
        LUPI_HOST_DEVICE double crossSection(                              int a_hashIndex, bool a_sampling = false ) const ;
        LUPI_HOST_DEVICE double augmentedCrossSection(                     int a_hashIndex ) const { return( m_augmentedCrossSection[a_hashIndex] ); }
                                                                                /**< Returns the value of the of the augmented cross section the reaction at index *a_index*. */
        LUPI_HOST_DEVICE double reactionCrossSection( int a_reactionIndex, int a_hashIndex, bool a_sampling = false ) const {
                return( m_reactionCrossSections[a_reactionIndex]->crossSection( a_hashIndex, a_sampling ) ); }
                /**< Returns the reaction's cross section for the reaction at index *a_reactionIndex* and multi-group index *a_hashIndex*. */

        LUPI_HOST_DEVICE double depositionEnergy(   int a_hashIndex ) const { return( m_depositionEnergy[a_hashIndex] ); }
        LUPI_HOST_DEVICE double depositionMomentum( int a_hashIndex ) const { return( m_depositionMomentum[a_hashIndex] ); }
        LUPI_HOST_DEVICE double productionEnergy(   int a_hashIndex ) const { return( m_productionEnergy[a_hashIndex] ); }
        LUPI_HOST_DEVICE double gain(               int a_hashIndex, int a_particleIndex ) const ;

        LUPI_HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex );

        LUPI_HOST_DEVICE void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
        LUPI_HOST void write( FILE *a_file ) const ;
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
        Vector<double> m_projectileMultiGroupBoundariesCollapsed;
        Vector<HeatedCrossSectionMultiGroup *> m_heatedCrossSections;

    public:
        LUPI_HOST_DEVICE HeatedCrossSectionsMultiGroup( );
        LUPI_HOST_DEVICE ~HeatedCrossSectionsMultiGroup( );

        LUPI_HOST_DEVICE double minimumEnergy( ) const { return( m_projectileMultiGroupBoundariesCollapsed[0] ); }
        LUPI_HOST_DEVICE double maximumEnergy( ) const { return( m_projectileMultiGroupBoundariesCollapsed.back( ) ); }
        LUPI_HOST_DEVICE Vector<double> const &temperatures( ) const { return( m_temperatures ); }   /**< Returns the value of the **m_temperatures**. */

        LUPI_HOST void update( LUPI::StatusMessageReporting &a_smr, GIDI::ProtareSingle const &a_protare, SetupInfo &a_setupInfo, Transporting::MC const &a_settings, GIDI::Transporting::Particles const &a_particles, 
                GIDI::Styles::TemperatureInfos const &a_temperatureInfos, std::vector<GIDI::Reaction const *> const &a_reactions, 
                std::vector<GIDI::Reaction const *> const &a_orphanProducts, bool a_zeroReactions, GIDI::ExcludeReactionsSet const &a_reactionsToExclude );

        LUPI_HOST_DEVICE int multiGroupThresholdIndex( MCGIDI_VectorSizeType a_index ) const { return( m_multiGroupThresholdIndex[a_index] ); }
                                                                                                    /**< Returns the threshold for the reaction at index *a_index*. */
        LUPI_HOST_DEVICE Vector<double> const &projectileMultiGroupBoundariesCollapsed( ) const { return( m_projectileMultiGroupBoundariesCollapsed ); }
                                                                                                    /**< Returns the value of the **m_projectileMultiGroupBoundariesCollapsed**. */
        LUPI_HOST_DEVICE Vector<HeatedCrossSectionMultiGroup *> const &heatedCrossSections( ) const { return( m_heatedCrossSections ); }

        LUPI_HOST_DEVICE double threshold( MCGIDI_VectorSizeType a_index ) const { return( m_thresholds[a_index] ); }     /**< Returns the threshold for the reaction at index *a_index*. */

        LUPI_HOST_DEVICE double crossSection(                              int a_hashIndex, double a_temperature, bool a_sampling = false ) const ;
        LUPI_HOST_DEVICE void crossSectionVector( double a_temperature, double a_userFactor, int a_numberAllocated, double *a_crossSectionVector ) const ;
        LUPI_HOST_DEVICE double reactionCrossSection( int a_reactionIndex, int a_hashIndex, double a_temperature, bool a_sampling = false ) const ;
        LUPI_HOST_DEVICE double reactionCrossSection( int a_reactionIndex, double a_temperature, double a_energy_in ) const ;
        LUPI_HOST_DEVICE int sampleReaction(                               int a_hashIndex, double a_temperature, double a_energy_in, double a_crossSection, 
                        double (*a_userrng)( void * ), void *a_rngState ) const ;

        LUPI_HOST_DEVICE double depositionEnergy(   int a_hashIndex, double a_temperature ) const ;
        LUPI_HOST_DEVICE double depositionMomentum( int a_hashIndex, double a_temperature ) const ;
        LUPI_HOST_DEVICE double productionEnergy(   int a_hashIndex, double a_temperature ) const ;
        LUPI_HOST_DEVICE double gain(               int a_hashIndex, double a_temperature, int a_particleIndex ) const ;

        LUPI_HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex );

        LUPI_HOST_DEVICE void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
        LUPI_HOST void write( FILE *a_file, int a_temperatureIndex ) const ;
        LUPI_HOST void print( ) const ;
};

/*
============================================================
================== NuclideGammaBranchInfo ==================
============================================================
*/
class NuclideGammaBranchInfo {

    private:
        double m_probability;                               /**< The probability that the level decays to state *m_residualStateIndex*. */
        double m_photonEmissionProbability;                 /**< The conditional probability the the decay emitted a photon. */
        double m_gammaEnergy;                               /**< The energy of the emitted photon. */
        int m_residualStateIndex;                           /**< The state the residual is left in after photon decay. */

    public:
        LUPI_HOST_DEVICE NuclideGammaBranchInfo( );
        LUPI_HOST NuclideGammaBranchInfo( PoPI::NuclideGammaBranchInfo const &a_nuclideGammaBranchInfo, std::map<std::string, int> &a_stateNamesToIndices );

        LUPI_HOST_DEVICE double probability( ) const { return( m_probability ); }                                /**< Returns the value of the **m_probability**. */
        LUPI_HOST_DEVICE double photonEmissionProbability( ) const { return( m_photonEmissionProbability ); }    /**< Returns the value of the **m_photonEmissionProbability**. */
        LUPI_HOST_DEVICE double gammaEnergy( ) const { return( m_gammaEnergy ); }                                /**< Returns the value of the **m_gammaEnergy**. */
        LUPI_HOST_DEVICE int residualStateIndex( ) const { return( m_residualStateIndex ); }                     /**< Returns the value of the **m_residualStateIndex**. */

        LUPI_HOST_DEVICE void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
        LUPI_HOST void print( ProtareSingle const *a_protareSingle, std::string const &a_indent, std::string const &a_iFormat,
                std::string const &a_energyFormat, std::string const &a_dFormat ) const ;
};

/*
============================================================
============== NuclideGammaBranchStateInfo =================
============================================================
*/
class NuclideGammaBranchStateInfo {

    private:
        char m_state[16];
        double m_multiplicity;                              /**< The average multiplicity of photons emitted including the emission of for sub-levels. */
        double m_averageGammaEnergy;                        /**< The average energy of photons emitted including the emission of for sub-levels. */
        Vector<int> m_branches;

    public:
        LUPI_HOST_DEVICE NuclideGammaBranchStateInfo( );
        LUPI_HOST NuclideGammaBranchStateInfo( PoPI::NuclideGammaBranchStateInfo const &a_nuclideGammaBranchingInfo, 
                std::vector<NuclideGammaBranchInfo *> &a_nuclideGammaBranchInfos, std::map<std::string, int> &a_stateNamesToIndices );

        LUPI_HOST_DEVICE double multiplicity( ) const { return( m_multiplicity ); }                           /**< Returns the value of the **m_multiplicity** member. */
        LUPI_HOST_DEVICE double averageGammaEnergy( ) const { return( m_averageGammaEnergy ); }               /**< Returns the value of the **m_averageGammaEnergy** member. */
        LUPI_HOST_DEVICE Vector<int> const &branches( ) const { return( m_branches ); }                       /**< Returns the value of the **m_branches** member. */

        LUPI_HOST_DEVICE void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
        LUPI_HOST void print( ProtareSingle const *a_protareSingle, std::string const &a_indent, std::string const &a_iFormat,
                std::string const &a_energyFormat, std::string const &a_dFormat ) const ;
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
        bool m_isCompleteParticle;
        double m_mass;
        double m_excitationEnergy;
        TwoBodyOrder m_twoBodyOrder;
        int m_neutronIndex;
        int m_initialStateIndex;                                /**< If the product has branching photons, then this is the state index to start the branching. */
        Functions::Function1d *m_multiplicity;
        Distributions::Distribution *m_distribution;
// still need *m_averageEnergy *m_averageMomentum;

        OutputChannel *m_outputChannel;

    public:
        LUPI_HOST_DEVICE Product( );
        LUPI_HOST Product( GIDI::Product const *a_product, SetupInfo &a_setupInfo, Transporting::MC const &a_settings, GIDI::Transporting::Particles const &a_particles,
                bool a_isFission );
        LUPI_HOST Product( PoPI::Database const &a_pop, std::string const &a_ID, std::string const &a_label );
        LUPI_HOST_DEVICE ~Product( );

        LUPI_HOST String const &ID( ) const { return( m_ID ); }                                  /**< Returns the value of the **m_ID**. */
        LUPI_HOST_DEVICE int index( ) const { return( m_index ); }                               /**< Returns the value of the **m_index**. */
        LUPI_HOST_DEVICE int userParticleIndex( ) const { return( m_userParticleIndex ); }       /**< Returns the value of the **m_userParticleIndex**. */
        LUPI_HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex );
        LUPI_HOST_DEVICE String label( ) const { return( m_label ); }                            /**< Returns the value of the **m_label**. */
        LUPI_HOST_DEVICE bool isCompleteParticle( ) const { return( m_isCompleteParticle ); }     /**< Returns the value of the **m_isCompleteParticle**. */
        LUPI_HOST_DEVICE double mass( ) const { return( m_mass ); }                              /**< Returns the value of the **m_mass**. */
        LUPI_HOST_DEVICE double excitationEnergy( ) const { return( m_excitationEnergy ); }      /**< Returns the value of the **m_excitationEnergy**. */
        LUPI_HOST_DEVICE TwoBodyOrder twoBodyOrder( ) const { return( m_twoBodyOrder ); }      /**< Returns the value of the **m_twoBodyOrder**. */
        LUPI_HOST_DEVICE int neutronIndex( ) const { return( m_neutronIndex ); }                 /**< Returns the value of the **m_neutronIndex**. */
        LUPI_HOST_DEVICE double finalQ( double a_x1 ) const ;
        LUPI_HOST_DEVICE bool hasFission( ) const ;

// FIXME (1) see FIXME (1) in MC class.
        LUPI_HOST_DEVICE Functions::Function1d const *multiplicity( ) const { return( m_multiplicity ); }      /**< Returns the value of the **m_multiplicity**. */
        LUPI_HOST void setMultiplicity( Functions::Function1d *a_multiplicity ) { m_multiplicity = a_multiplicity; }
        LUPI_HOST_DEVICE double productAverageMultiplicity( int a_index, double a_projectileEnergy ) const ;
// FIXME (1) see FIXME (1) in MC class.
        LUPI_HOST_DEVICE Distributions::Distribution const *distribution( ) const { return( m_distribution ); }      /**< Returns the value of the **m_distribution**. */
        LUPI_HOST void distribution( Distributions::Distribution *a_distribution ) { m_distribution = a_distribution; }
// FIXME (1) see FIXME (1) in MC class.
        LUPI_HOST_DEVICE OutputChannel *outputChannel( ) { return( m_outputChannel ); }                  /**< Returns the value of the **m_outputChannel**. */

        LUPI_HOST_DEVICE void sampleProducts( Protare const *a_protare, double a_projectileEnergy, Sampling::Input &a_input, 
                double (*a_userrng)( void * ), void *a_rngState, Sampling::ProductHandler &a_products ) const ;
        LUPI_HOST_DEVICE void sampleFinalState( Protare const *a_protare, double a_projectileEnergy, Sampling::Input &a_input, 
                double (*a_userrng)( void * ), void *a_rngState, Sampling::ProductHandler &a_products ) const ;
        LUPI_HOST_DEVICE void angleBiasing( Reaction const *a_reaction, int a_pid, double a_temperature, double a_energy_in, double a_mu_lab, 
                double &a_probability, double &a_energy_out, double (*a_userrng)( void * ), void *a_rngState, double &a_cumulative_weight ) const ;
        LUPI_HOST_DEVICE void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
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
        LUPI_HOST_DEVICE DelayedNeutron( );
        LUPI_HOST DelayedNeutron( int a_index, GIDI::DelayedNeutron const *a_delayedNeutron, SetupInfo &a_setupInfo, Transporting::MC const &a_settings, GIDI::Transporting::Particles const &a_particles );
        LUPI_HOST_DEVICE ~DelayedNeutron( );

        LUPI_HOST_DEVICE int delayedNeutronIndex( ) const { return( m_delayedNeutronIndex ); }
        LUPI_HOST_DEVICE double rate( ) const { return( m_rate ); }
        LUPI_HOST_DEVICE Product const &product( ) const { return( m_product ); }
        LUPI_HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex );

        LUPI_HOST_DEVICE void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
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
        bool m_hasFinalStatePhotons;                /**< If **true**, *this* channel has a photon with finalState attribute. */
        int m_neutronIndex;

        Functions::Function1d_d1 *m_Q;             /**< The Q-function for the output channel. Note, this is currently always the *evaluated* form even when running with multi-group data. */
        Vector<Product *> m_products;
        Functions::Function1d *m_totalDelayedNeutronMultiplicity;
        Vector<DelayedNeutron *> m_delayedNeutrons;

    public:
        LUPI_HOST_DEVICE OutputChannel( );
        LUPI_HOST OutputChannel( GIDI::OutputChannel const *a_outputChannel, SetupInfo &a_setupInfo, Transporting::MC const &a_settings, GIDI::Transporting::Particles const &a_particles );
        LUPI_HOST_DEVICE ~OutputChannel( );

        LUPI_HOST_DEVICE Product *operator[]( MCGIDI_VectorSizeType a_index ) { return( m_products[a_index] ); }  /**< Returns a pointer to the product at index *a_index*. */

        LUPI_HOST_DEVICE int neutronIndex( ) const { return( m_neutronIndex ); }                 /**< Returns the value of the **m_neutronIndex**. */
        LUPI_HOST_DEVICE bool isTwoBody( ) const { return( m_channelType == ChannelType::twoBody ); }         /**< Returns true if output channel is two-body and false otherwise. */
        LUPI_HOST_DEVICE double finalQ( double a_x1 ) const ;
        LUPI_HOST_DEVICE bool isFission( ) const { return( m_isFission ); }                      /**< Returns the value of the **m_isFission**. */
        LUPI_HOST_DEVICE bool hasFission( ) const ;
// FIXME (1) see FIXME (1) in MC class.
        LUPI_HOST_DEVICE Functions::Function1d_d1 *Q( ) { return( m_Q ); }                       /**< Returns the pointer of the **m_Q** member. */

        LUPI_HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex );
// FIXME (1) see FIXME (1) in MC class.
        LUPI_HOST_DEVICE Vector<Product *> const &products( ) const { return( m_products ); }    /**< Returns the value of the **m_products**. */

        Vector<DelayedNeutron *> delayedNeutrons( ) const { return( m_delayedNeutrons ); }
        LUPI_HOST_DEVICE DelayedNeutron const *delayedNeutron( int a_index ) const { return( m_delayedNeutrons[a_index] ); }

        LUPI_HOST void moveProductsEtAlToReaction( std::vector<Product *> &a_products, Functions::Function1d **a_totalDelayedNeutronMultiplicity, 
                std::vector<DelayedNeutron *> &a_delayedNeutrons, std::vector<Functions::Function1d_d1 *> &a_Qs );
#ifdef MCGIDI_USE_OUTPUT_CHANNEL
        LUPI_HOST_DEVICE int referenceOrphanProductsToReaction( Vector<Product *> &a_associatedOrphanProducts, bool a_pushProducts ) const ;
#endif

        LUPI_HOST_DEVICE double productAverageMultiplicity( int a_index, double a_projectileEnergy ) const ;

        LUPI_HOST_DEVICE void sampleProducts( Protare const *a_protare, double a_projectileEnergy, Sampling::Input &a_input, 
                double (*a_userrng)( void * ), void *a_rngState, Sampling::ProductHandler &a_products ) const ;
        LUPI_HOST_DEVICE void angleBiasing( Reaction const *a_reaction, int a_pid, double a_temperature, double a_energy_in, double a_mu_lab, 
                double &a_probability, double &a_energy_out, double (*a_userrng)( void * ), void *a_rngState, double &a_cumulative_weight ) const ;
        LUPI_HOST_DEVICE void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
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
        int m_initialStateIndex;                            /**< If not -1, then reaction contains a branching gamma data with this index for the data in m_nuclideGammaBranchStateInfos member of its **ProtareSingle** instance. */
        bool m_hasFission;                                  /**< Is *true* if the reaction is a fission reaction and *false* otherwise. */
        double m_projectileMass;                            /**< The mass of the projectile. */
        double m_targetMass;                                /**< The mass of the target. */
        double m_crossSectionThreshold;                     /**< The threshold for the reaction. */
        double m_twoBodyThreshold;                          /**< This is the T_1 value needed to do two-body kinematics. */
        bool m_upscatterModelASupported;
        bool m_hasFinalStatePhotons;                        /**< If **true**, *this* reaction has a photon with finalState attribute. */
        Vector<double> m_upscatterModelACrossSection;       /**< The multi-group cross section to use for upscatter model A. */
        Vector<int> m_productIndices;                       /**< The list of all products *this* reaction can product. */
        Vector<int> m_userProductIndices;                   /**< The list of all products *this* reaction can product as user indices. */
        Vector<int> m_productMultiplicities;                /**< The list of all multiplicities for each product in *m_productIndices* . */
        Vector<int> m_productIndicesTransportable;          /**< The list of all transportabls products *this* reaction can product. */
        Vector<int> m_userProductIndicesTransportable;      /**< The list of all transportabls products *this* reaction can product as user indices. */

        Vector<Functions::Function1d_d1 *> m_Qs;            /**< A list of Q-functions that is used when the C macro MCGIDI_USE_OUTPUT is defined. */
        Vector<Product *> m_products;                       /**< A list of all transporting products directly or nested in m_outputChannel that is used instead of having m_outputChannel loop of all transporting products if the C macro MCGIDI_USE_OUTPUT_CHANNEL is not defined. */
        Functions::Function1d *m_totalDelayedNeutronMultiplicity;
        Vector<DelayedNeutron *> m_delayedNeutrons;         /**< A list of all delayedNeutrons that can be used instead of having m_outputChannel loop of all transporting products. For *m_products* for more details. */
                                                            /**< The total delayed neutron multiplicity used when the C macro MCGIDI_USE_OUTPUT is defined. */
#ifdef MCGIDI_USE_OUTPUT_CHANNEL
        OutputChannel *m_outputChannel;                     /**< The output channel for this reaction. Only used if the C macro MCGIDI_USE_OUTPUT is defined. */
#endif
        int m_associatedOrphanProductIndex;                 /**< The index in the Protare's m_orphanProducts member for the orphanProduct associated with this reaction. */
        Vector<Product *> m_associatedOrphanProducts;       /**< The list of products from the orphanProduct reaction. */ /* Do not delete entries as owned by orphanProduct reaction. */
// Still need m_availableEnergy and m_availableMomentum.

    public:
        LUPI_HOST_DEVICE Reaction( );
        LUPI_HOST Reaction( GIDI::Reaction const &a_reaction, SetupInfo &a_setupInfo, Transporting::MC const &a_settings, GIDI::Transporting::Particles const &a_particles,
                GIDI::Styles::TemperatureInfos const &a_temperatureInfos );
        LUPI_HOST_DEVICE ~Reaction( );

        inline LUPI_HOST_DEVICE void updateProtareSingleInfo( ProtareSingle *a_protareSingle, int a_reactionIndex ) {
                m_protareSingle = a_protareSingle;
                m_reactionIndex = a_reactionIndex;
        }
        LUPI_HOST_DEVICE ProtareSingle const *protareSingle( ) const { return( m_protareSingle ); }   /**< Returns the value of the **m_protareSingle**. */
        LUPI_HOST_DEVICE int reactionIndex( ) const { return( m_reactionIndex ); }       /**< Returns the value of the **m_reactionIndex**. */
        LUPI_HOST_DEVICE String const &label( ) const { return( m_label ); }             /**< Returns the value of the **m_label**. */
        LUPI_HOST_DEVICE int ENDF_MT( ) const { return( m_ENDF_MT ); }                   /**< Returns the value of the **m_ENDF_MT**. */
        LUPI_HOST_DEVICE int ENDL_C( ) const { return( m_ENDL_C ); }                     /**< Returns the value of the **m_ENDL_C**. */
        LUPI_HOST_DEVICE int ENDL_S( ) const { return( m_ENDL_S ); }                     /**< Returns the value of the **m_ENDL_S**. */
        LUPI_HOST_DEVICE int neutronIndex( ) const { return( m_neutronIndex ); }         /**< Returns the value of the **m_neutronIndex**. */
        LUPI_HOST_DEVICE int initialStateIndex( ) const { return( m_initialStateIndex ); }    /**< Returns the value of the **m_initialStateIndex** member. */
        LUPI_HOST_DEVICE double finalQ( double a_energy ) const ;
        LUPI_HOST_DEVICE bool hasFission( ) const { return( m_hasFission ); }            /**< Returns the value of the **m_hasFission**. */
        LUPI_HOST_DEVICE double projectileMass( ) const { return( m_projectileMass ); }  /**< Returns the value of the **m_projectileMass**. */
        LUPI_HOST_DEVICE double targetMass( ) const { return( m_targetMass ); }          /**< Returns the value of the **m_targetMass**. */
        LUPI_HOST_DEVICE double crossSectionThreshold( ) const { return( m_crossSectionThreshold ); }    /**< Returns the value of the **m_crossSectionThreshold**. */
        LUPI_HOST_DEVICE double twoBodyThreshold( ) const { return( m_twoBodyThreshold ); }              /**< Returns the value of the *m_twoBodyThreshold* member. */
        LUPI_HOST_DEVICE double crossSection( URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy ) const ;
        LUPI_HOST_DEVICE double crossSection( URR_protareInfos const &a_URR_protareInfos, double a_temperature, double a_energy ) const ;

        LUPI_HOST Vector<int> const &productIndices( ) const { return( m_productIndices ); }
        LUPI_HOST Vector<int> const &userProductIndices( ) const { return( m_userProductIndices ); }
        LUPI_HOST Product const *product( int a_index ) const { return( m_products[a_index] ); }
        LUPI_HOST int productMultiplicity( int a_index ) const ;
        LUPI_HOST int productMultiplicities( int a_index ) const {
                LUPI::deprecatedFunction( "MCGIDI::Reaction::productMultiplicities", "MCGIDI::Reaction::productMultiplicity", "" );
                return( productMultiplicity( a_index ) ); }                                 /**< This method is deprecated. Please use **productMultiplicity** instead. */
        LUPI_HOST_DEVICE double productAverageMultiplicity( int a_index, double a_projectileEnergy ) const ;
        LUPI_HOST Vector<int> const &productIndicesTransportable( ) const { return( m_productIndicesTransportable ); }
        LUPI_HOST Vector<int> const &userProductIndicesTransportable( ) const { return( m_userProductIndicesTransportable ); }

#ifdef MCGIDI_USE_OUTPUT_CHANNEL
        LUPI_HOST_DEVICE OutputChannel const *outputChannel( ) const { return( m_outputChannel ); }              /**< Returns the value of the **m_outputChannel**. */
#endif
        LUPI_HOST_DEVICE int associatedOrphanProductIndex( ) const { return( m_associatedOrphanProductIndex ); } /**< Returns the value of the **m_associatedOrphanProductIndex**. */
        LUPI_HOST_DEVICE void setAssociatedOrphanProductIndex( int a_associatedOrphanProductIndex ) { m_associatedOrphanProductIndex = a_associatedOrphanProductIndex; }
        LUPI_HOST_DEVICE void setAssociatedOrphanProduct( Reaction const *a_orphanProduct );
        LUPI_HOST_DEVICE void referenceOrphanProductsToReaction( Vector<Product *> &a_associatedOrphanProducts ) const ;
        LUPI_HOST_DEVICE bool upscatterModelASupported( ) const { return( m_upscatterModelASupported ); }
        LUPI_HOST_DEVICE Vector<double> const &upscatterModelACrossSection( ) const { return( m_upscatterModelACrossSection ); } 
                                                                                                            /**< Returns the value of the **m_upscatterModelACrossSection**. */

        LUPI_HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex );

        LUPI_HOST_DEVICE void sampleProducts( Protare const *a_protare, double a_projectileEnergy, Sampling::Input &a_input, 
                double (*a_userrng)( void * ), void *a_rngState, Sampling::ProductHandler &a_products, bool a_checkOrphanProducts = true ) const ;
        LUPI_HOST_DEVICE static void sampleNullProducts( Protare const &a_protare, double a_projectileEnergy, Sampling::Input &a_input, 
                double (*a_userrng)( void * ), void *a_rngState, Sampling::ProductHandler &a_products );
        LUPI_HOST_DEVICE double angleBiasing( int a_pid, double a_temperature, double a_energy_in, double a_mu_lab, double &a_energy_out, 
                double (*a_userrng)( void * ), void *a_rngState, double *a_cumulative_weight = nullptr, bool a_checkOrphanProducts = true ) const ;
        LUPI_HOST_DEVICE void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
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
        int m_electronIndex;                                /**< The PoPs index of the electron. */
        int m_userElectronIndex;                            /**< The electron particle index defined by the user. */
        String m_evaluation;                                /**< The evaluation string for the Protare. */
        GIDI::Frame m_projectileFrame;                      /**< The frame the projectile data are given in. */

        Vector<int> m_productIndices;                       /**< The list of all products *this* protare can product. */
        Vector<int> m_userProductIndices;                   /**< The list of all products *this* reaction can product as user indices. */
        Vector<int> m_productIndicesTransportable;          /**< The list of all transportabls products *this* protare can product. */
        Vector<int> m_userProductIndicesTransportable;      /**< The list of all transportabls products *this* reaction can product as user indices. */

        bool m_isTNSL_ProtareSingle;                        /**< If *this* is a ProtareSingle instance with TNSL data *true* and otherwise *false*. */

    public:
        LUPI_HOST_DEVICE Protare( ProtareType a_protareType );
        LUPI_HOST Protare( ProtareType a_protareType, GIDI::Protare const &a_protare, PoPI::Database const &a_pops, Transporting::MC const &a_settings );
        virtual LUPI_HOST_DEVICE ~Protare( );

        LUPI_HOST_DEVICE ProtareType protareType( ) const { return( m_protareType ); }                               /**< Returns the value of the **m_protareType** member. */    

        LUPI_HOST_DEVICE String const &projectileID( ) const { return( m_projectileID ); }                       /**< Returns the value of the **m_projectileID** member. */
        LUPI_HOST_DEVICE int projectileIndex( ) const { return( m_projectileIndex ); }                           /**< Returns the value of the **m_projectileIndex** member. */
        LUPI_HOST_DEVICE int projectileUserIndex( ) const { return( m_projectileUserIndex ); }                   /**< Returns the value of the **m_projectileUserIndex** member. */
        LUPI_HOST_DEVICE double projectileMass( ) const { return( m_projectileMass ); }                          /**< Returns the value of the **m_projectileMass** member. */
        LUPI_HOST_DEVICE double projectileExcitationEnergy( ) const { return( m_projectileExcitationEnergy ); }  /**< Returns the value of the **m_projectileExcitationEnergy** member. */

        LUPI_HOST_DEVICE String const &targetID( ) const { return( m_targetID ); }                               /**< Returns the value of the **m_targetID** member. */
        LUPI_HOST_DEVICE int targetIndex( ) const { return( m_targetIndex ); }                                   /**< Returns the value of the **m_targetIndex** member. */
        LUPI_HOST_DEVICE int targetUserIndex( ) const { return( m_targetUserIndex ); }                           /**< Returns the value of the **m_targetUserIndex** member. */
        LUPI_HOST_DEVICE double targetMass( ) const { return( m_targetMass ); }                                  /**< Returns the value of the **m_targetMass** member. */
        LUPI_HOST_DEVICE double targetExcitationEnergy( ) const { return( m_targetExcitationEnergy ); }          /**< Returns the value of the **m_targetExcitationEnergy** member. */

        LUPI_HOST_DEVICE int neutronIndex( ) const { return( m_neutronIndex ); }                                 /**< Returns the value of the **m_neutronIndex** member. */
        LUPI_HOST_DEVICE int userNeutronIndex( ) const { return( m_userNeutronIndex ); }                         /**< Returns the value of the **m_userNeutronIndex** member. */
        LUPI_HOST_DEVICE int photonIndex( ) const { return( m_photonIndex ); }                                   /**< Returns the value of the **m_photonIndex** member. */
        LUPI_HOST_DEVICE int userPhotonIndex( ) const { return( m_userPhotonIndex ); }                           /**< Returns the value of the **m_userPhotonIndex** member. */
        LUPI_HOST_DEVICE int electronIndex( ) const { return( m_electronIndex ); }                               /**< Returns the value of the **m_electronIndex** member. */
        LUPI_HOST_DEVICE int userElectronIndex( ) const { return( m_userElectronIndex ); }                       /**< Returns the value of the **m_userElectronIndex** member. */
        LUPI_HOST_DEVICE String evaluation( ) const { return( m_evaluation ); }                                  /**< Returns the value of the **m_evaluation** member. */
        LUPI_HOST GIDI::Frame projectileFrame( ) const { return( m_projectileFrame ); }                          /**< Returns the value of the **m_projectileFrame** member. */

        LUPI_HOST Vector<int> const &productIndices( bool a_transportablesOnly ) const ;
        LUPI_HOST void productIndices( std::set<int> const &a_indices, std::set<int> const &a_transportableIndices );
        LUPI_HOST Vector<int> const &userProductIndices( bool a_transportablesOnly ) const ;
        LUPI_HOST void setUserParticleIndex( int a_particleIndex, int a_userParticleIndex );

        LUPI_HOST_DEVICE bool isTNSL_ProtareSingle( ) const { return( m_isTNSL_ProtareSingle ); }                /**< Returns the value of the **m_isTNSL_ProtareSingle** member. */
        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE MCGIDI_VectorSizeType numberOfProtares( ) const MCGIDI_TRUE_VIRTUAL;                            /**< Returns the number of protares contained in *this*. */
        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE ProtareSingle const *protare( MCGIDI_VectorSizeType a_index ) const MCGIDI_TRUE_VIRTUAL;        /**< Returns the **a_index** - 1 Protare contained in *this*. */
        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE ProtareSingle       *protare( MCGIDI_VectorSizeType a_index )       MCGIDI_TRUE_VIRTUAL;        /**< Returns the **a_index** - 1 Protare contained in *this*. */
        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE ProtareSingle const *protareWithReaction( int a_index ) const MCGIDI_TRUE_VIRTUAL;              /**< Returns the *ProtareSingle* that contains the (*a_index* - 1) reaction. */

        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE double minimumEnergy( ) const MCGIDI_TRUE_VIRTUAL;                                              /**< Returns the minimum cross section domain. */
        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE double maximumEnergy( ) const MCGIDI_TRUE_VIRTUAL ;                                             /**< Returns the maximum cross section domain. */
        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE Vector<double> temperatures( MCGIDI_VectorSizeType a_index = 0 ) const MCGIDI_TRUE_VIRTUAL ;    /**< Returns the list of temperatures for the requested ProtareSingle. */

        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST Vector<double> const &projectileMultiGroupBoundaries( ) const MCGIDI_TRUE_VIRTUAL;
        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST Vector<double> const &projectileMultiGroupBoundariesCollapsed( ) const MCGIDI_TRUE_VIRTUAL;

        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE std::size_t numberOfReactions( ) const MCGIDI_TRUE_VIRTUAL;
        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE Reaction const *reaction( int a_index ) const MCGIDI_TRUE_VIRTUAL;
        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE std::size_t numberOfOrphanProducts( ) const MCGIDI_TRUE_VIRTUAL;
        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE Reaction const *orphanProduct( int a_index ) const MCGIDI_TRUE_VIRTUAL;

        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE bool hasFission( ) const MCGIDI_TRUE_VIRTUAL;

        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE int URR_index( ) const MCGIDI_TRUE_VIRTUAL;
        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE bool hasURR_probabilityTables( ) const MCGIDI_TRUE_VIRTUAL;
        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE double URR_domainMin( ) const MCGIDI_TRUE_VIRTUAL;
        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE double URR_domainMax( ) const MCGIDI_TRUE_VIRTUAL;
        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE bool reactionHasURR_probabilityTables( int a_index ) const MCGIDI_TRUE_VIRTUAL ;

        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE double threshold( MCGIDI_VectorSizeType a_index ) const MCGIDI_TRUE_VIRTUAL;

        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE double crossSection(                              URR_protareInfos const &a_URR_protareInfos,
                int a_hashIndex, double a_temperature, double a_energy, bool a_sampling = false ) const MCGIDI_TRUE_VIRTUAL;
        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE void crossSectionVector( double a_temperature, double a_userFactor, int a_numberAllocated,
                double *a_crossSectionVector ) const MCGIDI_TRUE_VIRTUAL;
        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE double reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_hashIndex,
                double a_temperature, double a_energy, bool a_sampling = false ) const MCGIDI_TRUE_VIRTUAL;
        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE double reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos,
                double a_temperature, double a_energy ) const MCGIDI_TRUE_VIRTUAL;
        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE int sampleReaction(                               URR_protareInfos const &a_URR_protareInfos, int a_hashIndex,
                double a_temperature, double a_energy, double a_crossSection, double (*a_userrng)( void * ), void *a_rngState ) const MCGIDI_TRUE_VIRTUAL;

        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE double depositionEnergy(   int a_hashIndex, double a_temperature, double a_energy ) const MCGIDI_TRUE_VIRTUAL;
        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE double depositionMomentum( int a_hashIndex, double a_temperature, double a_energy ) const MCGIDI_TRUE_VIRTUAL;
        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE double productionEnergy(   int a_hashIndex, double a_temperature, double a_energy ) const MCGIDI_TRUE_VIRTUAL;
        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE double gain(               int a_hashIndex, double a_temperature, double a_energy, int a_particleIndex ) const MCGIDI_TRUE_VIRTUAL;

        MCGIDI_VIRTUAL_FUNCTION LUPI_HOST_DEVICE Vector<double> const &upscatterModelAGroupVelocities( ) const MCGIDI_TRUE_VIRTUAL;

        LUPI_HOST_DEVICE void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
        LUPI_HOST_DEVICE void serialize2( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
        LUPI_HOST_DEVICE void serializeCommon( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
        LUPI_HOST_DEVICE long sizeOf( ) const ;
        LUPI_HOST_DEVICE long memorySize( );
        LUPI_HOST_DEVICE void incrementMemorySize( long &a_totalMemory, long &a_sharedMemory );
};

/*
============================================================
====================== ProtareSingle =======================
============================================================
*/
class ProtareSingle : public Protare {

    private:
        String m_interaction;                                                       /**< The protare's interaction string. */
        int m_URR_index;                                                            /**< The index of the protare in the URR_protareInfos list. If negative, not in list. */
        bool m_hasURR_probabilityTables;                                            /**< *true* if URR probability tables present and *false* otherwise. */
        double m_URR_domainMin;                                                     /**< If URR probability tables present this is the minimum of the projectile energy domain for the tables. */
        double m_URR_domainMax;                                                     /**< If URR probability tables present this is the maximum of the projectile energy domain for the tables. */
        Vector<double> m_projectileMultiGroupBoundaries;                            /**< The multi-group boundaries for the projectile. Only used if m_crossSectionLookupMode and/or m_other1dDataLookupMode is multiGroup. */
        Vector<double> m_projectileMultiGroupBoundariesCollapsed;                   /**< The collased, multi-group boundaries for the projectile. Only used if m_crossSectionLookupMode and/or m_other1dDataLookupMode is multiGroup. */ 
        Vector<double> m_upscatterModelAGroupVelocities;                            /**< The speed of the projectile at each multi-group boundary. Need by upscatter model A. */

        Vector<Reaction *> m_reactions;                                             /**< The list of reactions. */
        Vector<Reaction *> m_orphanProducts;                                        /**< The list of orphan products. */
        bool m_isPhotoAtomic;                                                       /**< *true* if photo-atomic protare and false otherwise. */
        bool m_continuousEnergy;                                                    /**< If *true*, protare has continuous energy cross sections; otherwise, multi-group cross sections. */
        bool m_fixedGrid;                                                           /**< If *true*, continuous energy cross sections are fixed grid. */
        HeatedCrossSectionsContinuousEnergy m_heatedCrossSections;                  /**< Stores all cross section data for total and all reactions for all requested temperatures. */
        HeatedCrossSectionsMultiGroup m_heatedMultigroupCrossSections;              /**< Stores all multi-group cross section data for total and all reactions for all requested temperatures. */

        Vector<NuclideGammaBranchStateInfo *> m_nuclideGammaBranchStateInfos;       /**< List of all gamma branches for a nuclide. */
        Vector<NuclideGammaBranchInfo *> m_branches;                                /**< Condensed data on a nuclide's gamma branch including the gamma's energy, probability and the nuclide's residual state. */

        LUPI_HOST void setupNuclideGammaBranchStateInfos( SetupInfo &a_setupInfo, GIDI::ProtareSingle const &a_protare );

    public:
        LUPI_HOST_DEVICE ProtareSingle( );
        LUPI_HOST ProtareSingle( LUPI::StatusMessageReporting &a_smr, GIDI::ProtareSingle const &a_protare, PoPI::Database const &a_pops, Transporting::MC &a_settings, 
                GIDI::Transporting::Particles const &a_particles, DomainHash const &a_domainHash, GIDI::Styles::TemperatureInfos const &a_temperatureInfos,
                std::set<int> const &a_reactionsToExclude, int a_reactionsToExcludeOffset = 0, bool a_allowFixedGrid = true );
        LUPI_HOST_DEVICE ~ProtareSingle( );

        LUPI_HOST_DEVICE bool isPhotoAtomic( ) const { return( m_isPhotoAtomic ); }
        LUPI_HOST_DEVICE bool continuousEnergy( ) const { return( m_continuousEnergy ); }
        LUPI_HOST_DEVICE bool fixedGrid( ) const { return( m_fixedGrid ); }
        LUPI_HOST_DEVICE HeatedCrossSectionsContinuousEnergy const &heatedCrossSections( ) const { return( m_heatedCrossSections ); }  /**< Returns a reference to the **m_heatedCrossSections** member. */
        LUPI_HOST_DEVICE HeatedCrossSectionsContinuousEnergy &heatedCrossSections( ) { return( m_heatedCrossSections ); }              /**< Returns a reference to the **m_heatedCrossSections** member. */
        LUPI_HOST_DEVICE HeatedCrossSectionsMultiGroup const &heatedMultigroupCrossSections( ) const { return( m_heatedMultigroupCrossSections ); } /**< Returns a reference to the **m_heatedMultigroupCrossSections** member. */

        LUPI_HOST_DEVICE const Vector<NuclideGammaBranchStateInfo *> &nuclideGammaBranchStateInfos( ) const { return( m_nuclideGammaBranchStateInfos ); }
                                                                                    /**< Returns a reference to the **m_nuclideGammaBranchStateInfos** member. */
        LUPI_HOST_DEVICE const Vector<NuclideGammaBranchInfo *> &branches( ) const { return( m_branches ); }
                                                                                    /**< Returns a reference to the **m_branches** member. */

// FIXME (1) see FIXME (1) in MC class.
        LUPI_HOST_DEVICE Vector<Reaction *> const &reactions( ) const { return( m_reactions ); }                 /**< Returns the value of the **m_reactions** member. */
// FIXME (1) see FIXME (1) in MC class.
        LUPI_HOST_DEVICE Vector<Reaction *> const &orphanProducts( ) const { return( m_orphanProducts ); }       /**< Returns the value of the **m_orphanProducts** member. */

        LUPI_HOST_DEVICE void sampleBranchingGammas( Sampling::Input &a_input, double a_projectileEnergy, int a_initialStateIndex, 
                double (*a_userrng)( void * ), void *a_rngState, Sampling::ProductHandler &a_products ) const ;
        LUPI_HOST void setUserParticleIndex2( int a_particleIndex, int a_userParticleIndex );

// The rest are virtual methods defined in the Protare class.

        LUPI_HOST_DEVICE MCGIDI_VectorSizeType numberOfProtares( ) const { return( 1 ); }                        /**< Returns the number of protares contained in *this*. */
        LUPI_HOST_DEVICE ProtareSingle const *protare( MCGIDI_VectorSizeType a_index ) const ;
        LUPI_HOST_DEVICE ProtareSingle       *protare( MCGIDI_VectorSizeType a_index );
        LUPI_HOST_DEVICE ProtareSingle const *protareWithReaction( int a_index ) const ;

        LUPI_HOST_DEVICE double minimumEnergy( ) const { 
            if( m_continuousEnergy ) return( m_heatedCrossSections.minimumEnergy( ) );
            return( m_heatedMultigroupCrossSections.minimumEnergy( ) ); }                                   /**< Returns the minimum cross section domain. */
        LUPI_HOST_DEVICE double maximumEnergy( ) const { 
            if( m_continuousEnergy ) return( m_heatedCrossSections.maximumEnergy( ) );
            return( m_heatedMultigroupCrossSections.maximumEnergy( ) ); }                                   /**< Returns the maximum cross section domain. */
        LUPI_HOST_DEVICE Vector<double> temperatures( MCGIDI_VectorSizeType a_index = 0 ) const ;

        LUPI_HOST Vector<double> const &projectileMultiGroupBoundaries( ) const { return( m_projectileMultiGroupBoundaries ); }
                                                                                                            /**< Returns the value of the **m_projectileMultiGroupBoundaries** member. */
        LUPI_HOST Vector<double> const &projectileMultiGroupBoundariesCollapsed( ) const { return( m_projectileMultiGroupBoundariesCollapsed ); }
                                                                                                            /**< Returns the value of the **m_projectileMultiGroupBoundariesCollapsed** member. */

        LUPI_HOST_DEVICE std::size_t numberOfReactions( ) const { return( m_reactions.size( ) ); }                       /**< Returns the number of reactions of *this*. */
        LUPI_HOST_DEVICE Reaction const *reaction( int a_index ) const { return( m_reactions[a_index] ); }               /**< Returns the (a_index-1)^th reaction of *this*. */
        LUPI_HOST_DEVICE std::size_t numberOfOrphanProducts( ) const { return( m_orphanProducts.size( ) ); }             /**< Returns the number of orphan products of *this*. */
        LUPI_HOST_DEVICE Reaction const *orphanProduct( int a_index ) const { return( m_orphanProducts[a_index] ); }     /**< Returns the (a_index-1)^th orphan product of *this*. */

        LUPI_HOST_DEVICE bool hasFission( ) const ;
        LUPI_HOST_DEVICE String interaction( ) const { return( m_interaction ); }

        LUPI_HOST_DEVICE int URR_index( ) const { return( m_URR_index ); }
        LUPI_HOST_DEVICE void URR_index( int a_URR_index ) { m_URR_index = a_URR_index; }
        LUPI_HOST_DEVICE bool inURR( double a_energy ) const ;
        LUPI_HOST_DEVICE bool hasURR_probabilityTables( ) const { return( m_hasURR_probabilityTables ); }
        LUPI_HOST_DEVICE double URR_domainMin( ) const { return( m_URR_domainMin ); }
        LUPI_HOST_DEVICE double URR_domainMax( ) const { return( m_URR_domainMax ); }
        LUPI_HOST_DEVICE bool reactionHasURR_probabilityTables( int a_index ) const { return( m_heatedCrossSections.reactionHasURR_probabilityTables( a_index ) ); }

        LUPI_HOST_DEVICE double threshold( MCGIDI_VectorSizeType a_index ) const {
            if( m_continuousEnergy ) return( m_heatedCrossSections.threshold( a_index ) );
            return( m_heatedMultigroupCrossSections.threshold( a_index ) ); }                                       /**< Returns the threshold for the reaction at index *a_index*. */

        LUPI_HOST_DEVICE double crossSection(                              URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, bool a_sampling = false ) const ;
        LUPI_HOST_DEVICE void crossSectionVector( double a_temperature, double a_userFactor, int a_numberAllocated, double *a_crossSectionVector ) const ;
        LUPI_HOST_DEVICE double reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, bool a_sampling = false ) const ;
        LUPI_HOST_DEVICE double reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos,                  double a_temperature, double a_energy ) const ;
        LUPI_HOST_DEVICE int sampleReaction(                               URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, double a_crossSection, double (*a_userrng)( void * ), void *a_rngState ) const ;

        LUPI_HOST_DEVICE double depositionEnergy(   int a_hashIndex, double a_temperature, double a_energy ) const ;
        LUPI_HOST_DEVICE double depositionMomentum( int a_hashIndex, double a_temperature, double a_energy ) const ;
        LUPI_HOST_DEVICE double productionEnergy(   int a_hashIndex, double a_temperature, double a_energy ) const ;
        LUPI_HOST_DEVICE double gain(               int a_hashIndex, double a_temperature, double a_energy, int a_particleIndex ) const ;

        LUPI_HOST_DEVICE Vector<double> const &upscatterModelAGroupVelocities( ) const { return( m_upscatterModelAGroupVelocities ); }   /**< Returns a reference to the **m_upscatterModelAGroupVelocities** member. */

        LUPI_HOST_DEVICE void serialize2( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
        LUPI_HOST_DEVICE long sizeOf2( ) const { return sizeof(*this); }
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
        LUPI_HOST_DEVICE ProtareComposite( );
        LUPI_HOST ProtareComposite( LUPI::StatusMessageReporting &a_smr, GIDI::ProtareComposite const &a_protare, PoPI::Database const &a_pops, Transporting::MC &a_settings, 
                GIDI::Transporting::Particles const &a_particles, DomainHash const &a_domainHash, GIDI::Styles::TemperatureInfos const &a_temperatureInfos,
                std::set<int> const &a_reactionsToExclude, int a_reactionsToExcludeOffset = 0, bool a_allowFixedGrid = true );
        LUPI_HOST_DEVICE ~ProtareComposite( );

        Vector<ProtareSingle *> protares( ) const { return( m_protares ); }       /**< Returns the value of the **m_protares** member. */
        LUPI_HOST void setUserParticleIndex2( int a_particleIndex, int a_userParticleIndex );

// The rest are virtual methods defined in the Protare class.

        LUPI_HOST_DEVICE MCGIDI_VectorSizeType numberOfProtares( ) const { return( m_protares.size( ) ); }     /**< Returns the number of protares contained in *this*. */
        LUPI_HOST_DEVICE ProtareSingle const *protare( MCGIDI_VectorSizeType a_index ) const ;
        LUPI_HOST_DEVICE ProtareSingle       *protare( MCGIDI_VectorSizeType a_index );
        LUPI_HOST_DEVICE ProtareSingle const *protareWithReaction( int a_index ) const ;

        LUPI_HOST_DEVICE double minimumEnergy( ) const { return( m_minimumEnergy ); }     /**< Returns the value of the **m_minimumEnergy** member. */
        LUPI_HOST_DEVICE double maximumEnergy( ) const { return( m_maximumEnergy ); }     /**< Returns the value of the **m_maximumEnergy** member. */
        LUPI_HOST_DEVICE Vector<double> temperatures( MCGIDI_VectorSizeType a_index = 0 ) const ;

        LUPI_HOST Vector<double> const &projectileMultiGroupBoundaries( ) const { return( m_protares[0]->projectileMultiGroupBoundaries( ) ); }    
                                                                            /**< Returns the value of the **m_projectileMultiGroupBoundaries** member. */
        LUPI_HOST Vector<double> const &projectileMultiGroupBoundariesCollapsed( ) const { return( m_protares[0]->projectileMultiGroupBoundariesCollapsed( ) ); }
                                                                            /**< Returns the value of the **m_projectileMultiGroupBoundariesCollapsed** member. */

        LUPI_HOST_DEVICE std::size_t numberOfReactions( ) const { return( m_numberOfReactions ); }
                                                                            /**< Returns the value of the **m_numberOfReactions** member. */
        LUPI_HOST_DEVICE Reaction const *reaction( int a_index ) const ;
        LUPI_HOST_DEVICE std::size_t numberOfOrphanProducts( ) const { return( m_numberOfOrphanProducts ); }
                                                                            /**< Returns the value of the **m_numberOfOrphanProducts** member. */
        LUPI_HOST_DEVICE Reaction const *orphanProduct( int a_index ) const ;

        LUPI_HOST_DEVICE bool hasFission( ) const ;

        LUPI_HOST_DEVICE int URR_index( ) const { return( -1 ); }
        LUPI_HOST_DEVICE bool hasURR_probabilityTables( ) const ;
        LUPI_HOST_DEVICE double URR_domainMin( ) const ;
        LUPI_HOST_DEVICE double URR_domainMax( ) const ;
        LUPI_HOST_DEVICE bool reactionHasURR_probabilityTables( int a_index ) const ;

        LUPI_HOST_DEVICE double threshold( MCGIDI_VectorSizeType a_index ) const ;

        LUPI_HOST_DEVICE double crossSection(                              URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, bool a_sampling = false ) const ;
        LUPI_HOST_DEVICE void crossSectionVector( double a_temperature, double a_userFactor, int a_numberAllocated, double *a_crossSectionVector ) const ;
        LUPI_HOST_DEVICE double reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, bool a_sampling = false ) const ;
        LUPI_HOST_DEVICE double reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos,                  double a_temperature, double a_energy ) const ;
        LUPI_HOST_DEVICE int sampleReaction(                               URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, double a_crossSection, double (*a_userrng)( void * ), void *a_rngState ) const ;

        LUPI_HOST_DEVICE double depositionEnergy(   int a_hashIndex, double a_temperature, double a_energy ) const ;
        LUPI_HOST_DEVICE double depositionMomentum( int a_hashIndex, double a_temperature, double a_energy ) const ;
        LUPI_HOST_DEVICE double productionEnergy(   int a_hashIndex, double a_temperature, double a_energy ) const ;
        LUPI_HOST_DEVICE double gain(               int a_hashIndex, double a_temperature, double a_energy, int a_particleIndex ) const ;

        LUPI_HOST_DEVICE Vector<double> const &upscatterModelAGroupVelocities( ) const { return( m_protares[0]->upscatterModelAGroupVelocities( ) ); }
                                                                            /**< Returns a reference to the **m_upscatterModelAGroupVelocities** member. */

        LUPI_HOST_DEVICE void serialize2( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
        LUPI_HOST_DEVICE long sizeOf2( ) const { return sizeof(*this); }
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
        LUPI_HOST_DEVICE ProtareTNSL( );
        LUPI_HOST ProtareTNSL( LUPI::StatusMessageReporting &a_smr, GIDI::ProtareTNSL const &a_protare, PoPI::Database const &a_pops, Transporting::MC &a_settings, 
                GIDI::Transporting::Particles const &a_particles, DomainHash const &a_domainHash, GIDI::Styles::TemperatureInfos const &a_temperatureInfos,
                std::set<int> const &a_reactionsToExclude, int a_reactionsToExcludeOffset = 0, bool a_allowFixedGrid = true );
        LUPI_HOST_DEVICE ~ProtareTNSL( );

        LUPI_HOST_DEVICE ProtareSingle const *protareWithElastic( ) const { return( m_protareWithElastic ); }        /**< Returns the **m_protareWithElastic** member. */
        LUPI_HOST_DEVICE ProtareSingle const *TNSL( ) const { return( m_TNSL ); }                                    /**< Returns the **m_TNSL** member. */
        LUPI_HOST_DEVICE ProtareSingle const *protareWithoutElastic( ) const { return( m_protareWithoutElastic ); }  /**< Returns the **m_protareWithoutElastic** member. */

        LUPI_HOST_DEVICE double TNSL_maximumEnergy( ) const { return( m_TNSL_maximumEnergy ); }
        LUPI_HOST_DEVICE double TNSL_maximumTemperature( ) const { return( m_TNSL_maximumTemperature ); }
        LUPI_HOST void setUserParticleIndex2( int a_particleIndex, int a_userParticleIndex );

// The rest are virtual methods defined in the Protare class.

        LUPI_HOST_DEVICE MCGIDI_VectorSizeType numberOfProtares( ) const { return( 2 ); }  /**< Always Returns 2. */
        LUPI_HOST_DEVICE ProtareSingle const *protare( MCGIDI_VectorSizeType a_index ) const ;
        LUPI_HOST_DEVICE ProtareSingle       *protare( MCGIDI_VectorSizeType a_index );
        LUPI_HOST_DEVICE ProtareSingle const *protareWithReaction( int a_index ) const ;

        LUPI_HOST_DEVICE double minimumEnergy( ) const { return( m_protareWithElastic->minimumEnergy( ) ); }   /**< Returns the minimum cross section domain. */
        LUPI_HOST_DEVICE double maximumEnergy( ) const { return( m_protareWithElastic->maximumEnergy( ) ); }   /**< Returns the maximum cross section domain. */
        LUPI_HOST_DEVICE Vector<double> temperatures( MCGIDI_VectorSizeType a_index = 0 ) const ;

        LUPI_HOST Vector<double> const &projectileMultiGroupBoundaries( ) const { return( m_protareWithElastic->projectileMultiGroupBoundaries( ) ); }
                                                                            /**< Returns the value of the **m_projectileMultiGroupBoundaries** member. */
        LUPI_HOST Vector<double> const &projectileMultiGroupBoundariesCollapsed( ) const { return( m_protareWithElastic->projectileMultiGroupBoundariesCollapsed( ) ); }
                                                                            /**< Returns the value of the **m_projectileMultiGroupBoundariesCollapsed** member. */

        LUPI_HOST_DEVICE std::size_t numberOfReactions( ) const { return( m_TNSL->numberOfReactions( ) + m_protareWithElastic->numberOfReactions( ) ); }
        LUPI_HOST_DEVICE Reaction const *reaction( int a_index ) const ;
        LUPI_HOST_DEVICE std::size_t numberOfOrphanProducts( ) const { return( m_protareWithElastic->numberOfOrphanProducts( ) ); }
                                                                            /**< Returns the number of orphan products in the normal ProtareSingle. */
        LUPI_HOST_DEVICE Reaction const *orphanProduct( int a_index ) const { return( m_protareWithElastic->orphanProduct( a_index ) ); }
                                                                            /**< Returns the (a_index - 1 )^th orphan product in the normal ProtareSingle. */

        LUPI_HOST_DEVICE bool hasFission( ) const { return( m_protareWithElastic->hasFission( ) ); }    /* Returns the normal ProtareSingle's hasFission value. */

        LUPI_HOST_DEVICE int URR_index( ) const { return( -1 ); }
        LUPI_HOST_DEVICE bool hasURR_probabilityTables( ) const { return( m_protareWithElastic->hasURR_probabilityTables( ) ); }
        LUPI_HOST_DEVICE double URR_domainMin( ) const { return( m_protareWithElastic->URR_domainMin( ) ); }
        LUPI_HOST_DEVICE double URR_domainMax( ) const { return( m_protareWithElastic->URR_domainMax( ) ); }
        LUPI_HOST_DEVICE bool reactionHasURR_probabilityTables( int a_index ) const ;

        LUPI_HOST_DEVICE double threshold( MCGIDI_VectorSizeType a_index ) const ;

        LUPI_HOST_DEVICE double crossSection(                              URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, bool a_sampling = false ) const ;
        LUPI_HOST_DEVICE void crossSectionVector( double a_temperature, double a_userFactor, int a_numberAllocated, double *a_crossSectionVector ) const ;
        LUPI_HOST_DEVICE double reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, bool a_sampling = false ) const ;
        LUPI_HOST_DEVICE double reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos,                  double a_temperature, double a_energy ) const ;
        LUPI_HOST_DEVICE int sampleReaction(                               URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, double a_crossSection, double (*a_userrng)( void * ), void *a_rngState ) const ;

        LUPI_HOST_DEVICE double depositionEnergy(   int a_hashIndex, double a_temperature, double a_energy ) const ;
        LUPI_HOST_DEVICE double depositionMomentum( int a_hashIndex, double a_temperature, double a_energy ) const ;
        LUPI_HOST_DEVICE double productionEnergy(   int a_hashIndex, double a_temperature, double a_energy ) const ;
        LUPI_HOST_DEVICE double gain(               int a_hashIndex, double a_temperature, double a_energy, int a_particleIndex ) const ;

        LUPI_HOST_DEVICE Vector<double> const &upscatterModelAGroupVelocities( ) const { return( m_protareWithElastic->upscatterModelAGroupVelocities( ) ); }
                                                                            /**< Returns a reference to the **m_upscatterModelAGroupVelocities** member. */

        LUPI_HOST_DEVICE void serialize2( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
        LUPI_HOST_DEVICE long sizeOf2( ) const { return sizeof(*this); }
};

/*
============================================================
=========================== Others =========================
============================================================
*/
LUPI_HOST Protare *protareFromGIDIProtare( LUPI::StatusMessageReporting &a_smr, GIDI::Protare const &a_protare, PoPI::Database const &a_pops, Transporting::MC &a_settings, GIDI::Transporting::Particles const &a_particles,
                DomainHash const &a_domainHash, GIDI::Styles::TemperatureInfos const &a_temperatureInfos, std::set<int> const &a_reactionsToExclude,
                int a_reactionsToExcludeOffset = 0, bool a_allowFixedGrid = true );
LUPI_HOST Vector<double> GIDI_VectorDoublesToMCGIDI_VectorDoubles( GIDI::Vector a_vector );
LUPI_HOST void addVectorItemsToSet( Vector<int> const &a_productIndicesFrom, std::set<int> &a_productIndicesTo );

LUPI_HOST_DEVICE double sampleBetaFromMaxwellian( double (*a_userrng)( void * ), void *a_rngState );
LUPI_HOST_DEVICE bool sampleTargetBetaForUpscatterModelA( Protare const *a_protare, double a_projectileEnergy, Sampling::Input &a_input,
                double (*a_userrng)( void * ), void *a_rngState );
LUPI_HOST_DEVICE void upScatterModelABoostParticle( Sampling::Input &a_input, double (*a_userrng)( void * ), void *a_rngState, Sampling::Product &a_product );
LUPI_HOST_DEVICE void MCGIDI_sampleKleinNishina( double a_k1, double (*a_userrng)( void * ), void *a_rngState, double *a_energyOut, double *a_mu );
LUPI_HOST_DEVICE int distributionTypeToInt( Distributions::Type a_type );
LUPI_HOST_DEVICE Distributions::Type intToDistributionType( int a_type );
LUPI_HOST_DEVICE void serializeProducts( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode, Vector<Product *> &a_products );
LUPI_HOST_DEVICE void serializeDelayedNeutrons( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode, Vector<DelayedNeutron *> &a_delayedNeutrons );
LUPI_HOST_DEVICE void serializeQs( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode, Vector<Functions::Function1d_d1 *> &a_Qs );

LUPI_HOST void convertACE_URR_probabilityTablesFromGIDI( GIDI::ProtareSingle const &a_protare, Transporting::MC &a_settings, SetupInfo &a_setupInfo );
LUPI_HOST_DEVICE Transporting::URR_mode serializeURR_mode( Transporting::URR_mode a_URR_mode, LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
LUPI_HOST_DEVICE ACE_URR_probabilityTables *serializeACE_URR_probabilityTables( ACE_URR_probabilityTables *a_ACE_URR_probabilityTables,
                LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );

LUPI_HOST_DEVICE Distributions::Distribution *serializeDistribution( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode, 
                Distributions::Distribution *a_distribution );

}           // End of namespace MCGIDI.

#endif      // End of MCGIDI_hpp_included

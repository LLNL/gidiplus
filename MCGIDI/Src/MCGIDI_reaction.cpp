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

/*! \class Reaction 
 * Class representing a **GNDS** <**reaction**> node with only data needed for Monte Carlo transport.
 */

/* *********************************************************************************************************//**
 * Default constructor used when broadcasting a Reaction as needed by MPI or GPUs.
 ***********************************************************************************************************/

LUPI_HOST_DEVICE Reaction::Reaction( ) :
        m_protareSingle( nullptr ),
        m_reactionIndex( -1 ),
        m_label( ),
        m_ENDF_MT( 0 ),
        m_ENDL_C( 0 ),
        m_ENDL_S( 0 ),
        m_neutronIndex( 0 ),
        m_initialStateIndex( -1 ),
        m_hasFission( false ),
        m_projectileMass( 0.0 ),
        m_targetMass( 0.0 ),
        m_crossSectionThreshold( 0.0 ),
        m_twoBodyThreshold( 0.0 ),
        m_upscatterModelASupported( false ),
        m_hasFinalStatePhotons( false ),
#ifdef MCGIDI_USE_OUTPUT_CHANNEL
        m_outputChannel( nullptr ),
#endif
        m_associatedOrphanProductIndex( -1 ) {

}

/* *********************************************************************************************************//**
 * @param a_reaction            [in]    The GIDI::Reaction whose data is to be used to construct *this*.
 * @param a_setupInfo           [in]    Used internally when constructing a Protare to pass information to other constructors.
 * @param a_settings            [in]    Used to pass user options to the *this* to instruct it which data are desired.
 * @param a_particles           [in]    List of transporting particles and their information (e.g., multi-group boundaries and fluxes).
 * @param a_temperatureInfos    [in]    The list of temperature data to extract from *a_protare*.
 ***********************************************************************************************************/

LUPI_HOST Reaction::Reaction( GIDI::Reaction const &a_reaction, SetupInfo &a_setupInfo, Transporting::MC const &a_settings, GIDI::Transporting::Particles const &a_particles,
                GIDI::Styles::TemperatureInfos const &a_temperatureInfos ) :
        m_protareSingle( nullptr ),
        m_reactionIndex( -1 ),
        m_label( a_reaction.label( ).c_str( ) ),
        m_ENDF_MT( a_reaction.ENDF_MT( ) ),
        m_ENDL_C( a_reaction.ENDL_C( ) ),
        m_ENDL_S( a_reaction.ENDL_S( ) ),
        m_neutronIndex( a_settings.neutronIndex( ) ),
        m_initialStateIndex( -1 ),
        m_hasFission( a_reaction.hasFission( ) ),
        m_projectileMass( a_setupInfo.m_protare.projectileMass( ) ),
        m_targetMass( a_setupInfo.m_protare.targetMass( ) ),
        m_crossSectionThreshold( a_reaction.crossSectionThreshold( ) ),
        m_twoBodyThreshold( a_reaction.twoBodyThreshold( ) ),
        m_upscatterModelASupported( ( a_setupInfo.m_protare.projectileIndex( ) != a_setupInfo.m_protare.photonIndex( ) ) &&
                                    ( a_setupInfo.m_protare.projectileIndex( ) != a_setupInfo.m_protare.electronIndex( ) ) &&
                                    ( a_setupInfo.m_reactionType == Transporting::Reaction::Type::Reactions ) ),
        m_associatedOrphanProductIndex( -1 ) {

    a_setupInfo.m_hasFinalStatePhotons = false;
#ifndef MCGIDI_USE_OUTPUT_CHANNEL
    OutputChannel *m_outputChannel;
#endif
    m_outputChannel = new OutputChannel( a_reaction.outputChannel( ), a_setupInfo, a_settings, a_particles );

    std::set<std::string> product_ids;

    a_reaction.productIDs( product_ids, a_particles, false );
    m_productIndices.reserve( product_ids.size( ) );
    m_userProductIndices.reserve( product_ids.size( ) );
    for( std::set<std::string>::iterator iter = product_ids.begin( ); iter != product_ids.end( ); ++iter ) {
        m_productIndices.push_back( a_settings.pops( )[*iter] );
        m_userProductIndices.push_back( -1 );
    }

    a_reaction.productIDs( product_ids, a_particles, true );
    m_productMultiplicities.reserve( product_ids.size( ) );
    m_productIndicesTransportable.reserve( product_ids.size( ) );
    m_userProductIndicesTransportable.reserve( product_ids.size( ) );
    for( std::set<std::string>::iterator iter = product_ids.begin( ); iter != product_ids.end( ); ++iter ) {
        m_productMultiplicities.push_back( a_reaction.productMultiplicity( *iter ) );
        m_productIndicesTransportable.push_back( a_settings.pops( )[*iter] );
        m_userProductIndicesTransportable.push_back( -1 );
    }

    if( m_upscatterModelASupported && ( a_settings.upscatterModel( ) == Sampling::Upscatter::Model::A ) ) {
        GIDI::Vector const &l_upscatterModelACrossSection = a_reaction.crossSection( ).get<GIDI::Functions::Gridded1d>( a_settings.upscatterModelALabel( ) )->data( );
        m_upscatterModelACrossSection.resize( l_upscatterModelACrossSection.size( ) );
        for( std::size_t i1 = 0; i1 < l_upscatterModelACrossSection.size( ); ++i1 ) m_upscatterModelACrossSection[i1] = l_upscatterModelACrossSection[i1];
    }

    m_hasFinalStatePhotons = a_setupInfo.m_hasFinalStatePhotons;
#ifndef MCGIDI_USE_OUTPUT_CHANNEL
    std::vector<Product *> products;
    std::vector<DelayedNeutron *> delayedNeutrons;
    std::vector<Functions::Function1d_d1 *> Qs;

    m_totalDelayedNeutronMultiplicity = nullptr;
    m_outputChannel->moveProductsEtAlToReaction( products, &m_totalDelayedNeutronMultiplicity, delayedNeutrons, Qs );

    m_products.resize( products.size( ) );
    for( std::size_t index = 0; index < products.size( ); ++index ) m_products[index] = products[index];

    m_delayedNeutrons.resize( delayedNeutrons.size( ) );
    for( std::size_t index = 0; index < delayedNeutrons.size( ); ++index ) m_delayedNeutrons[index] = delayedNeutrons[index];

    m_Qs.resize( Qs.size( ) );
    for( std::size_t index = 0; index < Qs.size( ); ++index ) m_Qs[index] = Qs[index];

    delete m_outputChannel;
#endif
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

LUPI_HOST_DEVICE Reaction::~Reaction( ) {

#ifdef MCGIDI_USE_OUTPUT_CHANNEL
    delete m_outputChannel;
#else
    delete m_totalDelayedNeutronMultiplicity;
    for( auto iter = m_products.begin( ); iter != m_products.end( ); ++iter ) delete *iter;
    for( auto iter = m_delayedNeutrons.begin( ); iter != m_delayedNeutrons.end( ); ++iter ) delete *iter;
    for( auto iter = m_Qs.begin( ); iter != m_Qs.end( ); ++iter ) delete *iter;
#endif
}
/* *********************************************************************************************************//**
 * Returns the Q-value for projectile energy *a_energy*. 
 *
 * @param a_URR_protareInfos    [in]    URR information.
 * @param a_hashIndex           [in]    Specifies the continuous energy or multi-group index.
 * @param a_temperature         [in]    The temperature of the target.
 * @param a_energy_in           [in]    The energy of the projectile.
 ***********************************************************************************************************/

LUPI_HOST_DEVICE double Reaction::finalQ( double a_energy ) const { 

#ifdef MCGIDI_USE_OUTPUT_CHANNEL
    return( m_outputChannel->finalQ( a_energy ) );
#else
    double Q = 0.0;
    for( auto Q_iter = m_Qs.begin( ); Q_iter != m_Qs.end( ); ++Q_iter ) Q += (*Q_iter)->evaluate( a_energy );

    return( Q );
#endif
}

/* *********************************************************************************************************//**
 * Returns the reaction's cross section for target temperature *a_temperature* and projectile energy *a_energy_in*.
 *
 * @param a_URR_protareInfos    [in]    URR information.
 * @param a_hashIndex           [in]    Specifies the continuous energy or multi-group index.
 * @param a_temperature         [in]    The temperature of the target.
 * @param a_energy_in           [in]    The energy of the projectile.
 ***********************************************************************************************************/

LUPI_HOST_DEVICE double Reaction::crossSection( URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy_in ) const {

    return( m_protareSingle->reactionCrossSection( m_reactionIndex, a_URR_protareInfos, a_hashIndex, a_temperature, a_energy_in, false ) );
}

/* *********************************************************************************************************//**
 * Returns the reaction's cross section for target temperature *a_temperature* and projectile energy *a_energy_in*.
 *
 * @param a_URR_protareInfos    [in]    URR information.
 * @param a_temperature         [in]    The temperature of the target.
 * @param a_energy_in           [in]    The energy of the projectile.
 ***********************************************************************************************************/

LUPI_HOST_DEVICE double Reaction::crossSection( URR_protareInfos const &a_URR_protareInfos, double a_temperature, double a_energy_in ) const {

    return( m_protareSingle->reactionCrossSection( m_reactionIndex, a_URR_protareInfos, a_temperature, a_energy_in ) );
}


/* *********************************************************************************************************//**
 * Returns the multiplicity for outgoing particle with pops id *a_id*. If the multiplicity is energy dependent,
 * the returned value is -1. For energy dependent multiplicities it is better to use the method **productAverageMultiplicity**.
 *
 * @param a_id                      [in]    The PoPs id of the requested particle.
 *
 * @return                                  The multiplicity value for the requested particle.
 ***********************************************************************************************************/

LUPI_HOST int Reaction::productMultiplicity( int a_id ) const {

    int i1 = 0;

    for( Vector<int>::iterator iter = m_productIndices.begin( ); iter != m_productIndices.end( ); ++iter, ++i1 ) {
        if( *iter == a_id ) return( m_productMultiplicities[i1] );
    }

    return( 0 );
}

/* *********************************************************************************************************//**
 * Returns the energy dependent multiplicity for outgoing particle with pops id *a_id*. The returned value may not
 * be an integer. Energy dependent multiplicity mainly occurs for photons and fission neutrons.
 *
 * @param a_index                   [in]    The PoPs id of the requested particle.
 * @param a_projectileEnergy        [in]    The energy of the projectile.
 *
 * @return                                  The multiplicity value for the requested particle.
 ***********************************************************************************************************/

LUPI_HOST_DEVICE double Reaction::productAverageMultiplicity( int a_index, double a_projectileEnergy ) const {

    double multiplicity = 0.0;

    if( m_crossSectionThreshold > a_projectileEnergy ) return( multiplicity );

    int i1 = 0;
    for( Vector<int>::iterator iter = m_productIndices.begin( ); iter != m_productIndices.end( ); ++iter, ++i1 ) {
        if( *iter == a_index ) {
            multiplicity = m_productMultiplicities[i1];
            break;
        }
    }

    if( multiplicity < 0 ) {
#ifdef MCGIDI_USE_OUTPUT_CHANNEL
        multiplicity = m_outputChannel->productAverageMultiplicity( a_index, a_projectileEnergy );
#else
        multiplicity = 0.0;
        for( auto productIter = m_products.begin( ); productIter != m_products.end( ); ++productIter ) {
            multiplicity += (*productIter)->productAverageMultiplicity( a_index, a_projectileEnergy );
        }

        if( ( m_totalDelayedNeutronMultiplicity != nullptr ) && ( a_index == m_neutronIndex ) ) {
            multiplicity += m_totalDelayedNeutronMultiplicity->evaluate( a_projectileEnergy );
        }
#endif
    }

    return( multiplicity );
}

/* *********************************************************************************************************//**
 * Updates the m_userParticleIndex to *a_userParticleIndex* for all particles with PoPs index *a_particleIndex*.
 *
 * @param a_particleIndex       [in]    The PoPs id of the particle whose userPid is to be set.
 * @param a_userParticleIndex   [in]    The particle id specified by the user.
 ***********************************************************************************************************/

LUPI_HOST void Reaction::setUserParticleIndex( int a_particleIndex, int a_userParticleIndex ) {

#ifdef MCGIDI_USE_OUTPUT_CHANNEL
    m_outputChannel->setUserParticleIndex( a_particleIndex, a_userParticleIndex );
#else
    for( auto productIter = m_products.begin( ); productIter != m_products.end( ); ++productIter ) {
        (*productIter)->setUserParticleIndex( a_particleIndex, a_userParticleIndex );
    }

    for( auto iter = m_delayedNeutrons.begin( ); iter != m_delayedNeutrons.end( ); ++iter ) 
        (*iter)->setUserParticleIndex( a_particleIndex, a_userParticleIndex );
#endif

    for( auto i1 = 0; i1 < m_productIndices.size( ); ++i1 ) {
        if( m_productIndices[i1] == a_particleIndex ) m_userProductIndices[i1] = a_userParticleIndex;
    }

    for( auto i1 = 0; i1 < m_productIndicesTransportable.size( ); ++i1 ) {
        if( m_productIndicesTransportable[i1] == a_particleIndex ) m_userProductIndicesTransportable[i1] = a_userParticleIndex;
    }
}

/* *********************************************************************************************************//**
 * Updates the m_userParticleIndex to *a_userParticleIndex* for all particles with PoPs index *a_particleIndex*.
 *
 * @param a_particleIndex       [in]    The PoPs id of the particle whose userPid is to be set.
 * @param a_userParticleIndex   [in]    The particle id specified by the user.
 ***********************************************************************************************************/

LUPI_HOST_DEVICE void Reaction::setAssociatedOrphanProduct( Reaction const *a_orphanProduct ) {

    a_orphanProduct->referenceOrphanProductsToReaction( m_associatedOrphanProducts );
}

/* *********************************************************************************************************//**
 * Updates the m_userParticleIndex to *a_userParticleIndex* for all particles with PoPs index *a_particleIndex*.
 *
 * @param a_particleIndex       [in]    The PoPs id of the particle whose userPid is to be set.
 * @param a_userParticleIndex   [in]    The particle id specified by the user.
 ***********************************************************************************************************/

LUPI_HOST_DEVICE void Reaction::referenceOrphanProductsToReaction( Vector<Product *> &a_associatedOrphanProducts ) const {

#ifdef MCGIDI_USE_OUTPUT_CHANNEL
    a_associatedOrphanProducts.reserve( m_outputChannel->referenceOrphanProductsToReaction( a_associatedOrphanProducts, false ) );
    m_outputChannel->referenceOrphanProductsToReaction( a_associatedOrphanProducts, true );
#else
    a_associatedOrphanProducts.reserve( m_products.size( ) );
    for( auto productIter = m_products.begin( ); productIter != m_products.end( ); ++productIter ) {
        a_associatedOrphanProducts.push_back( *productIter );
    }
#endif
}

/* *********************************************************************************************************//**
 * This method adds sampled products to *a_products*.
 *
 * @param a_protare                 [in]    The Protare this Reaction belongs to.
 * @param a_projectileEnergy        [in]    The energy of the projectile.
 * @param a_input                   [in]    Sample options requested by user.
 * @param a_userrng                 [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState                [in]    The current state for the random number generator.
 * @param a_products                [in]    The object to add all sampled products to.
 * @param a_checkOrphanProducts     [in]    If true, associated orphan products are also sampled.
 ***********************************************************************************************************/

LUPI_HOST_DEVICE void Reaction::sampleProducts( Protare const *a_protare, double a_projectileEnergy, Sampling::Input &a_input, 
                double (*a_userrng)( void * ), void *a_rngState, Sampling::ProductHandler &a_products, bool a_checkOrphanProducts ) const {

    double projectileEnergy = a_projectileEnergy;

    a_input.m_reaction = this;
    a_input.m_projectileMass = m_projectileMass;
    a_input.m_targetMass = m_targetMass;

    a_input.m_dataInTargetFrame = false;
    if( upscatterModelASupported( ) && ( a_input.m_upscatterModel == Sampling::Upscatter::Model::A ) ) {

        a_input.m_dataInTargetFrame = sampleTargetBetaForUpscatterModelA( a_protare, a_projectileEnergy, a_input, a_userrng, a_rngState );
        if( a_input.m_dataInTargetFrame ) projectileEnergy = a_input.m_projectileEnergy;
    }

#ifdef MCGIDI_USE_OUTPUT_CHANNEL
    m_outputChannel->sampleProducts( a_protare, projectileEnergy, a_input, a_userrng, a_rngState, a_products );
#else
    if( m_hasFinalStatePhotons ) {
        double random = a_userrng( a_rngState );
        double cumulative = 0.0;
        bool sampled = false;
        for( auto productIter = m_products.begin( ); productIter != m_products.end( ); ++productIter ) {
            cumulative += (*productIter)->multiplicity( )->evaluate( projectileEnergy );
            if( cumulative >= random ) {
                (*productIter)->sampleFinalState( a_protare, projectileEnergy, a_input, a_userrng, a_rngState, a_products );
                sampled = true;
                break;
            }
        }
        if( !sampled ) {     // BRB: FIXME: still need to code for continuum photon.
        } }
    else {
        for( auto productIter = m_products.begin( ); productIter != m_products.end( ); ++productIter ) {
            (*productIter)->sampleProducts( a_protare, projectileEnergy, a_input, a_userrng, a_rngState, a_products );
        }
    }

    if( m_totalDelayedNeutronMultiplicity != nullptr ) {
        double totalDelayedNeutronMultiplicity = m_totalDelayedNeutronMultiplicity->evaluate( a_projectileEnergy );

        if( a_userrng( a_rngState ) < totalDelayedNeutronMultiplicity ) {       // Assumes that totalDelayedNeutronMultiplicity < 1.0, which it is.
            double sum = 0.0;

            totalDelayedNeutronMultiplicity *= a_userrng( a_rngState );
            for( std::size_t i1 = 0; i1 < (std::size_t) m_delayedNeutrons.size( ); ++i1 ) {
                DelayedNeutron const *delayedNeutron1 = m_delayedNeutrons[i1];
                Product const &product = delayedNeutron1->product( );

                sum += product.multiplicity( )->evaluate( a_projectileEnergy );
                if( sum >= totalDelayedNeutronMultiplicity ) {
                    product.distribution( )->sample( a_projectileEnergy, a_input, a_userrng, a_rngState );
                    a_input.m_delayedNeutronIndex = delayedNeutron1->delayedNeutronIndex( );
                    a_input.m_delayedNeutronDecayRate = delayedNeutron1->rate( );
                    a_products.add( a_projectileEnergy, product.index( ), product.userParticleIndex( ), product.mass( ), a_input, a_userrng, a_rngState, false );
                    break;
                }
            }
        }
    }

#endif

    if( a_checkOrphanProducts ) {
        for( auto productIter = m_associatedOrphanProducts.begin( ); productIter != m_associatedOrphanProducts.end( ); ++productIter ) {
            (*productIter)->sampleProducts( a_protare, projectileEnergy, a_input, a_userrng, a_rngState, a_products );
        }
    }
}

/* *********************************************************************************************************//**
 * This method adds a null product to *a_products*. When running in multi-group mode, a sampled reaction may be rejected if the threshold 
 * is in the multi-group that the projectile is in. If this happens, only null products should be returned. This type of behavior was need
 * in MCAPM but is probably not needed for MCGIDI.
 *
 * @param a_protare                 [in]    The Protare this Reaction belongs to.
 * @param a_projectileEnergy        [in]    The energy of the projectile.
 * @param a_input                   [in]    Sample options requested by user.
 * @param a_userrng                 [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState                [in]    The current state for the random number generator.
 * @param a_products                [in]    The object to add all sampled products to.
 ***********************************************************************************************************/

LUPI_HOST_DEVICE void Reaction::sampleNullProducts( Protare const &a_protare, double a_projectileEnergy, Sampling::Input &a_input, 
        double (*a_userrng)( void * ), void *a_rngState, Sampling::ProductHandler &a_products ) {

    a_input.m_sampledType = Sampling::SampledType::uncorrelatedBody;
    a_input.m_dataInTargetFrame = false;
    a_input.m_frame = GIDI::Frame::lab;
    a_input.m_delayedNeutronIndex = -1;
    a_input.m_delayedNeutronDecayRate = 0.0;

    a_input.m_energyOut1 = a_projectileEnergy;
    a_input.m_mu = 1.0;
    a_input.m_phi = 0.0;

    a_products.add( a_projectileEnergy, a_protare.projectileIndex( ), a_protare.projectileUserIndex( ), a_protare.projectileMass( ), a_input, a_userrng, a_rngState, false );
}

/* *********************************************************************************************************//**
 * Returns the weight for a project with energy *a_energy_in* to cause this reaction to emitted a particle of index
 * *a_pid* at angle *a_mu_lab* as seen in the lab frame. If a particle is emitted, *a_energy_out* is its sampled outgoing energy. 
 *
 * @param a_pid                     [in]    The index of the particle to emit.
 * @param a_temperature             [in]    Specifies the temperature of the material.
 * @param a_energy_in               [in]    The energy of the incident particle.
 * @param a_mu_lab                  [in]    The desired mu in the lab frame for the emitted particle.
 * @param a_energy_out              [in]    The energy of the emitted outgoing particle.
 * @param a_userrng                 [in]    The random number generator.
 * @param a_rngState                [in]    The state to pass to the random number generator.
 * @param a_cumulative_weight       [in]    The cumulative multiplicity.
 * @param a_checkOrphanProducts     [in]    If true, associated orphan products are also sampled.
 *
 * @return                                  The weight that the particle is emitted into mu *a_mu_lab*.
 ***********************************************************************************************************/

LUPI_HOST_DEVICE double Reaction::angleBiasing( int a_pid, double a_temperature, double a_energy_in, double a_mu_lab, double &a_energy_out, 
                double (*a_userrng)( void * ), void *a_rngState, double *a_cumulative_weight, bool a_checkOrphanProducts ) const {

    double cumulative_weight1 = 0.0;
    if( a_cumulative_weight == nullptr ) a_cumulative_weight = &cumulative_weight1;
    double weight1 = 0.0;

#ifdef MCGIDI_USE_OUTPUT_CHANNEL
    m_outputChannel->angleBiasing( this, a_pid, a_temperature, a_energy_in, a_mu_lab, weight1, a_energy_out, a_userrng, a_rngState, *a_cumulative_weight );
#else

    for( auto productIter = m_products.begin( ); productIter != m_products.end( ); ++productIter ) {
        (*productIter)->angleBiasing( this, a_pid, a_temperature, a_energy_in, a_mu_lab, weight1, a_energy_out, a_userrng, a_rngState, *a_cumulative_weight );
    }

    if( ( m_totalDelayedNeutronMultiplicity != nullptr ) && ( a_pid == neutronIndex( ) ) ) {
        for( std::size_t i1 = 0; i1 < (std::size_t) m_delayedNeutrons.size( ); ++i1 ) {
            DelayedNeutron const *delayedNeutron1 = m_delayedNeutrons[i1];
            Product const &product = delayedNeutron1->product( );

            product.angleBiasing( this, a_pid, a_temperature, a_energy_in, a_mu_lab, weight1, a_energy_out, a_userrng, a_rngState, *a_cumulative_weight );
        }
    }
#endif

    if( a_checkOrphanProducts ) {
        for( auto productIter = m_associatedOrphanProducts.begin( ); productIter != m_associatedOrphanProducts.end( ); ++productIter ) {
            (*productIter)->angleBiasing( this, a_pid, a_temperature, a_energy_in, a_mu_lab, weight1, a_energy_out, a_userrng, 
                    a_rngState, *a_cumulative_weight );
        }
    }

    return( weight1 );
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

LUPI_HOST_DEVICE void Reaction::serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode ) {
    
    DATA_MEMBER_STRING( m_label, a_buffer, a_mode );
    DATA_MEMBER_INT( m_ENDF_MT, a_buffer, a_mode );
    DATA_MEMBER_INT( m_ENDL_C, a_buffer, a_mode );
    DATA_MEMBER_INT( m_ENDL_S, a_buffer, a_mode );
    DATA_MEMBER_INT( m_neutronIndex, a_buffer, a_mode );
    DATA_MEMBER_INT( m_initialStateIndex, a_buffer, a_mode );
    DATA_MEMBER_CAST( m_hasFission, a_buffer, a_mode, bool );
    DATA_MEMBER_FLOAT( m_projectileMass, a_buffer, a_mode );
    DATA_MEMBER_FLOAT( m_targetMass, a_buffer, a_mode );
    DATA_MEMBER_FLOAT( m_crossSectionThreshold, a_buffer, a_mode );
    DATA_MEMBER_FLOAT( m_twoBodyThreshold, a_buffer, a_mode );
    DATA_MEMBER_CAST( m_upscatterModelASupported, a_buffer, a_mode, bool );
    DATA_MEMBER_CAST( m_hasFinalStatePhotons, a_buffer, a_mode, bool );
    DATA_MEMBER_VECTOR_DOUBLE( m_upscatterModelACrossSection, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_INT( m_productIndices, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_INT( m_userProductIndices, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_INT( m_productIndicesTransportable, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_INT( m_userProductIndicesTransportable, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_INT( m_productMultiplicities, a_buffer, a_mode );

#ifdef MCGIDI_USE_OUTPUT_CHANNEL
    bool haveChannel = m_outputChannel != nullptr;
    DATA_MEMBER_CAST( haveChannel, a_buffer, a_mode, bool );
    if( haveChannel ) {
        if( a_mode == LUPI::DataBuffer::Mode::Unpack ) {
            if (a_buffer.m_placement != nullptr) {
                m_outputChannel = new(a_buffer.m_placement) OutputChannel();
                a_buffer.incrementPlacement( sizeof(OutputChannel));
            }
            else {
                m_outputChannel = new OutputChannel();
            } }
        else if( a_mode == LUPI::DataBuffer::Mode::Memory ) {
            a_buffer.incrementPlacement( sizeof( OutputChannel ) );
        }
        m_outputChannel->serialize( a_buffer, a_mode );
    }
#else
    serializeQs( a_buffer, a_mode, m_Qs );
    serializeProducts( a_buffer, a_mode, m_products );
    m_totalDelayedNeutronMultiplicity = serializeFunction1d( a_buffer, a_mode, m_totalDelayedNeutronMultiplicity );
    serializeDelayedNeutrons( a_buffer, a_mode, m_delayedNeutrons );
#endif

    DATA_MEMBER_INT( m_associatedOrphanProductIndex, a_buffer, a_mode );
    serializeProducts( a_buffer, a_mode, m_associatedOrphanProducts );

    if( a_mode == LUPI::DataBuffer::Mode::Unpack ) {
        m_protareSingle = nullptr;
        m_reactionIndex = -1;
    }
}

}

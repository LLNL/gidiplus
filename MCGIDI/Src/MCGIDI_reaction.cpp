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

HOST_DEVICE Reaction::Reaction( ) :
        m_protareSingle( nullptr ),
        m_reactionIndex( -1 ),
        m_label( ),
        m_ENDF_MT( 0 ),
        m_ENDL_C( 0 ),
        m_ENDL_S( 0 ),
        m_neutronIndex( 0 ),
        m_hasFission( false ),
        m_projectileMass( 0.0 ),
        m_targetMass( 0.0 ),
        m_crossSectionThreshold( 0.0 ),
        m_upscatterModelASupported( false ),
        m_outputChannel( ),
        m_associatedOrphanProductIndex( -1 ),
        m_associatedOrphanProduct( nullptr ) {

}

/* *********************************************************************************************************//**
 * @param a_reaction            [in]    The GIDI::Reaction whose data is to be used to construct *this*.
 * @param a_setupInfo           [in]    Used internally when constructing a Protare to pass information to other constructors.
 * @param a_settings            [in]    Used to pass user options to the *this* to instruct it which data are desired.
 * @param a_particles           [in]    List of transporting particles and their information (e.g., multi-group boundaries and fluxes).
 * @param a_temperatureInfos    [in]    The list of temperature data to extract from *a_protare*.
 ***********************************************************************************************************/

HOST Reaction::Reaction( GIDI::Reaction const &a_reaction, SetupInfo &a_setupInfo, Transporting::MC const &a_settings, GIDI::Transporting::Particles const &a_particles,
            GIDI::Styles::TemperatureInfos const &a_temperatureInfos ) :
        m_protareSingle( nullptr ),
        m_reactionIndex( -1 ),
        m_label( a_reaction.label( ).c_str( ) ),
        m_ENDF_MT( a_reaction.ENDF_MT( ) ),
        m_ENDL_C( a_reaction.ENDL_C( ) ),
        m_ENDL_S( a_reaction.ENDL_S( ) ),
        m_neutronIndex( a_settings.neutronIndex( ) ),
        m_hasFission( a_reaction.hasFission( ) ),
        m_projectileMass( a_setupInfo.m_protare.projectileMass( ) ),
        m_targetMass( a_setupInfo.m_protare.targetMass( ) ),
        m_crossSectionThreshold( a_reaction.crossSectionThreshold( ) ),
        m_upscatterModelASupported( ( a_setupInfo.m_protare.projectileIndex( ) != a_setupInfo.m_protare.photonIndex( ) ) &&
                                    ( a_setupInfo.m_reactionType == Transporting::Reaction::Type::Reactions ) ),
        m_outputChannel( a_reaction.outputChannel( ), a_setupInfo, a_settings, a_particles ),
        m_associatedOrphanProductIndex( -1 ),
        m_associatedOrphanProduct( nullptr ) {

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
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

HOST_DEVICE Reaction::~Reaction( ) {

}

/* *********************************************************************************************************//**
 * Returns the reaction's cross section for target temperature *a_temperature* and projectile energy *a_energy_in*.
 *
 * @param a_URR_protareInfos    [in]    URR information.
 * @param a_hashIndex           [in]    Specifies the continuous energy or multi-group index.
 * @param a_temperature         [in]    The temperature of the target.
 * @param a_energy_in           [in]    The energy of the projectile.
 ***********************************************************************************************************/

HOST_DEVICE double Reaction::crossSection( URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy_in ) const {

    return( m_protareSingle->reactionCrossSection( m_reactionIndex, a_URR_protareInfos, a_hashIndex, a_temperature, a_energy_in, false ) );
}

/* *********************************************************************************************************//**
 * Returns the reaction's cross section for target temperature *a_temperature* and projectile energy *a_energy_in*.
 *
 * @param a_URR_protareInfos    [in]    URR information.
 * @param a_temperature         [in]    The temperature of the target.
 * @param a_energy_in           [in]    The energy of the projectile.
 ***********************************************************************************************************/

HOST_DEVICE double Reaction::crossSection( URR_protareInfos const &a_URR_protareInfos, double a_temperature, double a_energy_in ) const {

    return( m_protareSingle->reactionCrossSection( m_reactionIndex, a_URR_protareInfos, a_temperature, a_energy_in ) );
}


/* *********************************************************************************************************//**
 * 
 *
 * @param a_id                      [in]    The PoPs id of the requested particle.
 *
 * @return                                  The multiplicity value for the requested particle.
 ***********************************************************************************************************/

HOST int Reaction::productMultiplicities( int a_id ) const {

    int i1 = 0;
    for( Vector<int>::iterator iter = m_productIndices.begin( ); iter != m_productIndices.end( ); ++iter, ++i1 ) {
        if( *iter == a_id ) return( m_productMultiplicities[i1] );
    }

    return( 0 );
}

/* *********************************************************************************************************//**
 * Updates the m_userParticleIndex to *a_userParticleIndex* for all particles with PoPs index *a_particleIndex*.
 *
 * @param a_particleIndex       [in]    The PoPs id of the particle whose userPid is to be set.
 * @param a_userParticleIndex   [in]    The particle id specified by the user.
 ***********************************************************************************************************/

HOST void Reaction::setUserParticleIndex( int a_particleIndex, int a_userParticleIndex ) {

    m_outputChannel.setUserParticleIndex( a_particleIndex, a_userParticleIndex );

    for( auto i1 = 0; i1 < m_productIndices.size( ); ++i1 ) {
        if( m_productIndices[i1] == a_particleIndex ) m_userProductIndices[i1] = a_userParticleIndex;
    }

    for( auto i1 = 0; i1 < m_productIndicesTransportable.size( ); ++i1 ) {
        if( m_productIndicesTransportable[i1] == a_particleIndex ) m_userProductIndicesTransportable[i1] = a_userParticleIndex;
    }
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
 ***********************************************************************************************************/

HOST_DEVICE void Reaction::sampleProducts( Protare const *a_protare, double a_projectileEnergy, Sampling::Input &a_input, 
                double (*a_userrng)( void * ), void *a_rngState, Sampling::ProductHandler &a_products ) const {

    a_input.m_reaction = this;
    a_input.m_projectileMass = m_projectileMass;
    a_input.m_targetMass = m_targetMass;

    a_input.m_dataInTargetFrame = false;
    if( upscatterModelASupported( ) && ( a_input.m_upscatterModel == Sampling::Upscatter::Model::A ) ) {
        double projectileEnergy = a_projectileEnergy;

        a_input.m_dataInTargetFrame = sampleTargetBetaForUpscatterModelA( a_protare, a_projectileEnergy, a_input, a_userrng, a_rngState );
        if( a_input.m_dataInTargetFrame ) projectileEnergy = a_input.m_projectileEnergy;
        m_outputChannel.sampleProducts( a_protare, projectileEnergy, a_input, a_userrng, a_rngState, a_products ); }
    else {
        m_outputChannel.sampleProducts( a_protare, a_projectileEnergy, a_input, a_userrng, a_rngState, a_products );
    }

    if( m_associatedOrphanProduct != nullptr ) m_associatedOrphanProduct->sampleProducts( a_protare, a_projectileEnergy, a_input, a_userrng, a_rngState, a_products );
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

HOST_DEVICE void Reaction::sampleNullProducts( Protare const &a_protare, double a_projectileEnergy, Sampling::Input &a_input, 
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
 * Returns the weith for a project with energy *a_energy_in* to cause this reaction to emitted a particle of index
 * *a_pid* at angle *a_mu_lab* as seen in the lab frame. If a particle is emitted, *a_energy_out* is its sampled outgoing energy. 
 *
 * @param a_pid                     [in]    The index of the particle to emit.
 * @param a_energy_in               [in]    The energy of the incident particle.
 * @param a_mu_lab                  [in]    The desired mu in the lab frame for the emitted particle.
 * @param a_energy_out              [in]    The energy of the emitted outgoing particle.
 * @param a_userrng                 [in]    The random number generator.
 * @param a_rngState                [in]    The state to pass to the random number generator.
 * @param a_cumulative_weight       [in]    The cumulative multiplicity.
 *
 * @return                                  The weith that the particle is emitted into mu *a_mu_lab*.
 ***********************************************************************************************************/

HOST_DEVICE double Reaction::angleBiasing( int a_pid, double a_energy_in, double a_mu_lab, double &a_energy_out, 
                double (*a_userrng)( void * ), void *a_rngState, double *a_cumulative_weight ) const {

    double cumulative_weight1 = 0.0;
    if( a_cumulative_weight == nullptr ) a_cumulative_weight = &cumulative_weight1;
    double weight1 = 0.0;

    m_outputChannel.angleBiasing( this, a_pid, a_energy_in, a_mu_lab, weight1, a_energy_out, a_userrng, a_rngState, *a_cumulative_weight );

    if( m_associatedOrphanProduct != nullptr ) {
        double cumulative_weight2 = 0.0;
        double energy_out2;
        double weight2 = m_associatedOrphanProduct->angleBiasing( a_pid, a_energy_in, a_mu_lab, energy_out2, a_userrng, a_rngState, &cumulative_weight2 );

        *a_cumulative_weight += cumulative_weight2;
        if( cumulative_weight2 > a_userrng( a_rngState ) * *a_cumulative_weight ) {
            weight1 = weight2;
            a_energy_out = energy_out2;
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

HOST_DEVICE void Reaction::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    
    DATA_MEMBER_STRING( m_label, a_buffer, a_mode );
    DATA_MEMBER_INT( m_ENDF_MT, a_buffer, a_mode );
    DATA_MEMBER_INT( m_ENDL_C, a_buffer, a_mode );
    DATA_MEMBER_INT( m_ENDL_S, a_buffer, a_mode );
    DATA_MEMBER_INT( m_neutronIndex, a_buffer, a_mode );
    DATA_MEMBER_CAST( m_hasFission, a_buffer, a_mode, bool );
    DATA_MEMBER_FLOAT( m_projectileMass, a_buffer, a_mode );
    DATA_MEMBER_FLOAT( m_targetMass, a_buffer, a_mode );
    DATA_MEMBER_FLOAT( m_crossSectionThreshold, a_buffer, a_mode );
    DATA_MEMBER_CAST( m_upscatterModelASupported, a_buffer, a_mode, bool );
    DATA_MEMBER_VECTOR_DOUBLE( m_upscatterModelACrossSection, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_INT( m_productIndices, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_INT( m_userProductIndices, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_INT( m_productIndicesTransportable, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_INT( m_userProductIndicesTransportable, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_INT( m_productMultiplicities, a_buffer, a_mode );
    m_outputChannel.serialize( a_buffer, a_mode );
    DATA_MEMBER_INT( m_associatedOrphanProductIndex, a_buffer, a_mode );

    if( a_mode == DataBuffer::Mode::Unpack ) {
        m_protareSingle = nullptr;
        m_reactionIndex = -1;
        m_associatedOrphanProduct = nullptr;          // To be filled in by Protare after orphanProducts are unpacked.
    }
}

}

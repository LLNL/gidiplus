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

/*! \class Product
 * This class represents a **GNDS** <**product**> node with only data needed for Monte Carlo transport.
 */

/* *********************************************************************************************************//**
 * Default constructor used when broadcasting a Protare as needed by MPI or GPUs.
 ***********************************************************************************************************/

LUPI_HOST_DEVICE Product::Product( ) :
        m_ID( ),
        m_index( 0 ),
        m_userParticleIndex( -1 ),
        m_mass( 0.0 ),
        m_excitationEnergy( 0.0 ),
        m_twoBodyOrder( TwoBodyOrder::notApplicable ),
        m_neutronIndex( 0 ),
        m_initialStateIndex( -1 ),
        m_multiplicity( nullptr ),
        m_distribution( nullptr ),
        m_outputChannel( nullptr ) {

}

/* *********************************************************************************************************//**
 * @param a_product             [in]    The GIDI::Product whose data is to be used to construct *this*.
 * @param a_setupInfo           [in]    Used internally when constructing a Protare to pass information to other constructors.
 * @param a_settings            [in]    Used to pass user options to the *this* to instruct it which data are desired.
 * @param a_particles           [in]    List of transporting particles and their information (e.g., multi-group boundaries and fluxes).
 * @param a_isFission           [in]    *true* if parent channel is a fission channel and *false* otherwise.
 ***********************************************************************************************************/

LUPI_HOST Product::Product( GIDI::Product const *a_product, SetupInfo &a_setupInfo, Transporting::MC const &a_settings, 
                GIDI::Transporting::Particles const &a_particles, bool a_isFission ) :
        m_ID( a_product->particle( ).ID( ).c_str( ) ),
        m_index( MCGIDI_popsIndex( a_settings.pops( ), a_product->particle( ).ID( ) ) ),
        m_userParticleIndex( -1 ),
        m_label( a_product->label( ).c_str( ) ),
        m_isCompleteParticle( a_product->isCompleteParticle( ) ),
        m_mass( a_product->particle( ).mass( "MeV/c**2" ) ),         // Includes nuclear excitation energy.
        m_excitationEnergy( a_product->particle( ).excitationEnergy( ).value( ) ),
        m_twoBodyOrder( a_setupInfo.m_twoBodyOrder ),
        m_neutronIndex( a_settings.neutronIndex( ) ),
        m_initialStateIndex( -1 ),
        m_multiplicity( Functions::parseMultiplicityFunction1d( a_setupInfo, a_settings, a_product->multiplicity( ) ) ),
        m_distribution( nullptr ),
        m_outputChannel( nullptr ) {

    a_setupInfo.m_product1Mass = mass( );                           // Includes nuclear excitation energy.
    a_setupInfo.m_initialStateIndex = -1;
    m_distribution = Distributions::parseGIDI( a_product->distribution( ), a_setupInfo, a_settings );
    m_initialStateIndex = a_setupInfo.m_initialStateIndex;

    GIDI::OutputChannel const *output_channel = a_product->outputChannel( );
    if( output_channel != nullptr ) m_outputChannel = new OutputChannel( output_channel, a_setupInfo, a_settings, a_particles );

    if( a_isFission && ( m_index == a_settings.neutronIndex( ) ) && a_settings.wantTerrellPromptNeutronDistribution( ) ) {
        Functions::Function1d_d1 *multiplicity1 = static_cast<Functions::Function1d_d1 *>( m_multiplicity );

        m_multiplicity = new Functions::TerrellFissionNeutronMultiplicityModel( -1.0, multiplicity1 );
    }
}

/* *********************************************************************************************************//**
 * @param a_pops                [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_ID                  [in]    The PoPs id for the product.
 * @param a_label               [in]    The **GNDS** label for the product.
 ***********************************************************************************************************/

LUPI_HOST Product::Product( PoPI::Database const &a_pops, std::string const &a_ID, std::string const &a_label ) :
        m_ID( a_ID.c_str( ) ),
        m_index( MCGIDI_popsIndex( a_pops, a_ID ) ),
        m_userParticleIndex( -1 ),
        m_label( a_label.c_str( ) ),
        m_mass( 0.0 ),                                  // FIXME, good for photon but nothing else. Still need to implement.
        m_excitationEnergy( 0.0 ),
        m_twoBodyOrder( TwoBodyOrder::notApplicable ),
        m_neutronIndex( a_pops[PoPI::IDs::neutron] ),
        m_initialStateIndex( -1 ),
        m_multiplicity( nullptr ),
        m_distribution( nullptr ),
        m_outputChannel( nullptr ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

LUPI_HOST_DEVICE Product::~Product( ) {

    delete m_multiplicity;

    Distributions::Type type = Distributions::Type::none;
    if( m_distribution != nullptr ) type = m_distribution->type( );
    switch( type ) {
    case Distributions::Type::none:
        break;
    case Distributions::Type::unspecified:
        delete static_cast<Distributions::Unspecified *>( m_distribution );
        break;
    case Distributions::Type::angularTwoBody:
        delete static_cast<Distributions::AngularTwoBody *>( m_distribution );
        break;
    case Distributions::Type::KalbachMann:
        delete static_cast<Distributions::KalbachMann *>( m_distribution );
        break;
    case Distributions::Type::uncorrelated:
        delete static_cast<Distributions::Uncorrelated *>( m_distribution );
        break;
    case Distributions::Type::branching3d:
        delete static_cast<Distributions::Branching3d *>( m_distribution );
        break;
    case Distributions::Type::energyAngularMC:
        delete static_cast<Distributions::EnergyAngularMC *>( m_distribution );
        break;
    case Distributions::Type::angularEnergyMC:
        delete static_cast<Distributions::AngularEnergyMC *>( m_distribution );
        break;
    case Distributions::Type::coherentPhotoAtomicScattering:
        delete static_cast<Distributions::CoherentPhotoAtomicScattering *>( m_distribution );
        break;
    case Distributions::Type::incoherentPhotoAtomicScattering:
        delete static_cast<Distributions::IncoherentPhotoAtomicScattering *>( m_distribution );
        break;
    case Distributions::Type::incoherentPhotoAtomicScatteringElectron:
        delete static_cast<Distributions::IncoherentPhotoAtomicScatteringElectron *>( m_distribution );
        break;
    case Distributions::Type::pairProductionGamma:
        delete static_cast<Distributions::PairProductionGamma *>( m_distribution );
        break;
    case Distributions::Type::coherentElasticTNSL:
        delete static_cast<Distributions::CoherentElasticTNSL *>( m_distribution );
        break;
    case Distributions::Type::incoherentElasticTNSL:
        delete static_cast<Distributions::IncoherentElasticTNSL *>( m_distribution );
        break;
    }

    delete m_outputChannel;
}

/* *********************************************************************************************************//**
 * Updates the m_userParticleIndex to *a_userParticleIndex* for all particles with PoPs index *a_particleIndex*.
 *  
 * @param a_particleIndex       [in]    The PoPs id of the particle whose userPid is to be set.
 * @param a_userParticleIndex   [in]    The particle id specified by the user.
 ***********************************************************************************************************/

LUPI_HOST void Product::setUserParticleIndex( int a_particleIndex, int a_userParticleIndex ) {

    if( m_index == a_particleIndex ) m_userParticleIndex = a_userParticleIndex;
#ifdef MCGIDI_USE_OUTPUT_CHANNEL
    if( m_outputChannel != nullptr ) m_outputChannel->setUserParticleIndex( a_particleIndex, a_userParticleIndex );
#endif
}

/* *********************************************************************************************************//**
 * This method returns the final Q for *this* by getting its output channel's finalQ.
 *
 * @param a_x1                  [in]    The energy of the projectile.
 *
 * @return                              The Q-value at product energy *a_x1*.
 ***********************************************************************************************************/

LUPI_HOST_DEVICE double Product::finalQ( double a_x1 ) const {

#ifdef MCGIDI_USE_OUTPUT_CHANNEL
    if( m_outputChannel != nullptr ) return( m_outputChannel->finalQ( a_x1 ) );
#endif
    return( m_excitationEnergy );
}

/* *********************************************************************************************************//**
 * This method returns *true* if the output channel or any of its sub-output channels is a fission channel and *false* otherwise.
 *
 * @return                              *true* if any sub-output channel is a fission channel and *false* otherwise.
 ***********************************************************************************************************/

LUPI_HOST_DEVICE bool Product::hasFission( ) const {

#ifdef MCGIDI_USE_OUTPUT_CHANNEL
    if( m_outputChannel != nullptr ) return( m_outputChannel->hasFission( ) );
#endif
    return( false );
}

/* *********************************************************************************************************//**
 * Returns the energy dependent multiplicity for outgoing particle with pops id *a_id*. The returned value may not
 * be an integer. Energy dependent multiplicities mainly occurs for photons and fission neutrons.
 *
 * @param a_id                      [in]    The PoPs id of the requested particle.
 * @param a_projectileEnergy        [in]    The energy of the projectile.
 *
 * @return                                  The multiplicity value for the requested particle.
 ***********************************************************************************************************/

LUPI_HOST_DEVICE double Product::productAverageMultiplicity( int a_id, double a_projectileEnergy ) const {

    double multiplicity1 = 0.0;

    if( a_id == m_index ) {
        if( ( m_multiplicity->domainMin( ) <= a_projectileEnergy ) && ( m_multiplicity->domainMax( ) >= a_projectileEnergy ) )
            multiplicity1 += m_multiplicity->evaluate( a_projectileEnergy );
    }
#ifdef MCGIDI_USE_OUTPUT_CHANNEL
    if( m_outputChannel != nullptr ) multiplicity1 += m_outputChannel->productAverageMultiplicity( a_id, a_projectileEnergy );
#endif

    return( multiplicity1 );
}

/* *********************************************************************************************************//**
 * This method adds sampled products to *a_products*.
 *
 * @param a_protare                 [in]    The Protare this Reaction belongs to.
 * @param a_projectileEnergy        [in]    The energy of the projectile.
 * @param a_input                   [in]    Sample options requested by user.
 * @param a_userrng                 [in]    The random number gnerator.
 * @param a_rngState                [in]    The state for the random number gnerator.
 * @param a_products                [in]    The object to add all sampled products to.
 ***********************************************************************************************************/

LUPI_HOST_DEVICE void Product::sampleProducts( Protare const *a_protare, double a_projectileEnergy, Sampling::Input &a_input, 
                double (*a_userrng)( void * ), void *a_rngState, Sampling::ProductHandler &a_products ) const {

#ifdef MCGIDI_USE_OUTPUT_CHANNEL
    if( m_outputChannel != nullptr ) {
        m_outputChannel->sampleProducts( a_protare, a_projectileEnergy, a_input, a_userrng, a_rngState, a_products ); }
    else {
#endif
        if( m_twoBodyOrder == TwoBodyOrder::secondParticle ) {
            a_products.add( a_projectileEnergy, index( ), userParticleIndex( ), mass( ), a_input, a_userrng, a_rngState, index( ) == a_protare->photonIndex( ) ); }
        else {
            int _multiplicity = m_multiplicity->sampleBoundingInteger( a_projectileEnergy, a_userrng, a_rngState );
            int __multiplicity = _multiplicity;

            for( ; _multiplicity > 0; --_multiplicity ) {
                m_distribution->sample( a_projectileEnergy, a_input, a_userrng, a_rngState );
                a_input.m_delayedNeutronIndex = -1;
                a_input.m_delayedNeutronDecayRate = 0.0;
                a_products.add( a_projectileEnergy, index( ), userParticleIndex( ), mass( ), a_input, a_userrng, a_rngState, index( ) == a_protare->photonIndex( ) );
            }
            if( m_initialStateIndex >= 0 ) {
                if( __multiplicity == 0 ) {
                    ProtareSingle const *protare( static_cast<ProtareSingle const *>( a_protare ) );
                    protare->sampleBranchingGammas( a_input, a_projectileEnergy, m_initialStateIndex, a_userrng, a_rngState, a_products );
                }
            }
        }
#ifdef MCGIDI_USE_OUTPUT_CHANNEL
    }
#endif
}

/* *********************************************************************************************************//**
 * This method adds sampled products to *a_products*. In particular, the product is a capture reaction 
 * primary gamma what has a finalState attribute. This gamma is added as well as the gammas from the
 * gamma cascade.
 *
 * @param a_protare                 [in]    The Protare this Reaction belongs to.
 * @param a_projectileEnergy        [in]    The energy of the projectile.
 * @param a_input                   [in]    Sample options requested by user.
 * @param a_userrng                 [in]    The random number gnerator.
 * @param a_rngState                [in]    The state for the random number gnerator.
 * @param a_products                [in]    The object to add all sampled products to.
 ***********************************************************************************************************/

LUPI_HOST_DEVICE void Product::sampleFinalState( Protare const *a_protare, double a_projectileEnergy, Sampling::Input &a_input, 
                double (*a_userrng)( void * ), void *a_rngState, Sampling::ProductHandler &a_products ) const {

    m_distribution->sample( a_projectileEnergy, a_input, a_userrng, a_rngState );
    a_input.m_delayedNeutronIndex = -1;
    a_input.m_delayedNeutronDecayRate = 0.0;
    a_products.add( a_projectileEnergy, index( ), userParticleIndex( ), mass( ), a_input, a_userrng, a_rngState, index( ) == a_protare->photonIndex( ) );

    if( m_initialStateIndex >= 0 ) {
        ProtareSingle const *protare( static_cast<ProtareSingle const *>( a_protare ) );
        protare->sampleBranchingGammas( a_input, a_projectileEnergy, m_initialStateIndex, a_userrng, a_rngState, a_products );
    }
}

/* *********************************************************************************************************//**
 * Returns the weight for a projectile with energy *a_energy_in* to cause this channel to emitted a particle of index
 * *a_pid* at angle *a_mu_lab* as seen in the lab frame. If a particle is emitted, *a_energy_out* is its sampled outgoing energy.
 *
 * @param a_reaction                [in]    The reaction containing the particle which this distribution describes.
 * @param a_pid                     [in]    The index of the particle to emit.
 * @param a_temperature             [in]    Specifies the temperature of the material.
 * @param a_energy_in               [in]    The energy of the incident particle.
 * @param a_mu_lab                  [in]    The desired mu in the lab frame for the emitted particle.
 * @param a_weight                  [in]    The weight of emitting outgoing particle into lab angle *a_mu_lab*.
 * @param a_energy_out              [in]    The energy of the emitted outgoing particle.
 * @param a_userrng                 [in]    The random number generator.
 * @param a_rngState                [in]    The state to pass to the random number generator.
 * @param a_cumulative_weight       [in]    The sum of the multiplicity for other outgoing particles with index *a_pid*.
 ***********************************************************************************************************/

LUPI_HOST_DEVICE void Product::angleBiasing( Reaction const *a_reaction, int a_pid, double a_temperature, double a_energy_in, double a_mu_lab, 
                double &a_weight, double &a_energy_out, double (*a_userrng)( void * ), void *a_rngState, double &a_cumulative_weight ) const {

#ifdef MCGIDI_USE_OUTPUT_CHANNEL
    if( m_outputChannel != nullptr ) {
        m_outputChannel->angleBiasing( a_reaction, a_pid, a_temperature, a_energy_in, a_mu_lab, a_weight, a_energy_out, a_userrng, a_rngState, a_cumulative_weight ); }
    else {
#endif
        if( index( ) != a_pid ) return;

        double probability = 0.0;
        double energy_out = 0.0;

        if( a_cumulative_weight == 0.0 ) a_energy_out = 0.0;

        if( m_multiplicity->type( ) == Function1dType::branching ) { // Needs to handle F1_Branching.
            }
        else {
            probability = m_distribution->angleBiasing( a_reaction, a_temperature, a_energy_in, a_mu_lab, a_userrng, a_rngState, energy_out );
        }

        double weight = m_multiplicity->evaluate( a_energy_in ) * probability;
        a_cumulative_weight += weight;
        if( weight > a_userrng( a_rngState ) * a_cumulative_weight ) {
            a_weight = weight;
            a_energy_out = energy_out;
        }
#ifdef MCGIDI_USE_OUTPUT_CHANNEL
    }
#endif
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

LUPI_HOST_DEVICE void Product::serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode ) {

    DATA_MEMBER_STRING( m_ID, a_buffer, a_mode );
    DATA_MEMBER_INT( m_index, a_buffer, a_mode );
    DATA_MEMBER_INT( m_userParticleIndex, a_buffer, a_mode );
    DATA_MEMBER_STRING( m_label, a_buffer, a_mode );
    DATA_MEMBER_CAST( m_isCompleteParticle, a_buffer, a_mode, bool );
    DATA_MEMBER_FLOAT( m_mass, a_buffer, a_mode );
    DATA_MEMBER_FLOAT( m_excitationEnergy, a_buffer, a_mode );

    int twoBodyOrder = 0;
    switch( m_twoBodyOrder ) {
    case TwoBodyOrder::notApplicable :
        break;
    case TwoBodyOrder::firstParticle :
        twoBodyOrder = 1;
        break;
    case TwoBodyOrder::secondParticle :
        twoBodyOrder = 2;
        break;
    }
    DATA_MEMBER_INT( twoBodyOrder , a_buffer, a_mode );
    if( a_mode == LUPI::DataBuffer::Mode::Unpack ) {
        switch( twoBodyOrder ) {
        case 0 :
            m_twoBodyOrder = TwoBodyOrder::notApplicable;
            break;
        case 1 :
            m_twoBodyOrder = TwoBodyOrder::firstParticle;
            break;
        case 2 :
            m_twoBodyOrder = TwoBodyOrder::secondParticle;
            break;
        }
    }

    DATA_MEMBER_INT( m_neutronIndex, a_buffer, a_mode );
    DATA_MEMBER_INT( m_initialStateIndex, a_buffer, a_mode );

    m_multiplicity = serializeFunction1d( a_buffer, a_mode, m_multiplicity );
    m_distribution = serializeDistribution( a_buffer, a_mode, m_distribution );

#ifdef MCGIDI_USE_OUTPUT_CHANNEL
    bool haveChannel = m_outputChannel != nullptr;
    DATA_MEMBER_CAST( haveChannel, a_buffer, a_mode, bool );
    if( haveChannel && a_mode == LUPI::DataBuffer::Mode::Unpack ) {
        if (a_buffer.m_placement != nullptr) {
            m_outputChannel = new(a_buffer.m_placement) OutputChannel();
            a_buffer.incrementPlacement( sizeof(OutputChannel));
        }
        else {
            m_outputChannel = new OutputChannel();
        }
    }
    if( haveChannel && a_mode == LUPI::DataBuffer::Mode::Memory ) {
        a_buffer.incrementPlacement( sizeof(OutputChannel));
    }
    if( haveChannel ) m_outputChannel->serialize( a_buffer, a_mode );
#endif
}

}

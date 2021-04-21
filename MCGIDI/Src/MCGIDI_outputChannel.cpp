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
 * This class represents a **GNDS** <**outputChannel**> node with only data needed for Monte Carlo transport.
 */


/* *********************************************************************************************************//**
 * Default constructor used when broadcasting a Protare as needed by MPI or GPUs.
 ***********************************************************************************************************/

HOST_DEVICE OutputChannel::OutputChannel( ) :
        m_channelType( ChannelType::none ),
        m_isFission( false ),
        m_neutronIndex( 0 ),
        m_Q( nullptr ),
        m_products( ),
        m_totalDelayedNeutronMultiplicity( nullptr ) {

}

/* *********************************************************************************************************//**
 * @param a_outputChannel       [in]    The GIDI::OutputChannel whose data is to be used to construct *this*.
 * @param a_setupInfo           [in]    Used internally when constructing a Protare to pass information to other constructors.
 * @param a_settings            [in]    Used to pass user options to the *this* to instruct it which data are desired.
 * @param a_particles           [in]    List of transporting particles and their information (e.g., multi-group boundaries and fluxes).
 ***********************************************************************************************************/

HOST OutputChannel::OutputChannel( GIDI::OutputChannel const *a_outputChannel, SetupInfo &a_setupInfo, Transporting::MC const &a_settings, GIDI::Transporting::Particles const &a_particles ) :
        m_channelType( ChannelType::none ),
        m_isFission( false ),
        m_neutronIndex( a_settings.neutronIndex( ) ),
        m_Q( nullptr ),
        m_products( ),
        m_totalDelayedNeutronMultiplicity( nullptr ) {

    if( a_outputChannel != nullptr ) {
        m_channelType = a_outputChannel->twoBody( ) ? ChannelType::twoBody : ChannelType::uncorrelatedBodies;
        m_isFission = a_outputChannel->isFission( );

        m_Q = Functions::parseFunction1d( a_outputChannel->Q( ).get<GIDI::Functions::Function1dForm>( 0 ) );
        if( a_setupInfo.m_isPairProduction ) {
            double domainMin = m_Q->domainMin( ), domainMax = m_Q->domainMax( );

            delete m_Q;
            m_Q = new Functions::Constant1d( domainMin, domainMax, 0.0 );
        }
        a_setupInfo.m_Q = m_Q->evaluate( 0 );                                  // Needed for NBodyPhaseSpace.

        GIDI::Suite const &products = a_outputChannel->products( );
        if( m_channelType == ChannelType::twoBody ) {
            if( !a_setupInfo.m_protare.isTNSL_ProtareSingle( ) ) {
                GIDI::Product const *product = products.get<GIDI::Product>( 1 );
                a_setupInfo.m_product2Mass = product->particle( ).mass( "MeV/c**2" );         // Includes nuclear excitation energy.
            }
        }

        std::size_t size = 0;
        std::set<std::size_t> productsToDo;
        for( std::size_t i1 = 0; i1 < a_outputChannel->products( ).size( ); ++i1 ) {
            GIDI::Product const *product = products.get<GIDI::Product>( i1 );

            if( ( product->outputChannel( ) != nullptr ) || a_settings.sampleNonTransportingParticles( ) || a_particles.hasParticle( product->particle( ).ID( ) ) )
                productsToDo.insert( i1 );
        }
        size = productsToDo.size( );
        if( a_setupInfo.m_isPairProduction ) {
            size += 2;
            size = 2;                               // This is a kludge until the ENDL to GNDS translator is fixed.
        }
        m_products.reserve( size );

        if( a_setupInfo.m_isPairProduction ) {
            std::string ID( PoPI::IDs::photon );
            std::string label = ID;

            Product *product = new Product( a_settings.pops( ), ID, label );
            product->multiplicity( new Functions::Constant1d( a_setupInfo.m_domainMin, a_setupInfo.m_domainMax, 1.0, 0.0 ) );
            product->distribution( new Distributions::PairProductionGamma( a_setupInfo, true ) );
            m_products.push_back( product );

            label += "__a";
            product = new Product( a_settings.pops( ), ID, label );
            product->multiplicity( new Functions::Constant1d( a_setupInfo.m_domainMin, a_setupInfo.m_domainMax, 1.0, 0.0 ) );
            product->distribution( new Distributions::PairProductionGamma( a_setupInfo, false ) );
            m_products.push_back( product );
        }

        for( std::size_t i1 = 0; i1 < a_outputChannel->products( ).size( ); ++i1 ) {
            if( productsToDo.find( i1 ) == productsToDo.end( ) ) continue;

            GIDI::Product const *product = products.get<GIDI::Product>( i1 );

            if( a_setupInfo.m_isPairProduction ) {
                if( !a_settings.sampleNonTransportingParticles( ) ) continue;
                if( a_setupInfo.m_protare.targetIndex( ) != MCGIDI_popsIndex( a_settings.pops( ), product->particle( ).ID( ) ) ) continue;
            }
            a_setupInfo.m_twoBodyOrder = TwoBodyOrder::notApplicable;
            if( m_channelType == ChannelType::twoBody ) a_setupInfo.m_twoBodyOrder = ( ( i1 == 0 ? TwoBodyOrder::firstParticle : TwoBodyOrder::secondParticle ) );
            m_products.push_back( new Product( product, a_setupInfo, a_settings, a_particles, m_isFission ) );
        }

        if( ( a_settings.delayedNeutrons( ) == GIDI::Transporting::DelayedNeutrons::on ) && a_particles.hasParticle( PoPI::IDs::neutron ) ) {
            GIDI::FissionFragmentData const &fissionFragmentData = a_outputChannel->fissionFragmentData( );
            GIDI::Suite const &delayedNeutrons = fissionFragmentData.delayedNeutrons( );

            if( delayedNeutrons.size( ) > 0 ) {
                bool missingData = false;
                GIDI::Axes axes;
                GIDI::Functions::XYs1d totalDelayedNeutronMultiplicity( axes, ptwXY_interpolationLinLin );

                m_delayedNeutrons.reserve( delayedNeutrons.size( ) );
                for( std::size_t i1 = 0; i1 < delayedNeutrons.size( ); ++i1 ) {
                    GIDI::DelayedNeutron const *delayedNeutron = delayedNeutrons.get<GIDI::DelayedNeutron>( i1 );
                    GIDI::Product const &product = delayedNeutron->product( );
                    GIDI::Suite const &multiplicity = product.multiplicity( );

                    GIDI::Functions::Function1dForm const *form1d = multiplicity.get<GIDI::Functions::Function1dForm>( 0 );

                    if( form1d->type( ) == GIDI::FormType::unspecified1d ) {
                        missingData = true;
                        break;
                    }

                    if( form1d->type( ) != GIDI::FormType::XYs1d ) {
                        std::cerr << "OutputChannel::OutputChannel: GIDI::DelayedNeutron multiplicity type != GIDI::FormType::XYs1d" << std::endl;
                        missingData = true;
                        break;
                    }

                    GIDI::Functions::XYs1d const *multiplicityXYs1d = static_cast<GIDI::Functions::XYs1d const *>( form1d );
                    totalDelayedNeutronMultiplicity += *multiplicityXYs1d;

                    m_delayedNeutrons.push_back( new DelayedNeutron( static_cast<int>( i1 ), delayedNeutron, a_setupInfo, a_settings, a_particles ) );
                }
                if( !missingData ) m_totalDelayedNeutronMultiplicity = new Functions::XYs1d( totalDelayedNeutronMultiplicity );
            }
        }
    }
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

HOST_DEVICE OutputChannel::~OutputChannel( ) {

    delete m_Q;
    for( MCGIDI_VectorSizeType i1 = 0; i1 < m_products.size( ); ++i1 ) delete m_products[i1];

    delete m_totalDelayedNeutronMultiplicity;
    for( MCGIDI_VectorSizeType i1 = 0; i1 < m_delayedNeutrons.size( ); ++i1 ) delete m_delayedNeutrons[i1];
}

/* *********************************************************************************************************//**
 * This method returns the final Q for *this* by getting its final Q plus any sub-output channel's finalQ.
 *
 * @param a_x1                  [in]    The energy of the projectile.
 *
 * @return                              The Q-value at product energy *a_x1*.
 ***********************************************************************************************************/

HOST_DEVICE double OutputChannel::finalQ( double a_x1 ) const {

    double final_Q = m_Q->evaluate( a_x1 );

    for( MCGIDI_VectorSizeType i1 = 0; i1 < m_products.size( ); ++i1 ) final_Q += m_products[i1]->finalQ( a_x1 );
    return( final_Q );
}

/* *********************************************************************************************************//**
 * This method returns *true* if the output channel or any of its sub-output channels is a fission channel and *false* otherwise.
 *
 * @return                              *true* if *this* or any sub-output channel is a fission channel and *false* otherwise.
 ***********************************************************************************************************/

HOST_DEVICE bool OutputChannel::hasFission( ) const {

    if( m_isFission ) return( true );
    for( MCGIDI_VectorSizeType i1 = 0; i1 < m_products.size( ); ++i1 ) {
        if( m_products[i1]->hasFission( ) ) return( true );
    }
    return( false );
}

/* *********************************************************************************************************//**
 * Updates the m_userParticleIndex to *a_userParticleIndex* for all particles with PoPs index *a_particleIndex*.
 *  
 * @param a_particleIndex       [in]    The PoPs id of the particle whose userPid is to be set.
 * @param a_userParticleIndex   [in]    The particle id specified by the user.
 ***********************************************************************************************************/

HOST void OutputChannel::setUserParticleIndex( int a_particleIndex, int a_userParticleIndex ) {

    for( auto iter = m_products.begin( ); iter != m_products.end( ); ++iter ) (*iter)->setUserParticleIndex( a_particleIndex, a_userParticleIndex );
    for( auto iter = m_delayedNeutrons.begin( ); iter != m_delayedNeutrons.end( ); ++iter ) (*iter)->setUserParticleIndex( a_particleIndex, a_userParticleIndex );
}

/* *********************************************************************************************************//**
 * This method adds sampled products to *a_products*.
 *
 * @param a_protare                 [in]    The Protare this Reaction belongs to.
 * @param a_projectileEnergy        [in]    The energy of the projectile.
 * @param a_input                   [in]    Sample options requested by user.
 * @param a_userrng                 [in]    The random number generator.
 * @param a_rngState                [in]    The state to pass to the random number generator.
 * @param a_products                [in]    The object to add all sampled products to.
 ***********************************************************************************************************/

HOST_DEVICE void OutputChannel::sampleProducts( Protare const *a_protare, double a_projectileEnergy, Sampling::Input &a_input, 
                double (*a_userrng)( void * ), void *a_rngState, Sampling::ProductHandler &a_products ) const {

    for( Vector<Product *>::const_iterator iter = m_products.begin( ); iter != m_products.end( ); ++iter )
        (*iter)->sampleProducts( a_protare, a_projectileEnergy, a_input, a_userrng, a_rngState, a_products );

    if( m_totalDelayedNeutronMultiplicity != nullptr ) {
        double totalDelayedNeutronMultiplicity = m_totalDelayedNeutronMultiplicity->evaluate( a_projectileEnergy );

        if( a_userrng( a_rngState ) < totalDelayedNeutronMultiplicity ) {       // Assumes that totalDelayedNeutronMultiplicity < 1.0, which it is.
            double sum = 0.0;

            totalDelayedNeutronMultiplicity *= a_userrng( a_rngState );
            for( std::size_t i1 = 0; i1 < (std::size_t) m_delayedNeutrons.size( ); ++i1 ) {
                DelayedNeutron const *delayedNeutron1( delayedNeutron( i1 ) );
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
}

/* *********************************************************************************************************//**
 * Returns the probability for a project with energy *a_energy_in* to cause this channel to emitted a particle of index
 * *a_pid* at angle *a_mu_lab* as seen in the lab frame. If a particle is emitted, *a_energy_out* is its sampled outgoing energy.
 *
 * @param a_reaction                [in]    The reaction containing the particle which this distribution describes.
 * @param a_pid                     [in]    The index of the particle to emit.
 * @param a_energy_in               [in]    The energy of the incident particle.
 * @param a_mu_lab                  [in]    The desired mu in the lab frame for the emitted particle.
 * @param a_weight                  [in]    The probability of emitting outgoing particle into lab angle *a_mu_lab*.
 * @param a_energy_out              [in]    The energy of the emitted outgoing particle.
 * @param a_userrng                 [in]    The random number generator.
 * @param a_rngState                [in]    The state to pass to the random number generator.
 * @param a_cumulative_weight       [in]    The sum of the multiplicity for other outgoing particles with index *a_pid*.
 ***********************************************************************************************************/

HOST_DEVICE void OutputChannel::angleBiasing( Reaction const *a_reaction, int a_pid, double a_energy_in, double a_mu_lab, double &a_weight, double &a_energy_out,
                double (*a_userrng)( void * ), void *a_rngState, double &a_cumulative_weight ) const {

    for( Vector<Product *>::const_iterator iter = m_products.begin( ); iter != m_products.end( ); ++iter )
        (*iter)->angleBiasing( a_reaction, a_pid, a_energy_in, a_mu_lab, a_weight, a_energy_out, a_userrng, a_rngState, a_cumulative_weight );

    if( ( m_totalDelayedNeutronMultiplicity != nullptr ) && ( a_pid == neutronIndex( ) ) ) {
        for( std::size_t i1 = 0; i1 < (std::size_t) m_delayedNeutrons.size( ); ++i1 ) {
            DelayedNeutron const *delayedNeutron1( delayedNeutron( i1 ) );
            Product const &product = delayedNeutron1->product( );

            product.angleBiasing( a_reaction, a_pid, a_energy_in, a_mu_lab, a_weight, a_energy_out, a_userrng, a_rngState, a_cumulative_weight );
        }
    }
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

HOST_DEVICE void OutputChannel::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    int channelType = 0;
    switch( m_channelType ) {
    case ChannelType::none :
        break;
    case ChannelType::twoBody :
        channelType = 1;
        break;
    case ChannelType::uncorrelatedBodies :
        channelType = 2;
        break;
    }
    DATA_MEMBER_INT( channelType, a_buffer, a_mode );
    if( a_mode == DataBuffer::Mode::Unpack ) {
        switch( channelType ) {
        case 0 :
            m_channelType = ChannelType::none;
            break;
        case 1 :
            m_channelType = ChannelType::twoBody;
            break;
        case 2 :
            m_channelType = ChannelType::uncorrelatedBodies;
            break;
        }
    }

    DATA_MEMBER_CAST( m_isFission, a_buffer, a_mode, bool );
    DATA_MEMBER_INT( m_neutronIndex, a_buffer, a_mode );

    m_Q = serializeFunction1d( a_buffer, a_mode, m_Q );

    std::size_t vectorSize = m_products.size( );
    int vectorSizeInt = (int) vectorSize;
    DATA_MEMBER_INT( vectorSizeInt, a_buffer, a_mode );
    vectorSize = (std::size_t) vectorSizeInt;

    if( a_mode == DataBuffer::Mode::Unpack ) {
        m_products.resize( vectorSize, &a_buffer.m_placement );
        for( std::size_t vectorIndex = 0; vectorIndex < vectorSize; ++vectorIndex ) {
            if (a_buffer.m_placement != nullptr) {
                m_products[vectorIndex] = new(a_buffer.m_placement) Product;
                a_buffer.incrementPlacement( sizeof(Product));
            }
            else {
                m_products[vectorIndex] = new Product;
            }
        }
    }
    if( a_mode == DataBuffer::Mode::Memory ) {
        a_buffer.m_placement += m_products.internalSize();
        a_buffer.incrementPlacement( sizeof(Product)*vectorSize);
    }
    for( std::size_t vectorIndex = 0; vectorIndex < vectorSize; ++vectorIndex ) {
        m_products[vectorIndex]->serialize( a_buffer, a_mode );
    }

    m_totalDelayedNeutronMultiplicity = serializeFunction1d( a_buffer, a_mode, m_totalDelayedNeutronMultiplicity );

    vectorSize = m_delayedNeutrons.size( );
    vectorSizeInt = (int) vectorSize;
    DATA_MEMBER_INT( vectorSizeInt, a_buffer, a_mode );
    vectorSize = (std::size_t) vectorSizeInt;
    if( a_mode == DataBuffer::Mode::Unpack ) {
        m_delayedNeutrons.resize( vectorSize, &a_buffer.m_placement );
        for( std::size_t vectorIndex = 0; vectorIndex < vectorSize; ++vectorIndex ) {
            if (a_buffer.m_placement != nullptr) {
                m_delayedNeutrons[vectorIndex] = new(a_buffer.m_placement) DelayedNeutron;
                a_buffer.incrementPlacement( sizeof( DelayedNeutron ) );
            }
            else {
                m_delayedNeutrons[vectorIndex] = new DelayedNeutron;
            }
        }
    }
    if( a_mode == DataBuffer::Mode::Memory ) {
        a_buffer.m_placement += m_delayedNeutrons.internalSize();
        a_buffer.incrementPlacement(sizeof( DelayedNeutron ) * vectorSize);
    }
    for( std::size_t vectorIndex = 0; vectorIndex < vectorSize; ++vectorIndex ) {
        m_delayedNeutrons[vectorIndex]->serialize( a_buffer, a_mode );
    }
}

}

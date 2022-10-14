/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <map>

#include "MCGIDI.hpp"

namespace MCGIDI {

/* *********************************************************************************************************//**
 * Returns the proper **MCGIDI** protare base on the type of **GIDI** protare.
 *
 * @param a_smr                         [Out]   If errors are not to be thrown, then the error is reported via this instance.
 * @param a_protare                     [in]    The GIDI::Protare whose data is to be used to construct *this*.
 * @param a_pops                        [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_settings                    [in]    Used to pass user options to the *this* to instruct it which data are desired.
 * @param a_particles                   [in]    List of transporting particles and their information (e.g., multi-group boundaries and fluxes).
 * @param a_domainHash                  [in]    The hash data used when looking up a cross section.
 * @param a_temperatureInfos            [in]    The list of temperature data to extract from *a_protare*.
 * @param a_reactionsToExclude          [in]    A list of reaction to not include in the MCGIDI::Protare.
 * @param a_reactionsToExcludeOffset    [in]    The starting index for the reactions in this ProtareSingle.
 * @param a_allowFixedGrid              [in]    For internal (i.e., MCGIDI) use only. Users must use the default value.
 ***********************************************************************************************************/

MCGIDI_HOST Protare *protareFromGIDIProtare( LUPI::StatusMessageReporting &a_smr, GIDI::Protare const &a_protare, PoPI::Database const &a_pops, Transporting::MC &a_settings, 
                GIDI::Transporting::Particles const &a_particles, DomainHash const &a_domainHash, GIDI::Styles::TemperatureInfos const &a_temperatureInfos, 
                std::set<int> const &a_reactionsToExclude, int a_reactionsToExcludeOffset, bool a_allowFixedGrid ) {

    Protare *protare( nullptr );

    if( a_protare.protareType( ) == GIDI::ProtareType::single ) {
        protare = new ProtareSingle( a_smr, static_cast<GIDI::ProtareSingle const &>( a_protare ), a_pops, a_settings, a_particles, a_domainHash, 
                a_temperatureInfos, a_reactionsToExclude, a_reactionsToExcludeOffset, a_allowFixedGrid ); }
    else if( a_protare.protareType( ) == GIDI::ProtareType::composite ) {
        protare = new ProtareComposite( a_smr, static_cast<GIDI::ProtareComposite const &>( a_protare ), a_pops, a_settings, a_particles, a_domainHash,
                a_temperatureInfos, a_reactionsToExclude, a_reactionsToExcludeOffset, false ); }
    else if( a_protare.protareType( ) == GIDI::ProtareType::TNSL ) {
        protare = new ProtareTNSL( a_smr, static_cast<GIDI::ProtareTNSL const &>( a_protare ), a_pops, a_settings, a_particles, a_domainHash,
                a_temperatureInfos, a_reactionsToExclude, a_reactionsToExcludeOffset, false );
    }

    return( protare );
}

/*! \class Protare
 * Base class for the *MCGIDI* protare classes.
 */

/* *********************************************************************************************************//**
 * @param a_protareType         [in]    The enum for the type of Protare (i.e., single, composite or TNSL).
 *
 * Default constructor used when broadcasting a Protare as needed by MPI or GPUs.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE Protare::Protare( ProtareType a_protareType ) :
        m_protareType( a_protareType ),
        m_projectileID( ),
        m_projectileIndex( -1 ),
        m_projectileMass( 0.0 ),
        m_projectileExcitationEnergy( 0.0 ),

        m_targetID( ),
        m_targetIndex( -1 ),
        m_targetMass( 0.0 ),
        m_targetExcitationEnergy( 0.0 ),

        m_neutronIndex( -1 ),
        m_userNeutronIndex( -1 ),
        m_photonIndex( -1 ),
        m_userPhotonIndex( -1 ),
        m_electronIndex( -1 ),
        m_userElectronIndex( -1 ),
        m_evaluation( ),
        m_projectileFrame( GIDI::Frame::lab ),

        m_isTNSL_ProtareSingle( false ) {

}

/* *********************************************************************************************************//**
 * Default base Protare constructor.
 *
 * @param a_protareType         [in]    The enum for the type of Protare (i.e., single, composite or TNSL).
 * @param a_protare             [in]    The GIDI::Protare whose data is to be used to construct *this*.
 * @param a_pops                [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_settings            [in]    Used to pass user options to the *this* to instruct it which data are desired.
 ***********************************************************************************************************/

MCGIDI_HOST Protare::Protare( ProtareType a_protareType, GIDI::Protare const &a_protare, PoPI::Database const &a_pops, Transporting::MC const &a_settings ) :
        m_protareType( a_protareType ),
        m_projectileID( a_protare.projectile( ).ID( ).c_str( ) ),
        m_projectileMass( a_protare.projectile( ).mass( "MeV/c**2" ) ),          // Includes nuclear excitation energy.
        m_projectileExcitationEnergy( a_protare.projectile( ).excitationEnergy( ).value( ) ),

        m_targetID( a_protare.target( ).ID( ).c_str( ) ),
        m_targetMass( a_protare.target( ).mass( "MeV/c**2" ) ),                  // Includes nuclear excitation energy.
        m_targetExcitationEnergy( a_protare.target( ).excitationEnergy( ).value( ) ),

        m_neutronIndex( a_settings.neutronIndex( ) ),
        m_photonIndex( a_settings.photonIndex( ) ),
        m_electronIndex( a_settings.electronIndex( ) ),
        m_evaluation( a_protare.evaluation( ).c_str( ) ),
        m_projectileFrame( a_protare.projectileFrame( ) ),

        m_isTNSL_ProtareSingle( a_protare.isTNSL_ProtareSingle( ) ) {

    m_projectileIndex = a_pops[a_protare.projectile( ).ID( )];
    m_targetIndex = a_pops[a_protare.target( ).ID( )];
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE Protare::~Protare( ) {

}

/* *********************************************************************************************************//**
 * Returns the list product indices. If *a_transportablesOnly* is true, the list only includes transportable particle.
 *
 * @param a_transportablesOnly  [in]    If true, only transportable particle indices are added to *a_indices*, otherwise, all particle indices are added.
 ***********************************************************************************************************/

MCGIDI_HOST Vector<int> const &Protare::productIndices( bool a_transportablesOnly ) const {

    if( a_transportablesOnly ) return( m_productIndicesTransportable );
    return( m_productIndices );
}

/* *********************************************************************************************************//**
 * Sets *this* members *m_productIndices* and *m_productIndicesTransportable* to *a_indices* and *a_transportableIndices* respectively.
 *
 * @param a_indices                 [out]   The list of indices for the outgoing particles (i.e., products).
 * @param a_transportableIndices    [in]    The list of transportable indices for the outgoing particles (i.e., products).
 ***********************************************************************************************************/

MCGIDI_HOST void Protare::productIndices( std::set<int> const &a_indices, std::set<int> const &a_transportableIndices ) {

    m_productIndices.reserve( a_indices.size( ) );
    m_userProductIndices.reserve( a_indices.size( ) );
    for( std::set<int>::const_iterator iter = a_indices.begin( ); iter != a_indices.end( ); ++iter ) {
        m_productIndices.push_back( *iter );
        m_userProductIndices.push_back( -1 );
    }

    m_productIndicesTransportable.reserve( a_transportableIndices.size( ) );
    m_userProductIndicesTransportable.reserve( a_transportableIndices.size( ) );
    for( std::set<int>::const_iterator iter = a_transportableIndices.begin( ); iter != a_transportableIndices.end( ); ++iter ) {
        m_productIndicesTransportable.push_back( *iter );
        m_userProductIndicesTransportable.push_back( -1 );
    }
}

/* *********************************************************************************************************//**
 * Returns the list product indices. If *a_transportablesOnly* is true, the list only includes transportable particle.
 *
 * @param a_transportablesOnly  [in]    If true, only transportable particle indices are added to *a_indices*, otherwise, all particle indices are added.
 ***********************************************************************************************************/

MCGIDI_HOST Vector<int> const &Protare::userProductIndices( bool a_transportablesOnly ) const {

    if( a_transportablesOnly ) return( m_userProductIndicesTransportable );
    return( m_userProductIndices );
}

/* *********************************************************************************************************//**
 * Updates the m_userParticleIndex to *a_userParticleIndex* for all particles with PoPs index *a_particleIndex*.
 *
 * @param a_particleIndex       [in]    The PoPs id of the particle whose userPid is to be set.
 * @param a_userParticleIndex   [in]    The particle id specified by the user.
 ***********************************************************************************************************/

MCGIDI_HOST void Protare::setUserParticleIndex( int a_particleIndex, int a_userParticleIndex ) {

    if( m_projectileIndex == a_particleIndex ) m_projectileUserIndex = a_userParticleIndex;
    if( m_targetIndex == a_particleIndex ) m_targetUserIndex = a_userParticleIndex;
    if( m_neutronIndex == a_particleIndex ) m_userNeutronIndex = a_userParticleIndex;
    if( m_photonIndex == a_particleIndex ) m_userPhotonIndex = a_userParticleIndex;
    if( m_electronIndex == a_particleIndex ) m_userElectronIndex = a_userParticleIndex;

    for( auto i1 = 0; i1 < m_productIndices.size( ); ++i1 ) {
        if( m_productIndices[i1] == a_particleIndex ) m_userProductIndices[i1] = a_userParticleIndex;
    }

    for( auto i1 = 0; i1 < m_productIndicesTransportable.size( ); ++i1 ) {
        if( m_productIndicesTransportable[i1] == a_particleIndex ) m_userProductIndicesTransportable[i1] = a_userParticleIndex;
    }
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE void Protare::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    int protareType = 0;
    if( a_mode != DataBuffer::Mode::Unpack ) {
        switch( m_protareType ) {
        case ProtareType::single :
            break;
        case ProtareType::composite :
            protareType = 1;
            break;
        case ProtareType::TNSL :
            protareType = 2;
            break;
        }
    }
    DATA_MEMBER_INT( protareType, a_buffer, a_mode );
    if( a_mode == DataBuffer::Mode::Unpack ) {
        switch( protareType ) {
        case 0 :
            m_protareType = ProtareType::single;
            break;
        case 1 :
            m_protareType = ProtareType::composite;
            break;
        case 2 :
            m_protareType = ProtareType::TNSL;
            break;
        }
    }

    DATA_MEMBER_STRING( m_projectileID, a_buffer, a_mode );
    DATA_MEMBER_INT( m_projectileIndex, a_buffer, a_mode );
    DATA_MEMBER_INT( m_projectileUserIndex, a_buffer, a_mode );
    DATA_MEMBER_FLOAT( m_projectileMass, a_buffer, a_mode );
    DATA_MEMBER_FLOAT( m_projectileExcitationEnergy, a_buffer, a_mode );

    DATA_MEMBER_STRING( m_targetID, a_buffer, a_mode );
    DATA_MEMBER_INT( m_targetIndex, a_buffer, a_mode );
    DATA_MEMBER_INT( m_targetUserIndex, a_buffer, a_mode );
    DATA_MEMBER_FLOAT( m_targetMass, a_buffer, a_mode );
    DATA_MEMBER_FLOAT( m_targetExcitationEnergy, a_buffer, a_mode );

    DATA_MEMBER_INT( m_neutronIndex, a_buffer, a_mode );
    DATA_MEMBER_INT( m_userNeutronIndex, a_buffer, a_mode );
    DATA_MEMBER_INT( m_photonIndex, a_buffer, a_mode );
    DATA_MEMBER_INT( m_userPhotonIndex, a_buffer, a_mode );
    DATA_MEMBER_INT( m_electronIndex, a_buffer, a_mode );
    DATA_MEMBER_INT( m_userElectronIndex, a_buffer, a_mode );
    DATA_MEMBER_STRING( m_evaluation, a_buffer, a_mode );

    int frame = 0;
    if( m_projectileFrame == GIDI::Frame::centerOfMass ) frame = 1;
    DATA_MEMBER_INT( frame, a_buffer, a_mode );
    m_projectileFrame = GIDI::Frame::lab;
    if( frame == 1 ) m_projectileFrame = GIDI::Frame::centerOfMass;

    DATA_MEMBER_VECTOR_INT( m_productIndices, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_INT( m_userProductIndices, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_INT( m_productIndicesTransportable, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_INT( m_userProductIndicesTransportable, a_buffer, a_mode );
    DATA_MEMBER_CAST( m_isTNSL_ProtareSingle, a_buffer, a_mode, bool );
}

/* *********************************************************************************************************//**
 * This method counts the number of bytes of memory allocated by *this*. 
 * This is an improvement to the internalSize() method of getting memory size.
 ***********************************************************************************************************/
MCGIDI_HOST_DEVICE long Protare::memorySize( ) {

    DataBuffer buf;
    // Written this way for debugger to modify buf.m_placementStart here for easier double checking.
    buf.m_placement = buf.m_placementStart + sizeOf();
    serialize(buf, DataBuffer::Mode::Memory);
    return( ( buf.m_placement - buf.m_placementStart ) + ( buf.m_sharedPlacement - buf.m_sharedPlacementStart ) );
}

/* *********************************************************************************************************//**
 * This method counts the number of bytes of memory allocated by *this* and puts it into a_totalMemory.
 * If shared memory is used, the size of shared memory is a_sharedMemory. If using shared memory,
 * the host code only needs to allocate (a_totalMemory - a_sharedMemory) in main memory.
 ***********************************************************************************************************/
MCGIDI_HOST_DEVICE void Protare::incrementMemorySize( long &a_totalMemory, long &a_sharedMemory ) {

    DataBuffer buf;         // Written this way for debugger to modify buf.m_placementStart here for easier double checking.

    buf.m_placement = buf.m_placementStart + sizeOf( );
    serialize( buf, DataBuffer::Mode::Memory );
    a_totalMemory += buf.m_placement - buf.m_placementStart;
    a_sharedMemory += buf.m_sharedPlacement - buf.m_sharedPlacementStart;
}

/*! \class ProtareSingle
 * Class representing a **GNDS** <**reactionSuite**> node with only data needed for Monte Carlo transport. The
 * data are also stored in a way that is better suited for Monte Carlo transport. For example, cross section data
 * for each reaction are not stored with its reaction, but within the HeatedCrossSections member of the Protare.
 */

/* *********************************************************************************************************//**
 * Default constructor used when broadcasting a Protare as needed by MPI or GPUs.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE ProtareSingle::ProtareSingle( ) :
        Protare( ProtareType::single ),
        m_URR_index( -1 ),
        m_hasURR_probabilityTables( false ),
        m_URR_domainMin( -1.0 ),
        m_URR_domainMax( -1.0 ),
        m_projectileMultiGroupBoundaries( 0 ),
        m_projectileMultiGroupBoundariesCollapsed( 0 ),
        m_projectileFixedGrid( 0 ),
        m_reactions( 0 ),
        m_orphanProducts( 0 ) {

}

/* *********************************************************************************************************//**
 * @param a_smr                         [Out]   If errors are not to be thrown, then the error is reported via this instance.
 * @param a_protare                     [in]    The GIDI::Protare whose data is to be used to construct *this*.
 * @param a_pops                        [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_settings                    [in]    Used to pass user options to the *this* to instruct it which data are desired.
 * @param a_particles                   [in]    List of transporting particles and their information (e.g., multi-group boundaries and fluxes).
 * @param a_domainHash                  [in]    The hash data used when looking up a cross section.
 * @param a_temperatureInfos            [in]    The list of temperature data to extract from *a_protare*.
 * @param a_reactionsToExclude          [in]    A list of reaction to not include in the MCGIDI::Protare.
 * @param a_reactionsToExcludeOffset    [in]    The starting index for the reactions in this ProtareSingle.
 * @param a_allowFixedGrid              [in]    For internal (i.e., MCGIDI) use only. Users must use the default value.
 ***********************************************************************************************************/

MCGIDI_HOST ProtareSingle::ProtareSingle( LUPI::StatusMessageReporting &a_smr, GIDI::ProtareSingle const &a_protare, PoPI::Database const &a_pops, 
                Transporting::MC &a_settings, GIDI::Transporting::Particles const &a_particles, DomainHash const &a_domainHash, 
                GIDI::Styles::TemperatureInfos const &a_temperatureInfos, std::set<int> const &a_reactionsToExclude, int a_reactionsToExcludeOffset, 
                bool a_allowFixedGrid ) :
        Protare( ProtareType::single, a_protare, a_pops, a_settings ),
        m_interaction( a_protare.interaction( ).c_str( ) ),
        m_URR_index( -1 ),
        m_hasURR_probabilityTables( false ),
        m_URR_domainMin( -1.0 ),
        m_URR_domainMax( -1.0 ),
        m_projectileMultiGroupBoundaries( 0 ),
        m_projectileMultiGroupBoundariesCollapsed( 0 ),
        m_projectileFixedGrid( 0 ),
        m_reactions( 0 ),
        m_orphanProducts( 0 ),
        m_heatedCrossSections( ),
        m_heatedMultigroupCrossSections( ) {

    if( !a_protare.isPhotoAtomic( ) ) {
        std::set<std::string> incompleteParticles;
        a_protare.incompleteParticles( a_settings, incompleteParticles );
        for( auto particle = a_particles.particles( ).begin( ); particle != a_particles.particles( ).end( ); ++particle ) {
            if( incompleteParticles.count( particle->first ) != 0 ) {
                std::string message = "Requested particle '" + particle->first + "' is incomplete in '" + a_protare.realFileName( ) + "'.";
                if( a_settings.throwOnError( ) ) {
                    throw std::runtime_error( message.c_str( ) ); }
                else {
                    smr_setReportError2p( a_smr.smr( ), 0, 0, message.c_str( ) );
                }
            }
        }
    }

    SetupInfo setupInfo( *this );
    setupInfo.m_formatVersion = a_protare.formatVersion( );

    GIDI::Transporting::Particles particles;
    for( std::map<std::string, GIDI::Transporting::Particle>::const_iterator particle = a_particles.particles( ).begin( ); particle != a_particles.particles( ).end( ); ++particle ) {
        setupInfo.m_particleIndices[particle->first] = a_pops[particle->first];

        if( ( m_interaction == GIDI_MapInteractionAtomicChars ) && 
                !( ( particle->first == PoPI::IDs::photon ) || ( particle->first == PoPI::IDs::electron ) ) ) continue;
        particles.add( particle->second );
    }

    GIDI::Transporting::MG multiGroupSettings( a_settings.projectileID( ), GIDI::Transporting::Mode::MonteCarloContinuousEnergy, a_settings.delayedNeutrons( ) );
    multiGroupSettings.setThrowOnError( a_settings.throwOnError( ) );

    setupInfo.m_distributionLabel = a_temperatureInfos[0].griddedCrossSection( );

    a_settings.styles( &a_protare.styles( ) );

    switch( a_settings.crossSectionLookupMode( ) ) {
    case Transporting::LookupMode::Data1d::continuousEnergy :
        m_continuousEnergy = true;
        break;
    case Transporting::LookupMode::Data1d::multiGroup :
        m_continuousEnergy = false;
        if( a_settings.upscatterModel( ) == Sampling::Upscatter::Model::B ) {
            multiGroupSettings.setMode( GIDI::Transporting::Mode::multiGroupWithSnElasticUpScatter ); }
        else {
            multiGroupSettings.setMode( GIDI::Transporting::Mode::multiGroup );
        }
        break;
    default :
        throw std::runtime_error( "ProtareSingle::ProtareSingle: invalid lookupMode" );
    }
    m_fixedGrid = a_allowFixedGrid && ( a_protare.projectile( ).ID( ) == PoPI::IDs::photon ) && ( a_settings.fixedGridPoints( ).size( ) > 0 );

    setupNuclideGammaBranchStateInfos( setupInfo, a_protare );
    convertACE_URR_probabilityTablesFromGIDI( a_protare, a_settings,  setupInfo );

    if( ( a_settings.crossSectionLookupMode( ) == Transporting::LookupMode::Data1d::multiGroup ) || 
        ( a_settings.other1dDataLookupMode( ) == Transporting::LookupMode::Data1d::multiGroup ) ) {

        GIDI::Suite const *transportables = nullptr;
        if( setupInfo.m_formatVersion.major( ) > 1 ) {
            GIDI::Styles::HeatedMultiGroup const &heatedMultiGroup = *a_protare.styles( ).get<GIDI::Styles::HeatedMultiGroup>( a_settings.label( ) );
           transportables = &heatedMultiGroup.transportables( ); }
        else {
            std::vector<GIDI::Suite::const_iterator> tags = a_protare.styles( ).findAllOfMoniker( GIDI_multiGroupStyleChars );

            if( tags.size( ) != 1 ) throw std::runtime_error( "MCGIDI::ProtareSingle::ProtareSingle: What is going on here?" );
            GIDI::Styles::MultiGroup const &multiGroup = static_cast<GIDI::Styles::MultiGroup const &>( **tags[0] );
            transportables = &multiGroup.transportables( );
        }

        GIDI::Transportable const transportable = *transportables->get<GIDI::Transportable>( a_protare.projectile( ).ID( ) );
        m_projectileMultiGroupBoundaries = transportable.groupBoundaries( );
        GIDI::Transporting::Particle const *particle = a_particles.particle( a_protare.projectile( ).ID( ) );
        m_projectileMultiGroupBoundariesCollapsed = particle->multiGroup( ).boundaries( );
    }

    std::vector<GIDI::Reaction const *> GIDI_reactions;
    std::set<std::string> product_ids;
    std::set<std::string> product_ids_transportable;
    GIDI::Reaction const *nuclearPlusCoulombInterferenceReaction = nullptr;
    if( a_settings.nuclearPlusCoulombInterferenceOnly( ) ) nuclearPlusCoulombInterferenceReaction = a_protare.nuclearPlusCoulombInterferenceOnlyReaction( );

    for( std::size_t reactionIndex = 0; reactionIndex < a_protare.reactions( ).size( ); ++reactionIndex ) {
        if( a_reactionsToExclude.find( static_cast<int>( reactionIndex + a_reactionsToExcludeOffset ) ) != a_reactionsToExclude.end( ) ) continue;

        GIDI::Reaction const *GIDI_reaction = a_protare.reaction( reactionIndex );

        if( !GIDI_reaction->active( ) ) continue;

        if( m_continuousEnergy ) {
            if( GIDI_reaction->crossSectionThreshold( ) >= a_settings.energyDomainMax( ) ) continue; }
        else {
            GIDI::Vector multi_group_cross_section = GIDI_reaction->multiGroupCrossSection( a_smr, multiGroupSettings, a_temperatureInfos[0] );
            GIDI::Vector vector = GIDI::collapse( multi_group_cross_section, a_settings, a_particles, 0.0 );

            std::size_t i1 = 0;
            for( ; i1 < vector.size( ); ++i1 ) if( vector[i1] != 0.0 ) break;
            if( i1 == vector.size( ) ) continue;
        }
        if( a_settings.ignoreENDF_MT5( ) && ( GIDI_reaction->ENDF_MT( ) == 5 ) && ( a_reactionsToExclude.size( ) == 0 ) ) continue;

        GIDI_reaction->productIDs( product_ids, particles, false );
        GIDI_reaction->productIDs( product_ids_transportable, particles, true );

        if( ( reactionIndex == 0 ) && a_settings.nuclearPlusCoulombInterferenceOnly( ) && a_protare.onlyRutherfordScatteringPresent( ) ) continue;
        if( ( reactionIndex == 0 ) && ( nuclearPlusCoulombInterferenceReaction != nullptr ) ) {
            GIDI_reactions.push_back( nuclearPlusCoulombInterferenceReaction ); }
        else {
            GIDI_reactions.push_back( GIDI_reaction );
        }
    }

    bool zeroReactions = GIDI_reactions.size( ) == 0;   // Happens when all reactions are skipped in the prior loop.
    if( zeroReactions ) GIDI_reactions.push_back( a_protare.reaction( 0 ) );    // Special case where no reaction in the protare is wanted so the first one is used but its cross section is set to 0.0 at all energies.

    setupInfo.m_reactionType = Transporting::Reaction::Type::Reactions;
    m_reactions.reserve( GIDI_reactions.size( ) );
    for( auto GIDI_reaction = GIDI_reactions.begin( ); GIDI_reaction != GIDI_reactions.end( ); ++GIDI_reaction ) {
        setupInfo.m_reaction = *GIDI_reaction;
        setupInfo.m_isPairProduction = (*GIDI_reaction)->isPairProduction( );
        setupInfo.m_isPhotoAtomicIncoherentScattering = (*GIDI_reaction)->isPhotoAtomicIncoherentScattering( );
        setupInfo.m_initialStateIndex = -1;
        Reaction *reaction = new Reaction( **GIDI_reaction, setupInfo, a_settings, particles, a_temperatureInfos );
        setupInfo.m_initialStateIndices[(*GIDI_reaction)->label( )] = setupInfo.m_initialStateIndex;
        reaction->updateProtareSingleInfo( this, static_cast<int>( m_reactions.size( ) ) );
        m_reactions.push_back( reaction );
    }

    std::set<int> product_indices;
    for( std::set<std::string>::iterator iter = product_ids.begin( ); iter != product_ids.end( ); ++iter ) product_indices.insert( a_pops[*iter] );
    std::set<int> product_indices_transportable;
    for( std::set<std::string>::iterator iter = product_ids_transportable.begin( ); iter != product_ids_transportable.end( ); ++iter ) product_indices_transportable.insert( a_pops[*iter] );
    productIndices( product_indices, product_indices_transportable );

    if( a_settings.sampleNonTransportingParticles( ) || particles.hasParticle( PoPI::IDs::photon ) ) {
        setupInfo.m_reactionType = Transporting::Reaction::Type::OrphanProducts;
        m_orphanProducts.reserve( a_protare.orphanProducts( ).size( ) );
        for( std::size_t reactionIndex = 0; reactionIndex < a_protare.orphanProducts( ).size( ); ++reactionIndex ) {
            GIDI::Reaction const *GIDI_reaction = a_protare.orphanProduct( reactionIndex );

            if( GIDI_reaction->crossSectionThreshold( ) >= a_settings.energyDomainMax( ) ) continue;

            setupInfo.m_reaction = GIDI_reaction;
            Reaction *reaction = new Reaction( *GIDI_reaction, setupInfo, a_settings, particles, a_temperatureInfos );
            reaction->updateProtareSingleInfo( this, static_cast<int>( m_orphanProducts.size( ) ) );
            m_orphanProducts.push_back( reaction );

            GIDI::Functions::Reference1d const *reference( GIDI_reaction->crossSection( ).get<GIDI::Functions::Reference1d>( 0 ) );
            std::string xlink = reference->xlink( );
            GIDI::Ancestry const *ancestry = a_protare.findInAncestry( xlink );
            if( ancestry == nullptr ) throw std::runtime_error( "Could not find xlink for orphan product - 1." );
            ancestry = ancestry->ancestor( );
            if( ancestry == nullptr ) throw std::runtime_error( "Could not find xlink for orphan product - 2." );
            if( ancestry->moniker( ) != GIDI_crossSectionSumChars ) {
                ancestry = ancestry->ancestor( );
                if( ancestry == nullptr ) throw std::runtime_error( "Could not find xlink for orphan product - 3." );
            }
            GIDI::Sums::CrossSectionSum const *crossSectionSum = static_cast<GIDI::Sums::CrossSectionSum const *>( ancestry );
            GIDI::Sums::Summands const &summands = crossSectionSum->summands( );
            for( std::size_t i1 = 0; i1 < summands.size( ); ++i1 ) {
                GIDI::Sums::Summand::Base const *summand = summands[i1];

                GIDI::Ancestry const *ancestry = a_protare.findInAncestry( summand->href( ) );
                if( ancestry == nullptr ) throw std::runtime_error( "Could not find href for summand - 1." );
                ancestry = ancestry->ancestor( );
                if( ancestry == nullptr ) throw std::runtime_error( "Could not find href for summand - 2." );

                GIDI::Reaction const *GIDI_reaction2 = static_cast<GIDI::Reaction const *>( ancestry );
                for( MCGIDI_VectorSizeType reactionIndex2 = 0; reactionIndex2 < m_reactions.size( ); ++reactionIndex2 ) {
                    std::string label( m_reactions[reactionIndex2]->label( ).c_str( ) );

                    if( label == GIDI_reaction2->label( ) ) {
                        m_reactions[reactionIndex2]->associatedOrphanProductIndex( static_cast<int>( m_orphanProducts.size( ) ) - 1 );
                        m_reactions[reactionIndex2]->associatedOrphanProduct( reaction );
                        break;
                    }
                }
            }
        }
    }

    std::vector<GIDI::Reaction const *> GIDI_orphanProducts;
    for( std::size_t reactionIndex = 0; reactionIndex < a_protare.orphanProducts( ).size( ); ++reactionIndex ) {
        GIDI::Reaction const *GIDI_reaction = a_protare.orphanProduct( reactionIndex );

        if( GIDI_reaction->crossSectionThreshold( ) >= a_settings.energyDomainMax( ) ) continue;
        GIDI_orphanProducts.push_back( GIDI_reaction );
    }

    if( m_continuousEnergy ) {
        m_heatedCrossSections.update( a_smr, setupInfo, a_settings, particles, a_domainHash, a_temperatureInfos, GIDI_reactions, GIDI_orphanProducts,
                m_fixedGrid, zeroReactions );
        m_hasURR_probabilityTables = m_heatedCrossSections.hasURR_probabilityTables( );
        m_URR_domainMin = m_heatedCrossSections.URR_domainMin( );
        m_URR_domainMax = m_heatedCrossSections.URR_domainMax( ); }
    else {
        m_heatedMultigroupCrossSections.update( a_smr, a_protare, setupInfo, a_settings, particles, a_temperatureInfos, GIDI_reactions, GIDI_orphanProducts,
                zeroReactions );
    }

    if( ( photonIndex( ) != projectileIndex( ) ) && ( electronIndex( ) != projectileIndex( ) ) && ( a_settings.upscatterModel( ) == Sampling::Upscatter::Model::A ) ) {
        GIDI::Styles::Base const *style = a_protare.styles( ).get<GIDI::Styles::Base>( a_settings.upscatterModelALabel( ) );

        if( style->moniker( ) == GIDI_SnElasticUpScatterStyleChars ) style = a_protare.styles( ).get<GIDI::Styles::Base>( style->derivedStyle( ) );
        if( style->moniker( ) != GIDI_heatedMultiGroupStyleChars ) throw GIDI::Exception( "Label does not yield a heatedMultiGroup style." );

        GIDI::Styles::HeatedMultiGroup const &heatedMultiGroup = *static_cast<GIDI::Styles::HeatedMultiGroup const *>( style );
        std::vector<double> const &boundaries = heatedMultiGroup.groupBoundaries( a_protare.projectile( ).ID( ) );

        m_upscatterModelAGroupVelocities.resize( boundaries.size( ) );
        for( std::size_t i1 = 0; i1 < boundaries.size( ); ++i1 ) m_upscatterModelAGroupVelocities[i1] = MCGIDI_particleBeta( projectileMass( ), boundaries[i1] );
    }
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE ProtareSingle::~ProtareSingle( ) {

    for( Vector<NuclideGammaBranchInfo *>::const_iterator iter = m_branches.begin( ); iter < m_branches.end( ); ++iter ) delete *iter;
    for( Vector<NuclideGammaBranchStateInfo *>::const_iterator iter = m_nuclideGammaBranchStateInfos.begin( ); iter < m_nuclideGammaBranchStateInfos.end( ); ++iter ) delete *iter;
    for( Vector<Reaction *>::const_iterator iter = m_reactions.begin( ); iter < m_reactions.end( ); ++iter ) delete *iter;
    for( Vector<Reaction *>::const_iterator iter = m_orphanProducts.begin( ); iter < m_orphanProducts.end( ); ++iter ) delete *iter;
}

/* *********************************************************************************************************//**
 * Updates the m_userParticleIndex to *a_userParticleIndex* for all particles with PoPs index *a_particleIndex*.
 *
 * @param a_particleIndex       [in]    The PoPs id of the particle whose userPid is to be set.
 * @param a_userParticleIndex   [in]    The particle id specified by the user.
 ***********************************************************************************************************/

MCGIDI_HOST void ProtareSingle::setUserParticleIndex( int a_particleIndex, int a_userParticleIndex ) {

    Protare::setUserParticleIndex( a_particleIndex, a_userParticleIndex );

    m_heatedCrossSections.setUserParticleIndex( a_particleIndex, a_userParticleIndex );
    m_heatedMultigroupCrossSections.setUserParticleIndex( a_particleIndex, a_userParticleIndex );
    for( auto iter = m_reactions.begin( ); iter < m_reactions.end( ); ++iter ) (*iter)->setUserParticleIndex( a_particleIndex, a_userParticleIndex );
    for( auto iter = m_orphanProducts.begin( ); iter < m_orphanProducts.end( ); ++iter ) (*iter)->setUserParticleIndex( a_particleIndex, a_userParticleIndex );
}

/* *********************************************************************************************************//**
 * Returns the pointer representing the protare (i.e., *this*) if *a_index* is 0 and nullptr otherwise.
 *
 * @param a_index               [in]    Must always be 0.
 *
 * @return                              Returns the pointer representing *this*.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE ProtareSingle const *ProtareSingle::protare( MCGIDI_VectorSizeType a_index ) const {

    if( a_index != 0 ) return( nullptr );
    return( this );
}

/* *********************************************************************************************************//**
 * Returns the pointer representing the protare (i.e., *this*) if *a_index* is 0 and nullptr otherwise.
 *
 * @param a_index               [in]    Must always be 0.
 *
 * @return                              Returns the pointer representing *this*.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE ProtareSingle *ProtareSingle::protare( MCGIDI_VectorSizeType a_index ) {

    if( a_index != 0 ) return( nullptr );
    return( this );
}

/* *********************************************************************************************************//**
 * Returns the pointer to the **this** if (*a_index* - 1)th is a value reaction index and nullptr otherwise.
 * 
 * @param a_index               [in]    Index of the reaction.
 * 
 * @return                              Pointer to the requested protare or nullptr if invalid *a_index*..
 ***********************************************************************************************************/
 
MCGIDI_HOST_DEVICE ProtareSingle const *ProtareSingle::protareWithReaction( int a_index ) const {
 
    if( a_index < 0 ) return( nullptr );
    if( static_cast<std::size_t>( a_index ) < numberOfReactions( ) ) return( this );
    return( nullptr );
}

/* *********************************************************************************************************//**
 * Returns the list of temperatures for *this*.
 *
 * @param a_index               [in]    Index of the reqested ProtareSingle. Must be 0.
 *
 * @return                              Vector of doubles.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE Vector<double> ProtareSingle::temperatures( MCGIDI_VectorSizeType a_index ) const {

    if( a_index != 0 ) MCGIDI_THROW( "ProtareSingle::temperatures: a_index not 0." );
    if( m_continuousEnergy ) return( m_heatedCrossSections.temperatures( ) );
    return( m_heatedMultigroupCrossSections.temperatures( ) );
}

/* *********************************************************************************************************//**
 * Sets up the nuclear gamma branching data needed to sample gamma decays.
 *
 * @param a_setupInfo           [in]    Used internally when constructing a Protare to pass information to other constructors.
 * @param a_protare             [in]    The GIDI::Protare** whose data is to be used to construct gamma branching data.
 ***********************************************************************************************************/

MCGIDI_HOST void ProtareSingle::setupNuclideGammaBranchStateInfos( SetupInfo &a_setupInfo, GIDI::ProtareSingle const &a_protare ) {

    PoPI::NuclideGammaBranchStateInfos const &nuclideGammaBranchStateInfos = a_protare.nuclideGammaBranchStateInfos( );
    std::vector<NuclideGammaBranchInfo *> nuclideGammaBranchInfos;

    for( std::size_t i1 = 0; i1 < nuclideGammaBranchStateInfos.size( ); ++i1 )
            a_setupInfo.m_stateNamesToIndices[nuclideGammaBranchStateInfos[i1]->state( )] = (int) i1;

    m_nuclideGammaBranchStateInfos.reserve( nuclideGammaBranchStateInfos.size( ) );
    for( std::size_t i1 = 0; i1 < nuclideGammaBranchStateInfos.size( ); ++i1 ) {
        m_nuclideGammaBranchStateInfos.push_back( new NuclideGammaBranchStateInfo( *nuclideGammaBranchStateInfos[i1], nuclideGammaBranchInfos,
                a_setupInfo.m_stateNamesToIndices ) );
    }

    m_branches.reserve( nuclideGammaBranchInfos.size( ) );
    for( std::size_t i1 = 0; i1 < nuclideGammaBranchInfos.size( ); ++i1 ) m_branches.push_back( nuclideGammaBranchInfos[i1] );
}

/* *********************************************************************************************************//**
 * Returns true if *this* has a fission reaction and false otherwise.
 *
 * @return                              true is if *this* has a fission reaction and false otherwise.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE bool ProtareSingle::hasFission( ) const {

    for( Vector<Reaction *>::const_iterator iter = m_reactions.begin( ); iter < m_reactions.end( ); ++iter ) {
        if( (*iter)->hasFission( ) ) return( true );
    }
    return( false );
}

/* *********************************************************************************************************//**
 * Returns true if *a_energy* with unresolved resonance region (URR) of *this* and false otherwise.
 *
 * @return                              true is if *this* has a URR data.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE bool ProtareSingle::inURR( double a_energy ) const {

    if( a_energy < m_URR_domainMin ) return( false );
    if( a_energy > m_URR_domainMax ) return( false );

    return( true );
}

/* *********************************************************************************************************//**
 * Samples gammas from a nuclide electro-magnetic decay.
 *
 * @param a_input               [in]    Sample options requested by user.
 * @param a_projectileEnergy    [in]    The energy of the projectile.
 * @param a_initialStateIndex   [in]    The index in *m_nuclideGammaBranchStateInfos* whose nuclide data are used for sampling.
 * @param a_userrng             [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState            [in]    The current state for the random number generator.
 * @param a_products            [in]    The object to add all sampled gammas to.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE void ProtareSingle::sampleBranchingGammas( Sampling::Input &a_input, double a_projectileEnergy, int a_initialStateIndex, 
                double (*a_userrng)( void * ), void *a_rngState, Sampling::ProductHandler &a_products ) const {

    NuclideGammaBranchStateInfo *nuclideGammaBranchStateInfo = m_nuclideGammaBranchStateInfos[a_initialStateIndex];
    Vector<int> const &branches = nuclideGammaBranchStateInfo->branches( );

    double random = a_userrng( a_rngState );
    double sum = 0.0;
    for( MCGIDI_VectorSizeType i1 = 0; i1 < branches.size( ); ++i1 ) {
        NuclideGammaBranchInfo *NuclideGammaBranchInfo = m_branches[branches[i1]];

        sum += NuclideGammaBranchInfo->probability( );
        if( sum >= random ) {
            if( NuclideGammaBranchInfo->photonEmissionProbability( ) > a_userrng( a_rngState ) ) {
                a_input.m_sampledType = Sampling::SampledType::photon;
                a_input.m_dataInTargetFrame = false;
                a_input.m_frame = GIDI::Frame::lab;

                a_input.m_energyOut1 = NuclideGammaBranchInfo->gammaEnergy( );
                a_input.m_mu = 1.0 - a_userrng( a_rngState );
                a_input.m_phi = 2.0 * M_PI * a_userrng( a_rngState );

                a_products.add( a_projectileEnergy, photonIndex( ), userPhotonIndex( ), 0.0, a_input, a_userrng, a_rngState, true );
            }

            if( NuclideGammaBranchInfo->residualStateIndex( ) >= 0 )
                sampleBranchingGammas( a_input, a_projectileEnergy, NuclideGammaBranchInfo->residualStateIndex( ), a_userrng, a_rngState, a_products );
            break;
        }
    }
}

/* *********************************************************************************************************//**
 * Returns the total cross section for target temperature *a_temperature* and projectile energy *a_energy*. 
 * *a_sampling* is only used for multi-group cross section look up.
 *
 * @param a_URR_protareInfos    [in]    URR information.
 * @param a_hashIndex           [in]    Specifies the continuous energy or multi-group index.
 * @param a_temperature         [in]    The temperature of the target.
 * @param a_energy              [in]    The energy of the projectile.
 * @param a_sampling            [in]    Used for multi-group look up. If *true*, use augmented cross sections.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE double ProtareSingle::crossSection( URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, bool a_sampling ) const {

    if( m_continuousEnergy ) return( m_heatedCrossSections.crossSection( a_URR_protareInfos, m_URR_index, a_hashIndex, a_temperature, a_energy ) );

    return( m_heatedMultigroupCrossSections.crossSection( a_hashIndex, a_temperature, a_sampling ) );
}

/* *********************************************************************************************************//**
 * Adds the energy dependent, total cross section corresponding to the temperature *a_temperature* multiplied by *a_userFactor* to *a_crossSectionVector*.
 * 
 * @param   a_temperature               [in]        Specifies the temperature of the material.
 * @param   a_userFactor                [in]        User factor which all cross sections are multiplied by.
 * @param   a_numberAllocated           [in]        The length of memory allocated for *a_crossSectionVector*.
 * @param   a_crossSectionVector        [in/out]   The energy dependent, total cross section to add cross section data to.
 ***********************************************************************************************************/
 
MCGIDI_HOST_DEVICE void ProtareSingle::crossSectionVector( double a_temperature, double a_userFactor, int a_numberAllocated, double *a_crossSectionVector ) const {

    if( m_continuousEnergy ) {
        if( !m_fixedGrid ) MCGIDI_THROW( "ProtareSingle::crossSectionVector: continuous energy cannot be supported." );
        m_heatedCrossSections.crossSectionVector( a_temperature, a_userFactor, a_numberAllocated, a_crossSectionVector ); }
    else {
        m_heatedMultigroupCrossSections.crossSectionVector( a_temperature, a_userFactor, a_numberAllocated, a_crossSectionVector );
    }
}

/* *********************************************************************************************************//**
 * Returns the reaction's cross section for the reaction at index *a_reactionIndex*, for target temperature *a_temperature* and projectile energy *a_energy*. 
 * *a_sampling* is only used for multi-group cross section look up.
 *
 * @param a_reactionIndex       [in]    The index of the reaction.
 * @param a_URR_protareInfos    [in]    URR information.
 * @param a_hashIndex           [in]    Specifies the continuous energy or multi-group index.
 * @param a_temperature         [in]    The temperature of the target.
 * @param a_energy              [in]    The energy of the projectile.
 * @param a_sampling            [in]    Used for multi-group look up. If *true*, use augmented cross sections.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE double ProtareSingle::reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, bool a_sampling ) const {

    if( m_continuousEnergy ) return( m_heatedCrossSections.reactionCrossSection( a_reactionIndex, a_URR_protareInfos, m_URR_index, a_hashIndex, a_temperature, a_energy ) );

    return( m_heatedMultigroupCrossSections.reactionCrossSection( a_reactionIndex, a_hashIndex, a_temperature, a_sampling ) );
}

/* *********************************************************************************************************//**
 * Returns the reaction's cross section for the reaction at index *a_reactionIndex*, for target temperature *a_temperature* and projectile energy *a_energy*.
 *
 * @param a_reactionIndex       [in]    The index of the reaction.
 * @param a_URR_protareInfos    [in]    URR information.
 * @param a_temperature         [in]    The temperature of the target.
 * @param a_energy              [in]    The energy of the projectile.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE double ProtareSingle::reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, double a_temperature, double a_energy ) const {

    if( m_continuousEnergy ) return( m_heatedCrossSections.reactionCrossSection( a_reactionIndex, a_URR_protareInfos, m_URR_index, a_temperature, a_energy ) );

    return( m_heatedMultigroupCrossSections.reactionCrossSection( a_reactionIndex, a_temperature, a_energy ) );
}

/* *********************************************************************************************************//**
 * Returns the index of a sampled reaction for a target with termpature *a_temperature*, a projectile with energy *a_energy* and total cross section 
 * *a_crossSection*. Random numbers are obtained via *a_userrng* and *a_rngState*.
 *
 * @param a_URR_protareInfos    [in]    URR information.
 * @param a_hashIndex           [in]    Specifies the continuous energy or multi-group index.
 * @param a_temperature         [in]    The temperature of the target.
 * @param a_energy              [in]    The energy of the projectile.
 * @param a_crossSection        [in]    The total cross section.
 * @param a_userrng             [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState            [in]    The current state for the random number generator.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE int ProtareSingle::sampleReaction( URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, 
                double a_crossSection, double (*a_userrng)( void * ), void *a_rngState ) const {

    if( m_continuousEnergy ) return( m_heatedCrossSections.sampleReaction( a_URR_protareInfos, m_URR_index, a_hashIndex, a_temperature, a_energy, 
            a_crossSection, a_userrng, a_rngState ) );

    return( m_heatedMultigroupCrossSections.sampleReaction( a_hashIndex, a_temperature, a_energy, a_crossSection, a_userrng, a_rngState ) );
}

/* *********************************************************************************************************//**
 * Returns the index of a sampled reaction for a target with termpature *a_temperature*, a projectile with energy *a_energy* and total cross section
 * *a_crossSection*. Random numbers are obtained via *a_userrng* and *a_rngState*.
 *
 * @param a_hashIndex           [in]    Specifies the continuous energy or multi-group index.
 * @param a_temperature         [in]    The temperature of the target.
 * @param a_energy              [in]    The energy of the projectile.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE double ProtareSingle::depositionEnergy( int a_hashIndex, double a_temperature, double a_energy ) const {

    if( m_continuousEnergy ) return( m_heatedCrossSections.depositionEnergy( a_hashIndex, a_temperature, a_energy ) );

    return( m_heatedMultigroupCrossSections.depositionEnergy( a_hashIndex, a_temperature ) );
}

/* *********************************************************************************************************//**
 * Returns the index of a sampled reaction for a target with termpature *a_temperature*, a projectile with energy *a_energy* and total cross section
 * *a_crossSection*. Random numbers are obtained via *a_userrng* and *a_rngState*.
 *
 * @param a_hashIndex           [in]    Specifies the continuous energy or multi-group index.
 * @param a_temperature         [in]    The temperature of the target.
 * @param a_energy              [in]    The energy of the projectile.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE double ProtareSingle::depositionMomentum( int a_hashIndex, double a_temperature, double a_energy ) const {

    if( m_continuousEnergy ) return( m_heatedCrossSections.depositionMomentum( a_hashIndex, a_temperature, a_energy ) );

    return( m_heatedMultigroupCrossSections.depositionMomentum( a_hashIndex, a_temperature ) );
}

/* *********************************************************************************************************//**
 * Returns the index of a sampled reaction for a target with termpature *a_temperature*, a projectile with energy *a_energy* and total cross section
 * *a_crossSection*. Random numbers are obtained via *a_userrng* and *a_rngState*.
 *
 * @param a_hashIndex           [in]    Specifies the continuous energy or multi-group index.
 * @param a_temperature         [in]    The temperature of the target.
 * @param a_energy              [in]    The energy of the projectile.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE double ProtareSingle::productionEnergy( int a_hashIndex, double a_temperature, double a_energy ) const {

    if( m_continuousEnergy ) return( m_heatedCrossSections.productionEnergy( a_hashIndex, a_temperature, a_energy ) );

    return( m_heatedMultigroupCrossSections.productionEnergy( a_hashIndex, a_temperature ) );
}

/* *********************************************************************************************************//**
 * Returns the index of a sampled reaction for a target with termpature *a_temperature*, a projectile with energy *a_energy* and total cross section
 * *a_crossSection*. Random numbers are obtained via *a_userrng* and *a_rngState*.
 *
 * @param a_hashIndex           [in]    Specifies the continuous energy or multi-group index.
 * @param a_temperature         [in]    The temperature of the target.
 * @param a_energy              [in]    The energy of the projectile.
 * @param a_particleIndex       [in]    The index of the particle whose gain is to be returned.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE double ProtareSingle::gain( int a_hashIndex, double a_temperature, double a_energy, int a_particleIndex ) const {

    if( m_continuousEnergy ) return( m_heatedCrossSections.gain( a_hashIndex, a_temperature, a_energy, a_particleIndex ) );

    return( m_heatedMultigroupCrossSections.gain( a_hashIndex, a_temperature, a_particleIndex ) );
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

MCGIDI_HOST_DEVICE void ProtareSingle::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    Protare::serialize( a_buffer, a_mode );

    MCGIDI_VectorSizeType vectorSize;
    DataBuffer *workingBuffer = &a_buffer;

    DATA_MEMBER_STRING( m_interaction, a_buffer, a_mode );
    DATA_MEMBER_INT( m_URR_index, a_buffer, a_mode );
    DATA_MEMBER_CAST( m_hasURR_probabilityTables, a_buffer, a_mode, bool );
    DATA_MEMBER_FLOAT( m_URR_domainMin, a_buffer, a_mode );
    DATA_MEMBER_FLOAT( m_URR_domainMax, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_projectileMultiGroupBoundaries, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_projectileMultiGroupBoundariesCollapsed, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_projectileFixedGrid, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_upscatterModelAGroupVelocities, a_buffer, a_mode );

    vectorSize = m_nuclideGammaBranchStateInfos.size( );
    int vectorSizeInt = (int) vectorSize;
    DATA_MEMBER_INT( vectorSizeInt, *workingBuffer, a_mode );
    vectorSize = (MCGIDI_VectorSizeType) vectorSizeInt;

    if( a_mode == DataBuffer::Mode::Unpack ) {
        m_nuclideGammaBranchStateInfos.resize( vectorSize, &(workingBuffer->m_placement) );
        for( MCGIDI_VectorSizeType vectorIndex = 0; vectorIndex < vectorSize; ++vectorIndex ) {
            if (workingBuffer->m_placement != nullptr) {
                m_nuclideGammaBranchStateInfos[vectorIndex] = new(workingBuffer->m_placement) NuclideGammaBranchStateInfo;
                workingBuffer->incrementPlacement( sizeof( NuclideGammaBranchStateInfo ) );
            }
            else {
                m_nuclideGammaBranchStateInfos[vectorIndex] = new NuclideGammaBranchStateInfo;
            }
        }
    }
    if( a_mode == DataBuffer::Mode::Memory ) {
        a_buffer.m_placement += m_nuclideGammaBranchStateInfos.internalSize();
        a_buffer.incrementPlacement( sizeof( NuclideGammaBranchStateInfo ) * vectorSize );
    }
    for( MCGIDI_VectorSizeType vectorIndex = 0; vectorIndex < vectorSize; ++vectorIndex ) {
        m_nuclideGammaBranchStateInfos[vectorIndex]->serialize( *workingBuffer, a_mode );
    }

    vectorSize = m_branches.size( );
    vectorSizeInt = (int) vectorSize;
    DATA_MEMBER_INT( vectorSizeInt, *workingBuffer, a_mode );
    vectorSize = (MCGIDI_VectorSizeType) vectorSizeInt;

    if( a_mode == DataBuffer::Mode::Unpack ) {
        m_branches.resize( vectorSize, &(workingBuffer->m_placement) );
        for( MCGIDI_VectorSizeType vectorIndex = 0; vectorIndex < vectorSize; ++vectorIndex ) {
            if (workingBuffer->m_placement != nullptr) {
                m_branches[vectorIndex] = new(workingBuffer->m_placement) NuclideGammaBranchInfo;
                workingBuffer->incrementPlacement( sizeof( NuclideGammaBranchInfo ) );
            }
            else {
                m_branches[vectorIndex] = new NuclideGammaBranchInfo;
            }
        }
    }
    if( a_mode == DataBuffer::Mode::Memory ) {
        a_buffer.m_placement += m_branches.internalSize();
        workingBuffer->incrementPlacement( sizeof( NuclideGammaBranchInfo ) * vectorSize );
    }
    for( MCGIDI_VectorSizeType vectorIndex = 0; vectorIndex < vectorSize; ++vectorIndex ) {
        m_branches[vectorIndex]->serialize( *workingBuffer, a_mode );
    }

    vectorSize = m_reactions.size( );
    vectorSizeInt = (int) vectorSize;
    DATA_MEMBER_INT( vectorSizeInt, *workingBuffer, a_mode );
    vectorSize = (MCGIDI_VectorSizeType) vectorSizeInt;

    if( a_mode == DataBuffer::Mode::Unpack ) {
        m_reactions.resize( vectorSize, &(workingBuffer->m_placement) );
        for( MCGIDI_VectorSizeType vectorIndex = 0; vectorIndex < vectorSize; ++vectorIndex ) {
            if (workingBuffer->m_placement != nullptr) {
                m_reactions[vectorIndex] = new(workingBuffer->m_placement) Reaction;
                workingBuffer->incrementPlacement( sizeof(Reaction));
            }
            else {
                m_reactions[vectorIndex] = new Reaction;
            }
        }
    }
    if( a_mode == DataBuffer::Mode::Memory ) {
        a_buffer.m_placement += m_reactions.internalSize();
        a_buffer.incrementPlacement( sizeof(Reaction) * vectorSize);
    }
    for( MCGIDI_VectorSizeType vectorIndex = 0; vectorIndex < vectorSize; ++vectorIndex ) {
        m_reactions[vectorIndex]->serialize( *workingBuffer, a_mode );
        m_reactions[vectorIndex]->updateProtareSingleInfo( this, static_cast<int>( vectorIndex ) );
    }

    vectorSize = m_orphanProducts.size( );
    vectorSizeInt = (int) vectorSize;
    DATA_MEMBER_INT( vectorSizeInt, *workingBuffer, a_mode );
    vectorSize = (MCGIDI_VectorSizeType) vectorSizeInt;

    if( a_mode == DataBuffer::Mode::Unpack ) {
        m_orphanProducts.resize( vectorSize, &(workingBuffer->m_placement) );
        for( MCGIDI_VectorSizeType vectorIndex = 0; vectorIndex < vectorSize; ++vectorIndex ) {
            if (workingBuffer->m_placement != nullptr) {
                m_orphanProducts[vectorIndex] = new(workingBuffer->m_placement) Reaction;
                workingBuffer->incrementPlacement( sizeof(Reaction));
            }
            else {
                m_orphanProducts[vectorIndex] = new Reaction;
            }
        }
    }

    if( a_mode == DataBuffer::Mode::Memory ) {
        a_buffer.m_placement += m_orphanProducts.internalSize( );
        a_buffer.incrementPlacement( sizeof( Reaction ) * vectorSize );
    }

    for( MCGIDI_VectorSizeType vectorIndex = 0; vectorIndex < vectorSize; ++vectorIndex ) {
        m_orphanProducts[vectorIndex]->serialize( *workingBuffer, a_mode );
        m_orphanProducts[vectorIndex]->updateProtareSingleInfo( this, static_cast<int>( vectorIndex ) );
    }

    if( a_mode == DataBuffer::Mode::Unpack ) {
        for( MCGIDI_VectorSizeType i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
            int associatedOrphanProductIndex = m_reactions[i1]->associatedOrphanProductIndex( );

            if( associatedOrphanProductIndex >= 0 ) m_reactions[i1]->associatedOrphanProduct( m_orphanProducts[associatedOrphanProductIndex] );
        }
    }

    DATA_MEMBER_CAST( m_continuousEnergy, *workingBuffer, a_mode, bool );
    DATA_MEMBER_CAST( m_fixedGrid, *workingBuffer, a_mode, bool );
    m_heatedCrossSections.serialize( *workingBuffer, a_mode );
    m_heatedMultigroupCrossSections.serialize( *workingBuffer, a_mode );
}

}

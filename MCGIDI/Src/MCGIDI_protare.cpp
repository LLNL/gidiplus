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

HOST Protare *protareFromGIDIProtare( GIDI::Protare const &a_protare, PoPI::Database const &a_pops, Transporting::MC &a_settings, 
                GIDI::Transporting::Particles const &a_particles, DomainHash const &a_domainHash, GIDI::Styles::TemperatureInfos const &a_temperatureInfos, 
                std::set<int> const &a_reactionsToExclude, int a_reactionsToExcludeOffset, bool a_allowFixedGrid ) {

    Protare *protare( nullptr );

    if( a_protare.protareType( ) == GIDI::ProtareType::single ) {
        protare = new ProtareSingle( static_cast<GIDI::ProtareSingle const &>( a_protare ), a_pops, a_settings, a_particles, a_domainHash, 
                a_temperatureInfos, a_reactionsToExclude, a_reactionsToExcludeOffset, a_allowFixedGrid ); }
    else if( a_protare.protareType( ) == GIDI::ProtareType::composite ) {
        protare = new ProtareComposite( static_cast<GIDI::ProtareComposite const &>( a_protare ), a_pops, a_settings, a_particles, a_domainHash,
                a_temperatureInfos, a_reactionsToExclude, a_reactionsToExcludeOffset, false ); }
    else if( a_protare.protareType( ) == GIDI::ProtareType::TNSL ) {
        protare = new ProtareTNSL( static_cast<GIDI::ProtareTNSL const &>( a_protare ), a_pops, a_settings, a_particles, a_domainHash,
                a_temperatureInfos, a_reactionsToExclude, a_reactionsToExcludeOffset, false );
    }

    return( protare );
}


/*! \class Protare
 * Base class for the *MCGIDI* protare classes.
 */

/* *********************************************************************************************************//**
 * Default constructor used when broadcasting a Protare as needed by MPI or GPUs.
 ***********************************************************************************************************/

HOST_DEVICE Protare::Protare( ProtareType a_protareType ) :
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
        m_evaluation( ),
        m_projectileFrame( GIDI::Frame::lab ),

        m_isTNSL_ProtareSingle( false ) {

}

/* *********************************************************************************************************//**
 * Default base Protare constructor.
 *
 * @param a_protare             [in]    The GIDI::Protare whose data is to be used to construct *this*.
 * @param a_pops                [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_settings            [in]    Used to pass user options to the *this* to instruct it which data are desired.
 ***********************************************************************************************************/

HOST Protare::Protare( ProtareType a_protareType, GIDI::Protare const &a_protare, PoPI::Database const &a_pops, Transporting::MC const &a_settings ) :
        m_protareType( a_protareType ),
        m_projectileID( a_protare.projectile( ).ID( ).c_str( ) ),
        m_projectileMass( a_protare.projectile( ).mass( "MeV/c**2" ) ),          // Includes nuclear excitation energy.
        m_projectileExcitationEnergy( a_protare.projectile( ).excitationEnergy( ).value( ) ),

        m_targetID( a_protare.target( ).ID( ).c_str( ) ),
        m_targetMass( a_protare.target( ).mass( "MeV/c**2" ) ),                  // Includes nuclear excitation energy.
        m_targetExcitationEnergy( a_protare.target( ).excitationEnergy( ).value( ) ),

        m_neutronIndex( a_settings.neutronIndex( ) ),
        m_photonIndex( a_settings.photonIndex( ) ),
        m_evaluation( a_protare.evaluation( ).c_str( ) ),
        m_projectileFrame( a_protare.projectileFrame( ) ),

        m_isTNSL_ProtareSingle( a_protare.isTNSL_ProtareSingle( ) ) {

    m_projectileIndex = a_pops[a_protare.projectile( ).ID( )];
    m_targetIndex = a_pops[a_protare.target( ).ID( )];
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

HOST_DEVICE Protare::~Protare( ) {

}

/* *********************************************************************************************************//**
 * Returns the list product indices. If *a_transportablesOnly* is true, the list only includes transportable particle.
 *
 * @param a_transportablesOnly  [in]    If true, only transportable particle indices are added to *a_indices*, otherwise, all particle indices are added.
 ***********************************************************************************************************/

HOST Vector<int> const &Protare::productIndices( bool a_transportablesOnly ) const {

    if( a_transportablesOnly ) return( m_productIndicesTransportable );
    return( m_productIndices );
}

/* *********************************************************************************************************//**
 * Sets *this* members *m_productIndices* and *m_productIndicesTransportable* to *a_indices* and *a_transportableIndices* respectively.
 *
 * @param a_indices                 [out]   The list of indices for the outgoing particles (i.e., products).
 * @param a_transportableIndices    [in]    The list of transportable indices for the outgoing particles (i.e., products).
 ***********************************************************************************************************/

HOST void Protare::productIndices( std::set<int> const &a_indices, std::set<int> const &a_transportableIndices ) {

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

HOST Vector<int> const &Protare::userProductIndices( bool a_transportablesOnly ) const {

    if( a_transportablesOnly ) return( m_userProductIndicesTransportable );
    return( m_userProductIndices );
}

/* *********************************************************************************************************//**
 * Updates the m_userParticleIndex to *a_userParticleIndex* for all particles with PoPs index *a_particleIndex*.
 *
 * @param a_particleIndex       [in]    The PoPs id of the particle whose userPid is to be set.
 * @param a_userParticleIndex   [in]    The particle id specified by the user.
 ***********************************************************************************************************/

HOST void Protare::setUserParticleIndex( int a_particleIndex, int a_userParticleIndex ) {

    if( m_projectileIndex == a_particleIndex ) m_projectileUserIndex = a_userParticleIndex;
    if( m_targetIndex == a_particleIndex ) m_targetUserIndex = a_userParticleIndex;
    if( m_neutronIndex == a_particleIndex ) m_userNeutronIndex = a_userParticleIndex;
    if( m_photonIndex == a_particleIndex ) m_userPhotonIndex = a_userParticleIndex;

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

HOST_DEVICE void Protare::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    int protareType = 0;
    if( a_mode == DataBuffer::Mode::Pack ) {
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
 * This method counts the number of bytes of memory allocated by *this*. That is the member needed by *this* that is greater than
 * sizeof( *this );
 ***********************************************************************************************************/

HOST_DEVICE long Protare::internalSize( ) const {

    return( projectileID( ).internalSize( ) + targetID( ).internalSize( ) + evaluation( ).internalSize( ) +
            m_productIndices.internalSize( ) + m_productIndicesTransportable.internalSize( ) + m_userProductIndices.internalSize( ) + 
            m_userProductIndicesTransportable.internalSize( ) );
}

/*! \class ProtareSingle
 * Class representing a **GNDS** <**reactionSuite**> node with only data needed for Monte Carlo transport. The
 * data are also stored in a way that is better suited for Monte Carlo transport. For example, cross section data
 * for each reaction are not stored with its reaction, but within the HeatedCrossSections member of the Protare.
 */

/* *********************************************************************************************************//**
 * Default constructor used when broadcasting a Protare as needed by MPI or GPUs.
 ***********************************************************************************************************/

HOST_DEVICE ProtareSingle::ProtareSingle( ) :
        Protare( ProtareType::single ),
        m_URR_index( -1 ),
        m_hasURR_probabilityTables( false ),
        m_URR_domainMin( -1.0 ),
        m_URR_domainMax( -1.0 ),
        m_projectileMultiGroupBoundaries( 0 ),
        m_projectileFixedGrid( 0 ),
        m_reactions( 0 ),
        m_orphanProducts( 0 ) {

}

/* *********************************************************************************************************//**
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

HOST ProtareSingle::ProtareSingle( GIDI::ProtareSingle const &a_protare, PoPI::Database const &a_pops, Transporting::MC &a_settings, 
                GIDI::Transporting::Particles const &a_particles, DomainHash const &a_domainHash, GIDI::Styles::TemperatureInfos const &a_temperatureInfos,
                std::set<int> const &a_reactionsToExclude, int a_reactionsToExcludeOffset, bool a_allowFixedGrid ) :
        Protare( ProtareType::single, a_protare, a_pops, a_settings ),
        m_URR_index( -1 ),
        m_hasURR_probabilityTables( false ),
        m_URR_domainMin( -1.0 ),
        m_URR_domainMax( -1.0 ),
        m_projectileMultiGroupBoundaries( 0 ),
        m_projectileFixedGrid( 0 ),
        m_reactions( 0 ),
        m_orphanProducts( 0 ),
        m_heatedCrossSections( ),
        m_heatedMultigroupCrossSections( ) {

    SetupInfo setupInfo( *this );
    for( std::map<std::string, GIDI::Transporting::Particle>::const_iterator particle = a_particles.particles( ).begin( ); particle != a_particles.particles( ).end( ); ++particle ) {
        setupInfo.m_particleIndices[particle->first] = a_pops[particle->first];
    }

    GIDI::Transporting::MG multiGroupSettings( a_settings.projectileID( ), GIDI::Transporting::Mode::MonteCarloContinuousEnergy, a_settings.delayedNeutrons( ) );
    GIDI::Transporting::Particles particles;

    setupInfo.m_distributionLabel = a_temperatureInfos[0].griddedCrossSection( );
    if( setupInfo.m_distributionLabel == "" ) a_temperatureInfos[0].heatedMultiGroup( );

    a_settings.styles( &a_protare.styles( ) );

    switch( a_settings.crossSectionLookupMode( ) ) {
    case Transporting::LookupMode::Data1d::continuousEnergy :
        m_continuousEnergy = true;
        break;
    case Transporting::LookupMode::Data1d::multiGroup :
        m_continuousEnergy = false;
        multiGroupSettings.setMode( GIDI::Transporting::Mode::multiGroup );
        break;
    default :
        THROW( "ProtareSingle::ProtareSingle: invalid lookupMode" );
    }
    m_fixedGrid = a_allowFixedGrid && ( a_protare.projectile( ).ID( ) == PoPI::IDs::photon ) && ( a_settings.fixedGridPoints( ).size( ) > 0 );

    setupNuclideGammaBranchStateInfos( setupInfo, a_protare );

    if( ( a_settings.crossSectionLookupMode( ) == Transporting::LookupMode::Data1d::multiGroup ) || 
        ( a_settings.other1dDataLookupMode( ) == Transporting::LookupMode::Data1d::multiGroup ) ) {

        std::vector<GIDI::Suite::const_iterator> tags = a_protare.styles( ).findAllOfMoniker( multiGroupStyleMoniker );

        if( tags.size( ) != 1 ) THROW( "Protare::Protare: What is going on here?" );
        GIDI::Styles::MultiGroup const &multiGroup = static_cast<GIDI::Styles::MultiGroup const &>( **tags[0] );

        GIDI::Suite const &transportables = multiGroup.transportables( );
        GIDI::Transportable const transportable = *transportables.get<GIDI::Transportable>( a_protare.projectile( ).ID( ) );
        m_projectileMultiGroupBoundaries = transportable.groupBoundaries( );
    }

    std::vector<GIDI::Reaction const *> GIDI_reactions;
    std::vector<GIDI::Reaction const *> GIDI_orphanProducts;
    std::set<std::string> product_ids;
    std::set<std::string> product_ids_transportable;

    for( std::size_t reactionIndex = 0; reactionIndex < a_protare.reactions( ).size( ); ++reactionIndex ) {
        if( a_reactionsToExclude.find( static_cast<int>( reactionIndex + a_reactionsToExcludeOffset ) ) != a_reactionsToExclude.end( ) ) continue;

        GIDI::Reaction const *GIDI_reaction = a_protare.reaction( reactionIndex );

        if( !GIDI_reaction->active( ) ) continue;

        if( m_continuousEnergy ) {
            if( GIDI_reaction->crossSectionThreshold( ) >= a_settings.energyDomainMax( ) ) continue; }
        else {
            GIDI::Vector multi_group_cross_section = GIDI_reaction->multiGroupCrossSection( multiGroupSettings, a_temperatureInfos[0] );

            std::size_t i1 = 0;
            for( ; i1 < multi_group_cross_section.size( ); ++i1 ) if( multi_group_cross_section[i1] != 0.0 ) break;
            if( i1 == multi_group_cross_section.size( ) ) continue;
        }
        if( a_settings.ignoreENDF_MT5( ) && ( GIDI_reaction->ENDF_MT( ) == 5 ) && ( a_reactionsToExclude.size( ) == 0 ) ) continue;

        GIDI_reaction->productIDs( product_ids, a_particles, false );
        GIDI_reaction->productIDs( product_ids_transportable, a_particles, true );

        GIDI_reactions.push_back( GIDI_reaction );
    }

    setupInfo.m_reactionType = Transporting::Reaction::Type::Reactions;
    m_reactions.reserve( a_protare.reactions( ).size( ) );
    for( auto GIDI_reaction = GIDI_reactions.begin( ); GIDI_reaction != GIDI_reactions.end( ); ++GIDI_reaction ) {
        setupInfo.m_reaction = *GIDI_reaction;
        setupInfo.m_isPairProduction = (*GIDI_reaction)->isPairProduction( );
        Reaction *reaction2 = new Reaction( **GIDI_reaction, setupInfo, a_settings, a_particles, a_temperatureInfos );
        reaction2->updateProtareSingleInfo( this, static_cast<int>( m_reactions.size( ) ) );
        m_reactions.push_back( reaction2 );
    }

    std::set<int> product_indices;
    for( std::set<std::string>::iterator iter = product_ids.begin( ); iter != product_ids.end( ); ++iter ) product_indices.insert( a_pops[*iter] );
    std::set<int> product_indices_transportable;
    for( std::set<std::string>::iterator iter = product_ids_transportable.begin( ); iter != product_ids_transportable.end( ); ++iter ) product_indices_transportable.insert( a_pops[*iter] );
    productIndices( product_indices, product_indices_transportable );

    if( a_settings.sampleNonTransportingParticles( ) || a_particles.hasParticle( PoPI::IDs::photon ) ) {
        setupInfo.m_reactionType = Transporting::Reaction::Type::OrphanProducts;
        m_orphanProducts.reserve( a_protare.orphanProducts( ).size( ) );
        for( std::size_t reactionIndex = 0; reactionIndex < a_protare.orphanProducts( ).size( ); ++reactionIndex ) {
            GIDI::Reaction const *GIDI_reaction = a_protare.orphanProduct( reactionIndex );

            if( GIDI_reaction->crossSectionThreshold( ) >= a_settings.energyDomainMax( ) ) continue;

            setupInfo.m_reaction = GIDI_reaction;
            Reaction *reaction2 = new Reaction( *GIDI_reaction, setupInfo, a_settings, a_particles, a_temperatureInfos );
            reaction2->updateProtareSingleInfo( this, static_cast<int>( m_orphanProducts.size( ) ) );
            m_orphanProducts.push_back( reaction2 );

            GIDI::Functions::Reference1d const *reference( GIDI_reaction->crossSection( ).get<GIDI::Functions::Reference1d>( 0 ) );
            std::string xlink = reference->xlink( );
            GIDI::Ancestry const *ancestry = a_protare.findInAncestry( xlink );
            if( ancestry == nullptr ) THROW( "Could not find xlink for orphan product - 1." );
            ancestry = ancestry->ancestor( );
            if( ancestry == nullptr ) THROW( "Could not find xlink for orphan product - 2." );
            if( ancestry->moniker( ) != crossSectionSumMoniker ) {
                ancestry = ancestry->ancestor( );
                if( ancestry == nullptr ) THROW( "Could not find xlink for orphan product - 3." );
            }
            GIDI::Sums::CrossSectionSum const *crossSectionSum = static_cast<GIDI::Sums::CrossSectionSum const *>( ancestry );
            GIDI::Sums::Summands const &summands = crossSectionSum->summands( );
            for( std::size_t i1 = 0; i1 < summands.size( ); ++i1 ) {
                GIDI::Sums::Summand::Base const *summand = summands[i1];

                GIDI::Ancestry const *ancestry = a_protare.findInAncestry( summand->href( ) );
                if( ancestry == nullptr ) THROW( "Could not find href for summand - 1." );
                ancestry = ancestry->ancestor( );
                if( ancestry == nullptr ) THROW( "Could not find href for summand - 2." );

                GIDI::Reaction const *GIDI_reaction2 = static_cast<GIDI::Reaction const *>( ancestry );
                for( MCGIDI_VectorSizeType reactionIndex2 = 0; reactionIndex2 < m_reactions.size( ); ++reactionIndex2 ) {
                    std::string label( m_reactions[reactionIndex2]->label( ).c_str( ) );

                    if( label == GIDI_reaction2->label( ) ) {
                        m_reactions[reactionIndex2]->associatedOrphanProductIndex( static_cast<int>( m_orphanProducts.size( ) ) - 1 );
                        m_reactions[reactionIndex2]->associatedOrphanProduct( reaction2 );
                        break;
                    }
                }
            }
        }
    }

    if( m_continuousEnergy ) {
        m_heatedCrossSections.update( setupInfo, a_settings, a_particles, a_domainHash, a_temperatureInfos, GIDI_reactions, GIDI_orphanProducts, m_fixedGrid );
        m_hasURR_probabilityTables = m_heatedCrossSections.hasURR_probabilityTables( );
        m_URR_domainMin = m_heatedCrossSections.URR_domainMin( );
        m_URR_domainMax = m_heatedCrossSections.URR_domainMax( ); }
    else {
        m_heatedMultigroupCrossSections.update( a_protare, setupInfo, a_settings, a_particles, a_temperatureInfos, GIDI_reactions, GIDI_orphanProducts );
    }

    if( ( photonIndex( ) != projectileIndex( ) ) && ( a_settings.upscatterModel( ) == Sampling::Upscatter::Model::A ) ) {
        GIDI::Styles::MultiGroup const *multiGroup = a_protare.styles( ).multiGroup( a_settings.upscatterModelALabel( ) );
        if( multiGroup == nullptr ) THROW( "No such upscatter model A label" );

        GIDI::Transportable const *transportable = multiGroup->transportables( ).get<GIDI::Transportable>( a_protare.projectile( ).ID( ) );
        GIDI::Group const &group = transportable->group( );
        GIDI::Grid const &grid = group.grid( );
        std::vector<double> const &boundaries = grid.data( );

        m_upscatterModelAGroupVelocities.resize( boundaries.size( ) );
        for( std::size_t i1 = 0; i1 < boundaries.size( ); ++i1 ) m_upscatterModelAGroupVelocities[i1] = MCGIDI_particleBeta( projectileMass( ), boundaries[i1] );
    }
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

HOST_DEVICE ProtareSingle::~ProtareSingle( ) {

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

HOST void ProtareSingle::setUserParticleIndex( int a_particleIndex, int a_userParticleIndex ) {

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

HOST_DEVICE ProtareSingle const *ProtareSingle::protare( MCGIDI_VectorSizeType a_index ) const {

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
 
HOST_DEVICE ProtareSingle const *ProtareSingle::protareWithReaction( int a_index ) const {
 
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

HOST_DEVICE Vector<double> ProtareSingle::temperatures( MCGIDI_VectorSizeType a_index ) const {

    if( a_index != 0 ) THROW( "ProtareSingle::temperatures: a_index not 0." );
    if( m_continuousEnergy ) return( m_heatedCrossSections.temperatures( ) );
    return( m_heatedMultigroupCrossSections.temperatures( ) );
}

/* *********************************************************************************************************//**
 * Sets up the nuclear gamma braching data needed to sample gamma decays.
 *
 * @param a_setupInfo           [in]    Used internally when constructing a Protare to pass information to other constructors.
 * @param a_protare             [in]    The GIDI::Protare** whose data is to be used to construct gamma branching data.
 ***********************************************************************************************************/

HOST void ProtareSingle::setupNuclideGammaBranchStateInfos( SetupInfo &a_setupInfo, GIDI::ProtareSingle const &a_protare ) {

    PoPI::NuclideGammaBranchStateInfos const &nuclideGammaBranchStateInfos = a_protare.nuclideGammaBranchStateInfos( );
    std::vector<NuclideGammaBranchInfo *> nuclideGammaBranchInfos;

    for( std::size_t i1 = 0; i1 < nuclideGammaBranchStateInfos.size( ); ++i1 ) a_setupInfo.m_stateNamesToIndices[nuclideGammaBranchStateInfos[i1]->state( )] = (int) i1;

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

HOST_DEVICE bool ProtareSingle::hasFission( ) const {

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

HOST_DEVICE bool ProtareSingle::inURR( double a_energy ) const {

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

HOST_DEVICE void ProtareSingle::sampleBranchingGammas( Sampling::Input &a_input, double a_projectileEnergy, int a_initialStateIndex, 
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

HOST_DEVICE double ProtareSingle::crossSection( URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, bool a_sampling ) const {

    if( m_continuousEnergy ) return( m_heatedCrossSections.crossSection( a_URR_protareInfos, m_URR_index, a_hashIndex, a_temperature, a_energy ) );

    return( m_heatedMultigroupCrossSections.crossSection( a_hashIndex, a_temperature, a_sampling ) );
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

HOST_DEVICE double ProtareSingle::reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, bool a_sampling ) const {

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

HOST_DEVICE double ProtareSingle::reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, double a_temperature, double a_energy ) const {

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

HOST_DEVICE int ProtareSingle::sampleReaction( URR_protareInfos const &a_URR_protareInfos, int a_hashIndex, double a_temperature, double a_energy, 
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

HOST_DEVICE double ProtareSingle::depositionEnergy( int a_hashIndex, double a_temperature, double a_energy ) const {

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

HOST_DEVICE double ProtareSingle::depositionMomentum( int a_hashIndex, double a_temperature, double a_energy ) const {

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

HOST_DEVICE double ProtareSingle::productionEnergy( int a_hashIndex, double a_temperature, double a_energy ) const {

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

HOST_DEVICE double ProtareSingle::gain( int a_hashIndex, double a_temperature, double a_energy, int a_particleIndex ) const {

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

HOST_DEVICE void ProtareSingle::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    Protare::serialize( a_buffer, a_mode );

    MCGIDI_VectorSizeType vectorSize;
    DataBuffer *workingBuffer = &a_buffer;

    DATA_MEMBER_INT( m_URR_index, a_buffer, a_mode );
    DATA_MEMBER_CAST( m_hasURR_probabilityTables, a_buffer, a_mode, bool );
    DATA_MEMBER_FLOAT( m_URR_domainMin, a_buffer, a_mode );
    DATA_MEMBER_FLOAT( m_URR_domainMax, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_projectileMultiGroupBoundaries, a_buffer, a_mode );
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

/* *********************************************************************************************************//**
 * This method counts the number of bytes of memory allocated by *this*. That is the member needed by *this* that is greater than
 * sizeof( *this );
 ***********************************************************************************************************/

HOST_DEVICE long ProtareSingle::internalSize( ) const {

    long size = Protare::internalSize( ) + m_nuclideGammaBranchStateInfos.internalSize( );

    for( MCGIDI_VectorSizeType index = 0; index < m_nuclideGammaBranchStateInfos.size(); ++index ) {
        size += sizeof( *m_nuclideGammaBranchStateInfos[index] ) + m_nuclideGammaBranchStateInfos[index]->internalSize( );
    }

    size += m_branches.internalSize( );
    for( MCGIDI_VectorSizeType index = 0; index < m_branches.size(); ++index ) {
        size += sizeof( *m_branches[index] ) + m_branches[index]->internalSize( );
    }

    size += m_reactions.internalSize( );
    for( MCGIDI_VectorSizeType reactionIndex = 0; reactionIndex < m_reactions.size(); ++reactionIndex ) {
        size += sizeof( *m_reactions[reactionIndex] ) + m_reactions[reactionIndex]->internalSize( );
    }

    size += m_orphanProducts.internalSize();
    for( MCGIDI_VectorSizeType reactionIndex = 0; reactionIndex < m_orphanProducts.size(); ++reactionIndex ) {
        size += sizeof(*m_orphanProducts[reactionIndex]) + m_orphanProducts[reactionIndex]->internalSize();
    }

    size += m_heatedCrossSections.internalSize( );
    size += m_heatedMultigroupCrossSections.internalSize( );
    size += m_projectileMultiGroupBoundaries.internalSize( );
    size += m_projectileFixedGrid.internalSize( );
    size += m_upscatterModelAGroupVelocities.internalSize( );

    return( size );
}

}

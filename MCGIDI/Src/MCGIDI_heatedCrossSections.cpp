/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "MCGIDI.hpp"

#ifndef MCGIDI_CrossSectionLinearSubSearch
    #ifndef MCGIDI_CrossSectionBinarySubSearch
        #define MCGIDI_CrossSectionBinarySubSearch
    #endif
#endif

namespace MCGIDI {

/*
============================================================
======= HeatedReactionCrossSectionContinuousEnergy =========
============================================================
*/
HOST_DEVICE HeatedReactionCrossSectionContinuousEnergy::HeatedReactionCrossSectionContinuousEnergy( ) :
        m_offset( 0 ),
        m_threshold( 0.0 ),
        m_crossSection( ),
        m_URR_probabilityTables( nullptr ) {

}
/*
============================================================
*/
HOST HeatedReactionCrossSectionContinuousEnergy::HeatedReactionCrossSectionContinuousEnergy( int a_offset, double a_threshold, Vector<double> &a_crossSection ) :
        m_offset( a_offset ),
        m_threshold( a_threshold ),
        m_crossSection( a_crossSection ),
        m_URR_probabilityTables( nullptr ) {

}
/*
============================================================
*/
HOST HeatedReactionCrossSectionContinuousEnergy::HeatedReactionCrossSectionContinuousEnergy( double a_threshold, GIDI::Functions::Ys1d const &a_crossSection, Probabilities::ProbabilityBase2d *a_URR_probabilityTables ) :
        m_offset( a_crossSection.start( ) ),
        m_threshold( a_threshold ),
        m_crossSection( a_crossSection.Ys( ) ),
        m_URR_probabilityTables( a_URR_probabilityTables ) {

}
/*
============================================================
*/
HOST_DEVICE double HeatedReactionCrossSectionContinuousEnergy::URR_domainMin( ) const {

    if( m_URR_probabilityTables != nullptr ) return( m_URR_probabilityTables->domainMin( ) );

    return( -1.0 );    
}
/*
============================================================
*/
HOST_DEVICE double HeatedReactionCrossSectionContinuousEnergy::URR_domainMax( ) const {

    if( m_URR_probabilityTables != nullptr ) return( m_URR_probabilityTables->domainMax( ) );

    return( -1.0 );    
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

HOST_DEVICE void HeatedReactionCrossSectionContinuousEnergy::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    DATA_MEMBER_INT( m_offset, a_buffer, a_mode );
    DATA_MEMBER_FLOAT(  m_threshold, a_buffer, a_mode  );
    DATA_MEMBER_VECTOR_DOUBLE( m_crossSection, a_buffer, a_mode );
    m_URR_probabilityTables = serializeProbability2d( a_buffer, a_mode, m_URR_probabilityTables );
}

/* *********************************************************************************************************//**
 * This method counts the number of bytes of memory allocated by *this*. That is the member needed by *this* that is greater than
 * sizeof( *this );
 ***********************************************************************************************************/

HOST_DEVICE long HeatedReactionCrossSectionContinuousEnergy::sizeOf( ) const { 

    long size = (long) ( sizeof( *this ) + m_crossSection.internalSize( ) );

    if( m_URR_probabilityTables != nullptr ) size += m_URR_probabilityTables->sizeOf( ) + m_URR_probabilityTables->internalSize( );

    return( size );
}

/*
============================================================
=================== ContinuousEnergyGain ===================
============================================================
*/

/* *********************************************************************************************************//**
 ***********************************************************************************************************/
ContinuousEnergyGain::ContinuousEnergyGain( ) :
        m_particleIndex( -1 ),
        m_userParticleIndex( -1 ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

ContinuousEnergyGain::ContinuousEnergyGain( int a_particleIndex, std::size_t a_size ) :
        m_particleIndex( a_particleIndex ),
        m_userParticleIndex( -1 ),
        m_gain( a_size, 0.0 ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

ContinuousEnergyGain &ContinuousEnergyGain::operator=( ContinuousEnergyGain const &a_continuousEnergyGain ) {

    m_particleIndex = a_continuousEnergyGain.particleIndex( );
    m_userParticleIndex = a_continuousEnergyGain.userParticleIndex( );
    m_gain = a_continuousEnergyGain.gain( );

    return( *this );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

HOST_DEVICE double ContinuousEnergyGain::gain( int a_energy_index, double a_energy_fraction ) const {

    return( a_energy_fraction * m_gain[a_energy_index] + ( 1.0 - a_energy_fraction ) * m_gain[a_energy_index+1] );
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

HOST_DEVICE void ContinuousEnergyGain::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    DATA_MEMBER_INT( m_particleIndex, a_buffer, a_mode );
    DATA_MEMBER_INT( m_userParticleIndex, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_gain, a_buffer, a_mode );
}

/* *********************************************************************************************************//**
 * This method counts the number of bytes of memory allocated by *this*. That is the member needed by *this* that is greater than
 * sizeof( *this );
 ***********************************************************************************************************/

HOST_DEVICE long ContinuousEnergyGain::sizeOf( ) const {

    return( sizeof( *this ) + m_gain.internalSize( ) );
}

/*
============================================================
=========== HeatedCrossSectionContinuousEnergy =============
============================================================
*/
HOST_DEVICE HeatedCrossSectionContinuousEnergy::HeatedCrossSectionContinuousEnergy( ) :
        m_hashIndices( ),
        m_energies( ),
        m_totalCrossSection( ),
        m_depositionEnergy( ),
        m_depositionMomentum( ),
        m_productionEnergy( ),
        m_gains( ),
        m_reactionsInURR_region( ),
        m_reactionCrossSections( ) {

}
/*
============================================================
*/
HOST HeatedCrossSectionContinuousEnergy::HeatedCrossSectionContinuousEnergy( SetupInfo &a_setupInfo, Transporting::MC const &a_settings, GIDI::Transporting::Particles const &a_particles, 
                DomainHash const &a_domainHash, GIDI::Styles::TemperatureInfo const &a_temperatureInfo, std::vector<GIDI::Reaction const *> const &a_reactions,
                bool a_fixedGrid ) :
        m_hashIndices( ),
        m_energies( ),
        m_totalCrossSection( ),
        m_depositionEnergy( ),
        m_depositionMomentum( ),
        m_productionEnergy( ),
        m_gains( ),
        m_reactionsInURR_region( ),
        m_reactionCrossSections( ) {

    std::string label( a_temperatureInfo.griddedCrossSection( ) );
    std::string URR_label( a_temperatureInfo.URR_probabilityTables( ) );

// BRB6
    GIDI::Styles::GriddedCrossSection const &griddedCrossSectionStyle = 
            static_cast<GIDI::Styles::GriddedCrossSection const &>( *a_settings.styles( )->get<GIDI::Styles::Base>( label ) );
    GIDI::Grid const &grid = griddedCrossSectionStyle.grid( );

    std::vector<double> const &energies = grid.data( );
    std::vector<double> const &fixedGridPoints = a_settings.fixedGridPoints( );
    std::vector<int> fixedGridIndices( fixedGridPoints.size( ) );
    if( a_fixedGrid ) {
        for( int i1 = 0; i1 < static_cast<int>( fixedGridPoints.size( ) ); ++i1 ) {
            fixedGridIndices[i1] = static_cast<int>( binarySearchVector( fixedGridPoints[i1], energies ) );
        }
        m_energies = fixedGridPoints; }
    else {
        m_energies = energies;
    }

    m_hashIndices = a_domainHash.map( m_energies );

    int reactionIndex = 0;
    GIDI::Axes axes;
    std::vector<double> dummy;
    GIDI::Functions::Ys1d totalCrossSection( axes, ptwXY_interpolationLinLin, 0, dummy );
    GIDI::Functions::Ys1d fixedGridCrossSection( axes, ptwXY_interpolationLinLin, 0, fixedGridPoints );
    m_reactionCrossSections.resize( a_reactions.size( ) );
    for( std::vector<GIDI::Reaction const *>::const_iterator reactionIter = a_reactions.begin( ); reactionIter != a_reactions.end( ); ++reactionIter, ++reactionIndex ) {
        GIDI::Suite const &reactionCrossSectionSuite = (*reactionIter)->crossSection( );
        GIDI::Functions::Ys1d const *reactionCrossSection3 = reactionCrossSectionSuite.get<GIDI::Functions::Ys1d>( label );
        Probabilities::ProbabilityBase2d *URR_probabilityTables = nullptr;

        if( a_settings.want_URR_probabilityTables( ) & ( URR_label != "" ) ) {
            if( reactionCrossSectionSuite.has( URR_label ) ) {
                GIDI::Functions::URR_probabilityTables1d const &URR_probability_tables1d( *reactionCrossSectionSuite.get<GIDI::Functions::URR_probabilityTables1d>( URR_label ) );
                URR_probabilityTables = Probabilities::parseProbability2d( URR_probability_tables1d.function2d( ), nullptr );
            }
        }

        if( a_fixedGrid ) {
            GIDI::Functions::Ys1d *reactionCrossSection4 = &fixedGridCrossSection;

            int start = 0;

            if( energies[reactionCrossSection3->start( )] > fixedGridPoints[0] ) {
                start = static_cast<MCGIDI_VectorSizeType>( binarySearchVector( energies[reactionCrossSection3->start( )], fixedGridPoints ) ) + 1;
            }

            for( int i1 = 0; i1 < start; ++i1 ) reactionCrossSection4->set( i1, 0.0 );
            for( int i1 = start; i1 < static_cast<int>( fixedGridPoints.size( ) ); ++i1 ) {
                int index = fixedGridIndices[i1];
                double fraction = ( fixedGridPoints[i1] - energies[index] ) / ( energies[index+1] - energies[index] );

                index -= reactionCrossSection3->start( );
                reactionCrossSection4->set( i1, ( 1.0 - fraction ) * (*reactionCrossSection3)[index] + fraction * (*reactionCrossSection3)[index+1] );
            }
            reactionCrossSection3 = &fixedGridCrossSection;
        }
        m_reactionCrossSections[reactionIndex] = new HeatedReactionCrossSectionContinuousEnergy( (*reactionIter)->crossSectionThreshold( ), *reactionCrossSection3, URR_probabilityTables );
        totalCrossSection += *reactionCrossSection3;
    }
    m_totalCrossSection.resize( totalCrossSection.length( ), 0.0 );
    for( MCGIDI_VectorSizeType i1 = 0; i1 < static_cast<MCGIDI_VectorSizeType>( totalCrossSection.size( ) ); ++i1 ) m_totalCrossSection[i1+totalCrossSection.start()] = totalCrossSection[i1];

    if( hasURR_probabilityTables( ) ) {
        std::vector<int> reactions_in_URR_region;

        for( int reactionIndex = 0; reactionIndex < numberOfReactions( ); ++reactionIndex ) {
            if( m_reactionCrossSections[reactionIndex]->threshold( ) < URR_domainMax( ) ) reactions_in_URR_region.push_back( reactionIndex );
        }

        m_reactionsInURR_region.resize( reactions_in_URR_region.size( ) );
        for( std::size_t i1 = 0; i1 < reactions_in_URR_region.size( ); ++i1 ) m_reactionsInURR_region[i1] = reactions_in_URR_region[i1];
    }

    m_depositionEnergy.resize( totalCrossSection.length( ), 0.0 );
    m_depositionMomentum.resize( totalCrossSection.length( ), 0.0 );
    m_productionEnergy.resize( totalCrossSection.length( ), 0.0 );

    m_gains.resize( a_particles.particles( ).size( ) );
    int i1 = 0;
    int projectileGainIndex = -1;
    for( std::map<std::string, GIDI::Transporting::Particle>::const_iterator particle = a_particles.particles( ).begin( ); particle != a_particles.particles( ).end( );
                    ++particle, ++i1 ) {
        int particleIndex = a_setupInfo.m_particleIndices[particle->first];

        if( particleIndex == a_setupInfo.m_protare.projectileIndex( ) ) projectileGainIndex = i1;
        m_gains[i1] = ContinuousEnergyGain( particleIndex, totalCrossSection.length( ) );
    }

    std::vector< std::vector<double> > gains( a_particles.particles( ).size( ) );
    reactionIndex = 0;
    for( std::vector<GIDI::Reaction const *>::const_iterator reactionIter = a_reactions.begin( ); reactionIter != a_reactions.end( ); ++reactionIter, ++reactionIndex ) {
        HeatedReactionCrossSectionContinuousEnergy &MCGIDI_reaction_cross_section = *m_reactionCrossSections[reactionIndex];
        int offset = MCGIDI_reaction_cross_section.offset( );

        GIDI::Functions::Function1dForm const &available_energy = *(*reactionIter)->availableEnergy( ).get<GIDI::Functions::Function1dForm>( 0 );
        GIDI::Functions::Function1dForm const &available_momentum = *(*reactionIter)->availableMomentum( ).get<GIDI::Functions::Function1dForm>( 0 );

        std::vector<double> deposition_energy( m_energies.size( ), 0.0 );
        std::vector<double> deposition_momentum( m_energies.size( ), 0.0 );
        std::vector<double> production_energy( m_energies.size( ), 0.0 );
        for( std::size_t i1 = 0; i1 < gains.size( ); ++i1 ) gains[i1] = std::vector<double>( m_energies.size( ), 0.0 );

        for( int energy_index = offset; energy_index < m_energies.size( ); ++energy_index ) {
            double energy = m_energies[energy_index];

            deposition_energy[energy_index] = available_energy.evaluate( energy );
            deposition_momentum[energy_index] = available_momentum.evaluate( energy );
            production_energy[energy_index] = deposition_energy[energy_index] - energy;

            int i1 = 0;
            for( std::map<std::string, GIDI::Transporting::Particle>::const_iterator particle = a_particles.particles( ).begin( ); particle != a_particles.particles( ).end( ); 
                    ++particle, ++i1 ) {
                double product_energy, product_momentum, product_gain;

                if( (*reactionIter)->isPairProduction( ) && ( particle->first == PoPI::IDs::photon ) ) {
                    product_energy = 2 * PoPI_electronMass_MeV_c2;
                    product_momentum = 0.0;
                    product_gain = 2.0; }
                else {
                    (*reactionIter)->continuousEnergyProductData( particle->first, energy, product_energy, product_momentum, product_gain );
                }
                if( i1 == projectileGainIndex ) --product_gain;

                deposition_energy[energy_index] -= product_energy;
                deposition_momentum[energy_index] -= product_momentum;
                gains[i1][energy_index] = product_gain;
            }
        }

        for( int energy_index = offset; energy_index < m_energies.size( ); ++energy_index ) {
            double cross_section = MCGIDI_reaction_cross_section.crossSection( energy_index - offset );

            m_depositionEnergy[energy_index] += cross_section * deposition_energy[energy_index];
            m_depositionMomentum[energy_index] += cross_section * deposition_momentum[energy_index];
            m_productionEnergy[energy_index] += cross_section * production_energy[energy_index];
            for( MCGIDI_VectorSizeType i1 = 0; i1 < m_gains.size( ); ++i1 ) {
                m_gains[i1].adjustGain( energy_index, cross_section * gains[i1][energy_index] );
            }
        }
    }
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

HOST_DEVICE HeatedCrossSectionContinuousEnergy::~HeatedCrossSectionContinuousEnergy( ) {

    for( Vector<HeatedReactionCrossSectionContinuousEnergy *>::const_iterator iter = m_reactionCrossSections.begin( ); iter < m_reactionCrossSections.end( ); ++iter ) delete *iter;
}
/*
=========================================================
*/
HOST_DEVICE int HeatedCrossSectionContinuousEnergy::evaluationInfo( int a_hashIndex, double a_energy, double *a_energyFraction ) const {

    *a_energyFraction = 1.0;

    if( a_energy <= m_energies[0] ) return( 0 );
    if( a_energy >= m_energies.back( ) ) {
        *a_energyFraction = 0.0;
        return( (int) ( m_energies.size( ) - 2 ) );
    }

    int index1 = m_hashIndices[a_hashIndex];

#ifdef MCGIDI_CrossSectionLinearSubSearch
    while( m_energies[index1] > a_energy ) --index1;
    while( m_energies[index1] < a_energy ) ++index1;
    --index1;
#endif

#ifdef MCGIDI_CrossSectionBinarySubSearch
    int index2 = m_hashIndices[a_hashIndex];
    int index3 = m_energies.size( ) - 1;
    if( ( a_hashIndex + 1 ) < m_hashIndices.size( ) ) index3 = m_hashIndices[a_hashIndex+1] + 1;
    if( index3 == m_energies.size( ) ) --index3;
    if( index2 != index3 ) index2 = (int) binarySearchVectorBounded( a_energy, m_energies, index2, index3, false );
#endif

#ifdef MCGIDI_CrossSectionBinarySubSearch
    #ifdef MCGIDI_CrossSectionLinearSubSearch
        if( index1 != index2 ) {
            std::cerr << "Help " << index1 << "  " << index2 << std::endl;
        }
    #endif
    index1 = index2;
#endif

    *a_energyFraction = ( m_energies[index1+1] - a_energy ) / ( m_energies[index1+1] - m_energies[index1] );
    return( index1 );
}
/*
=========================================================
*/
HOST_DEVICE bool HeatedCrossSectionContinuousEnergy::hasURR_probabilityTables( ) const {

    for( MCGIDI_VectorSizeType i1 = 0; i1 < m_reactionCrossSections.size( ); ++i1 ) {
        HeatedReactionCrossSectionContinuousEnergy *reactionCrossSection = m_reactionCrossSections[i1];

        if( reactionCrossSection->hasURR_probabilityTables( ) ) return( true );
    }

    return( false );
}
/*
=========================================================
*/
HOST_DEVICE double HeatedCrossSectionContinuousEnergy::URR_domainMin( ) const {

    for( MCGIDI_VectorSizeType i1 = 0; i1 < m_reactionCrossSections.size( ); ++i1 ) {
        HeatedReactionCrossSectionContinuousEnergy *reactionCrossSection = m_reactionCrossSections[i1];

        if( reactionCrossSection->hasURR_probabilityTables( ) ) return( reactionCrossSection->URR_domainMin( ) );
    }

    return( -1.0 );
}
/*
=========================================================
*/
HOST_DEVICE double HeatedCrossSectionContinuousEnergy::URR_domainMax( ) const {

    for( MCGIDI_VectorSizeType i1 = 0; i1 < m_reactionCrossSections.size( ); ++i1 ) {
        HeatedReactionCrossSectionContinuousEnergy *reactionCrossSection = m_reactionCrossSections[i1];

        if( reactionCrossSection->hasURR_probabilityTables( ) ) return( reactionCrossSection->URR_domainMax( ) );
    }

    return( -1.0 );
}
/*
=========================================================
*/
HOST_DEVICE double HeatedCrossSectionContinuousEnergy::crossSection( URR_protareInfos const &a_URR_protareInfos, int a_URR_index, int a_hashIndex, double a_energy, bool a_sampling ) const {

    double energy_fraction;
    int energy_index = evaluationInfo( a_hashIndex, a_energy, &energy_fraction );

    if( a_URR_index >= 0 ) {
        URR_protareInfo const &URR_protare_info = a_URR_protareInfos[a_URR_index];

        if( URR_protare_info.m_inURR ) {
            double cross_section = 0.0;

            for( MCGIDI_VectorSizeType i1 = 0; i1 < m_reactionsInURR_region.size( ); ++i1 ) {
                cross_section += reactionCrossSection2( m_reactionsInURR_region[i1], a_URR_protareInfos, a_URR_index, a_energy, energy_index, energy_fraction, false );
            }

            return( cross_section );
        }
    }

    return( energy_fraction * m_totalCrossSection[energy_index] + ( 1.0 - energy_fraction ) * m_totalCrossSection[energy_index+1] );
}
/*
=========================================================
*/
HOST_DEVICE double HeatedCrossSectionContinuousEnergy::reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_URR_index, int a_hashIndex, double a_energy, bool a_sampling ) const {

    double energyFraction;

    int energyIndex = evaluationInfo( a_hashIndex, a_energy, &energyFraction );
    return( reactionCrossSection2( a_reactionIndex, a_URR_protareInfos, a_URR_index, a_energy, energyIndex, energyFraction ) );
}
/*
=========================================================
*/
HOST_DEVICE double HeatedCrossSectionContinuousEnergy::reactionCrossSection2( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_URR_index, 
        double a_energy, int a_energyIndex, double a_energyFraction, bool a_sampling ) const {

    HeatedReactionCrossSectionContinuousEnergy const &reaction = *m_reactionCrossSections[a_reactionIndex];
    double URR_cross_section_factor = 1.0;

    if( a_URR_index >= 0 ) {
        URR_protareInfo const &URR_protare_info = a_URR_protareInfos[a_URR_index];

        if( URR_protare_info.m_inURR ) {
            if( reaction.URR_probabilityTables( ) != nullptr ) URR_cross_section_factor = reaction.URR_probabilityTables( )->sample( a_energy, URR_protare_info.m_rng_Value, nullptr, nullptr );
        }
    }

    return( URR_cross_section_factor * ( a_energyFraction * reaction.crossSection( a_energyIndex ) + ( 1.0 - a_energyFraction ) * reaction.crossSection( a_energyIndex+1 ) ) );
}

/*
=========================================================
*/
HOST_DEVICE double HeatedCrossSectionContinuousEnergy::reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_URR_index, 
                double a_energy_in ) const {

    int energyIndex = static_cast<int>( binarySearchVector( a_energy_in, m_energies ) );
    double energyFraction;

    if( energyIndex < 0 ) {
        if( energyIndex == -1 ) {
            energyIndex = static_cast<int>( m_energies.size( ) ) - 2;
            energyFraction = 0.0; }
        else {
            energyIndex = 0;
            energyFraction = 1.0;
        } }
    else {
        energyFraction = ( m_energies[energyIndex+1] - a_energy_in ) / ( m_energies[energyIndex+1] - m_energies[energyIndex] );
    }

    return( reactionCrossSection2( a_reactionIndex, a_URR_protareInfos, a_URR_index, a_energy_in, energyIndex, energyFraction, false ) );
}

/* *********************************************************************************************************//**
 * Returns the index of a sampled reaction for a projectile with energy *a_energy* and total cross section
 * *a_crossSection*. Random numbers are obtained via *a_userrng* and *a_rngState*.
 *
 * @param a_hashIndex           [in]    Specifies the action of this method.
 * @param a_energy              [in]    The energy of the projectile.
 ***********************************************************************************************************/

HOST_DEVICE double HeatedCrossSectionContinuousEnergy::depositionEnergy( int a_hashIndex, double a_energy ) const {

    double energy_fraction;
    int energy_index = evaluationInfo( a_hashIndex, a_energy, &energy_fraction );

    return( energy_fraction * m_depositionEnergy[energy_index] + ( 1.0 - energy_fraction ) * m_depositionEnergy[energy_index+1] );
}

/* *********************************************************************************************************//**
 * Returns the index of a sampled reaction for a projectile with energy *a_energy* and total cross section
 * *a_crossSection*. Random numbers are obtained via *a_userrng* and *a_rngState*.
 *
 * @param a_hashIndex           [in]    Specifies the action of this method.
 * @param a_energy              [in]    The energy of the projectile.
 ***********************************************************************************************************/

HOST_DEVICE double HeatedCrossSectionContinuousEnergy::depositionMomentum( int a_hashIndex, double a_energy ) const {

    double energy_fraction;
    int energy_index = evaluationInfo( a_hashIndex, a_energy, &energy_fraction );

    return( energy_fraction * m_depositionMomentum[energy_index] + ( 1.0 - energy_fraction ) * m_depositionMomentum[energy_index+1] );
}

/* *********************************************************************************************************//**
 * Returns the index of a sampled reaction for a projectile with energy *a_energy* and total cross section
 * *a_crossSection*. Random numbers are obtained via *a_userrng* and *a_rngState*.
 *
 * @param a_hashIndex           [in]    Specifies the action of this method.
 * @param a_energy              [in]    The energy of the projectile.
 ***********************************************************************************************************/

HOST_DEVICE double HeatedCrossSectionContinuousEnergy::productionEnergy( int a_hashIndex, double a_energy ) const {

    double energy_fraction;
    int energy_index = evaluationInfo( a_hashIndex, a_energy, &energy_fraction );

    return( energy_fraction * m_productionEnergy[energy_index] + ( 1.0 - energy_fraction ) * m_productionEnergy[energy_index+1] );
}

/* *********************************************************************************************************//**
 * Returns the index of a sampled reaction for a projectile with energy *a_energy* and total cross section
 * *a_crossSection*. Random numbers are obtained via *a_userrng* and *a_rngState*.
 *
 * @param a_hashIndex           [in]    Specifies the action of this method.
 * @param a_energy              [in]    The energy of the projectile.
 * @param a_particleIndex       [in]    The index of the particle whose gain is requested.
 ***********************************************************************************************************/

HOST_DEVICE double HeatedCrossSectionContinuousEnergy::gain( int a_hashIndex, double a_energy, int a_particleIndex ) const {

    double energy_fraction;
    int energy_index = evaluationInfo( a_hashIndex, a_energy, &energy_fraction );

    for( MCGIDI_VectorSizeType i1 = 0; i1 < m_gains.size( ); ++i1 ) {
        if( a_particleIndex == m_gains[i1].particleIndex( ) ) return( m_gains[i1].gain( energy_index, energy_fraction ) );
    }

    return( 0.0 );
}

/* *********************************************************************************************************//**
 * Updates the m_userParticleIndex to *a_userParticleIndex* for all particles with PoPs index *a_particleIndex*.
 *
 * @param a_particleIndex       [in]    The PoPs id of the particle whose userPid is to be set.
 * @param a_userParticleIndex   [in]    The particle id specified by the user.
 ***********************************************************************************************************/

HOST void HeatedCrossSectionContinuousEnergy::setUserParticleIndex( int a_particleIndex, int a_userParticleIndex ) {

    for( auto iter = m_gains.begin( ); iter != m_gains.end( ); ++iter ) iter->setUserParticleIndex( a_particleIndex, a_userParticleIndex );
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

HOST_DEVICE void HeatedCrossSectionContinuousEnergy::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    DATA_MEMBER_VECTOR_INT( m_hashIndices, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_energies, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_totalCrossSection, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_depositionEnergy, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_depositionMomentum, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_productionEnergy, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_INT( m_reactionsInURR_region, a_buffer, a_mode );

    MCGIDI_VectorSizeType vectorSize = m_reactionCrossSections.size( );
    int vectorSizeInt = (int) vectorSize;
    DATA_MEMBER_INT( vectorSizeInt, a_buffer, a_mode );
    vectorSize = (MCGIDI_VectorSizeType) vectorSizeInt;

    if( a_mode == DataBuffer::Mode::Unpack ) m_reactionCrossSections.resize( vectorSize, &a_buffer.m_placement );
    for( MCGIDI_VectorSizeType memberIndex = 0; memberIndex < vectorSize; ++memberIndex ) {
        if( a_mode == DataBuffer::Mode::Unpack ) {
            if( a_buffer.m_placement != nullptr ) {
                m_reactionCrossSections[memberIndex] = new(a_buffer.m_placement) HeatedReactionCrossSectionContinuousEnergy;
                a_buffer.incrementPlacement( sizeof( HeatedReactionCrossSectionContinuousEnergy ) ); }
            else {
                m_reactionCrossSections[memberIndex] = new HeatedReactionCrossSectionContinuousEnergy;
            }
        }
        m_reactionCrossSections[memberIndex]->serialize( a_buffer, a_mode );
    }

    vectorSize = m_gains.size( );
    vectorSizeInt = (int) vectorSize;
    DATA_MEMBER_INT( vectorSizeInt, a_buffer, a_mode );
    vectorSize = (MCGIDI_VectorSizeType) vectorSizeInt;
    if( a_mode == DataBuffer::Mode::Unpack ) m_gains.resize( vectorSize, &a_buffer.m_placement );
    for( MCGIDI_VectorSizeType memberIndex = 0; memberIndex < vectorSize; ++memberIndex ) {
        m_gains[memberIndex].serialize( a_buffer, a_mode );
    }
}
/*
=========================================================
*/
HOST_DEVICE long HeatedCrossSectionContinuousEnergy::sizeOf( ) const {

    long size = (long) ( sizeof( *this ) + m_hashIndices.internalSize( ) + m_energies.internalSize( ) + 
            m_totalCrossSection.internalSize( ) + m_depositionEnergy.internalSize( ) + m_depositionMomentum.internalSize( ) + 
            m_productionEnergy.internalSize( ) + m_reactionCrossSections.internalSize( ) + m_reactionsInURR_region.size( ) );

    for( MCGIDI_VectorSizeType i1 = 0; i1 < m_reactionCrossSections.size( ); ++i1 ) size += m_reactionCrossSections[i1]->sizeOf( );
    for( MCGIDI_VectorSizeType i1 = 0; i1 < m_gains.size( ); ++i1 ) size += m_gains[i1].sizeOf( );
    return( size );
}

/*
============================================================
=========== HeatedCrossSectionsContinuousEnergy ============
============================================================
*/
HOST_DEVICE HeatedCrossSectionsContinuousEnergy::HeatedCrossSectionsContinuousEnergy( ) :
        m_temperatures( ),
        m_thresholds( ),
        m_heatedCrossSections( ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

HOST_DEVICE HeatedCrossSectionsContinuousEnergy::~HeatedCrossSectionsContinuousEnergy( ) {

    for( Vector<HeatedCrossSectionContinuousEnergy *>::const_iterator iter = m_heatedCrossSections.begin( ); iter != m_heatedCrossSections.end( ); ++iter ) delete *iter;
}
/*
=========================================================
*/
HOST void HeatedCrossSectionsContinuousEnergy::update( SetupInfo &a_setupInfo, Transporting::MC const &a_settings, GIDI::Transporting::Particles const &a_particles, 
                DomainHash const &a_domainHash, GIDI::Styles::TemperatureInfos const &a_temperatureInfos, std::vector<GIDI::Reaction const *> const &a_reactions, 
                std::vector<GIDI::Reaction const *> const &a_orphanProducts, bool a_fixedGrid ) {

    m_temperatures.reserve( a_temperatureInfos.size( ) );
    m_heatedCrossSections.reserve( a_temperatureInfos.size( ) );

    for( GIDI::Styles::TemperatureInfos::const_iterator iter = a_temperatureInfos.begin( ); iter != a_temperatureInfos.end( ); ++iter ) {
        m_temperatures.push_back( iter->temperature( ).value( ) );
        m_heatedCrossSections.push_back( new HeatedCrossSectionContinuousEnergy( a_setupInfo, a_settings, a_particles, a_domainHash, *iter, a_reactions, a_fixedGrid ) );
    }

    m_thresholds.resize( m_heatedCrossSections[0]->numberOfReactions( ) );
    for( int i1 = 0; i1 < m_heatedCrossSections[0]->numberOfReactions( ); ++i1 ) m_thresholds[i1] = m_heatedCrossSections[0]->threshold( i1 );
}
/*
=========================================================
*/
HOST_DEVICE double HeatedCrossSectionsContinuousEnergy::crossSection( URR_protareInfos const &a_URR_protareInfos, int a_URR_index, int a_hashIndex, double a_temperature, double a_energy, bool a_sampling ) const {

    int i1, number_of_temperatures = static_cast<int>( m_temperatures.size( ) );
    double cross_section;

    if( a_temperature <= m_temperatures[0] ) { 
        cross_section = m_heatedCrossSections[0]->crossSection( a_URR_protareInfos, a_URR_index, a_hashIndex, a_energy, a_sampling ); }
    else if( a_temperature >= m_temperatures.back( ) ) {
        cross_section = m_heatedCrossSections.back( )->crossSection( a_URR_protareInfos, a_URR_index, a_hashIndex, a_energy, a_sampling ); }
    else {
        for( i1 = 0; i1 < number_of_temperatures; ++i1 ) if( a_temperature < m_temperatures[i1] ) break;
        double fraction = ( a_temperature - m_temperatures[i1-1] ) / ( m_temperatures[i1] - m_temperatures[i1-1] );
        cross_section = ( 1. - fraction ) * m_heatedCrossSections[i1-1]->crossSection( a_URR_protareInfos, a_URR_index, a_hashIndex, a_energy, a_sampling )
                            + fraction * m_heatedCrossSections[i1]->crossSection( a_URR_protareInfos, a_URR_index, a_hashIndex, a_energy, a_sampling );
    }

    return( cross_section );
}
/*
=========================================================
*/
HOST_DEVICE double HeatedCrossSectionsContinuousEnergy::reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_URR_index, 
                int a_hashIndex, double a_temperature, double a_energy, bool a_sampling ) const {

    int i1, number_of_temperatures = static_cast<int>( m_temperatures.size( ) );
    double cross_section;

    if( a_temperature <= m_temperatures[0] ) {
        cross_section = m_heatedCrossSections[0]->reactionCrossSection( a_reactionIndex, a_URR_protareInfos, a_URR_index, a_hashIndex, a_energy, a_sampling ); }
    else if( a_temperature >= m_temperatures.back( ) ) {
        cross_section = m_heatedCrossSections.back( )->reactionCrossSection( a_reactionIndex, a_URR_protareInfos, a_URR_index, a_hashIndex, a_energy, a_sampling ); }
    else {
        for( i1 = 0; i1 < number_of_temperatures; ++i1 ) if( a_temperature < m_temperatures[i1] ) break;
        double fraction = ( a_temperature - m_temperatures[i1-1] ) / ( m_temperatures[i1] - m_temperatures[i1-1] );
        cross_section = ( 1. - fraction ) * m_heatedCrossSections[i1-1]->reactionCrossSection( a_reactionIndex, a_URR_protareInfos, a_URR_index, a_hashIndex, a_energy, a_sampling )
                            + fraction * m_heatedCrossSections[i1]->reactionCrossSection( a_reactionIndex, a_URR_protareInfos, a_URR_index, a_hashIndex, a_energy, a_sampling );
    }

    return( cross_section );
}

/*
=========================================================
*/
HOST_DEVICE double HeatedCrossSectionsContinuousEnergy::reactionCrossSection( int a_reactionIndex, URR_protareInfos const &a_URR_protareInfos, int a_URR_index,
                double a_temperature, double a_energy_in ) const {

    int i1, number_of_temperatures = static_cast<int>( m_temperatures.size( ) );
    double cross_section;

    if( a_temperature <= m_temperatures[0] ) {
        cross_section = m_heatedCrossSections[0]->reactionCrossSection( a_reactionIndex, a_URR_protareInfos, a_URR_index, a_energy_in ); }
    else if( a_temperature >= m_temperatures.back( ) ) {
        cross_section = m_heatedCrossSections.back( )->reactionCrossSection( a_reactionIndex, a_URR_protareInfos, a_URR_index, a_energy_in ); }
    else {
        for( i1 = 0; i1 < number_of_temperatures; ++i1 ) if( a_temperature < m_temperatures[i1] ) break;
        double fraction = ( a_temperature - m_temperatures[i1-1] ) / ( m_temperatures[i1] - m_temperatures[i1-1] );
        cross_section = ( 1. - fraction ) * m_heatedCrossSections[i1-1]->reactionCrossSection( a_reactionIndex, a_URR_protareInfos, a_URR_index, a_energy_in )
                            + fraction * m_heatedCrossSections[i1]->reactionCrossSection( a_reactionIndex, a_URR_protareInfos, a_URR_index, a_energy_in );
    }

    return( cross_section );
}

/*
=========================================================
*/
HOST_DEVICE int HeatedCrossSectionsContinuousEnergy::sampleReaction( URR_protareInfos const &a_URR_protareInfos, int a_URR_index, int a_hashIndex, double a_temperature, 
                double a_energy, double a_crossSection, double (*userrng)( void * ), void *rngState ) const {

    int i1, sampled_reaction_index, temperatureIndex1, temperatureIndex2, number_of_temperatures = static_cast<int>( m_temperatures.size( ) );
    double sampleCrossSection = a_crossSection * userrng( rngState );

    if( a_temperature <= m_temperatures[0] ) {
        temperatureIndex1 = 0;
        temperatureIndex2 = temperatureIndex1; }
    else if( a_temperature >= m_temperatures.back( ) ) {
        temperatureIndex1 = static_cast<int>( m_temperatures.size( ) ) - 1;
        temperatureIndex2 = temperatureIndex1; }
    else {
        for( i1 = 0; i1 < number_of_temperatures; ++i1 ) if( a_temperature < m_temperatures[i1] ) break;
        temperatureIndex1 = i1 - 1;
        temperatureIndex2 = i1;
    }

    int numberOfReactions = m_heatedCrossSections[0]->numberOfReactions( );
    double energyFraction1, energyFraction2, crossSectionSum = 0.0;

    HeatedCrossSectionContinuousEnergy &heatedCrossSection1 = *m_heatedCrossSections[temperatureIndex1];
    int energyIndex1 = heatedCrossSection1.evaluationInfo( a_hashIndex, a_energy, &energyFraction1 );

    if( temperatureIndex1 == temperatureIndex2 ) {
        for( sampled_reaction_index = 0; sampled_reaction_index < numberOfReactions; ++sampled_reaction_index ) {
            crossSectionSum += heatedCrossSection1.reactionCrossSection2( sampled_reaction_index, a_URR_protareInfos, a_URR_index, a_energy, energyIndex1, energyFraction1 );
            if( crossSectionSum >= sampleCrossSection ) break;
        } }
    else {
        double temperatureFraction2 = ( a_temperature - m_temperatures[temperatureIndex1] ) / ( m_temperatures[temperatureIndex2] - m_temperatures[temperatureIndex1] );
        double temperatureFraction1 = 1.0 - temperatureFraction2;
        HeatedCrossSectionContinuousEnergy &heatedCrossSection2 = *m_heatedCrossSections[temperatureIndex2];
        int energyIndex2 = heatedCrossSection2.evaluationInfo( a_hashIndex, a_energy, &energyFraction2 );

        for( sampled_reaction_index = 0; sampled_reaction_index < numberOfReactions; ++sampled_reaction_index ) {
            if( m_thresholds[sampled_reaction_index] >= a_energy ) continue;
            crossSectionSum += temperatureFraction1 * heatedCrossSection1.reactionCrossSection2( sampled_reaction_index, a_URR_protareInfos, a_URR_index, a_energy, energyIndex1, energyFraction1 );
            crossSectionSum += temperatureFraction2 * heatedCrossSection2.reactionCrossSection2( sampled_reaction_index, a_URR_protareInfos, a_URR_index, a_energy, energyIndex2, energyFraction2 );
            if( crossSectionSum >= sampleCrossSection ) break;
        }
    }

    if( sampled_reaction_index == numberOfReactions ) {
        if( crossSectionSum < ( 1.0 - 1e-8 ) * a_crossSection ) THROW( "HeatedCrossSectionsContinuousEnergy::sampleReaction: crossSectionSum less than a_crossSection" );
        for( sampled_reaction_index = 0; sampled_reaction_index < numberOfReactions; ++sampled_reaction_index ) {   // This should rarely happen so just pick the first reaction with non-zero cross section.
            if( heatedCrossSection1.reactionCrossSection2( sampled_reaction_index, a_URR_protareInfos, a_URR_index, a_energy, energyIndex1, energyFraction1, true ) > 0 ) break;
        }
    }

    return( sampled_reaction_index );
}

/* *********************************************************************************************************//**
 * Returns the deposition energy for target temperature *a_temperature* and projectile multi-group *a_hashIndex*.
 *
 * @param a_hashIndex           [in]    Specifies the action of this method.
 * @param a_temperature         [in]    The temperature of the target.
 * @param a_energy              [in]    The energy of the projectile.
 ***********************************************************************************************************/

HOST_DEVICE double HeatedCrossSectionsContinuousEnergy::depositionEnergy( int a_hashIndex, double a_temperature, double a_energy ) const {

    int i1, number_of_temperatures = static_cast<int>( m_temperatures.size( ) );
    double deposition_energy;

    if( a_temperature <= m_temperatures[0] ) { 
        deposition_energy = m_heatedCrossSections[0]->depositionEnergy( a_hashIndex, a_energy ); }
    else if( a_temperature >= m_temperatures.back( ) ) {
        deposition_energy = m_heatedCrossSections.back( )->depositionEnergy( a_hashIndex, a_energy ); }
    else {
        for( i1 = 0; i1 < number_of_temperatures; ++i1 ) if( a_temperature < m_temperatures[i1] ) break;
        double fraction = ( a_temperature - m_temperatures[i1-1] ) / ( m_temperatures[i1] - m_temperatures[i1-1] );
        deposition_energy = ( 1. - fraction ) * m_heatedCrossSections[i1-1]->depositionEnergy( a_hashIndex, a_energy )
                            + fraction * m_heatedCrossSections[i1]->depositionEnergy( a_hashIndex, a_energy );
    }

    return( deposition_energy );
}

/* *********************************************************************************************************//**
 * Returns the deposition momentum for target temperature *a_temperature* and projectile multi-group *a_hashIndex*.
 *
 * @param a_hashIndex           [in]    Specifies the action of this method.
 * @param a_temperature         [in]    The temperature of the target.
 * @param a_energy              [in]    The energy of the projectile.
 ***********************************************************************************************************/

HOST_DEVICE double HeatedCrossSectionsContinuousEnergy::depositionMomentum( int a_hashIndex, double a_temperature, double a_energy ) const {

    int i1, number_of_temperatures = static_cast<int>( m_temperatures.size( ) );
    double deposition_momentum;

    if( a_temperature <= m_temperatures[0] ) {
        deposition_momentum = m_heatedCrossSections[0]->depositionMomentum( a_hashIndex, a_energy ); }
    else if( a_temperature >= m_temperatures.back( ) ) {
        deposition_momentum = m_heatedCrossSections.back( )->depositionMomentum( a_hashIndex, a_energy ); }
    else {
        for( i1 = 0; i1 < number_of_temperatures; ++i1 ) if( a_temperature < m_temperatures[i1] ) break;
        double fraction = ( a_temperature - m_temperatures[i1-1] ) / ( m_temperatures[i1] - m_temperatures[i1-1] );
        deposition_momentum = ( 1. - fraction ) * m_heatedCrossSections[i1-1]->depositionMomentum( a_hashIndex, a_energy )
                            + fraction * m_heatedCrossSections[i1]->depositionMomentum( a_hashIndex, a_energy );
    }

    return( deposition_momentum );
}

/* *********************************************************************************************************//**
 * Returns the production momentum for target temperature *a_temperature* and projectile multi-group *a_hashIndex*.
 *
 * @param a_hashIndex           [in]    Specifies the action of this method.
 * @param a_temperature         [in]    The temperature of the target.
 * @param a_energy              [in]    The energy of the projectile.
 ***********************************************************************************************************/

HOST_DEVICE double HeatedCrossSectionsContinuousEnergy::productionEnergy( int a_hashIndex, double a_temperature, double a_energy ) const {

    int i1, number_of_temperatures = static_cast<int>( m_temperatures.size( ) );
    double production_energy;

    if( a_temperature <= m_temperatures[0] ) {
        production_energy = m_heatedCrossSections[0]->productionEnergy( a_hashIndex, a_energy ); }
    else if( a_temperature >= m_temperatures.back( ) ) {
        production_energy = m_heatedCrossSections.back( )->productionEnergy( a_hashIndex, a_energy ); }
    else {
        for( i1 = 0; i1 < number_of_temperatures; ++i1 ) if( a_temperature < m_temperatures[i1] ) break;
        double fraction = ( a_temperature - m_temperatures[i1-1] ) / ( m_temperatures[i1] - m_temperatures[i1-1] );
        production_energy = ( 1. - fraction ) * m_heatedCrossSections[i1-1]->productionEnergy( a_hashIndex, a_energy )
                            + fraction * m_heatedCrossSections[i1]->productionEnergy( a_hashIndex, a_energy );
    }

    return( production_energy );
}

/* *********************************************************************************************************//**
 * Returns the production momentum for target temperature *a_temperature* and projectile multi-group *a_hashIndex*.
 *
 * @param a_hashIndex           [in]    Specifies the action of this method.
 * @param a_temperature         [in]    The temperature of the target.
 * @param a_energy              [in]    The energy of the projectile.
 * @param a_particleIndex       [in]    The index of the particle whose gain is requested.
 ***********************************************************************************************************/

HOST_DEVICE double HeatedCrossSectionsContinuousEnergy::gain( int a_hashIndex, double a_temperature, double a_energy, int a_particleIndex ) const {

    int i1, number_of_temperatures = static_cast<int>( m_temperatures.size( ) );
    double production_energy;

    if( a_temperature <= m_temperatures[0] ) {
        production_energy = m_heatedCrossSections[0]->gain( a_hashIndex, a_energy, a_particleIndex ); }
    else if( a_temperature >= m_temperatures.back( ) ) {
        production_energy = m_heatedCrossSections.back( )->gain( a_hashIndex, a_energy, a_particleIndex ); }
    else {
        for( i1 = 0; i1 < number_of_temperatures; ++i1 ) if( a_temperature < m_temperatures[i1] ) break;
        double fraction = ( a_temperature - m_temperatures[i1-1] ) / ( m_temperatures[i1] - m_temperatures[i1-1] );
        production_energy = ( 1. - fraction ) * m_heatedCrossSections[i1-1]->gain( a_hashIndex, a_energy, a_particleIndex )
                            + fraction * m_heatedCrossSections[i1]->gain( a_hashIndex, a_energy, a_particleIndex );
    }

    return( production_energy );
}

/* *********************************************************************************************************//**
 * Updates the m_userParticleIndex to *a_userParticleIndex* for all particles with PoPs index *a_particleIndex*.
 *
 * @param a_particleIndex       [in]    The PoPs id of the particle whose userPid is to be set.
 * @param a_userParticleIndex   [in]    The particle id specified by the user.
 ***********************************************************************************************************/

HOST void HeatedCrossSectionsContinuousEnergy::setUserParticleIndex( int a_particleIndex, int a_userParticleIndex ) {

    for( auto iter = m_heatedCrossSections.begin( ); iter != m_heatedCrossSections.end( ); ++iter ) (*iter)->setUserParticleIndex( a_particleIndex, a_userParticleIndex );
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

HOST_DEVICE void HeatedCrossSectionsContinuousEnergy::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    DATA_MEMBER_VECTOR_DOUBLE( m_temperatures, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_thresholds, a_buffer, a_mode );

    MCGIDI_VectorSizeType vectorSize = m_heatedCrossSections.size( );
    int vectorSizeInt = (int) vectorSize;
    DATA_MEMBER_INT( vectorSizeInt, a_buffer, a_mode );
    vectorSize = (MCGIDI_VectorSizeType) vectorSizeInt;

    if( a_mode == DataBuffer::Mode::Unpack ) m_heatedCrossSections.resize( vectorSize, &a_buffer.m_placement );
    for( MCGIDI_VectorSizeType memberIndex = 0; memberIndex < vectorSize; ++memberIndex ) {
        if( a_mode == DataBuffer::Mode::Unpack ) {
            if( a_buffer.m_placement != nullptr ) {
                m_heatedCrossSections[memberIndex] = new(a_buffer.m_placement) HeatedCrossSectionContinuousEnergy;
                a_buffer.incrementPlacement( sizeof( HeatedCrossSectionContinuousEnergy ) ); }
            else {
                m_heatedCrossSections[memberIndex] = new HeatedCrossSectionContinuousEnergy;
            }
        }
        m_heatedCrossSections[memberIndex]->serialize( a_buffer, a_mode );
    }
}

/* *********************************************************************************************************//**
 * This method counts the number of bytes of memory allocated by *this*. That is the member needed by *this* that is greater than
 * sizeof( *this );
 ***********************************************************************************************************/

HOST_DEVICE long HeatedCrossSectionsContinuousEnergy::internalSize( ) const {

    long size = (long) ( m_temperatures.internalSize( ) + m_thresholds.internalSize( ) + m_heatedCrossSections.internalSize( ) );

    for( MCGIDI_VectorSizeType i1 = 0; i1 < m_heatedCrossSections.size( ); ++i1 ) size += m_heatedCrossSections[i1]->sizeOf( );
    return( size );
}

/*! \class MultiGroupGain
 * This class store a particles index and gain for a protare.
 */

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

HOST_DEVICE MultiGroupGain::MultiGroupGain( ) :
        m_particleIndex( -1 ),
        m_userParticleIndex( -1 ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

HOST MultiGroupGain::MultiGroupGain( int a_particleIndex, GIDI::Vector const &a_gain ) :
        m_particleIndex( a_particleIndex ),
        m_userParticleIndex( -1 ),
        m_gain( GIDI_VectorDoublesToMCGIDI_VectorDoubles( a_gain ) ) {

}

/* *********************************************************************************************************//**
 * @param a_multiGroupGain      [in]    The **MultiGroupGain** whose contents are to be copied.
 ***********************************************************************************************************/

HOST MultiGroupGain &MultiGroupGain::operator=( MultiGroupGain const &a_multiGroupGain ) {

    m_particleIndex = a_multiGroupGain.particleIndex( );
    m_userParticleIndex = a_multiGroupGain.userParticleIndex( );
    m_gain = a_multiGroupGain.gain( );

    return( *this );
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

HOST_DEVICE void MultiGroupGain::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    DATA_MEMBER_INT( m_particleIndex, a_buffer, a_mode );
    DATA_MEMBER_INT( m_userParticleIndex, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_gain, a_buffer, a_mode );
}


/*
============================================================
=========== HeatedReactionCrossSectionMultiGroup ===========
============================================================
*/
HOST_DEVICE HeatedReactionCrossSectionMultiGroup::HeatedReactionCrossSectionMultiGroup( ) {

}
/*
=========================================================
*/
HOST HeatedReactionCrossSectionMultiGroup::HeatedReactionCrossSectionMultiGroup( SetupInfo &a_setupInfo, Transporting::MC const &a_settings, int a_offset, std::vector<double> const &a_crossSection, 
            double a_threshold ) :
        m_threshold( a_threshold ),
        m_offset( a_offset ),
        m_crossSection( a_crossSection ),
        m_augmentedThresholdCrossSection( 0.0 ) {

    Vector<double> const &boundaries = a_setupInfo.m_protare.projectileMultiGroupBoundaries( );

    if( ( a_offset > 0 ) && ( boundaries[a_offset] < a_threshold ) ) {  // This uses the linear rejection above threshold in the group m_offset.
        if( ( boundaries[a_offset] < a_threshold ) && ( a_threshold < boundaries[a_offset+1] ) ) {
            m_augmentedThresholdCrossSection = m_crossSection[0] * 2.0 * ( a_threshold - boundaries[a_offset] ) / (  boundaries[a_offset+1] - a_threshold ); }
        else {
            m_crossSection[0] = 0.0;
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

HOST_DEVICE void HeatedReactionCrossSectionMultiGroup::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    DATA_MEMBER_FLOAT( m_threshold, a_buffer, a_mode  );
    DATA_MEMBER_INT( m_offset, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_crossSection, a_buffer, a_mode );
    DATA_MEMBER_FLOAT( m_augmentedThresholdCrossSection, a_buffer, a_mode  );
}

/*
============================================================
============= HeatedCrossSectionMultiGroup ================
============================================================
*/
HOST_DEVICE HeatedCrossSectionMultiGroup::HeatedCrossSectionMultiGroup( ) {

}
/*
=========================================================
*/
HOST HeatedCrossSectionMultiGroup::HeatedCrossSectionMultiGroup( GIDI::ProtareSingle const &a_protare, SetupInfo &a_setupInfo, 
                Transporting::MC const &a_settings, GIDI::Styles::TemperatureInfo const &a_temperatureInfo, 
                GIDI::Transporting::Particles const &a_particles, std::vector<GIDI::Reaction const *> const &a_reactions, std::string const &a_label ) :
        m_totalCrossSection( ),
        m_augmentedCrossSection( ),
        m_reactionCrossSections( ) {

    GIDI::Transporting::MG multi_group_settings( a_settings.projectileID( ), GIDI::Transporting::Mode::multiGroup, a_settings.delayedNeutrons( ) );

    GIDI::Axes axes;
    std::vector<double> dummy;
    GIDI::Functions::Ys1d totalCrossSection( axes, ptwXY_interpolationLinLin, 0, dummy );

    m_reactionCrossSections.reserve( a_reactions.size( ) );
    int index = 0;                                      // Only used for debugging.
    for( std::vector<GIDI::Reaction const *>::const_iterator reactionIter = a_reactions.begin( ); reactionIter != a_reactions.end( ); ++reactionIter, ++index ) {
        GIDI::Suite const &reactionCrossSectionSuite = (*reactionIter)->crossSection( );
        GIDI::Functions::Gridded1d const *reactionCrossSection3 = reactionCrossSectionSuite.get<GIDI::Functions::Gridded1d>( a_label );

        GIDI::Vector vector = GIDI::collapse( reactionCrossSection3->data( ), a_settings, a_particles, 0.0 );
        GIDI::Functions::Ys1d ys1d = GIDI::vector2GIDI_Ys1d( reactionCrossSection3->axes( ), vector );

        m_reactionCrossSections.push_back( new HeatedReactionCrossSectionMultiGroup( a_setupInfo, a_settings, ys1d.start( ), ys1d.Ys( ), (*reactionIter)->crossSectionThreshold( ) ) );

        totalCrossSection += ys1d;
    }

    m_totalCrossSection.resize( totalCrossSection.length( ), 0.0 );
    std::vector<double> const &Ys = totalCrossSection.Ys( );
    for( std::size_t i1 = 0; i1 < Ys.size( ); ++i1 ) m_totalCrossSection[i1+totalCrossSection.start( )] = Ys[i1];

    m_augmentedCrossSection.resize( totalCrossSection.length( ) );
    for( MCGIDI_VectorSizeType i1 = 0; i1 < m_augmentedCrossSection.size( ); ++i1 ) m_augmentedCrossSection[i1] = 0;
    for( MCGIDI_VectorSizeType i1 = 0; i1 < m_reactionCrossSections.size( ); ++i1 )
        m_augmentedCrossSection[m_reactionCrossSections[i1]->offset( )] += m_reactionCrossSections[i1]->augmentedThresholdCrossSection( );

    m_depositionEnergy = GIDI_VectorDoublesToMCGIDI_VectorDoubles( a_protare.multiGroupDepositionEnergy( multi_group_settings, a_temperatureInfo, a_particles ) );
    m_depositionMomentum = GIDI_VectorDoublesToMCGIDI_VectorDoubles( a_protare.multiGroupDepositionMomentum( multi_group_settings, a_temperatureInfo, a_particles ) );
    m_productionEnergy = GIDI_VectorDoublesToMCGIDI_VectorDoubles( a_protare.multiGroupQ( multi_group_settings, a_temperatureInfo, true ) );

    std::map<std::string, GIDI::Transporting::Particle> particles = a_particles.particles( );
    m_gains.resize( particles.size( ) );
    int i1 = 0;
    for( std::map<std::string, GIDI::Transporting::Particle>::const_iterator particle = particles.begin( ); particle != particles.end( ); ++particle, ++i1 ) {
        int particleIndex = a_setupInfo.m_particleIndices[particle->first];

        m_gains[i1] = MultiGroupGain( particleIndex, a_protare.multiGroupGain( multi_group_settings, a_temperatureInfo, particle->first ) );
    }
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

HOST_DEVICE HeatedCrossSectionMultiGroup::~HeatedCrossSectionMultiGroup( ) {

    for( Vector<HeatedReactionCrossSectionMultiGroup *>::const_iterator iter = m_reactionCrossSections.begin( ); iter < m_reactionCrossSections.end( ); ++iter ) delete *iter;
}

/* *********************************************************************************************************//**
 * Returns the multi-group cross section.
 *
 * @param a_hashIndex           [in]    The multi-group index.
 * @param a_sampling            [in]    Fix me.
 *
 * @return                              A vector of the length of the number of multi-group groups.
 ***********************************************************************************************************/

HOST_DEVICE double HeatedCrossSectionMultiGroup::crossSection( int a_hashIndex, bool a_sampling ) const { 

    double crossSection2 = m_totalCrossSection[a_hashIndex];

    if( a_sampling ) crossSection2 += m_augmentedCrossSection[a_hashIndex];

    return( crossSection2 );
}

/* *********************************************************************************************************//**
 * Returns the multi-group gain for particle with index *a_particleIndex*. If no particle is found, a Vector of all 0's is returned.
 *
 * @param a_particleIndex       [in]    The id of the particle whose gain is to be returned.
 * @param a_hashIndex           [in]    The multi-group index.
 *
 * @return                              A vector of the length of the number of multi-group groups.
 ***********************************************************************************************************/

HOST_DEVICE double HeatedCrossSectionMultiGroup::gain( int a_particleIndex, int a_hashIndex ) const {

    for( MCGIDI_VectorSizeType i1 = 0; i1 < m_gains.size( ); ++i1 ) {
        if( a_particleIndex == m_gains[i1].particleIndex( ) ) return( m_gains[i1].gain( a_hashIndex ) );
    }

    return( 0.0 );
}

/* *********************************************************************************************************//**
 * Updates the m_userParticleIndex to *a_userParticleIndex* for all particles with PoPs index *a_particleIndex*.
 *
 * @param a_particleIndex       [in]    The PoPs id of the particle whose userPid is to be set.
 * @param a_userParticleIndex   [in]    The particle id specified by the user.
 ***********************************************************************************************************/

HOST void HeatedCrossSectionMultiGroup::setUserParticleIndex( int a_particleIndex, int a_userParticleIndex ) {

    for( auto iter = m_gains.begin( ); iter != m_gains.end( ); ++iter ) iter->setUserParticleIndex( a_particleIndex, a_userParticleIndex );
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

HOST_DEVICE void HeatedCrossSectionMultiGroup::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    DATA_MEMBER_VECTOR_DOUBLE( m_totalCrossSection, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_augmentedCrossSection, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_depositionEnergy, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_depositionMomentum, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_productionEnergy, a_buffer, a_mode );

    MCGIDI_VectorSizeType vectorSize = m_reactionCrossSections.size( );
    int vectorSizeInt = (int) vectorSize;
    DATA_MEMBER_INT( vectorSizeInt, a_buffer, a_mode );
    vectorSize = (MCGIDI_VectorSizeType) vectorSizeInt;

    if( a_mode == DataBuffer::Mode::Unpack ) m_reactionCrossSections.resize( vectorSize, &a_buffer.m_placement );
    for( MCGIDI_VectorSizeType memberIndex = 0; memberIndex < vectorSize; ++memberIndex ) {
        if( a_mode == DataBuffer::Mode::Unpack ) {
            if( a_buffer.m_placement != nullptr ) {
                m_reactionCrossSections[memberIndex] = new(a_buffer.m_placement) HeatedReactionCrossSectionMultiGroup;
                a_buffer.incrementPlacement( sizeof( HeatedReactionCrossSectionMultiGroup ) ); }
            else {
                m_reactionCrossSections[memberIndex] = new HeatedReactionCrossSectionMultiGroup;
            }
        }
        m_reactionCrossSections[memberIndex]->serialize( a_buffer, a_mode );
    }

    vectorSize = m_gains.size( );
    vectorSizeInt = (int) vectorSize;
    DATA_MEMBER_INT( vectorSizeInt, a_buffer, a_mode );
    vectorSize = (MCGIDI_VectorSizeType) vectorSizeInt;
    if( a_mode == DataBuffer::Mode::Unpack ) m_gains.resize( vectorSize, &a_buffer.m_placement );
    for( MCGIDI_VectorSizeType memberIndex = 0; memberIndex < vectorSize; ++memberIndex ) {
        m_gains[memberIndex].serialize( a_buffer, a_mode );
    }
}
/*
=========================================================
*/
HOST_DEVICE long HeatedCrossSectionMultiGroup::sizeOf( ) const {

    long size = (long) ( sizeof( *this ) + m_totalCrossSection.internalSize( ) + m_augmentedCrossSection.internalSize( ) + 
            m_depositionEnergy.internalSize( ) + m_depositionMomentum.internalSize( ) + m_productionEnergy.internalSize( ) + 
            m_reactionCrossSections.internalSize( ) );

    for( MCGIDI_VectorSizeType i1 = 0; i1 < m_reactionCrossSections.size( ); ++i1 ) size += m_reactionCrossSections[i1]->sizeOf( );
    for( MCGIDI_VectorSizeType i1 = 0; i1 < m_gains.size( ); ++i1 ) size += m_gains[i1].sizeOf( );

    return( size );
}

/*! \class Protare
 * Base class for the protare sub-classes.
 */

/* *********************************************************************************************************//**
 * Generic constructor.
 ***********************************************************************************************************/

HOST_DEVICE HeatedCrossSectionsMultiGroup::HeatedCrossSectionsMultiGroup( ) {

}

/* *********************************************************************************************************//**
 * Generic constructor.
 ***********************************************************************************************************/

HOST_DEVICE HeatedCrossSectionsMultiGroup::~HeatedCrossSectionsMultiGroup( ) {

    for( Vector<HeatedCrossSectionMultiGroup *>::const_iterator iter = m_heatedCrossSections.begin( ); iter != m_heatedCrossSections.end( ); ++iter ) delete *iter;
}

/* *********************************************************************************************************//**
 * Fills in *this* with the requested temperature data.
 *
 * @param a_protare                     [in]    The GIDI::Protare used to constuct the Protare that *this* is a part of.
 * @param a_setupInfo                   [in]    Used internally when constructing a Protare to pass information to other components.
 * @param a_settings                    [in]    Used to pass user options to the *this* to instruct it which data are desired.
 * @param a_particles                   [in]    List of transporting particles and their information (e.g., multi-group boundaries and fluxes).
 * @param a_temperatureInfos            [in]    The list of temperatures to use.
 * @param a_reactions                   [in]    The list of reactions to use.
 * @param a_orphanProducts              [in]    The list of orphan products to use.
 ***********************************************************************************************************/

HOST void HeatedCrossSectionsMultiGroup::update( GIDI::ProtareSingle const &a_protare, SetupInfo &a_setupInfo, Transporting::MC const &a_settings, 
                GIDI::Transporting::Particles const &a_particles, GIDI::Styles::TemperatureInfos const &a_temperatureInfos, 
                std::vector<GIDI::Reaction const *> const &a_reactions, std::vector<GIDI::Reaction const *> const &a_orphanProducts ) {

    m_temperatures.reserve( a_temperatureInfos.size( ) );
    m_heatedCrossSections.reserve( a_temperatureInfos.size( ) );

    for( GIDI::Styles::TemperatureInfos::const_iterator iter = a_temperatureInfos.begin( ); iter != a_temperatureInfos.end( ); ++iter ) {
        m_temperatures.push_back( iter->temperature( ).value( ) );
        m_heatedCrossSections.push_back( new HeatedCrossSectionMultiGroup( a_protare, a_setupInfo, a_settings, *iter, a_particles, a_reactions, iter->heatedMultiGroup( ) ) );
    }

    m_thresholds.resize( m_heatedCrossSections[0]->numberOfReactions( ) );
    for( int i1 = 0; i1 < m_heatedCrossSections[0]->numberOfReactions( ); ++i1 ) m_thresholds[i1] = m_heatedCrossSections[0]->threshold( i1 );

    m_multiGroupThresholdIndex.resize( m_heatedCrossSections[0]->numberOfReactions( ) );
    for( int i1 = 0; i1 < m_heatedCrossSections[0]->numberOfReactions( ); ++i1 ) {
        m_multiGroupThresholdIndex[i1] = -1;
        if( m_thresholds[i1] > 0 ) m_multiGroupThresholdIndex[i1] = m_heatedCrossSections[0]->thresholdOffset( i1 );
    }

    m_projectileMultiGroupBoundaries = a_setupInfo.m_protare.projectileMultiGroupBoundaries( );
}

/* *********************************************************************************************************//**
 * Returns the total multi-group cross section for target temperature *a_temperature* and projectile multi-group *a_hashIndex*.
 *
 * @param a_hashIndex           [in]    The multi-group index.
 * @param a_temperature         [in]    The temperature of the target.
 * @param a_sampling            [in]    Used for multi-group look up. If *true*, use augmented cross sections.
 ***********************************************************************************************************/

HOST_DEVICE double HeatedCrossSectionsMultiGroup::crossSection( int a_hashIndex, double a_temperature, bool a_sampling ) const {

    int i1, number_of_temperatures = static_cast<int>( m_temperatures.size( ) );
    double cross_section;

    if( a_temperature <= m_temperatures[0] ) {
        cross_section = m_heatedCrossSections[0]->crossSection( a_hashIndex, a_sampling ); }
    else if( a_temperature >= m_temperatures.back( ) ) {
        cross_section = m_heatedCrossSections.back( )->crossSection( a_hashIndex, a_sampling ); }
    else {
        for( i1 = 0; i1 < number_of_temperatures; ++i1 ) if( a_temperature < m_temperatures[i1] ) break;
        double fraction = ( a_temperature - m_temperatures[i1-1] ) / ( m_temperatures[i1] - m_temperatures[i1-1] );

        cross_section = ( 1. - fraction ) * m_heatedCrossSections[i1-1]->crossSection( a_hashIndex, a_sampling )
                        + fraction * m_heatedCrossSections[i1]->crossSection( a_hashIndex, a_sampling );
    }

    return( cross_section );
}

/* *********************************************************************************************************//**
 * Returns the requested reaction's multi-group cross section for target temperature *a_temperature* and projectile multi-group *a_hashIndex*.
 *
 * @param a_reactionIndex       [in]    The index of the reaction.
 * @param a_hashIndex           [in]    The multi-group index.
 * @param a_temperature         [in]    The temperature of the target.
 * @param a_sampling            [in]    If *true*, use augmented cross sections.
 ***********************************************************************************************************/

HOST_DEVICE double HeatedCrossSectionsMultiGroup::reactionCrossSection( int a_reactionIndex, int a_hashIndex, double a_temperature, bool a_sampling ) const {

    int i1, number_of_temperatures = static_cast<int>( m_temperatures.size( ) );
    double cross_section;

    if( a_temperature <= m_temperatures[0] ) {
        cross_section = m_heatedCrossSections[0]->reactionCrossSection( a_reactionIndex, a_hashIndex, a_sampling ); }
    else if( a_temperature >= m_temperatures.back( ) ) {
        cross_section = m_heatedCrossSections.back( )->reactionCrossSection( a_reactionIndex, a_hashIndex, a_sampling ); }
    else {
        for( i1 = 0; i1 < number_of_temperatures; ++i1 ) if( a_temperature < m_temperatures[i1] ) break;
        double fraction = ( a_temperature - m_temperatures[i1-1] ) / ( m_temperatures[i1] - m_temperatures[i1-1] );
        cross_section = ( 1. - fraction ) * m_heatedCrossSections[i1-1]->reactionCrossSection( a_reactionIndex, a_hashIndex, a_sampling )
                            + fraction * m_heatedCrossSections[i1]->reactionCrossSection( a_reactionIndex, a_hashIndex, a_sampling );
    }

    return( cross_section );
}

/* *********************************************************************************************************//**
 * Returns the requested reaction's multi-group cross section for target temperature *a_temperature* and projectile multi-group *a_hashIndex*.
 *
 * @param a_reactionIndex       [in]    The index of the reaction.
 * @param a_temperature         [in]    The temperature of the target.
 * @param a_energy_in           [in]    The energy of the projectile.
 ***********************************************************************************************************/

HOST_DEVICE double HeatedCrossSectionsMultiGroup::reactionCrossSection( int a_reactionIndex, double a_temperature, double a_energy_in ) const {

    int energyIndex = static_cast<int>( binarySearchVector( a_energy_in, m_projectileMultiGroupBoundaries ) );

    if( energyIndex < 0 ) {
        energyIndex = 0;
        if( energyIndex == -1 ) energyIndex = static_cast<int>( m_projectileMultiGroupBoundaries.size( ) ) - 2;
    }

    return( reactionCrossSection( a_reactionIndex, energyIndex, a_temperature, false ) );
}

/* *********************************************************************************************************//**
 * Returns the requested reaction's multi-group cross section for target temperature *a_temperature* and projectile multi-group *a_hashIndex*.
 *
 * @param a_hashIndex           [in]    The multi-group index.
 * @param a_temperature         [in]    The temperature of the target.
 * @param a_energy              [in]    The energy of the projectile.
 * @param a_crossSection        [in]    The index of the reaction.
 * @param a_userrng             [in]    A random number generator that takes the state *a_rngState* and returns a double in the range [0.0, 1.0).
 * @param a_rngState            [in]    The current state for the random number generator.
 ***********************************************************************************************************/

HOST_DEVICE int HeatedCrossSectionsMultiGroup::sampleReaction( int a_hashIndex, double a_temperature, double a_energy, double a_crossSection, 
                double (*a_userrng)( void * ), void *a_rngState ) const {

    int i1, sampled_reaction_index, temperatureIndex1, temperatureIndex2, numberOfTemperatures = static_cast<int>( m_temperatures.size( ) );
    double sampleCrossSection = a_crossSection * a_userrng( a_rngState );

    if( a_temperature <= m_temperatures[0] ) {
        temperatureIndex1 = 0;
        temperatureIndex2 = temperatureIndex1; }
    else if( a_temperature >= m_temperatures.back( ) ) {
        temperatureIndex1 = static_cast<int>( m_temperatures.size( ) ) - 1;
        temperatureIndex2 = temperatureIndex1; }
    else {
        for( i1 = 0; i1 < numberOfTemperatures; ++i1 ) if( a_temperature < m_temperatures[i1] ) break;
        temperatureIndex1 = i1 - 1;
        temperatureIndex2 = i1;
    }

    int numberOfReactions = m_heatedCrossSections[0]->numberOfReactions( );
    double crossSectionSum = 0;
    HeatedCrossSectionMultiGroup &heatedCrossSection1 = *m_heatedCrossSections[temperatureIndex1];

    if( temperatureIndex1 == temperatureIndex2 ) {
        for( sampled_reaction_index = 0; sampled_reaction_index < numberOfReactions; ++sampled_reaction_index ) {
            crossSectionSum += heatedCrossSection1.reactionCrossSection( sampled_reaction_index, a_hashIndex, true );
            if( crossSectionSum >= sampleCrossSection ) break;
        } }
    else {
        double temperatureFraction2 = ( a_temperature - m_temperatures[temperatureIndex1] ) / ( m_temperatures[temperatureIndex2] - m_temperatures[temperatureIndex1] );
        double temperatureFraction1 = 1.0 - temperatureFraction2;
        HeatedCrossSectionMultiGroup &heatedCrossSection2 = *m_heatedCrossSections[temperatureIndex2];

        for( sampled_reaction_index = 0; sampled_reaction_index < numberOfReactions; ++sampled_reaction_index ) {
            if( m_thresholds[sampled_reaction_index] >= a_energy ) continue;
            crossSectionSum += temperatureFraction1 * heatedCrossSection1.reactionCrossSection( sampled_reaction_index, a_hashIndex, true );
            crossSectionSum += temperatureFraction2 * heatedCrossSection2.reactionCrossSection( sampled_reaction_index, a_hashIndex, true );
            if( crossSectionSum >= sampleCrossSection ) break;
        }
    }

    if( sampled_reaction_index == numberOfReactions ) return( MCGIDI_nullReaction );

    if( m_multiGroupThresholdIndex[sampled_reaction_index] == a_hashIndex ) {
        double energyAboveThreshold = a_energy - m_thresholds[sampled_reaction_index];

        if( energyAboveThreshold <= ( a_userrng( a_rngState ) * ( m_projectileMultiGroupBoundaries[a_hashIndex+1] - m_thresholds[sampled_reaction_index] ) ) )
            return( MCGIDI_nullReaction );
    }

    return( sampled_reaction_index );
}

/* *********************************************************************************************************//**
 * Returns the multi-group deposition energy for target temperature *a_temperature* and projectile multi-group *a_hashIndex*.
 *
 * @param a_hashIndex           [in]    The multi-group index.
 * @param a_temperature         [in]    The temperature of the target.
 *
 * @return                              The deposition energy.
 ***********************************************************************************************************/

HOST_DEVICE double HeatedCrossSectionsMultiGroup::depositionEnergy( int a_hashIndex, double a_temperature ) const {

    int i1, number_of_temperatures = static_cast<int>( m_temperatures.size( ) );
    double deposition_energy;

    if( a_temperature <= m_temperatures[0] ) {
        deposition_energy = m_heatedCrossSections[0]->depositionEnergy( a_hashIndex ); }
    else if( a_temperature >= m_temperatures.back( ) ) {
        deposition_energy = m_heatedCrossSections.back( )->depositionEnergy( a_hashIndex ); }
    else {
        for( i1 = 0; i1 < number_of_temperatures; ++i1 ) if( a_temperature < m_temperatures[i1] ) break;
        double fraction = ( a_temperature - m_temperatures[i1-1] ) / ( m_temperatures[i1] - m_temperatures[i1-1] );
        deposition_energy = ( 1. - fraction ) * m_heatedCrossSections[i1-1]->depositionEnergy( a_hashIndex )
                            + fraction * m_heatedCrossSections[i1]->depositionEnergy( a_hashIndex );
    }

    return( deposition_energy );
}

/* *********************************************************************************************************//**
 * Returns the multi-group deposition momentum for target temperature *a_temperature* and projectile multi-group *a_hashIndex*.
 *
 * @param a_hashIndex           [in]    The multi-group index.
 * @param a_temperature         [in]    The temperature of the target.
 *
 * @return                              The deposition energy.
 ***********************************************************************************************************/

HOST_DEVICE double HeatedCrossSectionsMultiGroup::depositionMomentum( int a_hashIndex, double a_temperature ) const {

    int i1, number_of_temperatures = static_cast<int>( m_temperatures.size( ) );
    double deposition_momentum;

    if( a_temperature <= m_temperatures[0] ) {
        deposition_momentum = m_heatedCrossSections[0]->depositionMomentum( a_hashIndex ); }
    else if( a_temperature >= m_temperatures.back( ) ) {
        deposition_momentum = m_heatedCrossSections.back( )->depositionMomentum( a_hashIndex ); }
    else {
        for( i1 = 0; i1 < number_of_temperatures; ++i1 ) if( a_temperature < m_temperatures[i1] ) break;
        double fraction = ( a_temperature - m_temperatures[i1-1] ) / ( m_temperatures[i1] - m_temperatures[i1-1] );
        deposition_momentum = ( 1. - fraction ) * m_heatedCrossSections[i1-1]->depositionMomentum( a_hashIndex )
                            + fraction * m_heatedCrossSections[i1]->depositionMomentum( a_hashIndex );
    }

    return( deposition_momentum );
}

/* *********************************************************************************************************//**
 * Returns the multi-group production energy for target temperature *a_temperature* and projectile multi-group *a_hashIndex*.
 *
 * @param a_hashIndex           [in]    The multi-group index.
 * @param a_temperature         [in]    The temperature of the target.
 *
 * @return                              The deposition energy.
 ***********************************************************************************************************/

HOST_DEVICE double HeatedCrossSectionsMultiGroup::productionEnergy( int a_hashIndex, double a_temperature ) const {

    int i1, number_of_temperatures = static_cast<int>( m_temperatures.size( ) );
    double production_energy;

    if( a_temperature <= m_temperatures[0] ) {
        production_energy = m_heatedCrossSections[0]->productionEnergy( a_hashIndex ); }
    else if( a_temperature >= m_temperatures.back( ) ) {
        production_energy = m_heatedCrossSections.back( )->productionEnergy( a_hashIndex ); }
    else {
        for( i1 = 0; i1 < number_of_temperatures; ++i1 ) if( a_temperature < m_temperatures[i1] ) break;
        double fraction = ( a_temperature - m_temperatures[i1-1] ) / ( m_temperatures[i1] - m_temperatures[i1-1] );

        production_energy = ( 1. - fraction ) * m_heatedCrossSections[i1-1]->productionEnergy( a_hashIndex )
                            + fraction * m_heatedCrossSections[i1]->productionEnergy( a_hashIndex );
    }

    return( production_energy );
}

/* *********************************************************************************************************//**
 * Returns the multi-group gain for particle with index *a_particleIndex*. If no particle is found, a Vector of all 0's is returned.
 *
 * @param a_hashIndex           [in]    The multi-group index.
 * @param a_temperature         [in]    The temperature of the target.
 * @param a_particleIndex       [in]    The id of the particle whose gain is to be returned.
 *
 * @return                              The multi-group gain.
 ***********************************************************************************************************/

HOST_DEVICE double HeatedCrossSectionsMultiGroup::gain( int a_hashIndex, double a_temperature, int a_particleIndex ) const {

    int i1, number_of_temperatures = static_cast<int>( m_temperatures.size( ) );

    if( a_temperature <= m_temperatures[0] ) {
        return( m_heatedCrossSections[0]->gain( a_particleIndex, a_hashIndex ) ); }
    else if( a_temperature >= m_temperatures.back( ) ) {
        return( m_heatedCrossSections.back( )->gain( a_particleIndex, a_hashIndex ) );
    }

    for( i1 = 0; i1 < number_of_temperatures; ++i1 ) if( a_temperature < m_temperatures[i1] ) break;
    double fraction = ( a_temperature - m_temperatures[i1-1] ) / ( m_temperatures[i1] - m_temperatures[i1-1] );

    double gain1 = m_heatedCrossSections[i1-1]->gain( a_particleIndex, a_hashIndex );
    double gain2 = m_heatedCrossSections[i1]->gain( a_particleIndex, a_hashIndex );

    return( ( 1. - fraction ) * gain1 + fraction * gain2 );
}

/* *********************************************************************************************************//**
 * Updates the m_userParticleIndex to *a_userParticleIndex* for all particles with PoPs index *a_particleIndex*.
 *
 * @param a_particleIndex       [in]    The PoPs id of the particle whose userPid is to be set.
 * @param a_userParticleIndex   [in]    The particle id specified by the user.
 ***********************************************************************************************************/

HOST void HeatedCrossSectionsMultiGroup::setUserParticleIndex( int a_particleIndex, int a_userParticleIndex ) {

    for( auto iter = m_heatedCrossSections.begin( ); iter != m_heatedCrossSections.end( ); ++iter ) (*iter)->setUserParticleIndex( a_particleIndex, a_userParticleIndex );
}

/* *********************************************************************************************************//**
 * This method serializes *this* for broadcasting as needed for MPI and GPUs. The method can count the number of required
 * bytes, pack *this* or unpack *this* depending on *a_mode*.
 *
 * @param a_buffer              [in]    The buffer to read or write data to depending on *a_mode*.
 * @param a_mode                [in]    Specifies the action of this method.
 ***********************************************************************************************************/

HOST_DEVICE void HeatedCrossSectionsMultiGroup::serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) {

    DATA_MEMBER_VECTOR_DOUBLE( m_temperatures, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_thresholds, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_INT( m_multiGroupThresholdIndex, a_buffer, a_mode );
    DATA_MEMBER_VECTOR_DOUBLE( m_projectileMultiGroupBoundaries, a_buffer, a_mode );

    MCGIDI_VectorSizeType vectorSize = m_heatedCrossSections.size( );
    int vectorSizeInt = (int) vectorSize;
    DATA_MEMBER_INT( vectorSizeInt, a_buffer, a_mode );
    vectorSize = (MCGIDI_VectorSizeType) vectorSizeInt;

    if( a_mode == DataBuffer::Mode::Unpack ) m_heatedCrossSections.resize( vectorSize, &a_buffer.m_placement );
    for( MCGIDI_VectorSizeType memberIndex = 0; memberIndex < vectorSize; ++memberIndex ) {
        if( a_mode == DataBuffer::Mode::Unpack ) {
            if( a_buffer.m_placement != nullptr ) {
                m_heatedCrossSections[memberIndex] = new(a_buffer.m_placement) HeatedCrossSectionMultiGroup;
                a_buffer.incrementPlacement( sizeof( HeatedCrossSectionMultiGroup ) ); }
            else {
                m_heatedCrossSections[memberIndex] = new HeatedCrossSectionMultiGroup;
            }
        }
        m_heatedCrossSections[memberIndex]->serialize( a_buffer, a_mode );
    }
}

/* *********************************************************************************************************//**
 * This method counts the number of bytes of memory allocated by *this*. That is the member needed by *this* that is greater than
 * sizeof( *this );
 ***********************************************************************************************************/

HOST_DEVICE long HeatedCrossSectionsMultiGroup::internalSize( ) const {

    long size = (long) ( m_temperatures.internalSize( ) + m_thresholds.internalSize( ) + m_multiGroupThresholdIndex.internalSize( ) + 
            m_projectileMultiGroupBoundaries.internalSize( ) + m_heatedCrossSections.internalSize( ) );

    for( MCGIDI_VectorSizeType i1 = 0; i1 < m_heatedCrossSections.size( ); ++i1 ) size += m_heatedCrossSections[i1]->sizeOf( );
    return( size );
}

}

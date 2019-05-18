/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdlib.h>
#include <algorithm>

#include "GIDI.hpp"

namespace GIDI {

/*! \class ProtareTNSL
 * Class to store <**reactionSuite**> nodes required for thermal neutron scattering law.
 */

/* *********************************************************************************************************//**
 * ProtareTNSL constructor.
 *
 * @param a_construction        [in]     Used to pass user options to the constructor.
 * @param a_protare             [in]     The non-TNSL **ProtareSingleton**.
 * @param a_TNSL                [in]     The TNSL **ProtareSingleton**.
 ***********************************************************************************************************/

ProtareTNSL::ProtareTNSL( Construction::Settings const &a_construction, ProtareSingleton *a_protare, ProtareSingleton *a_TNSL ) :
        m_protare( a_protare ),
        m_TNSL( a_TNSL ),
        m_elasticReaction( NULL ) {

    if( a_protare->projectile( ).ID( ) != PoPs::IDs::neutron ) throw std::runtime_error( "ProtareTNSL::ProtareTNSL: a_protare neutron as target." );
    if( a_TNSL->projectile( ).ID( ) != PoPs::IDs::neutron ) throw std::runtime_error( "ProtareTNSL::ProtareTNSL: a_TNSL not thermal neutron scattering protare." );

    projectile( a_protare->projectile( ) );
    target( a_protare->target( ) );

    for( std::size_t i1 = 0; i1 < m_protare->numberOfReactions( ); ++i1 ) {
        Reaction const *reaction = m_protare->reaction( i1 );

        if( reaction->ENDF_MT( ) == 2 ) {
            m_elasticReaction = reaction;
            break;
        }
    }

    if( m_elasticReaction == NULL ) throw std::runtime_error( "ProtareTNSL::ProtareTNSL: could not find elastic reaction in a_protare." );
// FIXME need to check that data are consistence.
}

/* *********************************************************************************************************//**
 ******************************************************************/

ProtareTNSL::~ProtareTNSL( ) {

    delete m_protare;
    delete m_TNSL;
}

/* *********************************************************************************************************//**
 * Returns the maximum usable multi-group index for the thermal neutron scattering law protare for the *a_settings* request.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_particles       [in]    The list of particles to be transported.
 *
 * @return                          The last multigroup index for which the TNSL protare has cross section data. Above this index, the cross section, et al. data must come from the regular protare.
 ******************************************************************/

std::size_t ProtareTNSL::maximumMultiGroupIndices( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const {

    std::size_t maximumMultiGroupIndex = 0;
#if 0                                       // This is commented out because the calling methods have qualifier const which does not allow m_maximumMultiGroupIndices to be modified.
    std::map<std::string,std::size_t>::iterator iter = m_maximumMultiGroupIndices.index( a_settings.label( ) );

    if( iter == map::end ) {
#endif
        std::size_t i1 = 0;
        Vector crossSection = m_TNSL->multiGroupCrossSection( a_settings, a_particles );

        for( ; i1 < crossSection.size( ); ++i1 ) {
            if( crossSection[i1] == 0 ) break;
        }
        if( i1 > 0 ) --i1;                   // Assume that the last non-zero multi-group in TSNL is incomplete.
        maximumMultiGroupIndex = i1;

// The check of multiplicity is needed because FUDGE has a inconsistency in what maximum energy to use for TNSL calculations.
// The following lines can be remove when this issue is resolved in FUDGE.
        Vector multiplicity = m_TNSL->multiGroupMultiplicity( a_settings, a_particles, PoPs::IDs::neutron );

        for( i1 = 0; i1 < multiplicity.size( ); ++i1 ) {
            if( multiplicity[i1] == 0 ) break;
        }
        if( i1 > 0 ) --i1;                   // Assume that the last non-zero multi-group in TSNL is incomplete.
        if( i1 < maximumMultiGroupIndex ) maximumMultiGroupIndex = i1;

#if 0

         m_maximumMultiGroupIndices.emplace( a_settings.label( ), maximumMultiGroupIndex ); }
    else {
        maximumMultiGroupIndex = iterator.second;
    }
#endif

    return( maximumMultiGroupIndex );
}

/* *********************************************************************************************************//**
 * Removes the elastic component from *a_vector* and adds in the TNSL component.
 *
 * @param   a_settings          [in]        Specifies the requested label.
 * @param   a_particles         [in]        The list of particles to be transported.
 * @param   a_vector            [in/out]    The vector from the non TNSL protare.
 * @param   a_vectorElastic     [in]        The vector from the elastic reactino from the non TNSL protare.
 * @param   a_vectorTNSL        [in]        The vector from the TNSL protare.
 ******************************************************************/

void ProtareTNSL::combineVectors( Settings::MG const &a_settings, Settings::Particles const &a_particles, Vector &a_vector, Vector const &a_vectorElastic, Vector const &a_vectorTNSL ) const {

    if( a_vectorTNSL.size( ) == 0 ) return;

    std::size_t maximumMultiGroupIndex = maximumMultiGroupIndices( a_settings, a_particles );

    for( std::size_t i1 = 0; i1 < maximumMultiGroupIndex; ++i1 ) a_vector[i1] += a_vectorTNSL[i1] - a_vectorElastic[i1];
}

/* *********************************************************************************************************//**
 * Removes the elastic component from *a_matrix* and adds in the TNSL component.
 *
 * @param   a_settings          [in]        Specifies the requested label.
 * @param   a_particles         [in]        The list of particles to be transported.
 * @param   a_matrix            [in/out]    The matrix from the non TNSL protare.
 * @param   a_matrixElastic     [in]        The matrix from the elastic reactino from the non TNSL protare.
 * @param   a_matrixTNSL        [in]        The matrix from the TNSL protare.
 ******************************************************************/

void ProtareTNSL::combineMatrices( Settings::MG const &a_settings, Settings::Particles const &a_particles, Matrix &a_matrix, Matrix const &a_matrixElastic, Matrix const &a_matrixTNSL ) const {

    if( a_matrixTNSL.size( ) == 0 ) return;

    std::size_t maximumMultiGroupIndex = maximumMultiGroupIndices( a_settings, a_particles );

    for( std::size_t i1 = 0; i1 < maximumMultiGroupIndex; ++i1 ) {
        Vector const &rowTNSL = a_matrixTNSL[i1];
        Vector const &rowElastic = a_matrixElastic[i1];
        Vector &row = a_matrix[i1];

        for( std::size_t i2 = 0; i2 < a_matrixTNSL.size( ); ++i2 ) row[i2] += rowTNSL[i2] - rowElastic[i2];
    }
}

/* *********************************************************************************************************//**
 * Returns the GNDS format version for the (a_index+1)^th Protare. The index **a_index** can only be 0 (normal protare) or 1 (TNSL protare).
 *
 * @param           a_index [in]    The index of the Protare whose format version is returned.
 *
 * @return                          The format version.
 ******************************************************************/

std::string const &ProtareTNSL::formatVersion( std::size_t a_index ) const {

    if( a_index == 0 ) return( m_protare->formatVersion( ) );
    if( a_index == 1 ) return( m_TNSL->formatVersion( ) );
    throw std::runtime_error( "ProtareTNSL::formatVersion: index can only be 0 or 1." );
}

/* *********************************************************************************************************//**
 * Returns the file name for the (a_index+1)^th Protare. The index **a_index** can only be 0 (normal protare) or 1 (TNSL protare).
 *
 * @param           a_index [in]    The index of the Protare whose file name is returned.
 *
 * @return                          The file name.
 ******************************************************************/

std::string const &ProtareTNSL::fileName( std::size_t a_index ) const {

    if( a_index == 0 ) return( m_protare->fileName( ) );
    if( a_index == 1 ) return( m_TNSL->fileName( ) );
    throw std::runtime_error( "ProtareTNSL::fileName: index can only be 0 or 1." );
}

/* *********************************************************************************************************//**
 * Returns the real file name for the (a_index+1)^th Protare. The index **a_index** can only be 0 (normal protare) or 1 (TNSL protare).
 *
 * @param           a_index [in]    The index of the Protare whose real file name is returned.
 *
 * @return                          The real file name.
 ******************************************************************/

std::string const &ProtareTNSL::realFileName( std::size_t a_index ) const {

    if( a_index == 0 ) return( m_protare->realFileName( ) );
    if( a_index == 1 ) return( m_TNSL->realFileName( ) );
    throw std::runtime_error( "ProtareTNSL::realFileName: index can only be 0 or 1." );
}

/* *********************************************************************************************************//**
 * Returns the list of libraries for the (a_index+1)^th contained Protare. The index **a_index** can only be 0 (normal protare) or 1 (TNSL protare).
 * 
 * @param           a_index     [in]    The index of the Protare whose libraries are to be returned.
 *
 * @return                              The list of libraries.
 ******************************************************************/

std::vector<std::string> ProtareTNSL::libraries( std::size_t a_index ) const {

    if( a_index == 0 ) return( m_protare->libraries( ) );
    if( a_index == 1 ) return( m_TNSL->libraries( ) );
    throw std::runtime_error( "ProtareTNSL::libraries: index can only be 0 or 1." );
}

/* *********************************************************************************************************//**
 * Returns the evaluation for the (a_index+1)^th Protare. The index **a_index** can only be 0 (normal protare) or 1 (TNSL protare).
 *
 * @param           a_index [in]    The index of the Protare whose evaluation is returned.
 *
 * @return                          The evaluation.
 ******************************************************************/

std::string const &ProtareTNSL::evaluation( std::size_t a_index ) const {

    if( a_index == 0 ) return( m_protare->evaluation( ) );
    if( a_index == 1 ) return( m_TNSL->evaluation( ) );
    throw std::runtime_error( "ProtareTNSL::evaluation: index can only be 0 or 1." );
}

/* *********************************************************************************************************//**
 * Returns the projectile frame for the (a_index+1)^th Protare. The index **a_index** can only be 0 (normal protare) or 1 (TNSL protare).
 *
 * @param           a_index [in]    The index of the Protare whose projectile frame is returned.
 *
 * @return                          The projectile frame.
 ******************************************************************/

frame ProtareTNSL::projectileFrame( std::size_t a_index ) const {

    if( a_index == 0 ) return( m_protare->projectileFrame( ) );
    if( a_index == 1 ) return( m_TNSL->projectileFrame( ) );
    throw std::runtime_error( "ProtareTNSL::projectileFrame: index can only be 0 or 1." );
}

/* *********************************************************************************************************//**
 * Returns the pointer representing the (a_index - 1)th **ProtareSingleton**.
 *
 * @param a_index               [in]    Index of the **ProtareSingleton** to return. Can only be 0 or 1.
 *
 * @return                              Pointer to the requested protare or NULL if invalid *a_index*..
 ***********************************************************************************************************/

ProtareSingleton const *ProtareTNSL::protare( std::size_t a_index ) const {

    if( a_index == 0 ) return( m_protare );
    if( a_index == 1 ) return( m_TNSL );
    return( NULL );
}

/* *********************************************************************************************************//**
 * Returns the threshold factor for the projectile hitting the target.
 *
 * @return              The threshold factor.
 ******************************************************************/

double ProtareTNSL::thresholdFactor( ) const {

    return( m_protare->thresholdFactor( ) );
}

/* *********************************************************************************************************//**
 * Returns the Documentation::Suite from the non TNSL protare.
 *
 * @return              The Documentation::Suite.
 ******************************************************************/

Documentation::Suite const &ProtareTNSL::documentations( ) const {

    return( m_protare->documentations( ) );
}

/* *********************************************************************************************************//**
 * Returns the style with label *a_label* from the non TNSL protare.
 *
 * @param           a_label     [in]    The label of the requested style.
 * @return                              The style with label *a_label*.
 ******************************************************************/

Styles::Base const &ProtareTNSL::style( std::string const a_label ) const {

    return( m_protare->style( a_label ) );
}

/* *********************************************************************************************************//**
 * Returns the Styles::Suite from the non TNSL protare.
 *
 * @return              The Styles::Suite.
 ******************************************************************/

Styles::Suite const &ProtareTNSL::styles( ) const {

    return( m_protare->styles( ) );
}

/* *********************************************************************************************************//**
 * Calls productIDs for each Protare contained in *this*.
 *
 * @param   a_ids                   [in]        Contains the list of particle ids.
 * @param   a_particles             [in]        The list of particles to be transported.
 * @param   a_transportablesOnly    [in]        If *true* only transportable particle ids are added to *a_ids*.
 ******************************************************************/

void ProtareTNSL::productIDs( std::set<std::string> &a_ids, Settings::Particles const &a_particles, bool a_transportablesOnly ) const {

    m_protare->productIDs( a_ids, a_particles, a_transportablesOnly );
    m_TNSL->productIDs( a_ids, a_particles, a_transportablesOnly );
}

/* *********************************************************************************************************//**
 * Determines the maximum Legredre order present in the multi-group transfer matrix for a give product for a give label.
 * Loops over all contained Protares to determine the maximum Legredre order.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_productID       [in]    The id of the requested product.
 * @return                          The maximum Legredre order. If no transfer matrix data are present for the requested product, -1 is returned.
 ***********************************************************************************************************/

int ProtareTNSL::maximumLegendreOrder( Settings::MG const &a_settings, std::string const &a_productID ) const {

    int maximumLegendreOrder1 = m_protare->maximumLegendreOrder( a_settings, a_productID );
    int maximumLegendreOrder2 = m_TNSL->maximumLegendreOrder( a_settings, a_productID );

    if( maximumLegendreOrder1 > maximumLegendreOrder2 ) return( maximumLegendreOrder1 );
    return( maximumLegendreOrder2 );
}

/* *********************************************************************************************************//**
 * Returns a list of all process temperature data from the non TNSL protare. For each temeprature, the labels for its
 *
 *   + heated cross section data,
 *   + gridded cross section data,
 *   + multi-group data, and
 *   + multi-group upscatter data.
 *
 * are returned. If no data are present for a give data type (e.g., gridded cross section, multi-group upscatter), its label is an empty std::string.
 *
 * @return  The list of temperatures and their labels via an Styles::TemperatureInfos instance. The Styles::TemperatureInfos class
 *          has many (if not all) the method of a std::vector.
 ***********************************************************************************************************/

Styles::TemperatureInfos ProtareTNSL::temperatures( ) const {

    return( m_protare->temperatures( ) );
}

/* *********************************************************************************************************//**
 * Returns the number of reactions from the non TNSL protare.
 *
 * @return              The total number of reactions.
 ******************************************************************/

std::size_t ProtareTNSL::numberOfReactions( ) const {

    return( m_protare->numberOfReactions( ) );
}

/* *********************************************************************************************************//**
 * Returns the (*a_index*+1)th reaction from the non TNSL protare.
 *
 * @param a_index           [in]    The index of the requested reaction.
 * @return                          The (*a_index*+1)th reaction.
 ***********************************************************************************************************/

Reaction const *ProtareTNSL::reaction( std::size_t a_index ) const {

    return( m_protare->reaction( a_index ) );
}

/* *********************************************************************************************************//**
 * Returns the number of orphanProduct's from the non TNSL protare.
 *  
 * @return              The total number of orphanProducts.
 ******************************************************************/
    
std::size_t ProtareTNSL::numberOfOrphanProducts( ) const {

    return( m_protare->numberOfOrphanProducts( ) );
}

/* *********************************************************************************************************//**
 * Returns the (*a_index*+1)th orphanProduct from the non TNSL protare.
 *
 * @param a_index           [in]    The index of the requested orphanProduct.
 * @return                          The (*a_index*+1)th orphanProduct.
 ***********************************************************************************************************/

Reaction const *ProtareTNSL::orphanProduct( std::size_t a_index ) const {

    return( m_protare->orphanProduct( a_index ) );
}

/* *********************************************************************************************************//**
 * Returns true if at least one reaction contains a fission channel.
 *
 * @return      true if at least one reaction contains a fission channel and false otherwise.
 ***********************************************************************************************************/

bool ProtareTNSL::hasFission( ) const {

    return( m_protare->hasFission( ) );
}

/* *********************************************************************************************************//**
 * Returns the requested Styles::MultiGroup from the styles from the non TNSL protare.
 *
 * @param a_label   [in]    Label for the requested Styles::MultiGroup.
 * @return                  The requested MultiGroup style.
 ***********************************************************************************************************/

Styles::MultiGroup const *ProtareTNSL::multiGroup( std::string const &a_label ) const {

    return( m_protare->multiGroup( a_label ) );
}

/* *********************************************************************************************************//**
 * Returns the multi-group boundaries for the requested label and product.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_productID       [in]    ID for the requested product.
 * @return                          List of multi-group boundaries.
 ***********************************************************************************************************/

std::vector<double> const &ProtareTNSL::groupBoundaries( Settings::MG const &a_settings, std::string const &a_productID ) const {

    return( m_protare->groupBoundaries( a_settings, a_productID ) );
}

/* *********************************************************************************************************//**
 * Returns the inverse speeds for the requested label from the non TNSL protare. The label must be for a heated multi-group style.
 *
 * @param   a_settings    [in]      Specifies the requested label.
 * @param   a_particles   [in]      The list of particles to be transported.
 * @return                          List of inverse speeds.
 ***********************************************************************************************************/

Vector ProtareTNSL::multiGroupInverseSpeed( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const {

    return( m_protare->multiGroupInverseSpeed( a_settings, a_particles ) );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total cross section for the requested label. This is summed over all reactions.
 *
 * @param   a_settings      [in]    Specifies the requested label.
 * @param   a_particles     [in]    The list of particles to be transported.
 * @return                          The requested multi-group cross section as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareTNSL::multiGroupCrossSection( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const {

    Vector vector = m_protare->multiGroupCrossSection( a_settings, a_particles );
    Vector vectorElastic = m_elasticReaction->multiGroupCrossSection( a_settings, a_particles );

    combineVectors( a_settings, a_particles, vector, vectorElastic, m_TNSL->multiGroupCrossSection( a_settings, a_particles ) );
    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total Q for the requested label. This is a cross section weighted multiplicity
 * summed over all reactions
 *
 * @param  a_settings       [in]    Specifies the requested label.
 * @param  a_particles      [in]    The list of particles to be transported.
 * @param  a_final          [in]    If false, only the Q for the primary reactions are return, otherwise, the Q for the final reactions.
 * @return                          The requested multi-group Q as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareTNSL::multiGroupQ( Settings::MG const &a_settings, Settings::Particles const &a_particles, bool a_final ) const {

    Vector vector = m_protare->multiGroupQ( a_settings, a_particles, a_final );
    Vector vectorElastic = m_elasticReaction->multiGroupQ( a_settings, a_particles, a_final );

    combineVectors( a_settings, a_particles, vector, vectorElastic, m_TNSL->multiGroupQ( a_settings, a_particles, a_final ) );
    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total multiplicity for the requested label for the requested product. This is a cross section weighted multiplicity.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param  a_particles      [in]    The list of particles to be transported.
 * @param a_productID       [in]    Id for the requested product.
 * @return                          The requested multi-group multiplicity as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareTNSL::multiGroupMultiplicity( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const {

    Vector vector = m_protare->multiGroupMultiplicity( a_settings, a_particles, a_productID );
    Vector vectorElastic = m_elasticReaction->multiGroupMultiplicity( a_settings, a_particles, a_productID );

    combineVectors( a_settings, a_particles, vector, vectorElastic, m_TNSL->multiGroupMultiplicity( a_settings, a_particles, a_productID ) );
    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total fission neutron multiplicity for the requested label. This is a cross section weighted multiplicity.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param  a_particles      [in]    The list of particles to be transported.
 * @return                          The requested multi-group fission neutron multiplicity as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareTNSL::multiGroupFissionNeutronMultiplicity( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const {

    return( m_protare->multiGroupFissionNeutronMultiplicity( a_settings, a_particles ) );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total product matrix for the requested label for the requested product id for the requested Legendre order.
 * If no data are found, an empty GIDI::Matrix is returned.
 *
 * @param a_settings        [in]    Specifies the requested label and if delayed neutrons should be included.
 * @param  a_particles      [in]    The list of particles to be transported.
 * @param a_productID       [in]    PoPs id for the requested product.
 * @param a_order           [in]    Requested product matrix, Legendre order.
 * @return                          The requested multi-group product matrix as a GIDI::Matrix.
 ***********************************************************************************************************/

Matrix ProtareTNSL::multiGroupProductMatrix( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID, int a_order ) const {

    Matrix matrix = m_protare->multiGroupProductMatrix( a_settings, a_particles, a_productID, a_order );
    Matrix matrixElastic = m_elasticReaction->multiGroupProductMatrix( a_settings, a_particles, a_productID, a_order );
    Matrix matrixTNSL = m_TNSL->multiGroupProductMatrix( a_settings, a_particles, a_productID, a_order );

    combineMatrices( a_settings, a_particles, matrix, matrixElastic, matrixTNSL );
    return( matrix );
}

/* *********************************************************************************************************//**
 * Like ProtareTNSL::multiGroupProductMatrix, but only returns the fission neutron, transfer matrix.
 *
 * @param a_settings        [in]    Specifies the requested label and if delayed neutrons should be included.
 * @param  a_particles      [in]    The list of particles to be transported.
 * @param a_order           [in]    Requested product matrix, Legendre order.
 * @return                          The requested multi-group neutron fission matrix as a GIDI::Matrix.
 ***********************************************************************************************************/

Matrix ProtareTNSL::multiGroupFissionMatrix( Settings::MG const &a_settings, Settings::Particles const &a_particles, int a_order ) const {

    return( m_protare->multiGroupFissionMatrix( a_settings, a_particles, a_order ) );
}

/* *********************************************************************************************************//**
 * Returns the multi-group transport correction for the requested label. The transport correction is calculated from the transfer matrix
 * for the projectile id for the Legendre order of *a_order + 1*.
 *
 * @param a_settings                [in]    Specifies the requested label.
 * @param  a_particles              [in]    The list of particles to be transported.
 * @param a_order                   [in]    Maximum Legendre order for transport. The returned transport correction is for the next higher Legender order.
 * @param a_transportCorrectionType [in]    Requested transport correction type.
 * @param a_temperature             [in]    The temperature of the flux to use when collapsing. Pass to the GIDI::collapse method.
 * @return                                  The requested multi-group transport correction as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareTNSL::multiGroupTransportCorrection( Settings::MG const &a_settings, Settings::Particles const &a_particles, int a_order, transportCorrectionType a_transportCorrectionType, double a_temperature ) const {

    if( a_transportCorrectionType == transportCorrection_None ) return( Vector( 0 ) );

    Matrix matrix( multiGroupProductMatrix( a_settings, a_particles, projectile( ).ID( ), a_order + 1 ) );
    Matrix matrixCollapsed = collapse( matrix, a_settings, a_particles, a_temperature, projectile( ).ID( ) );
    std::size_t size = matrixCollapsed.size( );
    std::vector<double> transportCorrection1( size, 0 );

    if( a_transportCorrectionType == transportCorrection_Pendlebury ) {
        for( std::size_t index = 0; index < size; ++index ) transportCorrection1[index] = matrixCollapsed[index][index];
    }
    return( Vector( transportCorrection1 ) );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total available energy for the requested label. This is a cross section weighted available energy
 * summed over all reactions.
 *
 * @param  a_settings   [in]    Specifies the requested label.
 * @param  a_particles  [in]    The list of particles to be transported.
 * @return                      The requested multi-group available energy as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareTNSL::multiGroupAvailableEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const {

    Vector vector = m_protare->multiGroupAvailableEnergy( a_settings, a_particles );
    Vector vectorElastic = m_elasticReaction->multiGroupAvailableEnergy( a_settings, a_particles );

    combineVectors( a_settings, a_particles, vector, vectorElastic, m_TNSL->multiGroupAvailableEnergy( a_settings, a_particles ) );
    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total average energy for the requested label for the requested product. This is a cross section weighted average energy
 * summed over all reactions.
 *
 * @param  a_settings       [in]    Specifies the requested label.
 * @param  a_particles      [in]    The list of particles to be transported.
 * @param  a_productID      [in]    Particle id for the requested product.
 * @return                          The requested multi-group average energy as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareTNSL::multiGroupAverageEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const {

    Vector vector = m_protare->multiGroupAverageEnergy( a_settings, a_particles, a_productID );
    Vector vectorElastic = m_elasticReaction->multiGroupAverageEnergy( a_settings, a_particles, a_productID );

    combineVectors( a_settings, a_particles, vector, vectorElastic, m_TNSL->multiGroupAverageEnergy( a_settings, a_particles, a_productID ) );
    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total deposition energy for the requested label. This is a cross section weighted deposition energy
 * summed over all reactions. The deposition energy is calculated by subtracting the average energy from each transportable particle
 * from the available energy. The list of transportable particles is specified via the list of particle specified in the *a_settings* argument.
 *
 * @param a_settings    [in]    Specifies the requested label and the products that are transported.
 * @param  a_particles  [in]    The list of particles to be transported.
 * @return                      The requested multi-group deposition energy as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareTNSL::multiGroupDepositionEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const {

    Vector vector = m_protare->multiGroupDepositionEnergy( a_settings, a_particles );
    Vector vectorElastic = m_elasticReaction->multiGroupDepositionEnergy( a_settings, a_particles );

    combineVectors( a_settings, a_particles, vector, vectorElastic, m_TNSL->multiGroupDepositionEnergy( a_settings, a_particles ) );
    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total available momentum for the requested label. This is a cross section weighted available momentum
 * summed over all reactions.
 *
 * @param a_settings    [in]    Specifies the requested label.
 * @param  a_particles  [in]    The list of particles to be transported.
 * @return                      The requested multi-group available momentum as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareTNSL::multiGroupAvailableMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const {

    Vector vector = m_protare->multiGroupAvailableMomentum( a_settings, a_particles );
    Vector vectorElastic = m_elasticReaction->multiGroupAvailableMomentum( a_settings, a_particles );

    combineVectors( a_settings, a_particles, vector, vectorElastic, m_TNSL->multiGroupAvailableMomentum( a_settings, a_particles ) );
    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total average momentum for the requested label for the requested product. This is a cross section weighted average momentum
 * summed over all reactions.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param  a_particles      [in]    The list of particles to be transported.
 * @param a_productID       [in]    Particle id for the requested product.
 * @return                          The requested multi-group average momentum as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareTNSL::multiGroupAverageMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const {

    Vector vector = m_protare->multiGroupAverageMomentum( a_settings, a_particles, a_productID );
    Vector vectorElastic = m_elasticReaction->multiGroupAverageMomentum( a_settings, a_particles, a_productID );

    combineVectors( a_settings, a_particles, vector, vectorElastic, m_TNSL->multiGroupAverageMomentum( a_settings, a_particles, a_productID ) );
    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total deposition momentum for the requested label. This is a cross section weighted deposition momentum
 * summed over all reactions. The deposition momentum is calculated by subtracting the average momentum from each transportable particle
 * from the available momentum. The list of transportable particles is specified via the list of particle specified in the *a_settings* argument.
 *
 * @param a_settings    [in]    Specifies the requested label.
 * @param a_particles  [in]    The list of particles to be transported.
 * @return                      The requested multi-group deposition momentum as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareTNSL::multiGroupDepositionMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const {

    Vector vector = m_protare->multiGroupDepositionMomentum( a_settings, a_particles );
    Vector vectorElastic = m_elasticReaction->multiGroupDepositionMomentum( a_settings, a_particles );

    combineVectors( a_settings, a_particles, vector, vectorElastic, m_TNSL->multiGroupDepositionMomentum( a_settings, a_particles ) );
    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, gain for the requested particle and label. This is a cross section weighted gain summed over all reactions.
 *
 * @param a_settings    [in]    Specifies the requested label.
 * @param a_particles   [in]    The list of particles to be transported.
 * @param a_productID   [in]    The PoPs' id for the particle whose gain is to be calculated.
 *
 * @return                      The requested multi-group gain as a **GIDI::Vector**.
 ***********************************************************************************************************/

Vector ProtareTNSL::multiGroupGain( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const {

    std::string const projectile_id = m_protare->projectile( ).ID( );    
    Vector vector = m_protare->multiGroupGain( a_settings, a_particles, a_productID );
    Vector vectorElastic = m_elasticReaction->multiGroupGain( a_settings, a_particles, a_productID, projectile_id );

    combineVectors( a_settings, a_particles, vector, vectorElastic, m_TNSL->multiGroupGain( a_settings, a_particles, a_productID ) );

    return( vector );
}

/* *********************************************************************************************************//**
 * This method always returns 1 since the projectile is always a neutron.
 *
 * @return      Always returns 1.
 ***********************************************************************************************************/

stringAndDoublePairs ProtareTNSL::muCutoffForCoulombPlusNuclearElastic( ) const {

    stringAndDoublePairs stringAndDoublePairs1;

    return( stringAndDoublePairs1 );
}

}

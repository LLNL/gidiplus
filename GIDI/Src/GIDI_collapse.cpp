/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "GIDI.hpp"

namespace GIDI {

static Vector collapseVector( Vector const &a_vector, std::vector<int> const &a_collapseIndices, 
        std::vector<double> const &a_weight, bool a_normalize );
static void multiGroupSetup( Transporting::MultiGroup const &a_boundaries, ptwXPoints **a_boundaries_xs,
                Transporting::Flux const &a_flux, ptwXYPoints **a_fluxes_xys, ptwXPoints **a_multiGroupFlux );

/* *********************************************************************************************************//**
 * Collapses a multi-group vector.
 *
 * @param a_vector                  [in]    The Vector to collapse.
 * @param a_settings                [in]    Specifies the uncollapsed and collapsed multi-group information and the flux.
 * @param a_particles               [in]    The list of particles to be transported.
 * @param a_temperature             [in]    The temperature of the flux to use when collapsing.
 *
 * @return                                  Returns the collapsed Vector.
 ***********************************************************************************************************/

Vector collapse( Vector const &a_vector, Transporting::Settings const &a_settings, Transporting::Particles const &a_particles, double a_temperature ) {

    Transporting::Particle const *projectile( a_particles.particle( a_settings.projectileID( ) ) );
    Transporting::ProcessedFlux const *flux( projectile->nearestProcessedFluxToTemperature( a_temperature ) );
    std::vector<double> const &multiGroupFlux( flux->multiGroupFlux( ) );
    std::vector<int> const &collapseIndices( projectile->collapseIndices( ) );

    return( collapseVector( a_vector, collapseIndices, multiGroupFlux, true ) );
}

/* *********************************************************************************************************//**
 * Collapses a multi-group vector.
 *
 * @param a_vector                  [in]    The Vector to collapse.
 * @param a_collapseIndices         [in]    Maps uncollapsed indices to collapsed indices.
 * @param a_weight                  [in]    The uncollapsed flux weighting.
 * @param a_normalize               [in]    If true, divide each collapsed value by it corresponding collapsed weight value.
 * @return                                  Returns the collapsed Vector.
 ***********************************************************************************************************/

static Vector collapseVector( Vector const &a_vector, std::vector<int> const &a_collapseIndices, 
        std::vector<double> const &a_weight, bool a_normalize ) {

    std::size_t n1( a_collapseIndices.size( ) - 1 );
    std::size_t index1( a_collapseIndices[0] );
    Vector vectorCollapsed( n1 );

    if( a_vector.size( ) > 0 ) {
        for( std::size_t i1 = 0; i1 < n1; ++i1 ) {
            std::size_t index2( a_collapseIndices[i1+1] );
            double fluxSum = 0;
            double valueSum = 0;

            for( std::size_t i2 = index1; i2 < index2; ++i2 ) {
                fluxSum += a_weight[i2];
                valueSum += a_weight[i2] * a_vector[i2];
            }
            if( a_normalize && ( fluxSum != 0 ) ) valueSum /= fluxSum;
            vectorCollapsed[i1] = valueSum;
            index1 = index2;
        }
    }

    return( vectorCollapsed );
}

/* *********************************************************************************************************//**
 * Collapses a multi-group matrix.
 *
 * @param a_matrix                  [in]    The Matrix to collapse.
 * @param a_settings                [in]    Specifies the uncollapsed and collapsed multi-group information and the flux.
 * @param a_particles               [in]    The list of particles to be transported.
 * @param a_temperature             [in]    The temperature of the flux to use when collapsing.
 * @param a_productID               [in]    Particle id of the outgoing particle.
 * @return                                  Returns the collapsed Matrix.
 ***********************************************************************************************************/

Matrix collapse( Matrix const &a_matrix, Transporting::Settings const &a_settings, Transporting::Particles const &a_particles, double a_temperature, std::string const &a_productID ) {

    if( a_matrix.size( ) == 0 ) return( a_settings.multiGroupZeroMatrix( a_particles, a_productID, true ) );

    Transporting::Particle const *projectile( a_particles.particle( a_settings.projectileID( ) ) );
    Transporting::ProcessedFlux const *flux( projectile->nearestProcessedFluxToTemperature( a_temperature ) );
    std::vector<double> const &multiGroupFlux( flux->multiGroupFlux( ) );
    std::vector<int> const &projectileCollapseIndices( projectile->collapseIndices( ) );

    Transporting::Particle const *product( a_particles.particle( a_productID ) );
    std::size_t n2 = product->numberOfGroups( );

    std::vector<int> productCollapseIndices( product->collapseIndices( ) );
    productCollapseIndices[0] = 0;
    productCollapseIndices[n2] = a_matrix[0].size( );

    std::vector<double> numberWeight( a_matrix[0].size( ), 1. );

    Matrix productCollapsed( 0, 0 );
    for( std::size_t i1 = 0; i1 < a_matrix.size( ); ++i1 ) {
        productCollapsed.push_back( collapseVector( a_matrix[i1], productCollapseIndices, numberWeight, false ) );
    }

    Matrix productCollapsedTranspose = productCollapsed.transpose( );
    Matrix collapsedTranspose( 0, 0 );
    for( std::size_t i2 = 0; i2 < n2; ++i2 ) {
        collapsedTranspose.push_back( collapseVector( productCollapsedTranspose[i2], projectileCollapseIndices, multiGroupFlux, true ) );
    }
    return( collapsedTranspose.transpose( ) );
}

/* *********************************************************************************************************//**
 * Transport correct a vector.
 *
 * @param a_vector                  [in]    The Vector to transport correct.
 * @param a_transportCorrection     [in]    The Vector that has the transport correction terms.
 * @return                                  Returns the collapsed Matrix.
 ***********************************************************************************************************/

Vector transportCorrect( Vector const &a_vector, Vector const &a_transportCorrection ) {

    return( a_vector - a_transportCorrection );
}

/* *********************************************************************************************************//**
 * Transport correct a Matrix.
 *
 * @param a_matrix                  [in]    The Matrix to transport correct.
 * @param a_transportCorrection     [in]    The Vector that has the transport correction terms.
 * @return                                  Returns the collapsed Matrix.
 ***********************************************************************************************************/

Matrix transportCorrect( Matrix const &a_matrix, Vector const &a_transportCorrection ) {

    std::size_t size = a_transportCorrection.size( );
    Matrix corrected( a_matrix );

    if( size == 0 ) return( corrected );
    if( a_matrix.size( ) == 0 ) {
        corrected = Matrix( size, size ); }
    else {
        if( size != a_matrix.size( ) ) throw Exception( "transportCorrect: matrix rows different than vector size." );
    }

    for( std::size_t index = 0; index < size; ++index ) corrected[index][index] -= a_transportCorrection[index];
    return( corrected );
}

/* *********************************************************************************************************//**
 * Returns a flux weighted multi-group version of the function *a_function*.
 *
 * @param a_boundaries              [in]    List of multi-group boundaries.
 * @param a_function                [in]    Function to multi-group.
 * @param a_flux                    [in]    Flux to use for weighting.
 * @return                                  Returns the multi-grouped Vector of *a_function*.
 ***********************************************************************************************************/

Vector multiGroupXYs1d( Transporting::MultiGroup const &a_boundaries, Functions::XYs1d const &a_function, Transporting::Flux const &a_flux ) {

    std::vector<double> const &boundaries = a_boundaries.boundaries( );
    ptwXPoints *boundaries_xs = ptwX_create( nullptr, boundaries.size( ), boundaries.size( ), &(boundaries[0]) );
    if( boundaries_xs == nullptr ) throw Exception( "GIDI::multiGroup: ptwX_create failed." );

    Transporting::Flux_order const &flux_order_0 = a_flux[0];
    double const *energies = flux_order_0.energies( );
    double const *fluxes = flux_order_0.fluxes( );
    ptwXYPoints *fluxes_xys = ptwXY_createFrom_Xs_Ys( nullptr, ptwXY_interpolationLinLin, ptwXY_interpolationToString( ptwXY_interpolationLinLin ), 
        12, 1e-3, flux_order_0.size( ), 10, flux_order_0.size( ), energies, fluxes, 0 );
    if( fluxes_xys == nullptr ) {
        ptwX_free( boundaries_xs );
        throw Exception( "GIDI::multiGroup: ptwXY_createFrom_Xs_Ys failed." );
    }

    ptwXPoints *multiGroupFlux = ptwXY_groupOneFunction( nullptr, fluxes_xys, boundaries_xs, ptwXY_group_normType_none, nullptr );
    if( multiGroupFlux == nullptr ) {
        ptwX_free( boundaries_xs );
        ptwXY_free( fluxes_xys );
        throw Exception( "GIDI::multiGroup: ptwXY_groupOneFunction failed." );
    }

    ptwXYPoints *ptwXY = ptwXY_clone2( nullptr, a_function.ptwXY( ) );
    ptwXPoints *groups = nullptr;
    if( ptwXY != nullptr ) {
        ptwXY_mutualifyDomains( nullptr, ptwXY, 1e-12, 1e-12, 1, fluxes_xys, 1e-12, 1e-12, 1 );
        groups = ptwXY_groupTwoFunctions( nullptr, ptwXY, fluxes_xys, boundaries_xs, ptwXY_group_normType_norm, multiGroupFlux );
    }
    ptwX_free( boundaries_xs );
    ptwXY_free( fluxes_xys );
    ptwX_free( multiGroupFlux );
    ptwXY_free( ptwXY );
    if( groups == nullptr ) throw Exception( "GIDI::multiGroup: ptwXY_groupTwoFunctions failed." );

    Vector vector( ptwX_length( nullptr, groups ), ptwX_getPointAtIndex( nullptr, groups, 0 ) );
    ptwX_free( groups );

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns a flux weighted multi-group version of the function *a_function1* * *a_function2*.
 *
 * @param a_boundaries              [in]    List of multi-group boundaries.
 * @param a_function                [in]    Function to multi-group.
 * @param a_flux                    [in]    Flux to use for weighting.
 *
 * @return                                  Returns the multi-grouped Vector of *a_function*.
 ***********************************************************************************************************/

Vector multiGroupTwoXYs1ds( Transporting::MultiGroup const &a_boundaries, Functions::XYs1d const &a_function1, 
                Functions::XYs1d const &a_function2, Transporting::Flux const &a_flux ) {

    ptwXPoints *boundaries_xs = nullptr, *multiGroupFlux = nullptr;
    ptwXYPoints *fluxes_xys = nullptr, *ptwXY1 = nullptr, *ptwXY2 = nullptr;
    std::string errorMessage( "GIDI::multiGroupTwoXYs1ds: ptwXY_clone2 for a_function1 failed." );
    

    multiGroupSetup( a_boundaries, &boundaries_xs, a_flux, &fluxes_xys, &multiGroupFlux );

    ptwXPoints *groups = nullptr;
    ptwXY1 = ptwXY_clone2( nullptr, a_function1.ptwXY( ) );
    if( ptwXY1 != nullptr ) {
        ptwXY2 = ptwXY_clone2( nullptr, a_function2.ptwXY( ) );
        if( ptwXY2 == nullptr ) {
            errorMessage = "GIDI::multiGroupTwoXYs1ds: ptwXY_clone2 for a_function2 failed."; }
        else {
            ptwXY_mutualifyDomains( nullptr, ptwXY1,     1e-12, 1e-12, 1, ptwXY2,     1e-12, 1e-12, 1 );
            ptwXY_mutualifyDomains( nullptr, ptwXY1,     1e-12, 1e-12, 1, fluxes_xys, 1e-12, 1e-12, 1 );
            ptwXY_mutualifyDomains( nullptr, fluxes_xys, 1e-12, 1e-12, 1, ptwXY2,     1e-12, 1e-12, 1 );
            groups = ptwXY_groupThreeFunctions( nullptr, ptwXY1, ptwXY2, fluxes_xys, boundaries_xs, ptwXY_group_normType_norm, multiGroupFlux );
        }
    }
    ptwX_free( boundaries_xs );
    ptwXY_free( fluxes_xys );
    ptwX_free( multiGroupFlux );
    ptwXY_free( ptwXY1 );
    ptwXY_free( ptwXY2 );

    if( groups == nullptr ) throw Exception( errorMessage );

    Vector vector( ptwX_length( nullptr, groups ), ptwX_getPointAtIndex( nullptr, groups, 0 ) );
    ptwX_free( groups );

    return( vector );
}

/* *********************************************************************************************************//**
 * Setups *a_boundaries_xs*, *a_fluxes_xys* and *a_multiGroupFlux* as needed by multi-grouping functions. Calling code are
 * responsible for free-ing *a_boundaries_xs*, *a_fluxes_xys* and *a_multiGroupFlux*.
 *
 * @param a_boundaries              [in]    List of multi-group boundaries.
 * @param a_boundaries_xs           [out]   A ptwXPoints representation of *a_boundaries*.
 * @param a_flux                    [in]    Flux to use for weighting.
 * @param a_fluxes_xys              [out]   A ptwXYPoints representation of *a_flux*.
 * @param a_multiGroupFlux          [out]   A ptwYPoints multi-grouped representation of *a_fluxes_xys*.
 ***********************************************************************************************************/

static void multiGroupSetup( Transporting::MultiGroup const &a_boundaries, ptwXPoints **a_boundaries_xs,
                Transporting::Flux const &a_flux, ptwXYPoints **a_fluxes_xys, ptwXPoints **a_multiGroupFlux ) {

    std::vector<double> const &boundaries = a_boundaries.boundaries( );
    *a_boundaries_xs = ptwX_create( nullptr, boundaries.size( ), boundaries.size( ), &(boundaries[0]) );
    if( *a_boundaries_xs == nullptr ) throw Exception( "GIDI::multiGroup: ptwX_create failed." );

    Transporting::Flux_order const &flux_order_0 = a_flux[0];
    double const *energies = flux_order_0.energies( );
    double const *fluxes = flux_order_0.fluxes( );
    *a_fluxes_xys = ptwXY_createFrom_Xs_Ys( nullptr, ptwXY_interpolationLinLin, ptwXY_interpolationToString( ptwXY_interpolationLinLin ),
        12, 1e-3, flux_order_0.size( ), 10, flux_order_0.size( ), energies, fluxes, 0 );
    if( *a_fluxes_xys == nullptr ) {
        *a_boundaries_xs = ptwX_free( *a_boundaries_xs );
        throw Exception( "GIDI::multiGroup: ptwXY_createFrom_Xs_Ys failed." );
    }

    *a_multiGroupFlux = ptwXY_groupOneFunction( nullptr, *a_fluxes_xys, *a_boundaries_xs, ptwXY_group_normType_none, nullptr );
    if( *a_multiGroupFlux == nullptr ) {
        *a_boundaries_xs = ptwX_free( *a_boundaries_xs );
        *a_fluxes_xys = ptwXY_free( *a_fluxes_xys );
        throw Exception( "GIDI::multiGroup: ptwXY_groupOneFunction failed." );
    }
}

}

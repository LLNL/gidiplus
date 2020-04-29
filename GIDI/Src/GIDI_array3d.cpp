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

/*! \class Array3d
 * Class to store a 3d array.
 */

/* *********************************************************************************************************//**
 *
 * @param a_node            [in]     The **pugi::xml_node** to be parsed and used to construct the Array3d.
 * @param a_useSystem_strtod    [in]    Flag passed to the function nfu_stringToListOfDoubles.
 ***********************************************************************************************************/

Array3d::Array3d( pugi::xml_node const &a_node, int a_useSystem_strtod ) :
        Form( a_node, FormType::array3d ),
        m_array( a_node, 3, a_useSystem_strtod ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Array3d::~Array3d( ) {

}

/* *********************************************************************************************************//**
 * Returns the matrix that represents the specified 3rd dimension. That is the matrix M[i][j] for all i, j of A2d[i][j][*a_index*].
 * This is mainly used for multi-group, Legendre expanded transfer matrices where a specific Legendre order is requested. This is,
 * the matrix represent the *energy_in* as rows and the *energy_outp* as columns for a specific Legendre order.
 *
 * @param a_index           [in]     The requested *index* for the 3rd dimension.
 ***********************************************************************************************************/

Matrix Array3d::matrix( std::size_t a_index ) const {

    if( size( ) <= a_index ) {
        Matrix matrix( 0, 0 );
        return( matrix );
    }

    std::size_t numberOfOrders = m_array.m_shape[2], rows = m_array.m_shape[0], columns = m_array.m_shape[1];
    Matrix matrix( rows, columns );

    std::size_t lengthSum = 0;
    for( std::size_t i1 = 0; i1 < m_array.m_numberOfStarts; ++i1 ) {
        std::size_t start = m_array.m_starts[i1];
        std::size_t length = m_array.m_lengths[i1];

        std::size_t energyInIndex = start / ( numberOfOrders * columns );
        std::size_t energyOutIndex = start % ( numberOfOrders * columns );
        std::size_t orderIndex = energyOutIndex % numberOfOrders;
        energyOutIndex /= numberOfOrders;

        std::size_t step = a_index - orderIndex;
        if( orderIndex > a_index ) {
            ++energyOutIndex;
            if( energyOutIndex >= columns ) {
                energyOutIndex = 0;
                ++energyInIndex;
            }
            step += numberOfOrders;
        }
        std::size_t dataIndex = lengthSum + step;
        for( ; step < length; step += numberOfOrders ) {
            matrix.set( energyInIndex, energyOutIndex, m_array.m_dValues[dataIndex] );
            ++energyOutIndex;
            if( energyOutIndex >= columns ) {
                energyOutIndex = 0;
                ++energyInIndex;
            }
            dataIndex += numberOfOrders;
        }
        lengthSum += length;
    }

    return( matrix );
}

}

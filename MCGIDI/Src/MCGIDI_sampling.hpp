/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#ifndef MCGIDI_sampling_hpp_included
#define MCGIDI_sampling_hpp_included 1

#include "MCGIDI_declareMacro.hpp"
#include "MCGIDI_vector.hpp"

namespace MCGIDI {

namespace Sampling {

enum class SampledType { firstTwoBody, secondTwoBody, uncorrelatedBody, unspecified, photon };

namespace Upscatter {

    enum class Model { none, A, B, BSnLimits };

}

/*
============================================================
================ ClientRandomNumberGenerator ===============
============================================================
*/
class ClientRandomNumberGenerator {
    private:
        double (*m_generator)( void * );                    /**< User supplied generator. */
        void *m_state;                                      /**< User supplied state. */

    public:
        HOST_DEVICE ClientRandomNumberGenerator( double (*a_generator)( void * ), void *a_state );

        HOST_DEVICE double (*generator( ))( void * ) { return( m_generator ); }
        HOST_DEVICE void *state( ) { return( m_state ); }
        HOST_DEVICE double Double( ) { return( m_generator( m_state ) ); }

// The following are deprecated.
        HOST_DEVICE double (*rng( ))( void * ) { return( generator( ) ); }
        HOST_DEVICE void *rngState( ) { return( state( ) ); }
        HOST_DEVICE double dRng( ) { return( Double( ) ); }
};

/*
============================================================
=================== Client Code RNG Data ===================
============================================================
*/
class ClientCodeRNGData : public ClientRandomNumberGenerator {

    public:
        HOST_DEVICE ClientCodeRNGData( double (*a_generator)( void * ), void *a_state );
};

/*
============================================================
=========================== Input ==========================
============================================================
*/
class Input {

    private:
        bool m_wantVelocity;                        /**< See member m_isVelocity in class Product for meaning. This is user input. */

    public:
        double m_temperature;                       /**< Set by user. */

        Upscatter::Model m_upscatterModel;           /**< BRB */
                                                // The rest of the members are set by MCGIDI methods.
                                                // These five are used for upscatter model A.
        bool m_dataInTargetFrame;                   /**< BRB */
        double m_projectileBeta;                    /**< BRB */
        double m_relativeMu;                        /**< BRB */
        double m_targetBeta;                        /**< BRB */
        double m_relativeBeta;                      /**< BRB */
        double m_projectileEnergy;                  /**< BRB */

        SampledType m_sampledType;                  /**< BRB */
        Reaction const *m_reaction;                 /**< BRB */

        double m_projectileMass;                    /**< BRB */
        double m_targetMass;                        /**< BRB */

        GIDI::Frame m_frame;                        /**< The frame the product data are returned in. */

        double m_mu;                                /**< The sampled mu = cos( theta ) for the product. */
        double m_phi;                               /**< The sampled phi for the product. */

        double m_energyOut1;                        /**< The sampled energy of the product. */
        double m_px_vx1;                            /**< Variable used for two-body sampling. */
        double m_py_vy1;                            /**< Variable used for two-body sampling. */
        double m_pz_vz1;                            /**< Variable used for two-body sampling. */

        double m_energyOut2;                        /**< The sampled energy of the second product for a two-body interaction. */
        double m_px_vx2;                            /**< Variable used for two-body sampling. */
        double m_py_vy2;                            /**< Variable used for two-body sampling. */
        double m_pz_vz2;                            /**< Variable used for two-body sampling. */

        int m_delayedNeutronIndex;                  /**< If the product is a delayed neutron, this is its index. */
        double m_delayedNeutronDecayRate;           /**< If the product is a delayed neutron, this is its decay rate. */

        HOST_DEVICE Input( bool a_wantVelocity, Upscatter::Model a_upscatterModel );

        HOST_DEVICE bool wantVelocity( ) const { return( m_wantVelocity ); }                            /**< BRB */

};

/*
============================================================
========================== Product =========================
============================================================
*/
class Product {

    public:
        SampledType m_sampledType;
        bool m_isVelocity;                      /**< If true, m_px_vx, m_py_vy and m_pz_vz are velocities otherwise momenta. */
        int m_productIndex;                     /**< The index of the sampled product. */
        int m_userProductIndex;                 /**< The user particle index of the sampled product. */
        double m_productMass;                   /**< The mass of the sampled product. */
        double m_kineticEnergy;                 /**< The kinetic energy of the sampled product. */
        double m_px_vx;                         /**< The velocity or momentum along the x-axis of the sampled product. */
        double m_py_vy;                         /**< The velocity or momentum along the y-axis of the sampled product. */
        double m_pz_vz;                         /**< The velocity or momentum along the z-axis of the sampled product. The z-axis is along the direction of the projectile's velolcity. */
        int m_delayedNeutronIndex;              /**< If the product is a delayed neutron, this is its index. */
        double m_delayedNeutronDecayRate;       /**< If the product is a delayed neutron, this is its decay rate. */
        double m_birthTimeSec;                  /**< Some products, like delayed fission neutrons, are to appear (be born) later. This is the time in seconds that such a particle should be born since the interaction. */
};

/*
============================================================
====================== ProductHandler ======================
============================================================
*/
class ProductHandler {

    public:
        HOST_DEVICE ProductHandler( ) {}
        HOST_DEVICE virtual ~ProductHandler( ) {}

        HOST_DEVICE virtual std::size_t size( ) = 0;
        HOST_DEVICE virtual void push_back( Product &a_product ) = 0;
        HOST_DEVICE virtual void clear( ) = 0;
        HOST_DEVICE void add( double a_projectileEnergy, int a_productIndex, int a_userProductIndex, double a_productMass, Input &a_input, 
                double (*a_userrng)( void * ), void *a_rngState, bool isPhoton );
};

/*
============================================================
================ StdVectorProductHandler ===================
============================================================
*/
#ifndef __CUDACC__
class StdVectorProductHandler : public ProductHandler {

    private:
        std::vector<Product> m_products;            /**< The list of products sampled. */

    public:
        StdVectorProductHandler( ) : m_products( ) { }
        ~StdVectorProductHandler( ) { }

        HOST_DEVICE virtual std::size_t size( ) { return( m_products.size( ) ); }
        HOST_DEVICE Product &operator[]( long a_index ) { return( m_products[a_index] ); }
        HOST_DEVICE std::vector<Product> &products( ) { return( m_products ); }
        HOST_DEVICE void push_back( Product &a_product ) { m_products.push_back( a_product ); }
        HOST_DEVICE void clear( ) { m_products.clear( ); }
};
#endif

/*
============================================================
============== MCGIDIVectorProductHandler ==================
============================================================
*/
class MCGIDIVectorProductHandler : public ProductHandler {

    private:
        Vector<Product> m_products;             /**< The list of products sampled. */

    public:
        HOST_DEVICE MCGIDIVectorProductHandler( MCGIDI_VectorSizeType a_size = 20 ) :
                m_products( ) {

            m_products.reserve( a_size );
        }
        HOST_DEVICE ~MCGIDIVectorProductHandler( ) {}

        HOST_DEVICE virtual std::size_t size( ) { return( m_products.size( ) ); }
        HOST_DEVICE Product const &operator[]( MCGIDI_VectorSizeType a_index ) const { return( m_products[a_index] ); }
        HOST_DEVICE Vector<Product> const &products( ) const { return( m_products ); }
        HOST_DEVICE void push_back( Product &a_product ) { m_products.push_back( a_product ); }
        HOST_DEVICE void clear( ) { m_products.clear( ); }
};

}       // End of namespace Sampling.

}       // End of namespace MCGIDI.

#endif      // End of MCGIDI_sampling_hpp_included

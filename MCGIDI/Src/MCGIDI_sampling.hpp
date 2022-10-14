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
        MCGIDI_HOST_DEVICE ClientRandomNumberGenerator( double (*a_generator)( void * ), void *a_state );

        MCGIDI_HOST_DEVICE double (*generator( ))( void * ) { return( m_generator ); }
        MCGIDI_HOST_DEVICE void *state( ) { return( m_state ); }
        MCGIDI_HOST_DEVICE double Double( ) { return( m_generator( m_state ) ); }

// The following are deprecated.
        MCGIDI_HOST_DEVICE double (*rng( ))( void * ) { return( generator( ) ); }
        MCGIDI_HOST_DEVICE void *rngState( ) { return( state( ) ); }
        MCGIDI_HOST_DEVICE double dRng( ) { return( Double( ) ); }
};

/*
============================================================
=================== Client Code RNG Data ===================
============================================================
*/
class ClientCodeRNGData : public ClientRandomNumberGenerator {

    public:
        MCGIDI_HOST_DEVICE ClientCodeRNGData( double (*a_generator)( void * ), void *a_state );
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

        Upscatter::Model m_upscatterModel;          /**< The upscatter model to use when sampling a target's velocity. */
                                                // The rest of the members are set by MCGIDI methods.
                                                // These five are used for upscatter model A.
        bool m_dataInTargetFrame;                   /**< **true if the data are in the target's frame and **false** otherwise. */
        double m_projectileBeta;                    /**< The beta = speed / c of the projectile. */
        double m_relativeMu;                        /**< BRB */
        double m_targetBeta;                        /**< The beta = speed / c of the target. */
        double m_relativeBeta;                      /**< The beta = speed / c of the relative speed between the projectile and the target.*/
        double m_projectileEnergy;                  /**< The energy of the projectile. */

        SampledType m_sampledType;                  /**< BRB */
        Reaction const *m_reaction;                 /**< The current reaction whose products are being sampled. */

        double m_projectileMass;                    /**< The mass of the projectile. */
        double m_targetMass;                        /**< The mass of the target. */

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

        MCGIDI_HOST_DEVICE Input( bool a_wantVelocity, Upscatter::Model a_upscatterModel );

        MCGIDI_HOST_DEVICE bool wantVelocity( ) const { return( m_wantVelocity ); }                            /**< BRB */

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
        MCGIDI_HOST_DEVICE ProductHandler( ) {}
        MCGIDI_HOST_DEVICE virtual ~ProductHandler( ) {}

        MCGIDI_HOST_DEVICE virtual std::size_t size( ) = 0;
        MCGIDI_HOST_DEVICE virtual void push_back( Product &a_product ) = 0;
        MCGIDI_HOST_DEVICE virtual void clear( ) = 0;
        MCGIDI_HOST_DEVICE void add( double a_projectileEnergy, int a_productIndex, int a_userProductIndex, double a_productMass, Input &a_input, 
                double (*a_userrng)( void * ), void *a_rngState, bool isPhoton );
};

/*
============================================================
================ StdVectorProductHandler ===================
============================================================
*/
#ifdef __CUDACC__

#define MCGIDI_CUDACC_numberOfProducts 1000

class StdVectorProductHandler : public ProductHandler {

    private:
        std::size_t m_size;
        Product m_products[1024];

    public:
        MCGIDI_HOST_DEVICE StdVectorProductHandler( ) : m_size( 0 ) { }
        MCGIDI_HOST_DEVICE ~StdVectorProductHandler( ) { }

        MCGIDI_HOST_DEVICE virtual std::size_t size( ) { return( m_size ); }
        MCGIDI_HOST_DEVICE Product &operator[]( long a_index ) { return( m_products[a_index] ); }
        MCGIDI_HOST_DEVICE void push_back( Product &a_product ) {
            if( m_size < MCGIDI_CUDACC_numberOfProducts ) {
                m_products[m_size] = a_product;
                ++m_size;
            }
        }
        MCGIDI_HOST_DEVICE void clear( ) { m_size = 0; }
};

#else
class StdVectorProductHandler : public ProductHandler {

    private:
        std::vector<Product> m_products;            /**< The list of products sampled. */

    public:
        MCGIDI_HOST_DEVICE StdVectorProductHandler( ) : m_products( ) { }
        MCGIDI_HOST_DEVICE ~StdVectorProductHandler( ) { }

        MCGIDI_HOST_DEVICE virtual std::size_t size( ) { return( m_products.size( ) ); }
        MCGIDI_HOST_DEVICE Product &operator[]( long a_index ) { return( m_products[a_index] ); }
        MCGIDI_HOST_DEVICE std::vector<Product> &products( ) { return( m_products ); }
        MCGIDI_HOST_DEVICE void push_back( Product &a_product ) { m_products.push_back( a_product ); }
        MCGIDI_HOST_DEVICE void clear( ) { m_products.clear( ); }
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
        MCGIDI_HOST_DEVICE MCGIDIVectorProductHandler( MCGIDI_VectorSizeType a_size = 20 ) :
                m_products( ) {

            m_products.reserve( a_size );
        }
        MCGIDI_HOST_DEVICE ~MCGIDIVectorProductHandler( ) {}

        MCGIDI_HOST_DEVICE virtual std::size_t size( ) { return( m_products.size( ) ); }
        MCGIDI_HOST_DEVICE Product const &operator[]( MCGIDI_VectorSizeType a_index ) const { return( m_products[a_index] ); }
        MCGIDI_HOST_DEVICE Vector<Product> const &products( ) const { return( m_products ); }
        MCGIDI_HOST_DEVICE void push_back( Product &a_product ) { m_products.push_back( a_product ); }
        MCGIDI_HOST_DEVICE void clear( ) { m_products.clear( ); }
};

}       // End of namespace Sampling.

}       // End of namespace MCGIDI.

#endif      // End of MCGIDI_sampling_hpp_included

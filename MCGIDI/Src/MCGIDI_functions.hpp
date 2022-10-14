/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#ifndef MCGIDI_functions_hpp_included
#define MCGIDI_functions_hpp_included 1

#include <math.h>

#include <nf_utilities.h>
#include <ptwXY.h>
#include "MCGIDI_dataBuffer.hpp"

namespace MCGIDI {

enum class Interpolation { LINLIN, LINLOG, LOGLIN, LOGLOG, FLAT, OTHER };
enum class Function1dType { none, constant, XYs, polyomial, gridded, regions, branching, TerrellFissionNeutronMultiplicityModel };
enum class Function2dType { none, XYs };
enum class ProbabilityBase1dType { none, xs_pdf_cdf };
enum class ProbabilityBase2dType { none, XYs, regions, isotropic, discreteGamma, primaryGamma, recoil, NBodyPhaseSpace, evaporation, 
        generalEvaporation, simpleMaxwellianFission, Watt, weightedFunctionals };

enum class ProbabilityBase3dType { none, XYs };

namespace Functions {

/*
============================================================
====================== FunctionBase ========================
============================================================
*/
class FunctionBase {

    private:
        int m_dimension;
        double m_domainMin;
        double m_domainMax;
        Interpolation m_interpolation;
        double m_outerDomainValue;

    public:
        MCGIDI_HOST_DEVICE FunctionBase( );
        MCGIDI_HOST   FunctionBase( GIDI::Functions::FunctionForm const &a_function );
        MCGIDI_HOST_DEVICE FunctionBase( int a_dimension, double a_domainMin, double a_domainMax, Interpolation a_interpolation, double a_outerDomainValue = 0 );
        MCGIDI_HOST_DEVICE virtual ~FunctionBase( ) = 0;

        MCGIDI_HOST_DEVICE Interpolation interpolation( ) const { return( m_interpolation ); }
        MCGIDI_HOST_DEVICE double domainMin( ) const { return( m_domainMin ); }
        MCGIDI_HOST_DEVICE double domainMax( ) const { return( m_domainMax ); }
        MCGIDI_HOST_DEVICE double outerDomainValue( ) const { return( m_outerDomainValue ); }
        MCGIDI_HOST_DEVICE virtual void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
======================== Function1d ========================
============================================================
*/
class Function1d : public FunctionBase {

    protected:
        Function1dType m_type;

    public:
        MCGIDI_HOST_DEVICE Function1d( );
        MCGIDI_HOST_DEVICE Function1d( double a_domainMin, double a_domainMax, Interpolation a_interpolation, double a_outerDomainValue = 0 );
        MCGIDI_HOST_DEVICE ~Function1d( );

        MCGIDI_HOST_DEVICE virtual int sampleBoundingInteger( double a_x1, double (*rng)( void * ), void *rngState ) const ;
        MCGIDI_HOST_DEVICE virtual double evaluate( double a_x1 ) const = 0;
        MCGIDI_HOST_DEVICE Function1dType type( ) { return( m_type ); }
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
======================== Constant1d ========================
============================================================
*/
class Constant1d : public Function1d {

    private:
        double m_value;

    public:
        MCGIDI_HOST_DEVICE Constant1d( );
        MCGIDI_HOST_DEVICE Constant1d( double a_domainMin, double a_domainMax, double a_value, double a_outerDomainValue = 0 );
        MCGIDI_HOST Constant1d( GIDI::Functions::Constant1d const &a_constant1d );
        MCGIDI_HOST_DEVICE ~Constant1d( );

        MCGIDI_HOST_DEVICE double evaluate( double a_x1 ) const { return( m_value ); }
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
=========================== XYs1d ==========================
============================================================
*/
class XYs1d : public Function1d {

    private:
        Vector<double> m_Xs;
        Vector<double> m_Ys;

    public:
        MCGIDI_HOST_DEVICE XYs1d( );
        MCGIDI_HOST XYs1d( Interpolation a_interpolation, Vector<double> a_Xs, Vector<double> a_Ys, double a_outerDomainValue = 0 );
        MCGIDI_HOST XYs1d( GIDI::Functions::XYs1d const &a_XYs1d );
        MCGIDI_HOST_DEVICE ~XYs1d( );

        MCGIDI_HOST_DEVICE double evaluate( double a_x1 ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
======================= Polynomial1d =======================
============================================================
*/          
class Polynomial1d : public Function1d {

    private:
        Vector<double> m_coefficients;
        Vector<double> m_coefficientsReversed;

    public:
        MCGIDI_HOST_DEVICE Polynomial1d( );
        MCGIDI_HOST Polynomial1d( double a_domainMin, double a_domainMax, Vector<double> const &a_coefficients, double a_outerDomainValue = 0 );
        MCGIDI_HOST Polynomial1d( GIDI::Functions::Polynomial1d const &a_polynomial1d );
        MCGIDI_HOST_DEVICE ~Polynomial1d( );

        MCGIDI_HOST_DEVICE Vector<double> const &coefficients( ) const { return( m_coefficients ); }
        MCGIDI_HOST_DEVICE double evaluate( double a_x1 ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
========================= Gridded1d ========================
============================================================
*/
class Gridded1d : public Function1d {

    private:
        Vector<double> m_grid;
        Vector<double> m_data;

    public:
        MCGIDI_HOST_DEVICE Gridded1d( );
        MCGIDI_HOST Gridded1d( GIDI::Functions::Gridded1d const &a_gridded1d );
        MCGIDI_HOST_DEVICE ~Gridded1d( );

        MCGIDI_HOST_DEVICE double evaluate( double a_x1 ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
========================= Regions1d ========================
============================================================
*/
class Regions1d : public Function1d {

    private:
        Vector<double> m_Xs;
        Vector<Function1d *> m_functions1d;

    public:
        MCGIDI_HOST_DEVICE Regions1d( );
        MCGIDI_HOST Regions1d( GIDI::Functions::Regions1d const &a_regions1d );
        MCGIDI_HOST_DEVICE ~Regions1d( );

        MCGIDI_HOST_DEVICE void append( Function1d *a_function1d );
        MCGIDI_HOST_DEVICE double evaluate( double a_x1 ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
======================== Branching1d =======================
============================================================
*/
class Branching1d : public Function1d {

    private:
        int m_initialStateIndex;

    public:
        MCGIDI_HOST_DEVICE Branching1d( );
        MCGIDI_HOST Branching1d( SetupInfo &a_setupInfo, GIDI::Functions::Branching1d const &a_branching1d );
        MCGIDI_HOST_DEVICE ~Branching1d( );

        MCGIDI_HOST_DEVICE int initialStateIndex( ) const { return( m_initialStateIndex ); }

        MCGIDI_HOST_DEVICE double evaluate( double a_x1 ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
========== TerrellFissionNeutronMultiplicityModel ==========
============================================================
*/
class TerrellFissionNeutronMultiplicityModel : public Function1d {

    private:
        double m_width;
        Function1d *m_multiplicity;

    public:
        MCGIDI_HOST_DEVICE TerrellFissionNeutronMultiplicityModel( );
        MCGIDI_HOST TerrellFissionNeutronMultiplicityModel( double a_width, Function1d *a_multiplicity );
        MCGIDI_HOST_DEVICE ~TerrellFissionNeutronMultiplicityModel( );

        MCGIDI_HOST_DEVICE int sampleBoundingInteger( double a_energy, double (*a_rng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE double evaluate( double a_energy ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
======================== Function2d ========================
============================================================
*/
class Function2d : public FunctionBase {

    protected:
        Function2dType m_type;

    public:
        MCGIDI_HOST_DEVICE Function2d( );
        MCGIDI_HOST Function2d( double a_domainMin, double a_domainMax, Interpolation a_interpolation, double a_outerDomainValue = 0 );
        MCGIDI_HOST_DEVICE ~Function2d( );

        MCGIDI_HOST_DEVICE Function2dType type( ) { return m_type; }
        MCGIDI_HOST_DEVICE virtual double evaluate( double a_x2, double a_x1 ) const = 0;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
=========================== XYs2d ==========================
============================================================
*/
class XYs2d : public Function2d {

    private:
        Vector<double> m_Xs;
        Vector<Function1d *> m_functions1d;

    public:
        MCGIDI_HOST_DEVICE XYs2d( );
        MCGIDI_HOST XYs2d( GIDI::Functions::XYs2d const &a_XYs2d );
        MCGIDI_HOST_DEVICE ~XYs2d( );

        MCGIDI_HOST_DEVICE double evaluate( double a_x2, double a_x1 ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
========================== others ==========================
============================================================
*/
MCGIDI_HOST Function1d *parseMultiplicityFunction1d( SetupInfo &a_setupInfo, Transporting::MC const &a_settings, GIDI::Suite const &a_suite );
MCGIDI_HOST Function1d *parseFunction1d( Transporting::MC const &a_settings, GIDI::Suite const &a_suite );
MCGIDI_HOST Function1d *parseFunction1d( GIDI::Functions::Function1dForm const *form1d );
MCGIDI_HOST Function2d *parseFunction2d( Transporting::MC const &a_settings, GIDI::Suite const &a_suite );
MCGIDI_HOST Function2d *parseFunction2d( GIDI::Functions::Function2dForm const *form2d );

}           // End of namespace Functions.

/*
============================================================
============================================================
================== namespace Probabilities ==================
============================================================
============================================================
*/
namespace Probabilities {

/*
============================================================
===================== ProbabilityBase ======================
============================================================
*/
class ProbabilityBase : public Functions::FunctionBase {

    protected:
        Vector<double> m_Xs;

    public:

        MCGIDI_HOST_DEVICE ProbabilityBase( );
        MCGIDI_HOST ProbabilityBase( GIDI::Functions::FunctionForm const &a_probabilty );
        MCGIDI_HOST ProbabilityBase( GIDI::Functions::FunctionForm const &a_probabilty, Vector<double> const &a_Xs );
        MCGIDI_HOST_DEVICE ~ProbabilityBase( );
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
===================== ProbabilityBase1d ====================
============================================================
*/
class ProbabilityBase1d : public ProbabilityBase {

    protected:
        ProbabilityBase1dType m_type;

    public:
        MCGIDI_HOST_DEVICE ProbabilityBase1d( );
        MCGIDI_HOST ProbabilityBase1d( GIDI::Functions::FunctionForm const &a_probabilty, Vector<double> const &a_Xs );
        MCGIDI_HOST_DEVICE ~ProbabilityBase1d( );

        MCGIDI_HOST_DEVICE ProbabilityBase1dType type( ) { return m_type; }
        MCGIDI_HOST_DEVICE virtual double evaluate( double a_x1 ) const = 0;
        MCGIDI_HOST_DEVICE virtual double sample( double a_rngValue, double (*a_userrng)( void * ), void *a_rngState ) const = 0;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
======================= Xs_pdf_cdf1d =======================
============================================================
*/
class Xs_pdf_cdf1d : public ProbabilityBase1d {

    private:
        Vector<double> m_pdf;
        Vector<double> m_cdf;

    public:
        MCGIDI_HOST_DEVICE Xs_pdf_cdf1d( );
        MCGIDI_HOST Xs_pdf_cdf1d( GIDI::Functions::Xs_pdf_cdf1d const &a_xs_pdf_cdf1d );
        MCGIDI_HOST_DEVICE ~Xs_pdf_cdf1d( );

        MCGIDI_HOST_DEVICE double evaluate( double a_x1 ) const ;
        MCGIDI_HOST_DEVICE double sample( double a_rngValue, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
===================== ProbabilityBase2d ====================
============================================================
*/
class ProbabilityBase2d : public ProbabilityBase {

    protected:
        ProbabilityBase2dType m_type;

    public:
        MCGIDI_HOST_DEVICE ProbabilityBase2d( );
        MCGIDI_HOST ProbabilityBase2d( GIDI::Functions::FunctionForm const &a_probabilty );
        MCGIDI_HOST ProbabilityBase2d( GIDI::Functions::FunctionForm const &a_probabilty, Vector<double> const &a_Xs );
        MCGIDI_HOST_DEVICE ~ProbabilityBase2d( );

        MCGIDI_HOST_DEVICE ProbabilityBase2dType type( ) { return m_type; }
        MCGIDI_HOST_DEVICE virtual double evaluate( double a_x2, double a_x1 ) const = 0;
        MCGIDI_HOST_DEVICE virtual double sample( double a_x2, double a_rngValue, double (*a_userrng)( void * ), void *a_rngState ) const = 0;
        MCGIDI_HOST_DEVICE virtual double sample2dOf3d( double a_x2, double a_rngValue, double (*a_userrng)( void * ), void *a_rngState, double *a_x1_1, double *a_x1_2 ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
========================== XYs2d ===========================
============================================================
*/
class XYs2d : public ProbabilityBase2d {

    private:
        Vector<ProbabilityBase1d *> m_probabilities;

    public:
        MCGIDI_HOST_DEVICE XYs2d( );
        MCGIDI_HOST XYs2d( GIDI::Functions::XYs2d const &a_XYs2d );
        MCGIDI_HOST_DEVICE ~XYs2d( );

        MCGIDI_HOST_DEVICE double evaluate( double a_x2, double a_x1 ) const ;
        MCGIDI_HOST_DEVICE double sample( double a_x2, double a_rngValue, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE double sample2dOf3d( double a_x2, double a_rngValue, double (*a_userrng)( void * ), void *a_rngState, double *a_x1_1, double *a_x1_2 ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
======================== Regions2d =========================
============================================================
*/
class Regions2d : public ProbabilityBase2d {

    private:
        Vector<ProbabilityBase2d *> m_probabilities;

    public:
        MCGIDI_HOST_DEVICE Regions2d( );
        MCGIDI_HOST Regions2d( GIDI::Functions::Regions2d const &a_regions2d );
        MCGIDI_HOST_DEVICE ~Regions2d( );

        MCGIDI_HOST_DEVICE double evaluate( double a_x2, double a_x1 ) const ;
        MCGIDI_HOST_DEVICE double sample( double a_x2, double a_rngValue, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
======================== Isotropic2d =======================
============================================================
*/
class Isotropic2d : public ProbabilityBase2d {

    public:
        MCGIDI_HOST_DEVICE Isotropic2d( );
        MCGIDI_HOST Isotropic2d( GIDI::Functions::Isotropic2d const &a_isotropic2d );
        MCGIDI_HOST_DEVICE ~Isotropic2d( );

        MCGIDI_HOST_DEVICE double evaluate( double a_x2, double a_x1 ) const { return( 0.5 ); }
        MCGIDI_HOST_DEVICE double sample( double a_x2, double a_rngValue, double (*a_userrng)( void * ), void *a_rngState ) const { return( 1. - 2. * a_rngValue ); }
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode ) { ProbabilityBase2d::serialize( a_buffer, a_mode ); }
};

/*
============================================================
====================== DiscreteGamma2d =====================
============================================================
*/
class DiscreteGamma2d : public ProbabilityBase2d {

    private:
        double m_value;

    public:
        MCGIDI_HOST_DEVICE DiscreteGamma2d( );
        MCGIDI_HOST DiscreteGamma2d( GIDI::Functions::DiscreteGamma2d const &a_discreteGamma2d );
        MCGIDI_HOST_DEVICE ~DiscreteGamma2d( );

        MCGIDI_HOST_DEVICE double evaluate( double a_x2, double a_x1 ) const { return( m_value ); }        // FIXME This is wrong, should be something like 1 when domainMin <= a_x1 <= domainMax ), I think. I.e., should be a probability.
        MCGIDI_HOST_DEVICE double sample( double a_x2, double a_rngValue, double (*a_userrng)( void * ), void *a_rngState ) const { return( m_value ); }
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
====================== PrimaryGamma2d =====================
============================================================
*/
class PrimaryGamma2d : public ProbabilityBase2d {

    private:
        double m_primaryEnergy;
        double m_massFactor;
        String m_finalState;

    public:
        MCGIDI_HOST_DEVICE PrimaryGamma2d( );
        MCGIDI_HOST PrimaryGamma2d( GIDI::Functions::PrimaryGamma2d const &a_primaryGamma2d, SetupInfo *a_setupInfo );
        MCGIDI_HOST_DEVICE ~PrimaryGamma2d( );

        double primaryEnergy( ) const { return( m_primaryEnergy ); }
        double massFactor( ) const { return( m_massFactor ); }
        String const &finalState( ) const { return( m_finalState ); }

        MCGIDI_HOST_DEVICE double evaluate( double a_x2, double a_x1 ) const ;
        MCGIDI_HOST_DEVICE double sample( double a_x2, double a_rngValue, double (*a_userrng)( void * ), void *a_rngState ) const { return( m_primaryEnergy + a_x2 * m_massFactor ); }
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
========================= Recoil2d =========================
============================================================
*/
class Recoil2d: public ProbabilityBase2d {

    private:
        String m_xlink;

    public:
        MCGIDI_HOST_DEVICE Recoil2d( );
        MCGIDI_HOST Recoil2d( GIDI::Functions::Recoil2d const &a_recoil2d );
        MCGIDI_HOST_DEVICE ~Recoil2d( );

        MCGIDI_HOST_DEVICE double evaluate( double a_x2, double a_x1 ) const ;
        MCGIDI_HOST_DEVICE double sample( double a_x2, double a_rngValue, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
==================== NBodyPhaseSpace2d =====================
============================================================
*/
class NBodyPhaseSpace2d : public ProbabilityBase2d {

    private:
        int m_numberOfProducts;
        double m_mass;
        double m_energy_in_COMFactor;
        double m_massFactor;
        double m_Q;
        ProbabilityBase1d *m_dist;

    public:
        MCGIDI_HOST_DEVICE NBodyPhaseSpace2d( );
        MCGIDI_HOST NBodyPhaseSpace2d( GIDI::Functions::NBodyPhaseSpace2d const &a_NBodyPhaseSpace2d, SetupInfo *a_setupInfo );
        MCGIDI_HOST_DEVICE ~NBodyPhaseSpace2d( );

        MCGIDI_HOST_DEVICE double evaluate( double a_x2, double a_x1 ) const ;
        MCGIDI_HOST_DEVICE double sample( double a_x2, double a_rngValue, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
====================== Evaporation2d =======================
============================================================
*/
class Evaporation2d: public ProbabilityBase2d {

    private:
        double m_U;
        Functions::Function1d *m_theta;

    public:
        MCGIDI_HOST_DEVICE Evaporation2d( );
        MCGIDI_HOST Evaporation2d( GIDI::Functions::Evaporation2d const &a_generalEvaporation2d );
        MCGIDI_HOST_DEVICE ~Evaporation2d( );

        MCGIDI_HOST_DEVICE double evaluate( double a_x2, double a_x1 ) const ;
        MCGIDI_HOST_DEVICE double sample( double a_x2, double a_rngValue, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
=================== GeneralEvaporation2d ===================
============================================================
*/
class GeneralEvaporation2d: public ProbabilityBase2d {

    private:
        Functions::Function1d *m_theta;
        ProbabilityBase1d *m_g;

    public:
        MCGIDI_HOST_DEVICE GeneralEvaporation2d( );
        MCGIDI_HOST GeneralEvaporation2d( GIDI::Functions::GeneralEvaporation2d const &a_generalEvaporation2d );
        MCGIDI_HOST_DEVICE ~GeneralEvaporation2d( );

        MCGIDI_HOST_DEVICE double evaluate( double a_x2, double a_x1 ) const ;
        MCGIDI_HOST_DEVICE double sample( double a_x2, double a_rngValue, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
================= SimpleMaxwellianFission2d ================
============================================================
*/
class SimpleMaxwellianFission2d: public ProbabilityBase2d {

    private:
        double m_U;
        Functions::Function1d *m_theta;

    public:
        MCGIDI_HOST_DEVICE SimpleMaxwellianFission2d( );
        MCGIDI_HOST SimpleMaxwellianFission2d( GIDI::Functions::SimpleMaxwellianFission2d const &a_simpleMaxwellianFission2d );
        MCGIDI_HOST_DEVICE ~SimpleMaxwellianFission2d( );

        MCGIDI_HOST_DEVICE double evaluate( double a_x2, double a_x1 ) const ;
        MCGIDI_HOST_DEVICE double sample( double a_x2, double a_rngValue, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
========================== Watt2d ==========================
============================================================
*/
class Watt2d : public ProbabilityBase2d {

    private:
        double m_U;
        Functions::Function1d *m_a;
        Functions::Function1d *m_b;

    public:
        MCGIDI_HOST_DEVICE Watt2d( );
        MCGIDI_HOST Watt2d( GIDI::Functions::Watt2d const &a_Watt2d );
        MCGIDI_HOST_DEVICE ~Watt2d( );

        MCGIDI_HOST_DEVICE double evaluate( double a_x2, double a_x1 ) const ;
        MCGIDI_HOST_DEVICE double sample( double a_x2, double a_rngValue, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
=================== WeightedFunctionals2d ==================
============================================================
*/
class WeightedFunctionals2d: public ProbabilityBase2d {

    private:
      Vector<Functions::Function1d *> m_weight;
      Vector<ProbabilityBase2d *> m_energy;

    public:
        MCGIDI_HOST_DEVICE WeightedFunctionals2d( );
        MCGIDI_HOST WeightedFunctionals2d( GIDI::Functions::WeightedFunctionals2d const &a_weightedFunctionals2d );
        MCGIDI_HOST_DEVICE ~WeightedFunctionals2d( );

        MCGIDI_HOST_DEVICE double evaluate( double a_x2, double a_x1 ) const ;
        MCGIDI_HOST_DEVICE double sample( double a_x2, double a_rngValue, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
===================== ProbabilityBase3d ====================
============================================================
*/
class ProbabilityBase3d : public ProbabilityBase {

    protected:
        ProbabilityBase3dType m_type;

    public:
        MCGIDI_HOST_DEVICE ProbabilityBase3d( );
        MCGIDI_HOST ProbabilityBase3d( GIDI::Functions::FunctionForm const &a_probabilty, Vector<double> const &a_Xs );
        MCGIDI_HOST_DEVICE ~ProbabilityBase3d( );

        MCGIDI_HOST_DEVICE ProbabilityBase3dType type( ) { return m_type; }
        MCGIDI_HOST_DEVICE virtual double evaluate( double a_x3, double a_x2, double a_x1 ) const = 0;
        MCGIDI_HOST_DEVICE virtual double sample( double a_x3, double a_x2_1, double a_x2_2, double a_rngValue, double (*a_userrng)( void * ), void *a_rngState ) const = 0;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
========================== XYs3d ===========================
============================================================
*/
class XYs3d : public ProbabilityBase3d {

    private:
        Vector<ProbabilityBase2d *> m_probabilities;

    public:
        MCGIDI_HOST_DEVICE XYs3d( );
        MCGIDI_HOST XYs3d( GIDI::Functions::XYs3d const &a_XYs3d );
        MCGIDI_HOST_DEVICE ~XYs3d( );

        MCGIDI_HOST_DEVICE double evaluate( double a_x3, double a_x2, double a_x1 ) const ;
        MCGIDI_HOST_DEVICE double sample( double a_x3, double a_x2_1, double a_x2_2, double a_rngValue, double (*a_userrng)( void * ), void *a_rngState ) const ;
        MCGIDI_HOST_DEVICE void serialize( DataBuffer &a_buffer, DataBuffer::Mode a_mode );
};

/*
============================================================
========================== others ==========================
============================================================
*/
MCGIDI_HOST ProbabilityBase1d *parseProbability1d( Transporting::MC const &a_settings, GIDI::Suite const &a_suite );
MCGIDI_HOST ProbabilityBase1d *parseProbability1d( GIDI::Functions::Function1dForm const *form1d );
MCGIDI_HOST ProbabilityBase2d *parseProbability2d( Transporting::MC const &a_settings, GIDI::Suite const &a_suite, SetupInfo *a_setupInfo );
MCGIDI_HOST ProbabilityBase2d *parseProbability2d( GIDI::Functions::Function2dForm const *form2d, SetupInfo *a_setupInfo );
MCGIDI_HOST ProbabilityBase3d *parseProbability3d( Transporting::MC const &a_settings, GIDI::Suite const &a_suite );
MCGIDI_HOST ProbabilityBase3d *parseProbability3d( GIDI::Functions::Function3dForm const *form3d );


}           // End of namespace Probabilities.

/*
============================================================
========================== others ==========================
============================================================
*/
MCGIDI_HOST_DEVICE Interpolation GIDI2MCGIDI_interpolation( ptwXY_interpolation a_interpolation );

MCGIDI_HOST_DEVICE Function1dType Function1dClass( Functions::Function1d *funct );
MCGIDI_HOST_DEVICE Functions::Function1d *serializeFunction1d( DataBuffer &a_buffer, DataBuffer::Mode a_mode, Functions::Function1d *a_function1d );

MCGIDI_HOST_DEVICE Function2dType Function2dClass( Functions::Function2d *funct );
MCGIDI_HOST_DEVICE Functions::Function2d *serializeFunction2d( DataBuffer &a_buffer, DataBuffer::Mode a_mode, Functions::Function2d *a_function2d );

MCGIDI_HOST_DEVICE ProbabilityBase1dType ProbabilityBase1dClass( Probabilities::ProbabilityBase1d *funct );
MCGIDI_HOST_DEVICE Probabilities::ProbabilityBase1d *serializeProbability1d( DataBuffer &a_buffer, DataBuffer::Mode a_mode, Probabilities::ProbabilityBase1d *a_probability1d );

MCGIDI_HOST_DEVICE ProbabilityBase2dType ProbabilityBase2dClass( Probabilities::ProbabilityBase2d *funct );
MCGIDI_HOST_DEVICE Probabilities::ProbabilityBase2d *serializeProbability2d( DataBuffer &a_buffer, DataBuffer::Mode a_mode, Probabilities::ProbabilityBase2d *a_probability2d );

MCGIDI_HOST_DEVICE ProbabilityBase3dType ProbabilityBase3dClass (Probabilities::ProbabilityBase3d *funct );
MCGIDI_HOST_DEVICE Probabilities::ProbabilityBase3d *serializeProbability3d( DataBuffer &a_buffer, DataBuffer::Mode a_mode, Probabilities::ProbabilityBase3d *a_probability3d );

}           // End of namespace MCGIDI.

#endif      // End of MCGIDI_functions_hpp_included

/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "GIDI.hpp"

#define calculatedAtThermalAttribute "calculatedAtThermal"
#define asymmetricAttribute "asymmetric"

namespace GIDI {

namespace DoubleDifferentialCrossSection {

/*! \class Base
 * Base class inherited by DoubleDifferentialCrossSection forms.
 */

/* *********************************************************************************************************//**
 * @param a_node            [in]    The **pugi::xml_node** to be parsed and used to construct the Base.
 * @param a_type            [in]    The formType for the DoubleDifferentialCrossSection form.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

Base::Base( pugi::xml_node const &a_node, formType a_type, Suite *a_parent ) :
        Form( a_node, a_type, a_parent ) {

}

/*! \class CoherentPhotoAtomicScattering
 * This is the **coherentPhotonScattering** style class.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options for parsing.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops            [in]    A PoPs::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs    [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                  This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

CoherentPhotoAtomicScattering::CoherentPhotoAtomicScattering( Construction::Settings const &a_construction, pugi::xml_node const &a_node, 
                PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs, Suite *a_parent ) :
        Base( a_node, f_coherentPhotonScattering, a_parent ),
        m_formFactor( data1dParse( a_construction, a_node.child( "formFactor" ).first_child( ), NULL ) ),
        m_realAnomalousFactor( data1dParseAllowEmpty( a_construction, a_node.child( "imaginaryAnomalousFactor" ).first_child( ), NULL ) ),
        m_imaginaryAnomalousFactor( data1dParseAllowEmpty( a_construction, a_node.child( "imaginaryAnomalousFactor" ).first_child( ), NULL ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

CoherentPhotoAtomicScattering::~CoherentPhotoAtomicScattering( ) {

    delete m_formFactor;
    delete m_realAnomalousFactor;
    delete m_imaginaryAnomalousFactor;
}

/*! \class IncoherentPhotoAtomicScattering
 * This is the **incoherentPhotonScattering** class.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options for parsing.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops            [in]    A PoPs::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs    [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                  This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

IncoherentPhotoAtomicScattering::IncoherentPhotoAtomicScattering( Construction::Settings const &a_construction, pugi::xml_node const &a_node, 
                PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs, Suite *a_parent ) :
        Base( a_node, f_incoherentPhotonScattering, a_parent ),
        m_scatteringFunction( data1dParse( a_construction, a_node.first_child( ), NULL ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

IncoherentPhotoAtomicScattering::~IncoherentPhotoAtomicScattering( ) {

    delete m_scatteringFunction;
}

namespace n_ThermalNeutronScatteringLaw {

/*! \class S_table
 * This is the **S_table** class.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options for parsing.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 ***********************************************************************************************************/

S_table::S_table( Construction::Settings const &a_construction, pugi::xml_node const &a_node ) :
        Form( a_node, f_generic, NULL ),
        m_function2d( NULL ) { // data2dParse( a_construction, a_node.first_child( ), NULL ) ) 

// FIXME BRB
//    m_function2d->setAncestor( this );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

S_table::~S_table( ) {

    delete m_function2d;

}

/*! \class CoherentElastic
 * This is the **CoherentElastic** class.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options for parsing.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops            [in]    A PoPs::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs    [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                  This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

CoherentElastic::CoherentElastic( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs, Suite *a_parent ) :
        Base( a_node, f_coherentElastic, a_parent ),
        m_S_table( a_construction, a_node.child( "S_table" ) ) {

    m_S_table.setAncestor( this );        
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

CoherentElastic::~CoherentElastic( ) {

}

/*! \class DebyeWaller
 * This is the **DebyeWaller** class.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options for parsing.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 ***********************************************************************************************************/

DebyeWaller::DebyeWaller( Construction::Settings const &a_construction, pugi::xml_node const &a_node ) :
        Form( a_node, f_generic, NULL ),
        m_function1d( data1dParse( a_construction, a_node.first_child( ), NULL ) ) {

    m_function1d->setAncestor( this );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

DebyeWaller::~DebyeWaller( ) {

    delete m_function1d;
}

/*! \class IncoherentElastic
 * This is the **IncoherentElastic** class.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options for parsing.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops            [in]    A PoPs::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs    [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                  This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

IncoherentElastic::IncoherentElastic( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs, Suite *a_parent ) :
        Base( a_node, f_coherentElastic, a_parent ),
        m_characteristicCrossSection( a_node.child( "characteristicCrossSection" ) ),
        m_DebyeWaller( a_construction, a_node.child( "DebyeWaller" ) ) {

    m_characteristicCrossSection.setAncestor( this );        
    m_DebyeWaller.setAncestor( this );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

IncoherentElastic::~IncoherentElastic( ) {

}

/*! \class Options
 * This is the **options** class.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options for parsing.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 ***********************************************************************************************************/

Options::Options( Construction::Settings const &a_construction, pugi::xml_node const &a_node ) :
        Form( a_node, f_generic, NULL ),
        m_calculatedAtThermal( strcmp( a_node.attribute( calculatedAtThermalAttribute ).value( ), "true" ) == 0 ),
        m_asymmetric( strcmp( a_node.attribute( asymmetricAttribute ).value( ), "true" ) == 0 ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Options::~Options( ) {

}

/*! \class T_effective
 * This is the **T_effective** class.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options for parsing.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 ***********************************************************************************************************/

T_effective::T_effective( Construction::Settings const &a_construction, pugi::xml_node const &a_node ) :
        Form( a_node, f_generic, NULL ),
        m_function1d( data1dParseAllowEmpty( a_construction, a_node.first_child( ), NULL ) ) {

    if( m_function1d != NULL ) m_function1d->setAncestor( this );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

T_effective::~T_effective( ) {

    delete m_function1d;
}

/*! \class ScatteringAtom
 * This is the **scatteringAtom** class.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options for parsing.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 ***********************************************************************************************************/

ScatteringAtom::ScatteringAtom( Construction::Settings const &a_construction, pugi::xml_node const &a_node ) :
        Form( a_node, f_generic, NULL ),
        m_mass( a_node.child( "mass" ) ),
        m_freeAtomCrossSection( a_node.child( "freeAtomCrossSection" ) ), 
        m_e_critical( a_node.child( "e_critical" ) ),
        m_e_max( a_node.child( "e_max" ) ),
        m_T_effective( a_construction, a_node.child( "T_effective" ) ) {

    m_mass.setAncestor( this );
    m_freeAtomCrossSection.setAncestor( this );
    m_e_critical.setAncestor( this );
    m_e_max.setAncestor( this );
    m_T_effective.setAncestor( this );

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

ScatteringAtom::~ScatteringAtom( ) {

}

/*! \class S_alpha_beta
 * This is the **S_alpha_beta** class.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options for parsing.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 ***********************************************************************************************************/

S_alpha_beta::S_alpha_beta( Construction::Settings const &a_construction, pugi::xml_node const &a_node ) :
        Form( a_node, f_generic, NULL ),
        m_function3d( NULL ) { // data3dParse( a_construction, a_node.first_child( ), NULL ) )

// FIXME BRB
//    m_function3d->setAncestor( this );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

S_alpha_beta::~S_alpha_beta( ) {

    delete m_function3d;
}

/*! \class IncoherentInelastic
 * This is the **IncoherentInelastic** class.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options for parsing.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed.
 * @param a_pops            [in]    A PoPs::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs    [in]    The *internal* PoPs::Database instance used to get particle indices and possibly other particle information.
 *                                  This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

IncoherentInelastic::IncoherentInelastic( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs, Suite *a_parent ) :
        Base( a_node, f_incoherentInelastic, a_parent ),
        m_options( a_construction, a_node.child( "options" ) ),
        m_scatteringAtoms( a_construction, scatteringAtomsMoniker, a_node, a_pops, a_internalPoPs, parseScatteringAtom, NULL ),
        m_S_alpha_beta( a_construction, a_node.child( "S_alpha_beta" ) ) {

    m_options.setAncestor( this );
    m_scatteringAtoms.setAncestor( this );
    m_S_alpha_beta.setAncestor( this );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

IncoherentInelastic::~IncoherentInelastic( ) {

}

}                   // End namespace n_ThermalNeutronScatteringLaw.

}                   // End namespace DoubleDifferentialCrossSection.

}                   // End namespace GIDI.

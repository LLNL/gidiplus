/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "GIDI.hpp"
#include <HAPI.hpp>

namespace GIDI {

namespace DoubleDifferentialCrossSection {

/*! \class Base
 * Base class inherited by DoubleDifferentialCrossSection forms.
 */

/* *********************************************************************************************************//**
 * @param a_node            [in]    The **HAPI::Node** to be parsed and used to construct the Base.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 * @param a_type            [in]    The FormType for the DoubleDifferentialCrossSection form.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

Base::Base( HAPI::Node const &a_node, SetupInfo &a_setupInfo, FormType a_type, Suite *a_parent ) :
        Form( a_node, a_setupInfo, a_type, a_parent ) {

}

/*! \class CoherentPhotoAtomicScattering
 * This is the **coherentPhotonScattering** style class.
 */

/* *********************************************************************************************************//**
 * @param a_construction    [in]    Used to pass user options for parsing.
 * @param a_node            [in]    The **HAPI::Node** to be parsed.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 * @param a_pops            [in]    A PoPI::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs    [in]    The *internal* PoPI::Database instance used to get particle indices and possibly other particle information.
 *                                  This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

CoherentPhotoAtomicScattering::CoherentPhotoAtomicScattering( Construction::Settings const &a_construction, HAPI::Node const &a_node,
		SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs, Suite *a_parent ) :
        Base( a_node, a_setupInfo, FormType::coherentPhotonScattering, a_parent ),
        m_formFactor( data1dParse( a_construction, a_node.child( GIDI_formFactorChars ).first_child( ), a_setupInfo, nullptr ) ),
        m_realAnomalousFactor( data1dParseAllowEmpty( a_construction, a_node.child( GIDI_realAnomalousFactorChars ).first_child( ), a_setupInfo, nullptr ) ),
        m_imaginaryAnomalousFactor( data1dParseAllowEmpty( a_construction, a_node.child( GIDI_imaginaryAnomalousFactorChars ).first_child( ), a_setupInfo, nullptr ) ) {

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
 * @param a_node            [in]    The **HAPI::Node** to be parsed.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 * @param a_pops            [in]    A PoPI::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs    [in]    The *internal* PoPI::Database instance used to get particle indices and possibly other particle information.
 *                                  This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

IncoherentPhotoAtomicScattering::IncoherentPhotoAtomicScattering( Construction::Settings const &a_construction, HAPI::Node const &a_node,
		SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs, Suite *a_parent ) :
        Base( a_node, a_setupInfo, FormType::incoherentPhotonScattering, a_parent ),
        m_scatteringFunction( data1dParse( a_construction, a_node.first_child( ), a_setupInfo, nullptr ) ) {

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
 * @param a_node            [in]    The **HAPI::Node** to be parsed.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 ***********************************************************************************************************/

S_table::S_table( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo ) :
        Form( a_node, a_setupInfo, FormType::generic, nullptr ),
        m_function2d( nullptr ) { // data2dParse( a_construction, a_node.first_child( ), a_setupInfo, nullptr ) ) 

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
 * @param a_node            [in]    The **HAPI::Node** to be parsed.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 * @param a_pops            [in]    A PoPI::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs    [in]    The *internal* PoPI::Database instance used to get particle indices and possibly other particle information.
 *                                  This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

CoherentElastic::CoherentElastic( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo,
		PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs, Suite *a_parent ) :
        Base( a_node, a_setupInfo, FormType::coherentElastic, a_parent ),
        m_S_table( a_construction, a_node.child( GIDI_S_tableChars ), a_setupInfo ) {

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
 * @param a_node            [in]    The **HAPI::Node** to be parsed.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 ***********************************************************************************************************/

DebyeWaller::DebyeWaller( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo ) :
        Form( a_node, a_setupInfo, FormType::generic, nullptr ),
        m_function1d( data1dParse( a_construction, a_node.first_child( ), a_setupInfo, nullptr ) ) {

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
 * @param a_node            [in]    The **HAPI::Node** to be parsed.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 * @param a_pops            [in]    A PoPI::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs    [in]    The *internal* PoPI::Database instance used to get particle indices and possibly other particle information.
 *                                  This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

IncoherentElastic::IncoherentElastic( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo,
		PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs, Suite *a_parent ) :
        Base( a_node, a_setupInfo, FormType::coherentElastic, a_parent ),
        m_characteristicCrossSection( a_node.child( GIDI_characteristicCrossSectionChars ), a_setupInfo ),
        m_DebyeWaller( a_construction, a_node.child( GIDI_DebyeWallerChars ), a_setupInfo ) {

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
 * @param a_node            [in]    The **HAPI::Node** to be parsed.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 ***********************************************************************************************************/

Options::Options( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo ) :
        Form( a_node, a_setupInfo, FormType::generic, nullptr ),
        m_calculatedAtThermal( strcmp( a_node.attribute_as_string( GIDI_calculatedAtThermalChars ).c_str( ), GIDI_trueChars ) == 0 ),
        m_asymmetric( strcmp( a_node.attribute_as_string( GIDI_asymmetricChars ).c_str( ), GIDI_trueChars ) == 0 ) {

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
 * @param a_node            [in]    The **HAPI::Node** to be parsed.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 ***********************************************************************************************************/

T_effective::T_effective( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo ) :
        Form( a_node, a_setupInfo, FormType::generic, nullptr ),
        m_function1d( data1dParseAllowEmpty( a_construction, a_node.first_child( ), a_setupInfo, nullptr ) ) {

    if( m_function1d != nullptr ) m_function1d->setAncestor( this );
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
 * @param a_node            [in]    The **HAPI::Node** to be parsed.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 ***********************************************************************************************************/

ScatteringAtom::ScatteringAtom( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo ) :
        Form( a_node, a_setupInfo, FormType::generic, nullptr ),
        m_mass( a_node.child( GIDI_massChars ), a_setupInfo ),
        m_freeAtomCrossSection( a_node.child( GIDI_freeAtomCrossSectionChars ), a_setupInfo ), 
        m_e_critical( a_node.child( GIDI_e_criticalChars ), a_setupInfo ),
        m_e_max( a_node.child( GIDI_e_maxChars ), a_setupInfo ),
        m_T_effective( a_construction, a_node.child( GIDI_T_effectiveChars ), a_setupInfo ) {

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
 * @param a_node            [in]    The **HAPI::Node** to be parsed.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 ***********************************************************************************************************/

S_alpha_beta::S_alpha_beta( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo ) :
        Form( a_node, a_setupInfo, FormType::generic, nullptr ),
        m_function3d( nullptr ) { // data3dParse( a_construction, a_node.first_child( ), a_setupInfo, nullptr ) )

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
 * @param a_node            [in]    The **HAPI::Node** to be parsed.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 * @param a_pops            [in]    A PoPI::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs    [in]    The *internal* PoPI::Database instance used to get particle indices and possibly other particle information.
 *                                  This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

IncoherentInelastic::IncoherentInelastic( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo,
		PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs, Suite *a_parent ) :
        Base( a_node, a_setupInfo, FormType::incoherentInelastic, a_parent ),
        m_options( a_construction, a_node.child( GIDI_optionsChars ), a_setupInfo ),
        m_scatteringAtoms( a_construction, GIDI_scatteringAtomsChars, a_node, a_setupInfo, a_pops, a_internalPoPs, parseScatteringAtom, nullptr ),
        m_S_alpha_beta( a_construction, a_node.child( GIDI_S_alpha_betaChars ), a_setupInfo ) {

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

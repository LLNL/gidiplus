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

namespace Distributions {

/*! \class Distribution
 * This is the base class inherited by all distribution form classes.
 */

/* *********************************************************************************************************//**
 *
 * @param a_moniker         [in]    The **GNDS** moniker for the distribution.
 * @param a_type            [in]    The FormType for the distribution form.
 * @param a_label           [in]    The label for the distribution.
 * @param a_productFrame    [in]    The frame the product data are in.
 ***********************************************************************************************************/

Distribution::Distribution( std::string const &a_moniker, FormType a_type, std::string const &a_label, Frame a_productFrame ) :
        Form( a_moniker, a_type, a_label ),
        m_productFrame( a_productFrame ) {

}

/* *********************************************************************************************************//**
 *
 * @param a_node            [in]    The **pugi::xml_node** to be parsed and used to construct the distribution.
 * @param a_type            [in]    The FormType for the distribution form.
 * @param a_parent          [in]    The **m_distribution** member of GIDI::Product this distribution form belongs to.
 ***********************************************************************************************************/

Distribution::Distribution( pugi::xml_node const &a_node, FormType a_type, Suite *a_parent ) :
        Form( a_node, a_type, a_parent ),
        m_productFrame( parseFrame( a_node, "productFrame" ) ) {

}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void Distribution::toXMLNodeStarter( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string attributes;

    attributes += a_writeInfo.addAttribute( "label", label( ) );
    attributes += a_writeInfo.addAttribute( "productFrame", frameToString( m_productFrame ) );
    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );
}

/*! \class MultiGroup3d
 * Class for the GNDS <**multiGroup3d**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction        [in]    Used to pass user options to the constructor.
 * @param a_node                [in]    The **pugi::xml_node** to be parsed and used to construct the MultiGroup3d.
 * @param a_parent              [in]    The **m_distribution** member of GIDI::Product this distribution form belongs to.
 ***********************************************************************************************************/

MultiGroup3d::MultiGroup3d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
    Distribution( a_node, FormType::multiGroup3d, a_parent ),
    m_gridded3d( a_construction, a_node.child( "gridded3d" ) ) {

}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void MultiGroup3d::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );

    toXMLNodeStarter( a_writeInfo, a_indent );
    m_gridded3d.toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class AngularTwoBody
 * Class for the GNDS <**angularTwoBody**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_label           [in]    The label for *this* form.
 * @param a_productFrame    [in]    The frame the product data as in.
 * @param a_angular         [in]    The 2-d functional angular representation.
 ***********************************************************************************************************/

AngularTwoBody::AngularTwoBody( std::string const &a_label, Frame a_productFrame, Functions::Function2dForm *a_angular ) :
        Distribution( angularTwoBodyMoniker, FormType::angularTwoBody, a_label, a_productFrame ),
        m_angular( a_angular ) {

    if( a_angular != NULL ) a_angular->setAncestor( this );
}

/* *********************************************************************************************************//**
 *
 * @param a_construction        [in]    Used to pass user options to the constructor.
 * @param a_node                [in]    The **pugi::xml_node** to be parsed and used to construct the AngularTwoBody.
 * @param a_parent              [in]    The **m_distribution** member of GIDI::Product this distribution form belongs to.
 ***********************************************************************************************************/

AngularTwoBody::AngularTwoBody( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Distribution( a_node, FormType::angularTwoBody, a_parent ),
        m_angular( data2dParse( a_construction, a_node.first_child( ), NULL ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

AngularTwoBody::~AngularTwoBody( ) {

    delete m_angular;
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/
 
void AngularTwoBody::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {
    
    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );

    toXMLNodeStarter( a_writeInfo, a_indent );
    if( m_angular != NULL ) m_angular->toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class KalbachMann
 * Class for the GNDS <**KalbachMann**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction        [in]    Used to pass user options to the constructor.
 * @param a_node                [in]    The **pugi::xml_node** to be parsed and used to construct the KalbachMann.
 * @param a_parent              [in]    The **m_distribution** member of GIDI::Product this distribution form belongs to.
 ***********************************************************************************************************/

KalbachMann::KalbachMann( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Distribution( a_node, FormType::KalbachMann, a_parent ),
        m_f( data2dParse( a_construction, a_node.child( "f" ).first_child( ), NULL ) ),
        m_r( data2dParse( a_construction, a_node.child( "r" ).first_child( ), NULL ) ),
        m_a( NULL ) {

    pugi::xml_node const &aNode = a_node.child( "a" );
    if( aNode.type( ) != pugi::node_null ) m_a = data2dParse( a_construction, aNode.first_child( ), NULL );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

KalbachMann::~KalbachMann( ) {

    delete m_f;
    delete m_r;
    delete m_a;
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/
 
void KalbachMann::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {
    
    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );

    toXMLNodeStarter( a_writeInfo, a_indent );
    m_f->toXMLList( a_writeInfo, indent2 );
    m_r->toXMLList( a_writeInfo, indent2 );
    if( m_a != NULL ) m_a->toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class EnergyAngular
 * Class for the GNDS <**energyAngular**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction        [in]    Used to pass user options to the constructor.
 * @param a_node                [in]    The **pugi::xml_node** to be parsed and used to construct the EnergyAngular.
 * @param a_parent              [in]    The **m_distribution** member of GIDI::Product this distribution form belongs to.
 ***********************************************************************************************************/

EnergyAngular::EnergyAngular( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Distribution( a_node, FormType::energyAngular, a_parent ),
        m_energyAngular( data3dParse( a_construction, a_node.first_child( ), NULL ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

EnergyAngular::~EnergyAngular( ) {

    delete m_energyAngular;
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/
 
void EnergyAngular::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {
    
    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );

    toXMLNodeStarter( a_writeInfo, a_indent );
    m_energyAngular->toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class EnergyAngularMC
 * Class for the GNDS <**energyAngularMC**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction        [in]    Used to pass user options to the constructor.
 * @param a_node                [in]    The **pugi::xml_node** to be parsed and used to construct the EnergyAngularMC.
 * @param a_parent              [in]    The **m_distribution** member of GIDI::Product this distribution form belongs to.
 ***********************************************************************************************************/

EnergyAngularMC::EnergyAngularMC( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Distribution( a_node, FormType::energyAngularMC, a_parent ),
        m_energy( data2dParse( a_construction, a_node.child( "energy" ).first_child( ), NULL ) ),
        m_energyAngular( data3dParse( a_construction, a_node.child( "energyAngular" ).first_child( ), NULL ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

EnergyAngularMC::~EnergyAngularMC( ) {

    delete m_energy;
    delete m_energyAngular;
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/
 
void EnergyAngularMC::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {
    
    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string indent3 = a_writeInfo.incrementalIndent( indent2 );

    toXMLNodeStarter( a_writeInfo, a_indent );

    a_writeInfo.addNodeStarter( indent2, "energy" );
    m_energy->toXMLList( a_writeInfo, indent3 );
    a_writeInfo.addNodeEnder( "energy" );

    a_writeInfo.addNodeStarter( indent2, "energyAngular" );
    m_energyAngular->toXMLList( a_writeInfo, indent3 );
    a_writeInfo.addNodeEnder( "energyAngular" );

    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class AngularEnergy
 * Class for the GNDS <**angularEnergy**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction        [in]    Used to pass user options to the constructor.
 * @param a_node                [in]    The **pugi::xml_node** to be parsed and used to construct the AngularEnergy.
 * @param a_parent              [in]    The **m_distribution** member of GIDI::Product this distribution form belongs to.
 ***********************************************************************************************************/

AngularEnergy::AngularEnergy( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Distribution( a_node, FormType::angularEnergy, a_parent ),
        m_angularEnergy( data3dParse( a_construction, a_node.first_child( ), NULL ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

AngularEnergy::~AngularEnergy( ) {

    delete m_angularEnergy;
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/
 
void AngularEnergy::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {
    
    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );

    toXMLNodeStarter( a_writeInfo, a_indent );
    m_angularEnergy->toXMLList( a_writeInfo, indent2 );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class AngularEnergyMC
 * Class for the GNDS <**angularEnergyMC**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction        [in]    Used to pass user options to the constructor.
 * @param a_node                [in]    The **pugi::xml_node** to be parsed and used to construct the AngularEnergyMC.
 * @param a_parent              [in]    The **m_distribution** member of GIDI::Product this distribution form belongs to.
 ***********************************************************************************************************/

AngularEnergyMC::AngularEnergyMC( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Distribution( a_node, FormType::angularEnergyMC, a_parent ),
        m_angular( data2dParse( a_construction, a_node.child( "angular" ).first_child( ), NULL ) ),
        m_angularEnergy( data3dParse( a_construction, a_node.child( "angularEnergy" ).first_child( ), NULL ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

AngularEnergyMC::~AngularEnergyMC( ) {

    delete m_angular;
    delete m_angularEnergy;
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/
 
void AngularEnergyMC::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {
    
    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string indent3 = a_writeInfo.incrementalIndent( indent2 );

    toXMLNodeStarter( a_writeInfo, a_indent );
    a_writeInfo.addNodeStarter( indent2, angularMoniker );
    m_angular->toXMLList( a_writeInfo, indent3 );
    a_writeInfo.addNodeEnder( angularMoniker );
    a_writeInfo.addNodeStarter( indent2, angularEnergyMoniker );
    m_angularEnergy->toXMLList( a_writeInfo, indent3 );
    a_writeInfo.addNodeEnder( angularEnergyMoniker );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class Uncorrelated
 * Class for the GNDS <**uncorrelated**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction        [in]    Used to pass user options to the constructor.
 * @param a_node                [in]    The **pugi::xml_node** to be parsed and used to construct the Uncorrelated.
 * @param a_parent              [in]    The **m_distribution** member of GIDI::Product this distribution form belongs to.
 ***********************************************************************************************************/

Uncorrelated::Uncorrelated( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Distribution( a_node, FormType::uncorrelated, a_parent ),
        m_angular( data2dParse( a_construction, a_node.child( "angular" ).first_child( ), NULL ) ),
        m_energy( data2dParse( a_construction, a_node.child( "energy" ).first_child( ), NULL ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Uncorrelated::~Uncorrelated( ) {

    delete m_angular;
    delete m_energy;
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/
 
void Uncorrelated::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string indent3 = a_writeInfo.incrementalIndent( indent2 );

    toXMLNodeStarter( a_writeInfo, a_indent );

    a_writeInfo.addNodeStarter( indent2, "angular", "" );
    m_angular->toXMLList( a_writeInfo, indent3 );
    a_writeInfo.addNodeEnder( "angular" );

    a_writeInfo.addNodeStarter( indent2, "energy", "" );
    m_energy->toXMLList( a_writeInfo, indent3 );
    a_writeInfo.addNodeEnder( "energy" );

    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class LLNLAngularEnergy
 * Class for the GNDS <**LLNLAngularEnergy**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction        [in]    Used to pass user options to the constructor.
 * @param a_node                [in]    The **pugi::xml_node** to be parsed and used to construct the LLNLAngularEnergy.
 * @param a_parent              [in]    The **m_distribution** member of GIDI::Product this distribution form belongs to.
 ***********************************************************************************************************/

LLNLAngularEnergy::LLNLAngularEnergy( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Distribution( a_node, FormType::LLNL_angularEnergy, a_parent ),
        m_angular( data2dParse( a_construction, a_node.child( "LLNLAngularOfAngularEnergy" ).first_child( ), NULL ) ),
        m_angularEnergy( data3dParse( a_construction, a_node.child( "LLNLAngularEnergyOfAngularEnergy" ).first_child( ), NULL ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

LLNLAngularEnergy::~LLNLAngularEnergy( ) {

    delete m_angular;
    delete m_angularEnergy;
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/
 
void LLNLAngularEnergy::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {
    
    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string indent3 = a_writeInfo.incrementalIndent( indent2 );

    toXMLNodeStarter( a_writeInfo, a_indent );
    a_writeInfo.addNodeStarter( indent2, LLNLAngularOfAngularEnergyMoniker );
    m_angular->toXMLList( a_writeInfo, indent3 );
    a_writeInfo.addNodeEnder( LLNLAngularOfAngularEnergyMoniker );
    a_writeInfo.addNodeStarter( indent2, LLNLAngularEnergyOfAngularEnergyMoniker );
    m_angularEnergy->toXMLList( a_writeInfo, indent3 );
    a_writeInfo.addNodeEnder( LLNLAngularEnergyOfAngularEnergyMoniker );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class CoherentPhotoAtomicScattering
 * Class for the GNDS <**coherentPhotoAtomicScattering**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction        [in]    Used to pass user options to the constructor.
 * @param a_node                [in]    The **pugi::xml_node** to be parsed and used to construct the CoherentPhotoAtomicScattering.
 * @param a_parent              [in]    The **m_distribution** member of GIDI::Product this distribution form belongs to.
 ***********************************************************************************************************/

CoherentPhotoAtomicScattering::CoherentPhotoAtomicScattering( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Distribution( a_node, FormType::coherentPhotonScattering, a_parent ),
        m_href( a_node.attribute( hrefAttribute ).value( ) ) {

}

/*! \class IncoherentPhotoAtomicScattering
 * Class for the GNDS <**incoherentPhotoAtomicScattering**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction        [in]    Used to pass user options to the constructor.
 * @param a_node                [in]    The **pugi::xml_node** to be parsed and used to construct the IncoherentPhotoAtomicScattering.
 * @param a_parent              [in]    The **m_distribution** member of GIDI::Product this distribution form belongs to.
 ***********************************************************************************************************/

IncoherentPhotoAtomicScattering::IncoherentPhotoAtomicScattering( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Distribution( a_node, FormType::incoherentPhotonScattering, a_parent ),
        m_href( a_node.attribute( hrefAttribute ).value( ) ) {

}

/*! \class ThermalNeutronScatteringLaw
 * Class for the GNDS <**incoherentPhotoAtomicScattering**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction        [in]    Used to pass user options to the constructor.
 * @param a_node                [in]    The **pugi::xml_node** to be parsed and used to construct the ThermalNeutronScatteringLaw.
 * @param a_parent              [in]    The **m_distribution** member of GIDI::Product this distribution form belongs to.
 ***********************************************************************************************************/

ThermalNeutronScatteringLaw::ThermalNeutronScatteringLaw( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Distribution( a_node, FormType::thermalNeutronScatteringLaw, a_parent ),
        m_href( a_node.attribute( hrefAttribute ).value( ) ) {

}

/*! \class Branching3d
 * Class for the GNDS <**branching3d**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction        [in]    Used to pass user options to the constructor.
 * @param a_node                [in]    The **pugi::xml_node** to be parsed and used to construct the Branching3d.
 * @param a_parent              [in]    The **m_distribution** member of GIDI::Product this distribution form belongs to.
 ***********************************************************************************************************/

Branching3d::Branching3d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Distribution( a_node, FormType::branching3d, a_parent ) {

}

/*! \class Reference3d
 * Class for the GNDS <**reference**> distribution node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction        [in]    Used to pass user options to the constructor.
 * @param a_node                [in]    The **pugi::xml_node** to be parsed and used to construct the Reference3d.
 * @param a_parent              [in]    The **m_distribution** member of GIDI::Product this distribution form belongs to.
 ***********************************************************************************************************/

Reference3d::Reference3d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Distribution( a_node, FormType::reference3d, a_parent ),
        m_href( a_node.attribute( hrefAttribute ).value( ) ) {

}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void Reference3d::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string attributes;

    attributes += a_writeInfo.addAttribute( "label", label( ) );
    attributes += a_writeInfo.addAttribute( "href", href( ) );
    a_writeInfo.addNodeStarterEnder( a_indent, moniker( ), attributes );
}

/*! \class Unspecified
 * Class for the GNDS <**unspecified**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction        [in]    Used to pass user options to the constructor.
 * @param a_node                [in]    The **pugi::xml_node** to be parsed and used to construct the Unspecified.
 * @param a_parent              [in]    The **m_distribution** member of GIDI::Product this distribution form belongs to.
 ***********************************************************************************************************/

Unspecified::Unspecified( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Distribution( a_node, FormType::unspecified, a_parent ) {

}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/
 
void Unspecified::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {
    
    toXMLNodeStarter( a_writeInfo, a_indent );
    a_writeInfo.addNodeEnder( moniker( ) );
}

}                     // End of namespace Distributions.

}                     // End of namespace GIDI.

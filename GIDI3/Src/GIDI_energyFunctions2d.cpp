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

/*! \class GeneralEvaporation2d
 * Class for the GNDS <**generalEvaporation**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction    [in]    Used to pass user options for parsing.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed to construct a GeneralEvaporation2d instance.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

GeneralEvaporation2d::GeneralEvaporation2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Function2dForm( a_construction, a_node, f_generalEvaporation2d, a_parent ),
        m_U( a_node.child( "U" ) ),
        m_theta( data1dParse( a_construction, a_node.child( "theta" ).first_child( ), NULL ) ),
        m_g( data1dParse( a_construction, a_node.child( "g" ).first_child( ), NULL ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

GeneralEvaporation2d::~GeneralEvaporation2d( ) {

    delete m_theta;
    delete m_g;
}

/* *********************************************************************************************************//**
 * Returns the domain minimum for the instance.
 *
 * @return              The domain minimum for the instance.
 ***********************************************************************************************************/

double GeneralEvaporation2d::domainMin( ) const {

    return( m_theta->domainMin( ) );
}

/* *********************************************************************************************************//**
 * Returns the domain maximum for the instance.
 *
 * @return              The domain maximum for the instance.
 ***********************************************************************************************************/

double GeneralEvaporation2d::domainMax( ) const {

    return( m_theta->domainMax( ) );
}

/* *********************************************************************************************************//**
 * Returns the value of the function evaluated at the specified projectile's energy and product's energy.
 * Currently not implemented.
 *
 * @param a_x2              [in]    The projectile's energy.
 * @param a_x1              [in]    The product's energy.
 * @return                          The value of the function evaluated at *a_x2*, and *a_x1*.
 ***********************************************************************************************************/
 
double GeneralEvaporation2d::evaluate( double a_x2, double a_x1 ) const {

    throw std::runtime_error( "GeneralEvaporation2d::evaluate: not implemented." );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 * @param       a_embedded          [in]        If *true*, *this* function is embedded in a higher dimensional function.
 * @param       a_inRegions         [in]        This is not used in this method.
 ***********************************************************************************************************/

void GeneralEvaporation2d::toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string attributes;

    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );

    m_U.toXMLList( a_writeInfo, indent2 );
    energy2dToXMLList( a_writeInfo, "theta", indent2, m_theta );
    energy2dToXMLList( a_writeInfo, "g", indent2, m_g );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class SimpleMaxwellianFission2d
 * Class for the GNDS <**simpleMaxwellianFission**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction    [in]    Used to pass user options for parsing.
 * @param a_node            [in]    The **pugi::xml_node** to be parsed to construct a SimpleMaxwellianFission2d instance.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/
 
SimpleMaxwellianFission2d::SimpleMaxwellianFission2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Function2dForm( a_construction, a_node, f_simpleMaxwellianFission2d, a_parent ),
        m_U( a_node.child( "U" ) ),
        m_theta( data1dParse( a_construction, a_node.child( "theta" ).first_child( ), NULL ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

SimpleMaxwellianFission2d::~SimpleMaxwellianFission2d( ) {

    delete m_theta;
}

/* *********************************************************************************************************//**
 * Returns the domain minimum for the instance.
 *
 * @return              The domain minimum for the instance.
 ***********************************************************************************************************/

double SimpleMaxwellianFission2d::domainMin( ) const {

    return( m_theta->domainMin( ) );
}

/* *********************************************************************************************************//**
 * Returns the domain maximum for the instance.
 *
 * @return              The domain maximum for the instance.
 ***********************************************************************************************************/
 
double SimpleMaxwellianFission2d::domainMax( ) const {

    return( m_theta->domainMax( ) );
}

/* *********************************************************************************************************//**
 * Returns the value of the function evaluated at the specified projectile's energy and product's energy.
 * Currently not implemented.
 *
 * @param a_x2              [in]    The projectile's energy.
 * @param a_x1              [in]    The product's energy.
 * @return                          The value of the function evaluated at *a_x2*, and *a_x1*.
 ***********************************************************************************************************/

double SimpleMaxwellianFission2d::evaluate( double a_x2, double a_x1 ) const {

    throw std::runtime_error( "SimpleMaxwellianFission2d::evaluate: not implemented." );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 * @param       a_embedded          [in]        If *true*, *this* function is embedded in a higher dimensional function.
 * @param       a_inRegions         [in]        This is not used in this method.
 ***********************************************************************************************************/

void SimpleMaxwellianFission2d::toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string attributes;

    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );

    m_U.toXMLList( a_writeInfo, indent2 );
    energy2dToXMLList( a_writeInfo, "theta", indent2, m_theta );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class Evaporation2d
 * Class for the GNDS <**evaporation**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction        [in]    Used to pass user options for parsing.
 * @param a_node                [in]    The **pugi::xml_node** to be parsed to construct a Evaporation2d instance.
 * @param a_parent              [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/
 
Evaporation2d::Evaporation2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Function2dForm( a_construction, a_node, f_evaporation2d, a_parent ),
        m_U( a_node.child( "U" ) ),
        m_theta( data1dParse( a_construction, a_node.child( "theta" ).first_child( ), NULL ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Evaporation2d::~Evaporation2d( ) {

    delete m_theta;
}

/* *********************************************************************************************************//**
 * Returns the domain minimum for the instance.
 *
 * @return              The domain minimum for the instance.
 ***********************************************************************************************************/

double Evaporation2d::domainMin( ) const {

    return( m_theta->domainMin( ) );
}

/* *********************************************************************************************************//**
 * Returns the domain maximum for the instance.
 *
 * @return              The domain maximum for the instance.
 ***********************************************************************************************************/

double Evaporation2d::domainMax( ) const {

    return( m_theta->domainMax( ) );
}

/* *********************************************************************************************************//**
 * Returns the value of the function evaluated at the specified projectile's energy and product's energy.
 * Currently not implemented.
 *
 * @param a_x2              [in]    The projectile's energy.
 * @param a_x1              [in]    The product's energy.
 * @return                          The value of the function evaluated at *a_x2*, and *a_x1*.
 ***********************************************************************************************************/

double Evaporation2d::evaluate( double a_x2, double a_x1 ) const {

    throw std::runtime_error( "Evaporation2d::evaluate: not implemented." );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 * @param       a_embedded          [in]        If *true*, *this* function is embedded in a higher dimensional function.
 * @param       a_inRegions         [in]        This is not used in this method.
 ***********************************************************************************************************/

void Evaporation2d::toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string attributes;

    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );

    m_U.toXMLList( a_writeInfo, indent2 );
    energy2dToXMLList( a_writeInfo, "theta", indent2, m_theta );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class Watt2d
 * Class for the GNDS <**Watt**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction        [in]    Used to pass user options for parsing.
 * @param a_node                [in]    The **pugi::xml_node** to be parsed to construct a Watt2d instance.
 * @param a_parent              [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/
 
Watt2d::Watt2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Function2dForm( a_construction, a_node, f_Watt2d, a_parent ),
        m_U( a_node.child( "U" ) ),
        m_a( data1dParse( a_construction, a_node.child( "a" ).first_child( ), NULL ) ),
        m_b( data1dParse( a_construction, a_node.child( "b" ).first_child( ), NULL ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Watt2d::~Watt2d( ) {

    delete m_a;
    delete m_b;
}

/* *********************************************************************************************************//**
 * Returns the domain minimum for the instance.
 *
 * @return              The domain minimum for the instance.
 ***********************************************************************************************************/

double Watt2d::domainMin( ) const {

    return( m_a->domainMin( ) );
}

/* *********************************************************************************************************//**
 * Returns the domain maximum for the instance.
*
 * @return              The domain maximum for the instance.
 ***********************************************************************************************************/

double Watt2d::domainMax( ) const {

    return( m_a->domainMax( ) );
}

/* *********************************************************************************************************//**
 * Returns the value of the function evaluated at the specified projectile's energy and product's energy.
 * Currently not implemented.
 *
 * @param a_x2              [in]    The projectile's energy.
 * @param a_x1              [in]    The product's energy.
 * @return                          The value of the function evaluated at *a_x2*, and *a_x1*.
 ***********************************************************************************************************/

double Watt2d::evaluate( double a_x2, double a_x1 ) const {

    throw std::runtime_error( "Watt2d::evaluate: not implemented." );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 * @param       a_embedded          [in]        If *true*, *this* function is embedded in a higher dimensional function.
 * @param       a_inRegions         [in]        This is not used in this method.
 ***********************************************************************************************************/

void Watt2d::toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string attributes;

    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );

    m_U.toXMLList( a_writeInfo, indent2 );
    energy2dToXMLList( a_writeInfo, "a", indent2, m_a );
    energy2dToXMLList( a_writeInfo, "b", indent2, m_b );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class MadlandNix2d
 * Class for the GNDS <**MadlandNix**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction        [in]    Used to pass user options for parsing.
 * @param a_node                [in]    The **pugi::xml_node** to be parsed to construct a MadlandNix2d instance.
 * @param a_parent              [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

MadlandNix2d::MadlandNix2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Function2dForm( a_construction, a_node, f_MadlandNix2d, a_parent ),
        m_EFL( a_node.child( "EFL" ) ),
        m_EFH( a_node.child( "EFH" ) ),
        m_T_M( data1dParse( a_construction, a_node.child( "T_M" ).first_child( ), NULL ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

MadlandNix2d::~MadlandNix2d( ) {

    delete m_T_M;
}

/* *********************************************************************************************************//**
 * Returns the domain minimum for the instance.
 *
 * @return              The domain minimum for the instance.
 ***********************************************************************************************************/

double MadlandNix2d::domainMin( ) const {

    return( m_T_M->domainMin( ) );
}

/* *********************************************************************************************************//**
 * Returns the domain maximum for the instance.
 *
 * @return              The domain maximum for the instance.
 ***********************************************************************************************************/

double MadlandNix2d::domainMax( ) const {

    return( m_T_M->domainMax( ) );
}

/* *********************************************************************************************************//**
 * Returns the value of the function evaluated at the specified projectile's energy and product's energy.
 * Currently not implemented.
 *
 * @param a_x2              [in]    The projectile's energy.
 * @param a_x1              [in]    The product's energy.
 * @return                          The value of the function evaluated at *a_x2*, and *a_x1*.
 ***********************************************************************************************************/

double MadlandNix2d::evaluate( double a_x2, double a_x1 ) const {

    throw std::runtime_error( "MadlandNix2d::evaluate: not implemented." );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 * @param       a_embedded          [in]        If *true*, *this* function is embedded in a higher dimensional function.
 * @param       a_inRegions         [in]        This is not used in this method.
 ***********************************************************************************************************/

void MadlandNix2d::toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string attributes;

    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );

    m_EFL.toXMLList( a_writeInfo, indent2 );
    m_EFH.toXMLList( a_writeInfo, indent2 );
    energy2dToXMLList( a_writeInfo, "T_M", indent2, m_T_M );
    a_writeInfo.addNodeEnder( moniker( ) );
}

/*! \class Weighted_function2d
 * Class for the GNDS <**weighted**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction        [in]    Used to pass user options for parsing.
 * @param a_node                [in]    The **pugi::xml_node** to be parsed to construct a Weighted_function2d instance.
 * @param a_parent              [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

Weighted_function2d::Weighted_function2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Function2dForm( a_construction, a_node, f_weighted_function2d, a_parent ) {

    pugi::xml_node child = a_node.first_child( );
    m_weight = data1dParse( a_construction, child, NULL );

    child = child.next_sibling( );
    m_energy = data2dParse( a_construction, child, NULL );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Weighted_function2d::~Weighted_function2d( ) {

    delete m_weight;
    delete m_energy;
}

/* *********************************************************************************************************//**
 * Returns the domain minimum for the instance.
 *
 * @return              The domain minimum for the instance.
 ***********************************************************************************************************/

double Weighted_function2d::domainMin( ) const {

    return( m_weight->domainMin( ) );
}

/* *********************************************************************************************************//**
 * Returns the domain maximum for the instance.
 *
 * @return              The domain maximum for the instance.
 ***********************************************************************************************************/
 
double Weighted_function2d::domainMax( ) const {

    return( m_weight->domainMax( ) );
}

/* *********************************************************************************************************//**
 * Returns the value of the function evaluated at the specified projectile's energy and product's energy.
 * Currently not implemented.
 *
 * @param a_x2              [in]    The projectile's energy.
 * @param a_x1              [in]    The product's energy.
 * @return                          The value of the function evaluated at *a_x2*, and *a_x1*.
 ***********************************************************************************************************/

double Weighted_function2d::evaluate( double a_x2, double a_x1 ) const {

    throw std::runtime_error( "Weighted_function2d::evaluate: not implemented." );
}

/*! \class WeightedFunctionals2d
 * Class for the GNDS <**weightedFunctionals**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction        [in]    Used to pass user options for parsing.
 * @param a_node                [in]    The **pugi::xml_node** to be parsed to construct a WeightedFunctionals2d instance.
 * @param a_parent              [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

WeightedFunctionals2d::WeightedFunctionals2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Function2dForm( a_construction, a_node, f_weightedFunctionals2d, a_parent ) {

    for( pugi::xml_node child = a_node.first_child( ); child; child = child.next_sibling( ) ) {
        std::string name( child.name( ) );

        if( name != "weighted" ) throw std::runtime_error( "WeightedFunctionals2d::WeightedFunctionals2d: bad child." );
        m_weighted_function2d.push_back( new Weighted_function2d( a_construction, child, NULL ) );
    }

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

WeightedFunctionals2d::~WeightedFunctionals2d( ) {

    for( std::vector<Weighted_function2d *>::iterator iter = m_weighted_function2d.begin( ); iter < m_weighted_function2d.end( ); ++iter ) delete *iter;
}

/* *********************************************************************************************************//**
 * Returns the domain minimum for the instance.
 *
 * @return              The domain minimum for the instance.
 ***********************************************************************************************************/
 
double WeightedFunctionals2d::domainMin( ) const {

    double domainMin1 = m_weighted_function2d[0]->domainMin( );

    for( std::vector<Weighted_function2d *>::const_iterator iter = m_weighted_function2d.begin( ); iter < m_weighted_function2d.end( ); ++iter ) {
        double domainMin2 = (*iter)->domainMin( );

        if( domainMin2 < domainMin1 ) domainMin1 = domainMin2;
    }
    return( domainMin1 );
}

/* *********************************************************************************************************//**
 * Returns the domain maximum for the instance.
 *
 * @return              The domain maximum for the instance.
 ***********************************************************************************************************/

double WeightedFunctionals2d::domainMax( ) const {

    double domainMax1 = m_weighted_function2d[0]->domainMax( );

    for( std::vector<Weighted_function2d *>::const_iterator iter = m_weighted_function2d.begin( ); iter < m_weighted_function2d.end( ); ++iter ) {
        double domainMax2 = (*iter)->domainMax( );

        if( domainMax2 < domainMax1 ) domainMax1 = domainMax2;
    }
    return( domainMax1 );
}

/* *********************************************************************************************************//**
 * Returns the value of the function evaluated at the specified projectile's energy and product's energy.
 * Currently not implemented.
 *
 * @param a_x2              [in]    The projectile's energy.
 * @param a_x1              [in]    The product's energy.
 * @return                          The value of the function evaluated at *a_x2*, and *a_x1*.
 ***********************************************************************************************************/

double WeightedFunctionals2d::evaluate( double a_x2, double a_x1 ) const {

    throw std::runtime_error( "WeightedFunctionals2d::evaluate: not implemented." );
}

/*! \class NBodyPhaseSpace2d
 * Class for the GNDS <**NBodyPhaseSpace**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction        [in]    Used to pass user options for parsing.
 * @param a_node                [in]    The **pugi::xml_node** to be parsed to construct a NBodyPhaseSpace2d instance.
 * @param a_parent              [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

NBodyPhaseSpace2d::NBodyPhaseSpace2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent ) :
        Function2dForm( a_construction, a_node, f_NBodyPhaseSpace2d, a_parent ),
        m_numberOfProducts( a_node.attribute( "numberOfProducts" ).as_int( ) ),
        m_mass( a_node.child( "mass" ) ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

NBodyPhaseSpace2d::~NBodyPhaseSpace2d( ) {

}

/* *********************************************************************************************************//**
 * Returns the domain minimum for the instance.
 *
 * @return              The domain minimum for the instance.
 ***********************************************************************************************************/

double NBodyPhaseSpace2d::domainMin( ) const {

    return( 0. );           // FIXME
}

/* *********************************************************************************************************//**
 * Returns the domain maximum for the instance.
 *
 * @return              The domain maximum for the instance.
 ***********************************************************************************************************/

double NBodyPhaseSpace2d::domainMax( ) const {

    return( 1. );           // FIXME
}

/* *********************************************************************************************************//**
 * Returns the value of the function evaluated at the specified projectile's energy and product's energy.
 * Currently not implemented.
 *
 * @param a_x2              [in]    The projectile's energy.
 * @param a_x1              [in]    The product's energy.
 * @return                          The value of the function evaluated at *a_x2*, and *a_x1*.
 ***********************************************************************************************************/

double NBodyPhaseSpace2d::evaluate( double a_x2, double a_x1 ) const {

    throw std::runtime_error( "NBodyPhaseSpace2d::evaluate: not implemented." );
}

}
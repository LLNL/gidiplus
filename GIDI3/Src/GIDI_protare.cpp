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

static bool sortTemperatures( Styles::TemperatureInfo const &lhs, Styles::TemperatureInfo const &rhs );

/*! \class Protare
 * Base class for the protare sub-classes.
 */

/* *********************************************************************************************************//**
 * Base Protare constructor.
 ***********************************************************************************************************/

Protare::Protare( ) :
        Ancestry( "" ),
        m_projectile( "", "", -1.0 ),
        m_target( "", "", -1.0 ) {
    
}

/* *********************************************************************************************************//**
 ******************************************************************/

Protare::~Protare( ) {

}

/* *********************************************************************************************************//**
 * Called by the constructs. This method does most of the parsing.
 *
 * @param a_node                        [in]    The protare (i.e., reactionSuite) node to be parsed and used to construct a Protare.
 * @param a_pops                        [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs                [in]    The internal PoPs::Database instance used to get particle indices and possibly other particle information.
 * @param a_targetRequiredInGlobalPoPs  [in]    If *true*, the target is required to be in **a_pops**.
 * @param a_requiredInPoPs              [in]    If *true*, no particle is required to be in **a_pops**.
 ***********************************************************************************************************/

void Protare::initialize( pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs, bool a_targetRequiredInGlobalPoPs,
                bool a_requiredInPoPs ) {

    moniker( a_node.name( ) );    

    std::string projectileID = a_node.attribute( "projectile" ).value( );
    m_projectile = ParticleInfo( projectileID, a_pops, a_internalPoPs, a_requiredInPoPs );

    std::string targetID = a_node.attribute( "target" ).value( );
    m_target = ParticleInfo( targetID, a_pops, a_internalPoPs, a_targetRequiredInGlobalPoPs && a_requiredInPoPs );
}

/*! \class ProtareSingleton
 * Class to store a GNDS <**reactionSuite**> node.
 */

/* *********************************************************************************************************//**
 * Parses a GNDS file to construct the Protare instance. Calls the initialize method which does most of the work.
 *
 * @param a_pops            [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_projectileID    [in]    The PoPs id of the projectile.
 * @param a_targetID        [in]    The PoPs id of the target.
 * @param a_evaluation      [in]    The evaluation string.
 ***********************************************************************************************************/

ProtareSingleton::ProtareSingleton( PoPs::Database const &a_pops, std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_evaluation ) :
        m_formatVersion( GIDI_format ),
        m_evaluation( a_evaluation ),
        m_projectileFrame( GIDI::lab ),
        m_thresholdFactor( 0.0 ),
        m_documentations( ),
        m_styles( ),
        m_reactions( reactionsMoniker ),
        m_orphanProducts( orphanProductsMoniker ),
        m_fissionComponents( fissionComponentsMoniker ) {

    moniker( GIDI_topLevelMoniker );
    m_documentations.setAncestor( this );
    m_styles.setAncestor( this );
    m_reactions.setAncestor( this );
    m_orphanProducts.setAncestor( this );
    m_sums.setAncestor( this );
    m_fissionComponents.setAncestor( this );

    projectile( ParticleInfo( a_projectileID, a_pops, a_pops, true ) );
    target( ParticleInfo( a_targetID, a_pops, a_pops, true ) );
}

/* *********************************************************************************************************//**
 * Parses a GNDS file to construct the Protare instance. Calls the initialize method which does most of the work.
 *
 * @param a_construction                [in]    Used to pass user options to the constructor.
 * @param a_fileName                    [in]    File containing a protare (i.e., reactionSuite) node that is parsed and used to construct the Protare.
 * @param a_fileType                    [in]    File type of a_fileType. Currently, only GIDI::XML is supported.
 * @param a_pops                        [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_libraries                   [in]    The list of libraries that were searched to find *this*.
 * @param a_targetRequiredInGlobalPoPs  [in]    If *true*, the target is required to be in **a_pops**.
 * @param a_requiredInPoPs              [in]    If *true*, no particle is required to be in **a_pops**.
 ***********************************************************************************************************/

ProtareSingleton::ProtareSingleton( Construction::Settings const &a_construction, std::string const &a_fileName, fileType a_fileType, 
                PoPs::Database const &a_pops, std::vector<std::string> const &a_libraries, bool a_targetRequiredInGlobalPoPs, bool a_requiredInPoPs ) :
        Protare( ),
        m_libraries( a_libraries ),
        m_fileName( a_fileName ),
        m_realFileName( realPath( a_fileName ) ),
        m_documentations( ),
        m_styles( ),
        m_reactions( reactionsMoniker ),
        m_orphanProducts( orphanProductsMoniker ),
        m_fissionComponents( fissionComponentsMoniker ) {

    pugi::xml_document doc;

    if( a_fileType != XML ) throw std::runtime_error( "Only XML file type supported." );

    pugi::xml_parse_result result = doc.load_file( a_fileName.c_str( ) );
    if( result.status != pugi::status_ok ) throw std::runtime_error( result.description( ) );

    pugi::xml_node protare = doc.first_child( );

    initialize( a_construction, protare, a_pops, a_targetRequiredInGlobalPoPs, a_requiredInPoPs );
}

/* *********************************************************************************************************//**
 * Parses a GNDS pugi::xml_node instance to construct the Protare instance. Calls the ProtareSingleton::initialize method which does most of the work.
 *
 * @param a_construction                [in]    Used to pass user options to the constructor.
 * @param a_node                        [in]    The **pugi::xml_node** to be parsed and used to construct the Protare.
 * @param a_pops                        [in]    A PoPs::Database instance used to get particle indices and possibly other particle information.
 * @param a_libraries                   [in]    The list of libraries that were searched to find *this*.
 * @param a_targetRequiredInGlobalPoPs  [in]    If *true*, the target is required to be in **a_pops**.
 * @param a_requiredInPoPs              [in]    If *true*, no particle is required to be in **a_pops**.
 ***********************************************************************************************************/

ProtareSingleton::ProtareSingleton( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops,
                std::vector<std::string> const &a_libraries, bool a_targetRequiredInGlobalPoPs, bool a_requiredInPoPs ) :
        Protare( ),
        m_libraries( a_libraries ),
        m_documentations( ),
        m_styles( ),
        m_reactions( reactionsMoniker ),
        m_orphanProducts( orphanProductsMoniker ),
        m_fissionComponents( fissionComponentsMoniker ) {

    initialize( a_construction, a_node, a_pops, a_targetRequiredInGlobalPoPs, a_requiredInPoPs );
}

/* *********************************************************************************************************//**
 * Called by the constructs. This method does most of the parsing.
 *
 * @param a_construction                [in]    Used to pass user options to the constructor.
 * @param a_node                        [in]    The protare (i.e., reactionSuite) node to be parsed and used to construct a Protare.
 * @param a_pops                        [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_targetRequiredInGlobalPoPs  [in]    If *true*, the target is required to be in **a_pops**.
 * @param a_requiredInPoPs              [in]    If *true*, no particle is required to be in **a_pops**.
 ***********************************************************************************************************/

void ProtareSingleton::initialize( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, bool a_targetRequiredInGlobalPoPs,
                bool a_requiredInPoPs ) {

    m_documentations.setAncestor( this );
    m_styles.setAncestor( this );
    m_reactions.setAncestor( this );
    m_orphanProducts.setAncestor( this );
    m_sums.setAncestor( this );
    m_fissionComponents.setAncestor( this );

    m_formatVersion = a_node.attribute( "format" ).value( );
    if( m_formatVersion != GIDI_format ) throw std::runtime_error( "unsupport GND format version" );

    pugi::xml_node internalPoPs = a_node.child( "PoPs" );
    m_internalPoPs.addDatabase( internalPoPs, true );
    m_internalPoPs.calculateNuclideGammaBranchStateInfos( m_nuclideGammaBranchStateInfos );

    Protare::initialize( a_node, a_pops, m_internalPoPs, a_targetRequiredInGlobalPoPs, a_requiredInPoPs );

    m_isTNSL_ProtareSingleton = false;
    m_thresholdFactor = 1.0;
    if( a_requiredInPoPs ) {
        PoPs::Base const &target1( a_pops.get<PoPs::Base>( target( ).pid( ) ) );
        if( target1.isUnorthodox( ) ) m_isTNSL_ProtareSingleton = target( ).pid( ).find( "Fission" ) == std::string::npos;  // FIXME, not a good way to determine this.
        m_thresholdFactor = 1.0 + projectile( ).mass( "amu" ) / target( ).mass( "amu" );
    }

    m_evaluation = a_node.attribute( "evaluation" ).value( );

    m_projectileFrame = parseFrame( a_node, "projectileFrame" );

    m_styles.parse( a_construction, a_node.child( stylesMoniker ), a_pops, m_internalPoPs, parseStylesSuite, NULL );
    m_reactions.parse( a_construction, a_node.child( reactionsMoniker ), a_pops, m_internalPoPs, parseReaction, &m_styles );
    m_orphanProducts.parse( a_construction, a_node.child( orphanProductsMoniker ), a_pops, m_internalPoPs, parseOrphanProduct, &m_styles );

    m_sums.parse( a_construction, a_node.child( sumsMoniker ), a_pops, m_internalPoPs );
    m_fissionComponents.parse( a_construction, a_node.child( fissionComponentsMoniker ), a_pops, m_internalPoPs, parseFissionComponent, &m_styles );
}

/* *********************************************************************************************************//**
 ******************************************************************/

ProtareSingleton::~ProtareSingleton( ) {

}

/* *********************************************************************************************************//**
 * Returns the pointer representing the protare (i.e., *this*) if *a_index* is 0 and NULL otherwise.
 *
 * @param a_index               [in]    Must always be 0.
 *
 * @return                              Returns the pointer representing *this*.
 ***********************************************************************************************************/

ProtareSingleton const *ProtareSingleton::protare( std::size_t a_index ) const {

    if( a_index != 0 ) return( NULL );
    return( this );
}

/* *********************************************************************************************************//**
 * Fills in a std::set with a unique list of all product indices produced by reactions and orphanProducts. 
 * If a_transportablesOnly is true, only transportable product incides are return.
 *
 * @param a_ids                 [out]   The unique list of product indices.
 * @param a_particles           [in]    The list of particles to be transported.
 * @param a_transportablesOnly  [in]    If true, only transportable product indices are added in the list.
 ***********************************************************************************************************/

void ProtareSingleton::productIDs( std::set<std::string> &a_ids, Settings::Particles const &a_particles, bool a_transportablesOnly ) const {

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction = m_reactions.get<Reaction>( i1 );

        reaction->productIDs( a_ids, a_particles, a_transportablesOnly );
    }

    for( std::size_t i1 = 0; i1 < m_orphanProducts.size( ); ++i1 ) {
        Reaction const *reaction = m_orphanProducts.get<Reaction>( i1 );

        reaction->productIDs( a_ids, a_particles, a_transportablesOnly );
    }
}

/* *********************************************************************************************************//**
 * Determines the maximum Legredre order present in the multi-group transfer matrix for a give product for a give label.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_productID       [in]    The id of the requested product.
 * @return                          The maximum Legredre order. If no transfer matrix data are present for the requested product, -1 is returned.
 ***********************************************************************************************************/

int ProtareSingleton::maximumLegendreOrder( Settings::MG const &a_settings, std::string const &a_productID ) const {

    int _maximumLegendreOrder = -1;

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction = m_reactions.get<Reaction>( i1 );
        int r_maximumLegendreOrder = reaction->maximumLegendreOrder( a_settings, a_productID );

        if( r_maximumLegendreOrder > _maximumLegendreOrder ) _maximumLegendreOrder = r_maximumLegendreOrder;
    }

    return( _maximumLegendreOrder );
}

/* *********************************************************************************************************//**
 * Returns a list of all process temperature data. For each temeprature, the labels for its 
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

Styles::TemperatureInfos ProtareSingleton::temperatures( ) const {

    std::size_t size( m_styles.size( ) );
    Styles::TemperatureInfos temperature_infos;

    for( std::size_t i1 = 0; i1 < size; ++i1 ) {
        Styles::Base const *style1 = m_styles.get<Styles::Base>( i1 );

        if( style1->moniker( ) == heatedStyleMoniker ) {
            PhysicalQuantity const &temperature = style1->temperature( );
            std::string heated_cross_section( style1->label( ) );
            std::string gridded_cross_section( "" );
            std::string URR_probability_tables( "" );
            std::string heated_multi_group( "" );
            std::string Sn_elastic_upscatter( "" );

            for( std::size_t i2 = 0; i2 < size; ++i2 ) {
                Styles::Base const *style2 = m_styles.get<Styles::Base>( i2 );

                if( style2->moniker( ) == multiGroupStyleMoniker ) continue;
                if( style2->temperature( ).value( ) != temperature.value( ) ) continue;

                if( style2->moniker( ) == griddedCrossSectionStyleMoniker ) {
                    gridded_cross_section = style2->label( ); }
                else if( style2->moniker( ) == URR_probabilityTablesStyleMoniker ) {
                    URR_probability_tables = style2->label( ); }
                else if( style2->moniker( ) == SnElasticUpScatterStyleMoniker ) {
                    Sn_elastic_upscatter = style2->label( ); }
                else if( style2->moniker( ) == heatedMultiGroupStyleMoniker ) {
                    heated_multi_group = style2->label( );
                }
            }
            temperature_infos.push_back( Styles::TemperatureInfo( temperature, heated_cross_section, gridded_cross_section, URR_probability_tables,
                    heated_multi_group, Sn_elastic_upscatter ) );
        }
    }

    std::sort( temperature_infos.begin( ), temperature_infos.end( ), sortTemperatures );

    return( temperature_infos );
}

/* *********************************************************************************************************//**
 * FOR INTERNAL USE ONLY.
 *
 * Determines if the temperature of lhs is less than that of rhs, or not.
 *
 * @param lhs   [in]
 * @param rhs   [in]
 * @return      true if temperature of lhs < rhs and false otherwise.
 ***********************************************************************************************************/

bool sortTemperatures( Styles::TemperatureInfo const &lhs, Styles::TemperatureInfo const &rhs ) {

    return( lhs.temperature( ).value( ) < rhs.temperature( ).value( ) );
}

/* *********************************************************************************************************//**
 * Returns the requested Styles::MultiGroup from the styles.
 *
 * @param a_label   [in]    Label for the requested Styles::MultiGroup.
 * @return
 ***********************************************************************************************************/

Styles::MultiGroup const *ProtareSingleton::multiGroup( std::string const &a_label ) const {

    Styles::Base const *style1 = m_styles.get<Styles::Base>( a_label );

    if( style1->moniker( ) == SnElasticUpScatterStyleMoniker ) style1 = m_styles.get<Styles::Base>( style1->derivedStyle( ) );
    if( style1->moniker( ) == heatedMultiGroupStyleMoniker ) {
        Styles::HeatedMultiGroup const *heatedMultiGroup = static_cast<Styles::HeatedMultiGroup const *>( style1 );
        style1 = m_styles.get<Styles::Base>( heatedMultiGroup->parameters( ) );
    }
    if( style1->moniker( ) != multiGroupStyleMoniker ) return( NULL );
    return( static_cast<Styles::MultiGroup const *>( style1 ) );
}

/* *********************************************************************************************************//**
 * Returns the multi-group boundaries for the requested label and product.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_productID       [in]    ID for the requested product.
 * @return                          List of multi-group boundaries.
 ***********************************************************************************************************/

std::vector<double> const &ProtareSingleton::groupBoundaries( Settings::MG const &a_settings, std::string const &a_productID ) const {

    Styles::Base const *style1 = m_styles.get<Styles::Base>( a_settings.label( ) );

    if( style1->moniker( ) == SnElasticUpScatterStyleMoniker ) style1 = m_styles.get<Styles::Base>( style1->derivedStyle( ) );

    if( style1->moniker( ) == multiGroupStyleMoniker ) {
        Styles::MultiGroup const &_multiGroupStyle = dynamic_cast<Styles::MultiGroup const &>( *style1 );
        return( _multiGroupStyle.groupBoundaries( a_productID ) ); }
    else if( style1->moniker( ) == heatedMultiGroupStyleMoniker ) {
        Styles::HeatedMultiGroup const &_heatedMultiGroupStyle = dynamic_cast<Styles::HeatedMultiGroup const &>( *style1 );
        return( _heatedMultiGroupStyle.groupBoundaries( a_productID ) );
    }
    throw std::runtime_error( "Style does not have groupBoundaries method." );
}

/* *********************************************************************************************************//**
 * Returns the inverse speeds for the requested label. The label must be for a heated multi-group style.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_particles       [in]    The list of particles to be transported.
 * @return                          List of inverse speeds.
 ***********************************************************************************************************/

Vector ProtareSingleton::multiGroupInverseSpeed( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const {

    Styles::Base const *_style = m_styles.getViaLineage<Styles::Base>( a_settings.label( ) );
    if( _style->moniker( ) == SnElasticUpScatterStyleMoniker ) _style = _style->getDerivedStyle( );
    Styles::HeatedMultiGroup const &_multiGroupStyle = dynamic_cast<Styles::HeatedMultiGroup const &>( *_style );
    return( _multiGroupStyle.inverseSpeed( ) );
}

/* *********************************************************************************************************//**
 * Returns true if at least one reaction contains a fission channel.
 *
 * @return  true if at least one reaction contains a fission channel.
 ***********************************************************************************************************/

bool ProtareSingleton::hasFission( ) const {

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction = m_reactions.get<Reaction>( i1 );

        if( reaction->hasFission( ) ) return( true );
    }
    return( false );
}

/* *********************************************************************************************************//**
 * Used by Ancestry to tranverse GNDS nodes. This method returns a pointer to a derived class' a_item member or NULL if none exists.
 *
 * @param a_item    [in]    The name of the class member whose pointer is to be return.
 * @return                  The pointer to the class member or NULL if class does not have a member named a_item.
 ***********************************************************************************************************/

Ancestry const *ProtareSingleton::findInAncestry3( std::string const &a_item ) const {

    if( a_item == stylesMoniker ) return( &m_styles );
    if( a_item == reactionsMoniker ) return( &m_reactions );
    if( a_item == orphanProductsMoniker ) return( &m_orphanProducts );
    if( a_item == sumsMoniker ) return( &m_sums );
    if( a_item == fissionComponentsMoniker ) return( &m_fissionComponents );

    return( NULL );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total cross section for the requested label. This is summed over all reactions.
 *
 * @param a_settings    [in]    Specifies the requested label.
 * @param a_particles   [in]    The list of particles to be transported.
 * @return                      The requested multi-group cross section as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareSingleton::multiGroupCrossSection( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const {

    Vector vector;

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction = m_reactions.get<Reaction>( i1 );

        vector += reaction->multiGroupCrossSection( a_settings, a_particles );
    }
    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total multiplicity for the requested label for the requested product. This is a cross section weighted multiplicity.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_particles       [in]    The list of particles to be transported.
 * @param a_productID       [in]    Id for the requested product.
 * @return                          The requested multi-group multiplicity as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareSingleton::multiGroupMultiplicity( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const {

    Vector vector( 0 );

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction = m_reactions.get<Reaction>( i1 );

        vector += reaction->multiGroupMultiplicity( a_settings, a_particles, a_productID );
    }

    for( std::size_t i1 = 0; i1 < m_orphanProducts.size( ); ++i1 ) {
        Reaction const *reaction = m_orphanProducts.get<Reaction>( i1 );

        vector += reaction->multiGroupMultiplicity( a_settings, a_particles, a_productID );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total fission neutron multiplicity for the requested label. This is a cross section weighted multiplicity.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_particles       [in]    The list of particles to be transported.
 * @return                          The requested multi-group fission neutron multiplicity as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareSingleton::multiGroupFissionNeutronMultiplicity( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const {

    Vector vector( 0 );

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction = m_reactions.get<Reaction>( i1 );

        if( reaction->hasFission( ) ) vector += reaction->multiGroupMultiplicity( a_settings, a_particles, PoPs::IDs::neutron );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total Q for the requested label. This is a cross section weighted multiplicity
 * summed over all reactions
 *
 * @param a_settings    [in]    Specifies the requested label.
 * @param a_particles   [in]    The list of particles to be transported.
 * @param a_final               If false, only the Q for the primary reactions are return, otherwise, the Q for the final reactions.
 * @return                      The requested multi-group Q as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareSingleton::multiGroupQ( Settings::MG const &a_settings, Settings::Particles const &a_particles, bool a_final ) const {

    Vector vector( 0 );

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction = m_reactions.get<Reaction>( i1 );

        vector += reaction->multiGroupQ( a_settings, a_particles, a_final );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total product matrix for the requested label for the requested product id for the requested Legendre order.
 * If no data are found, an empty GIDI::Matrix is returned.
 *
 * @param a_settings        [in]    Specifies the requested label and if delayed neutrons should be included.
 * @param a_particles       [in]    The list of particles to be transported.
 * @param a_productID       [in]    PoPs id for the requested product.
 * @param a_order           [in]    Requested product matrix, Legendre order.
 * @return                          The requested multi-group product matrix as a GIDI::Matrix.
 ***********************************************************************************************************/

Matrix ProtareSingleton::multiGroupProductMatrix( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID, int a_order ) const {

    Matrix matrix( 0, 0 );

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction = m_reactions.get<Reaction>( i1 );

        matrix += reaction->multiGroupProductMatrix( a_settings, a_particles, a_productID, a_order );
    }

    for( std::size_t i1 = 0; i1 < m_orphanProducts.size( ); ++i1 ) {
        Reaction const *reaction = m_orphanProducts.get<Reaction>( i1 );

        matrix += reaction->multiGroupProductMatrix( a_settings, a_particles, a_productID, a_order );
    }

    return( matrix );
}

/* *********************************************************************************************************//**
 * Like ProtareSingleton::multiGroupProductMatrix, but only returns the fission neutron, transfer matrix.
 *
 * @param a_settings    [in]    Specifies the requested label and if delayed neutrons should be included.
 * @param a_particles   [in]    The list of particles to be transported.
 * @param a_order       [in]    Requested product matrix, Legendre order.
 * @return                      The requested multi-group neutron fission matrix as a GIDI::Matrix.
 ***********************************************************************************************************/

Matrix ProtareSingleton::multiGroupFissionMatrix( Settings::MG const &a_settings, Settings::Particles const &a_particles, int a_order ) const {

    Matrix matrix( 0, 0 );

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction = m_reactions.get<Reaction>( i1 );

        if( reaction->hasFission( ) ) matrix += reaction->multiGroupFissionMatrix( a_settings, a_particles, a_order );
    }

    return( matrix );
}

/* *********************************************************************************************************//**
 * Returns the multi-group transport correction for the requested label. The transport correction is calculated from the transfer matrix
 * for the projectile id for the Legendre order of *a_order + 1*.
 *
 * @param a_settings                [in]    Specifies the requested label.
 * @param a_particles               [in]    The list of particles to be transported.
 * @param a_order                   [in]    Maximum Legendre order for transport. The returned transport correction is for the next higher Legender order.
 * @param a_transportCorrectionType [in]    Requested transport correction type.
 * @param a_temperature             [in]    The temperature of the flux to use when collapsing. Pass to the GIDI::collapse method.
 * @return                                  The requested multi-group transport correction as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareSingleton::multiGroupTransportCorrection( Settings::MG const &a_settings, Settings::Particles const &a_particles, int a_order, transportCorrectionType a_transportCorrectionType, double a_temperature ) const {

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
 * @param a_settings    [in]    Specifies the requested label.
 * @param a_particles   [in]    The list of particles to be transported.
 * @return                      The requested multi-group available energy as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareSingleton::multiGroupAvailableEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const {

    Vector vector( 0 );

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction = m_reactions.get<Reaction>( i1 );

        vector += reaction->multiGroupAvailableEnergy( a_settings, a_particles );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total average energy for the requested label for the requested product. This is a cross section weighted average energy
 * summed over all reactions.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_particles       [in]    The list of particles to be transported.
 * @param a_productID       [in]    Particle id for the requested product.
 * @return                          The requested multi-group average energy as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareSingleton::multiGroupAverageEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const {

    Vector vector( 0 );

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction = m_reactions.get<Reaction>( i1 );

        vector += reaction->multiGroupAverageEnergy( a_settings, a_particles, a_productID );
    }

    for( std::size_t i1 = 0; i1 < m_orphanProducts.size( ); ++i1 ) {
        Reaction const *reaction = m_orphanProducts.get<Reaction>( i1 );

        vector += reaction->multiGroupAverageEnergy( a_settings, a_particles, a_productID );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total deposition energy for the requested label. This is a cross section weighted deposition energy
 * summed over all reactions. The deposition energy is calculated by subtracting the average energy from each transportable particle
 * from the available energy. The list of transportable particles is specified via the list of particle specified in the *a_settings* argument.
 *
 * @param a_settings    [in]    Specifies the requested label and the products that are transported.
 * @param a_particles   [in]    The list of particles to be transported.
 * @return                      The requested multi-group deposition energy as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareSingleton::multiGroupDepositionEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const {

    std::map<std::string, Settings::Particle> const &products( a_particles.particles( ) );
    Vector vector = multiGroupAvailableEnergy( a_settings, a_particles );

    for( std::map<std::string, Settings::Particle>::const_iterator iter = products.begin( ); iter != products.end( ); ++iter ) {
        vector -= multiGroupAverageEnergy( a_settings, a_particles, iter->first );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total available momentum for the requested label. This is a cross section weighted available momentum
 * summed over all reactions.
 *
 * @param a_settings    [in]    Specifies the requested label.
 * @param a_particles   [in]    The list of particles to be transported.
 * @return                      The requested multi-group available momentum as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareSingleton::multiGroupAvailableMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const {

    Vector vector( 0 );

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction = m_reactions.get<Reaction>( i1 );

        vector += reaction->multiGroupAvailableMomentum( a_settings, a_particles );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total average momentum for the requested label for the requested product. This is a cross section weighted average momentum
 * summed over all reactions.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_particles       [in]    The list of particles to be transported.
 * @param a_productID       [in]    Particle id for the requested product.
 * @return                          The requested multi-group average momentum as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareSingleton::multiGroupAverageMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const {

    Vector vector( 0 );

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction = m_reactions.get<Reaction>( i1 );

        vector += reaction->multiGroupAverageMomentum( a_settings, a_particles, a_productID );
    }

    for( std::size_t i1 = 0; i1 < m_orphanProducts.size( ); ++i1 ) {
        Reaction const *reaction = m_orphanProducts.get<Reaction>( i1 );

        vector += reaction->multiGroupAverageMomentum( a_settings, a_particles, a_productID );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total deposition momentum for the requested label. This is a cross section weighted deposition momentum
 * summed over all reactions. The deposition momentum is calculated by subtracting the average momentum from each transportable particle
 * from the available momentum. The list of transportable particles is specified via the list of particle specified in the *a_settings* argument.
 *
 * @param a_settings    [in]    Specifies the requested label.
 * @param a_particles   [in]    The list of particles to be transported.
 * @return                      The requested multi-group deposition momentum as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareSingleton::multiGroupDepositionMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const {

    std::map<std::string, Settings::Particle> const &products( a_particles.particles( ) );
    Vector vector = multiGroupAvailableMomentum( a_settings, a_particles );

    for( std::map<std::string, Settings::Particle>::const_iterator iter = products.begin( ); iter != products.end( ); ++iter ) {
        vector -= multiGroupAverageMomentum( a_settings, a_particles, iter->first );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, gain for the requested particle and label. This is a cross section weighted gain summed over all reactions.
 *
 * @param a_settings    [in]    Specifies the requested label.
 * @param a_particles   [in]    The list of particles to be transported.
 * @param a_productID   [in]    The particle PoPs' id for the whose gain is to be calculated.
 *
 * @return                      The requested multi-group gain as a **GIDI::Vector**.
 ***********************************************************************************************************/

Vector ProtareSingleton::multiGroupGain( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const {

    Vector vector( 0 );
    std::string const projectile_ID = projectile( ).ID( );

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction = m_reactions.get<Reaction>( i1 );

        vector += reaction->multiGroupGain( a_settings, a_particles, a_productID, projectile_ID );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 *
 *
 * @return      A list of label, mu cutoff pairs.
 ***********************************************************************************************************/

stringAndDoublePairs ProtareSingleton::muCutoffForCoulombPlusNuclearElastic( ) const {

    stringAndDoublePairs muCutoffs;

    for( std::size_t i1 = 0; i1 < m_styles.size( ); ++i1 ) {
        Styles::Base const *style1 = m_styles.get<Styles::Base>( i1 );

        if( style1->moniker( ) == CoulombPlusNuclearElasticMuCutoffStyleMoniker ) {
            Styles::CoulombPlusNuclearElasticMuCutoff const *style2 = static_cast<Styles::CoulombPlusNuclearElasticMuCutoff const *>( style1 );
            
            stringAndDoublePair labelMu( style2->label( ), style2->muCutoff( ) );

            muCutoffs.push_back( labelMu );
        }
    }

    return( muCutoffs );
}

/* *********************************************************************************************************//**
 * Write *this* to a file in GNDS/XML format.
 *
 * @param       a_fileName          [in]        Name of file to save XML lines to.
 ***********************************************************************************************************/

void ProtareSingleton::saveAs( std::string const &a_fileName ) const {

    WriteInfo writeInfo;

    toXMLList( writeInfo, "" );

    std::ofstream fileio;
    fileio.open( a_fileName.c_str( ) );
    for( std::list<std::string>::iterator iter = writeInfo.m_lines.begin( ); iter != writeInfo.m_lines.end( ); ++iter ) {
        fileio << *iter << std::endl;
    }
    fileio.close( );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void ProtareSingleton::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string header = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>";
    std::string attributes;

    a_writeInfo.push_back( header );

    attributes  = a_writeInfo.addAttribute( "projectile", projectile( ).ID( ) );
    attributes += a_writeInfo.addAttribute( "target", target( ).ID( ) );
    attributes += a_writeInfo.addAttribute( "evaluation", evaluation( ) );
    attributes += a_writeInfo.addAttribute( "format", GIDI_format );
    attributes += a_writeInfo.addAttribute( "projectileFrame", frameToString( projectileFrame( ) ) );

    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );

    m_styles.toXMLList( a_writeInfo, indent2 );

    std::vector<std::string> pops_XMLList;
    m_internalPoPs.toXMLList( pops_XMLList, indent2 );
    for( std::vector<std::string>::iterator iter = pops_XMLList.begin( ); iter != pops_XMLList.end( ); ++iter ) a_writeInfo.push_back( *iter );

    m_reactions.toXMLList( a_writeInfo, indent2 );
    m_orphanProducts.toXMLList( a_writeInfo, indent2 );
    m_sums.toXMLList( a_writeInfo, indent2 );
    m_fissionComponents.toXMLList( a_writeInfo, indent2 );

    a_writeInfo.addNodeEnder( moniker( ) );
}

}

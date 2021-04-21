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
        m_target( "", "", -1.0 ),
        m_GNDS_target( "", "", -1.0 ) {
    
}

/* *********************************************************************************************************//**
 ******************************************************************/

Protare::~Protare( ) {

}

/* *********************************************************************************************************//**
 * Called by the constructs. This method does most of the parsing.
 *
 * @param a_node                        [in]    The protare (i.e., reactionSuite) node to be parsed and used to construct a Protare.
 * @param a_setupInfo                   [in]    Information create my the Protare constructor to help in parsing.
 * @param a_pops                        [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs                [in]    The internal PoPI::Database instance used to get particle indices and possibly other particle information.
 * @param a_targetRequiredInGlobalPoPs  [in]    If *true*, the target is required to be in **a_pops**.
 * @param a_requiredInPoPs              [in]    If *true*, no particle is required to be in **a_pops**.
 ***********************************************************************************************************/

void Protare::initialize( pugi::xml_node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs,
                bool a_targetRequiredInGlobalPoPs, bool a_requiredInPoPs ) {

    setMoniker( a_node.name( ) );    

    std::string projectileID = a_node.attribute( GIDI_projectileChars ).value( );
    m_projectile = ParticleInfo( projectileID, a_pops, a_internalPoPs, a_requiredInPoPs );

    std::string targetID = a_node.attribute( GIDI_targetChars ).value( );
    m_GNDS_target = ParticleInfo( targetID, a_pops, a_internalPoPs, a_targetRequiredInGlobalPoPs && a_requiredInPoPs );

    auto iter = a_setupInfo.m_particleSubstitution->find( m_GNDS_target.ID( ) );
    if( iter != a_setupInfo.m_particleSubstitution->end( ) ) {
        m_target = iter->second; }
    else {
        m_target = m_GNDS_target;
    }
}

/* *********************************************************************************************************//**
 * If the protare is a ProtareTNSL then summing over all reactions will include the standard protare's elastic cross section
 * in the domain of the TNSL data. The standard elastic cross section should not be added in this domain.
 * If needed, this function corrects the cross section for this over counting of the elastic cross section.
 * This method does nothing unless overwritten by the ProtareTNSL class.
 *
 * @param       a_label                     [in]    The label of the elastic cross section data to use if over counting needs to be corrected.
 * @param       a_crossSectionSum           [in]    The cross section to correct.
 ***********************************************************************************************************/

void Protare::TNSL_crossSectionSumCorrection( std::string const &a_label, Functions::XYs1d &a_crossSectionSum ) {

}

/* *********************************************************************************************************//**
 * If the protare is a ProtareTNSL then summing over all reactions will include the standard protare's elastic cross section
 * in the domain of the TNSL data. The standard elastic cross section should not be added in this domain.
 * If needed, this function corrects the cross section for this over counting of the elastic cross section.
 * This method does nothing unless overwritten by the ProtareTNSL class.
 *
 * @param       a_label                     [in]    The label of the elastic cross section data to use if over counting needs to be corrected.
 * @param       a_crossSectionSum           [in]    The cross section to correct.
 ***********************************************************************************************************/

void Protare::TNSL_crossSectionSumCorrection( std::string const &a_label, Functions::Ys1d &a_crossSectionSum ) {

}

/* *********************************************************************************************************//**
 * If the protare is a ProtareTNSL then summing over all reactions will include the standard protare's elastic cross section
 * in the domain of the TNSL data. The standard elastic cross section should not be added in this domain.
 * If needed, this function corrects the cross section for this over counting of the elastic cross section.
 * This method does nothing as the multi-group cross section data for the standard protare's elastic cross section are
 * zeroed when the data are read in. However, this method is added so codes do not have to check the type of data they are accessing.
 *
 * @param       a_label                     [in]    The label of the elastic cross section data to use if over counting needs to be corrected.
 * @param       a_crossSectionSum           [in]    The cross section to correct.
 ***********************************************************************************************************/

void Protare::TNSL_crossSectionSumCorrection( std::string const &a_label, Vector &a_crossSectionSum ) {

}

/* *********************************************************************************************************//**
 * Returns a list of all reaction indices whose ENDL C value is in the set *a_CValues*.
 *
 * @param       a_CValues                       [in]    A list of ENDL C values.
 * @param       a_checkActiveState              [in]    If true, all reactions whose active state is false are not included in the returned set even if their CValue match on in the list.
 *
 * @return                                      The list of reaction indices.
 ***********************************************************************************************************/

std::set<int> Protare::reactionIndicesMatchingENDLCValues( std::set<int> const &a_CValues, bool a_checkActiveState ) {

    std::set<int> indices;

    for( std::size_t i1 = 0; i1 < numberOfReactions( ); ++i1 ) {
        Reaction *reaction1 = reaction( i1 );

        if( a_checkActiveState && !reaction1->active( ) ) continue;
        if( a_CValues.find( reaction1->ENDL_C( ) ) != a_CValues.end( ) ) indices.insert( i1 );
    }

    return( indices );
}

/*! \class ProtareSingle
 * Class to store a GNDS <**reactionSuite**> node.
 */

/* *********************************************************************************************************//**
 * Parses a GNDS file to construct the Protare instance. Calls the initialize method which does most of the work.
 *
 * @param a_pops            [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_projectileID    [in]    The PoPs id of the projectile.
 * @param a_targetID        [in]    The PoPs id of the target.
 * @param a_evaluation      [in]    The evaluation string.
 * @param a_interaction     [in]    The interaction flag for the protare.
 * @param a_formatVersion   [in]    The GNDS format to use.
 ***********************************************************************************************************/

ProtareSingle::ProtareSingle( PoPI::Database const &a_pops, std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_evaluation,
                std::string const &a_interaction, std::string const &a_formatVersion ) :
        m_formatVersion( a_formatVersion ),
        m_evaluation( a_evaluation ),
        m_interaction( a_interaction ),
        m_projectileFrame( Frame::lab ),
        m_thresholdFactor( 0.0 ) {

    setMoniker( GIDI_topLevelChars );
    initialize( );

    setProjectile( ParticleInfo( a_projectileID, a_pops, a_pops, true ) );
    setTarget( ParticleInfo( a_targetID, a_pops, a_pops, true ) );
}

/* *********************************************************************************************************//**
 * Parses a GNDS file to construct the Protare instance. Calls the initialize method which does most of the work.
 *
 * @param a_construction                [in]    Used to pass user options to the constructor.
 * @param a_fileName                    [in]    File containing a protare (i.e., reactionSuite) node that is parsed and used to construct the Protare.
 * @param a_fileType                    [in]    File type of a_fileType. Currently, only GIDI::XML is supported.
 * @param a_pops                        [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_particleSubstitution        [in]    Map of particles to substitute with another particles.
 * @param a_libraries                   [in]    The list of libraries that were searched to find *this*.
 * @param a_interaction                 [in]    The interaction flag for the protare.
 * @param a_targetRequiredInGlobalPoPs  [in]    If *true*, the target is required to be in **a_pops**.
 * @param a_requiredInPoPs              [in]    If *true*, no particle is required to be in **a_pops**.
 ***********************************************************************************************************/

ProtareSingle::ProtareSingle( Construction::Settings const &a_construction, std::string const &a_fileName, FileType a_fileType, 
                PoPI::Database const &a_pops, ParticleSubstitution const &a_particleSubstitution, std::vector<std::string> const &a_libraries, 
                std::string const &a_interaction, bool a_targetRequiredInGlobalPoPs, bool a_requiredInPoPs ) :
        Protare( ),
        m_libraries( a_libraries ),
        m_interaction( a_interaction ),
        m_fileName( a_fileName ),
        m_realFileName( realPath( a_fileName ) ) {

    pugi::xml_document doc;

    if( a_fileType != FileType::XML ) throw Exception( "Only XML file type supported." );

    pugi::xml_parse_result result = doc.load_file( a_fileName.c_str( ) );
    if( result.status != pugi::status_ok ) throw Exception( result.description( ) );

    pugi::xml_node protare = doc.first_child( );

    SetupInfo setupInfo( this );
    ParticleSubstitution particleSubstitution( a_particleSubstitution );
    setupInfo.m_particleSubstitution = &particleSubstitution;

    initialize( a_construction, protare, setupInfo, a_pops, a_targetRequiredInGlobalPoPs, a_requiredInPoPs );
}

/* *********************************************************************************************************//**
 * Base initializer to be called by all constructors (directly or indirectly).
 ***********************************************************************************************************/

void ProtareSingle::initialize( ) {

    m_externalFiles.setMoniker( GIDI_externalFilesChars );
    m_externalFiles.setAncestor( this );

    m_styles.setMoniker( GIDI_stylesChars );
    m_styles.setAncestor( this );

    m_documentations.setAncestor( this );
    m_documentations.setMoniker( GIDI_documentations_1_10_Chars );

    m_reactions.setAncestor( this );
    m_reactions.setMoniker( GIDI_reactionsChars );

    m_orphanProducts.setAncestor( this );
    m_orphanProducts.setMoniker( GIDI_orphanProductsChars );

    m_sums.setAncestor( this );

    m_fissionComponents.setAncestor( this );
    m_fissionComponents.setMoniker( GIDI_fissionComponentsChars );
}

/* *********************************************************************************************************//**
 * Called by the constructs. This method does most of the parsing.
 *
 * @param a_construction                [in]    Used to pass user options to the constructor.
 * @param a_node                        [in]    The protare (i.e., reactionSuite) node to be parsed and used to construct a Protare.
 * @param a_setupInfo                   [in]    Information create my the Protare constructor to help in parsing.
 * @param a_pops                        [in]    A PoPs Database instance used to get particle indices and possibly other particle information.
 * @param a_targetRequiredInGlobalPoPs  [in]    If *true*, the target is required to be in **a_pops**.
 * @param a_requiredInPoPs              [in]    If *true*, no particle is required to be in **a_pops**.
 ***********************************************************************************************************/

void ProtareSingle::initialize( Construction::Settings const &a_construction, pugi::xml_node const &a_node, SetupInfo &a_setupInfo,
                PoPI::Database const &a_pops, bool a_targetRequiredInGlobalPoPs, bool a_requiredInPoPs ) {

    pugi::xml_node internalPoPs = a_node.child( GIDI_PoPsChars );
    m_internalPoPs.addDatabase( internalPoPs, true );
    std::vector<PoPI::Alias *> const aliases = m_internalPoPs.aliases( );
    for( auto alias = aliases.begin( ); alias != aliases.end( ); ++alias ) {
        a_setupInfo.m_particleSubstitution->insert( { (*alias)->pid( ), ParticleInfo( (*alias)->ID( ), a_pops, a_pops, true ) } );
    }

    Protare::initialize( a_node, a_setupInfo, a_pops, m_internalPoPs, a_targetRequiredInGlobalPoPs, a_requiredInPoPs );
    initialize( );

    m_formatVersion.setFormat( a_node.attribute( GIDI_formatChars ).value( ) );
    if( !m_formatVersion.supported( ) ) throw Exception( "unsupport GND format version" );
    a_setupInfo.m_formatVersion = m_formatVersion;

    if( m_formatVersion.major( ) > 1 ) m_interaction = a_node.attribute( GIDI_interactionChars ).value( );

    m_internalPoPs.calculateNuclideGammaBranchStateInfos( m_nuclideGammaBranchStateInfos );

    m_isTNSL_ProtareSingle = false;
    m_thresholdFactor = 1.0;
    if( a_pops.exists( target( ).pid( ) ) ) {
        std::string name( a_node.child( GIDI_reactionsChars ).first_child( ).child( GIDI_doubleDifferentialCrossSectionChars ).first_child( ).name( ) );

        m_isTNSL_ProtareSingle = name.find( "thermalNeutronScatteringLaw" ) != std::string::npos;
        m_thresholdFactor = 1.0 + projectile( ).mass( "amu" ) / target( ).mass( "amu" );            // BRB FIXME, I think only this statement needs to be in this if section.
    }

    m_evaluation = a_node.attribute( GIDI_evaluationChars ).value( );

    m_projectileFrame = parseFrame( a_node, a_setupInfo, GIDI_projectileFrameChars );

    m_externalFiles.parse( a_construction, a_node.child( GIDI_externalFilesChars ), a_setupInfo, a_pops, m_internalPoPs, parseExternalFilesSuite, nullptr );

    m_styles.parse( a_construction, a_node.child( GIDI_stylesChars ), a_setupInfo, a_pops, m_internalPoPs, parseStylesSuite, nullptr );

    Styles::Evaluated *evaluated = m_styles.get<Styles::Evaluated>( 0 );

    m_projectileEnergyMin = evaluated->projectileEnergyDomain( ).minimum( );
    m_projectileEnergyMax = evaluated->projectileEnergyDomain( ).maximum( );

    m_reactions.parse( a_construction, a_node.child( GIDI_reactionsChars ), a_setupInfo, a_pops, m_internalPoPs, parseReaction, &m_styles );
    m_orphanProducts.parse( a_construction, a_node.child( GIDI_orphanProductsChars ), a_setupInfo, a_pops, m_internalPoPs, parseOrphanProduct, &m_styles );

    m_sums.parse( a_construction, a_node.child( GIDI_sumsChars ), a_setupInfo, a_pops, m_internalPoPs );
    m_fissionComponents.parse( a_construction, a_node.child( GIDI_fissionComponentsChars ), a_setupInfo, a_pops, m_internalPoPs, parseFissionComponent, &m_styles );
}

/* *********************************************************************************************************//**
 ******************************************************************/

ProtareSingle::~ProtareSingle( ) {

}

/* *********************************************************************************************************//**
 * Returns the pointer representing the protare (i.e., *this*) if *a_index* is 0 and nullptr otherwise.
 *
 * @param a_index               [in]    Must always be 0.
 *
 * @return                              Returns the pointer representing *this*.
 ***********************************************************************************************************/

ProtareSingle *ProtareSingle::protare( std::size_t a_index ) {

    if( a_index != 0 ) return( nullptr );
    return( this );
}

/* *********************************************************************************************************//**
 * Returns the pointer representing the protare (i.e., *this*) if *a_index* is 0 and nullptr otherwise.
 *
 * @param a_index               [in]    Must always be 0.
 *
 * @return                              Returns the pointer representing *this*.
 ***********************************************************************************************************/

ProtareSingle const *ProtareSingle::protare( std::size_t a_index ) const {

    if( a_index != 0 ) return( nullptr );
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

void ProtareSingle::productIDs( std::set<std::string> &a_ids, Transporting::Particles const &a_particles, bool a_transportablesOnly ) const {

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction1 = m_reactions.get<Reaction>( i1 );

        if( !reaction1->active( ) ) continue;
        reaction1->productIDs( a_ids, a_particles, a_transportablesOnly );
    }

    for( std::size_t i1 = 0; i1 < m_orphanProducts.size( ); ++i1 ) {
        Reaction const *reaction1 = m_orphanProducts.get<Reaction>( i1 );

        if( !reaction1->active( ) ) continue;
        reaction1->productIDs( a_ids, a_particles, a_transportablesOnly );
    }
}

/* *********************************************************************************************************//**
 * Determines the maximum Legredre order present in the multi-group transfer matrix for a give product for a give label.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_temperatureInfo [in]    Specifies the temperature and labels use to lookup the requested data.
 * @param a_productID       [in]    The id of the requested product.
 *
 * @return                          The maximum Legredre order. If no transfer matrix data are present for the requested product, -1 is returned.
 ***********************************************************************************************************/

int ProtareSingle::maximumLegendreOrder( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const {

    int _maximumLegendreOrder = -1;

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction1 = m_reactions.get<Reaction>( i1 );

        if( !reaction1->active( ) ) continue;
        int r_maximumLegendreOrder = reaction1->maximumLegendreOrder( a_settings, a_temperatureInfo, a_productID );
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

Styles::TemperatureInfos ProtareSingle::temperatures( ) const {

    std::size_t size( m_styles.size( ) );
    Styles::TemperatureInfos temperature_infos;

    for( std::size_t i1 = 0; i1 < size; ++i1 ) {
        Styles::Base const *style1 = m_styles.get<Styles::Base>( i1 );

        if( style1->moniker( ) == GIDI_heatedStyleChars ) {
            PhysicalQuantity const &temperature = style1->temperature( );
            std::string heated_cross_section( style1->label( ) );
            std::string gridded_cross_section( "" );
            std::string URR_probability_tables( "" );
            std::string heated_multi_group( "" );
            std::string Sn_elastic_upscatter( "" );

            for( std::size_t i2 = 0; i2 < size; ++i2 ) {
                Styles::Base const *style2 = m_styles.get<Styles::Base>( i2 );

                if( style2->moniker( ) == GIDI_multiGroupStyleChars ) continue;
                if( style2->temperature( ).value( ) != temperature.value( ) ) continue;

                if( style2->moniker( ) == GIDI_griddedCrossSectionStyleChars ) {
                    gridded_cross_section = style2->label( ); }
                else if( style2->moniker( ) == GIDI_URR_probabilityTablesStyleChars ) {
                    URR_probability_tables = style2->label( ); }
                else if( style2->moniker( ) == GIDI_SnElasticUpScatterStyleChars ) {
                    Sn_elastic_upscatter = style2->label( ); }
                else if( style2->moniker( ) == GIDI_heatedMultiGroupStyleChars ) {
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
 * Returns the multi-group boundaries for the requested label and product.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_temperatureInfo [in]    Specifies the temperature and labels use to lookup the requested data.
 * @param a_productID       [in]    ID for the requested product.
 *
 * @return                          List of multi-group boundaries.
 ***********************************************************************************************************/

std::vector<double> const &ProtareSingle::groupBoundaries( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const {

    Styles::HeatedMultiGroup const *heatedMultiGroupStyle1 = m_styles.get<Styles::HeatedMultiGroup>( a_temperatureInfo.heatedMultiGroup( ) );

    return( heatedMultiGroupStyle1->groupBoundaries( a_productID ) );
}

/* *********************************************************************************************************//**
 * Returns the inverse speeds for the requested label. The label must be for a heated multi-group style.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_temperatureInfo [in]    Specifies the temperature and labels use to lookup the requested data.
 *
 * @return                          List of inverse speeds.
 ***********************************************************************************************************/

Vector ProtareSingle::multiGroupInverseSpeed( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const {

    Styles::HeatedMultiGroup const *heatedMultiGroupStyle1 = m_styles.get<Styles::HeatedMultiGroup>( a_temperatureInfo.heatedMultiGroup( ) );

    return( heatedMultiGroupStyle1->inverseSpeedData( ) );
}

/* *********************************************************************************************************//**
 * Returns true if at least one reaction contains a fission channel.
 *
 * @return  true if at least one reaction contains a fission channel.
 ***********************************************************************************************************/

bool ProtareSingle::hasFission( ) const {

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction1 = m_reactions.get<Reaction>( i1 );

        if( !reaction1->active( ) ) continue;
        if( reaction1->hasFission( ) ) return( true );
    }
    return( false );
}

/* *********************************************************************************************************//**
 * Used by Ancestry to tranverse GNDS nodes. This method returns a pointer to a derived class' a_item member or nullptr if none exists.
 *
 * @param a_item    [in]    The name of the class member whose pointer is to be return.
 * @return                  The pointer to the class member or nullptr if class does not have a member named a_item.
 ***********************************************************************************************************/

Ancestry *ProtareSingle::findInAncestry3( std::string const &a_item ) {

    if( a_item == GIDI_stylesChars ) return( &m_styles );
    if( a_item == GIDI_reactionsChars ) return( &m_reactions );
    if( a_item == GIDI_orphanProductsChars ) return( &m_orphanProducts );
    if( a_item == GIDI_sumsChars ) return( &m_sums );
    if( a_item == GIDI_fissionComponentsChars ) return( &m_fissionComponents );

    return( nullptr );
}

/* *********************************************************************************************************//**
 * Used by Ancestry to tranverse GNDS nodes. This method returns a pointer to a derived class' a_item member or nullptr if none exists.
 *
 * @param a_item    [in]    The name of the class member whose pointer is to be return.
 * @return                  The pointer to the class member or nullptr if class does not have a member named a_item.
 ***********************************************************************************************************/

Ancestry const *ProtareSingle::findInAncestry3( std::string const &a_item ) const {

    if( a_item == GIDI_stylesChars ) return( &m_styles );
    if( a_item == GIDI_reactionsChars ) return( &m_reactions );
    if( a_item == GIDI_orphanProductsChars ) return( &m_orphanProducts );
    if( a_item == GIDI_sumsChars ) return( &m_sums );
    if( a_item == GIDI_fissionComponentsChars ) return( &m_fissionComponents );

    return( nullptr );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total cross section for the requested label. This is summed over all reactions.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_temperatureInfo [in]    Specifies the temperature and labels use to lookup the requested data.
 *
 * @return                          The requested multi-group cross section as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareSingle::multiGroupCrossSection( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const {

    Vector vector;

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction1 = m_reactions.get<Reaction>( i1 );

        if( !reaction1->active( ) ) continue;
        vector += reaction1->multiGroupCrossSection( a_settings, a_temperatureInfo );
    }
    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total multiplicity for the requested label for the requested product. This is a cross section weighted multiplicity.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_temperatureInfo [in]    Specifies the temperature and labels use to lookup the requested data.
 * @param a_productID       [in]    Id for the requested product.
 *
 * @return                          The requested multi-group multiplicity as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareSingle::multiGroupMultiplicity( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const {

    Vector vector( 0 );

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction1 = m_reactions.get<Reaction>( i1 );

        if( !reaction1->active( ) ) continue;
        vector += reaction1->multiGroupMultiplicity( a_settings, a_temperatureInfo, a_productID );
    }

    for( std::size_t i1 = 0; i1 < m_orphanProducts.size( ); ++i1 ) {
        Reaction const *reaction1 = m_orphanProducts.get<Reaction>( i1 );

        if( !reaction1->active( ) ) continue;
        vector += reaction1->multiGroupMultiplicity( a_settings, a_temperatureInfo, a_productID );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total fission neutron multiplicity for the requested label. This is a cross section weighted multiplicity.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_temperatureInfo [in]    Specifies the temperature and labels use to lookup the requested data.
 *
 * @return                          The requested multi-group fission neutron multiplicity as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareSingle::multiGroupFissionNeutronMultiplicity( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const {

    Vector vector( 0 );

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction1 = m_reactions.get<Reaction>( i1 );

        if( !reaction1->active( ) ) continue;
        if( reaction1->hasFission( ) ) vector += reaction1->multiGroupMultiplicity( a_settings, a_temperatureInfo, PoPI::IDs::neutron );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total Q for the requested label. This is a cross section weighted multiplicity
 * summed over all reactions
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_temperatureInfo [in]    Specifies the temperature and labels use to lookup the requested data.
 * @param a_final           [in]    If false, only the Q for the primary reactions are return, otherwise, the Q for the final reactions.
 *
 * @return                          The requested multi-group Q as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareSingle::multiGroupQ( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, bool a_final ) const {

    Vector vector( 0 );

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction1 = m_reactions.get<Reaction>( i1 );

        if( !reaction1->active( ) ) continue;
        vector += reaction1->multiGroupQ( a_settings, a_temperatureInfo, a_final );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total product matrix for the requested label for the requested product id for the requested Legendre order.
 * If no data are found, an empty GIDI::Matrix is returned.
 *
 * @param a_settings        [in]    Specifies the requested label and if delayed neutrons should be included.
 * @param a_temperatureInfo [in]    Specifies the temperature and labels use to lookup the requested data.
 * @param a_particles       [in]    The list of particles to be transported.
 * @param a_productID       [in]    PoPs id for the requested product.
 * @param a_order           [in]    Requested product matrix, Legendre order.
 *
 * @return                          The requested multi-group product matrix as a GIDI::Matrix.
 ***********************************************************************************************************/

Matrix ProtareSingle::multiGroupProductMatrix( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, std::string const &a_productID, int a_order ) const {

    Matrix matrix( 0, 0 );

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction1 = m_reactions.get<Reaction>( i1 );

        if( !reaction1->active( ) ) continue;
        matrix += reaction1->multiGroupProductMatrix( a_settings, a_temperatureInfo, a_particles, a_productID, a_order );
    }

    for( std::size_t i1 = 0; i1 < m_orphanProducts.size( ); ++i1 ) {
        Reaction const *reaction1 = m_orphanProducts.get<Reaction>( i1 );

        if( !reaction1->active( ) ) continue;
        matrix += reaction1->multiGroupProductMatrix( a_settings, a_temperatureInfo, a_particles, a_productID, a_order );
    }

    return( matrix );
}

/* *********************************************************************************************************//**
 * Like ProtareSingle::multiGroupProductMatrix, but only returns the fission neutron, transfer matrix.
 *
 * @param a_settings        [in]    Specifies the requested label and if delayed neutrons should be included.
 * @param a_temperatureInfo [in]    Specifies the temperature and labels use to lookup the requested data.
 * @param a_particles       [in]    The list of particles to be transported.
 * @param a_order           [in]    Requested product matrix, Legendre order.
 *
 * @return                          The requested multi-group neutron fission matrix as a GIDI::Matrix.
 ***********************************************************************************************************/

Matrix ProtareSingle::multiGroupFissionMatrix( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, int a_order ) const {

    Matrix matrix( 0, 0 );

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction1 = m_reactions.get<Reaction>( i1 );

        if( !reaction1->active( ) ) continue;
        if( reaction1->hasFission( ) ) matrix += reaction1->multiGroupFissionMatrix( a_settings, a_temperatureInfo, a_particles, a_order );
    }

    return( matrix );
}

/* *********************************************************************************************************//**
 * Returns the multi-group transport correction for the requested label. The transport correction is calculated from the transfer matrix
 * for the projectile id for the Legendre order of *a_order + 1*.
 *
 * @param a_settings                [in]    Specifies the requested label.
 * @param a_temperatureInfo         [in]    Specifies the temperature and labels use to lookup the requested data.
 * @param a_particles               [in]    The list of particles to be transported.
 * @param a_order                   [in]    Maximum Legendre order for transport. The returned transport correction is for the next higher Legender order.
 * @param a_transportCorrectionType [in]    Requested transport correction type.
 * @param a_temperature             [in]    The temperature of the flux to use when collapsing. Pass to the GIDI::collapse method.
 *
 * @return                                  The requested multi-group transport correction as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareSingle::multiGroupTransportCorrection( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, int a_order, TransportCorrectionType a_transportCorrectionType, double a_temperature ) const {

    if( a_transportCorrectionType == TransportCorrectionType::None ) return( Vector( 0 ) );

    Matrix matrix( multiGroupProductMatrix( a_settings, a_temperatureInfo, a_particles, projectile( ).ID( ), a_order + 1 ) );
    Matrix matrixCollapsed = collapse( matrix, a_settings, a_particles, a_temperature, projectile( ).ID( ) );
    std::size_t size = matrixCollapsed.size( );
    std::vector<double> transportCorrection1( size, 0 );

    if( a_transportCorrectionType == TransportCorrectionType::Pendlebury ) {
        for( std::size_t index = 0; index < size; ++index ) transportCorrection1[index] = matrixCollapsed[index][index];
    }
    return( Vector( transportCorrection1 ) );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total available energy for the requested label. This is a cross section weighted available energy
 * summed over all reactions.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_temperatureInfo [in]    Specifies the temperature and labels use to lookup the requested data.
 *
 * @return                          The requested multi-group available energy as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareSingle::multiGroupAvailableEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const {

    Vector vector( 0 );

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction1 = m_reactions.get<Reaction>( i1 );

        if( !reaction1->active( ) ) continue;
        vector += reaction1->multiGroupAvailableEnergy( a_settings, a_temperatureInfo );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total average energy for the requested label for the requested product. This is a cross section weighted average energy
 * summed over all reactions.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_temperatureInfo [in]    Specifies the temperature and labels use to lookup the requested data.
 * @param a_productID       [in]    Particle id for the requested product.
 *
 * @return                          The requested multi-group average energy as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareSingle::multiGroupAverageEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const {

    Vector vector( 0 );

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction1 = m_reactions.get<Reaction>( i1 );

        if( !reaction1->active( ) ) continue;
        vector += reaction1->multiGroupAverageEnergy( a_settings, a_temperatureInfo, a_productID );
    }

    for( std::size_t i1 = 0; i1 < m_orphanProducts.size( ); ++i1 ) {
        Reaction const *reaction1 = m_orphanProducts.get<Reaction>( i1 );

        if( !reaction1->active( ) ) continue;
        vector += reaction1->multiGroupAverageEnergy( a_settings, a_temperatureInfo, a_productID );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total deposition energy for the requested label. This is a cross section weighted deposition energy
 * summed over all reactions. The deposition energy is calculated by subtracting the average energy from each transportable particle
 * from the available energy. The list of transportable particles is specified via the list of particle specified in the *a_settings* argument.
 *
 * @param a_settings    [in]    Specifies the requested label and the products that are transported.
 * @param a_temperatureInfo [in]    Specifies the temperature and labels use to lookup the requested data.
 * @param a_particles   [in]    The list of particles to be transported.
 *
 * @return                      The requested multi-group deposition energy as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareSingle::multiGroupDepositionEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles ) const {

    std::map<std::string, Transporting::Particle> const &products( a_particles.particles( ) );
    Vector vector = multiGroupAvailableEnergy( a_settings, a_temperatureInfo );

    for( std::map<std::string, Transporting::Particle>::const_iterator iter = products.begin( ); iter != products.end( ); ++iter ) {
        vector -= multiGroupAverageEnergy( a_settings, a_temperatureInfo, iter->first );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total available momentum for the requested label. This is a cross section weighted available momentum
 * summed over all reactions.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_temperatureInfo [in]    Specifies the temperature and labels use to lookup the requested data.
 *
 * @return                          The requested multi-group available momentum as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareSingle::multiGroupAvailableMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const {

    Vector vector( 0 );

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction1 = m_reactions.get<Reaction>( i1 );

        if( !reaction1->active( ) ) continue;
        vector += reaction1->multiGroupAvailableMomentum( a_settings, a_temperatureInfo );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total average momentum for the requested label for the requested product. This is a cross section weighted average momentum
 * summed over all reactions.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_temperatureInfo [in]    Specifies the temperature and labels use to lookup the requested data.
 * @param a_productID       [in]    Particle id for the requested product.
 *
 * @return                          The requested multi-group average momentum as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareSingle::multiGroupAverageMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const {

    Vector vector( 0 );

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction1 = m_reactions.get<Reaction>( i1 );

        if( !reaction1->active( ) ) continue;
        vector += reaction1->multiGroupAverageMomentum( a_settings, a_temperatureInfo, a_productID );
    }

    for( std::size_t i1 = 0; i1 < m_orphanProducts.size( ); ++i1 ) {
        Reaction const *reaction1 = m_orphanProducts.get<Reaction>( i1 );

        if( !reaction1->active( ) ) continue;
        vector += reaction1->multiGroupAverageMomentum( a_settings, a_temperatureInfo, a_productID );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, total deposition momentum for the requested label. This is a cross section weighted deposition momentum
 * summed over all reactions. The deposition momentum is calculated by subtracting the average momentum from each transportable particle
 * from the available momentum. The list of transportable particles is specified via the list of particle specified in the *a_settings* argument.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_temperatureInfo [in]    Specifies the temperature and labels use to lookup the requested data.
 * @param a_particles       [in]    The list of particles to be transported.
 *
 * @return                          The requested multi-group deposition momentum as a GIDI::Vector.
 ***********************************************************************************************************/

Vector ProtareSingle::multiGroupDepositionMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles ) const {

    std::map<std::string, Transporting::Particle> const &products( a_particles.particles( ) );
    Vector vector = multiGroupAvailableMomentum( a_settings, a_temperatureInfo );

    for( std::map<std::string, Transporting::Particle>::const_iterator iter = products.begin( ); iter != products.end( ); ++iter ) {
        vector -= multiGroupAverageMomentum( a_settings, a_temperatureInfo, iter->first );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 * Returns the multi-group, gain for the requested particle and label. This is a cross section weighted gain summed over all reactions.
 *
 * @param a_settings        [in]    Specifies the requested label.
 * @param a_temperatureInfo [in]    Specifies the temperature and labels use to lookup the requested data.
 * @param a_productID       [in]    The particle PoPs' id for the whose gain is to be calculated.
 *
 * @return                          The requested multi-group gain as a **GIDI::Vector**.
 ***********************************************************************************************************/

Vector ProtareSingle::multiGroupGain( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const {

    Vector vector( 0 );
    std::string const projectile_ID = projectile( ).ID( );

    for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
        Reaction const *reaction1 = m_reactions.get<Reaction>( i1 );

        if( !reaction1->active( ) ) continue;
        vector += reaction1->multiGroupGain( a_settings, a_temperatureInfo, a_productID, projectile_ID );
    }

    return( vector );
}

/* *********************************************************************************************************//**
 *
 *
 * @return      A list of label, mu cutoff pairs.
 ***********************************************************************************************************/

stringAndDoublePairs ProtareSingle::muCutoffForCoulombPlusNuclearElastic( ) const {

    stringAndDoublePairs muCutoffs;

    for( std::size_t i1 = 0; i1 < m_styles.size( ); ++i1 ) {
        Styles::Base const *style1 = m_styles.get<Styles::Base>( i1 );

        if( style1->moniker( ) == GIDI_CoulombPlusNuclearElasticMuCutoffStyleChars ) {
            Styles::CoulombPlusNuclearElasticMuCutoff const *style2 = static_cast<Styles::CoulombPlusNuclearElasticMuCutoff const *>( style1 );
            
            stringAndDoublePair labelMu( style2->label( ), style2->muCutoff( ) );

            muCutoffs.push_back( labelMu );
        }
    }

    return( muCutoffs );
}

/* *********************************************************************************************************//**
 * Returns the list of DelayedNeutronProduct instances.
 *
 * @return      a_delayedNeutronProducts        The list of delayed neutrons.
 ***********************************************************************************************************/

DelayedNeutronProducts ProtareSingle::delayedNeutronProducts( ) const {

    DelayedNeutronProducts delayedNeutronProducts1;

    if( hasFission( ) ) {
        for( std::size_t i1 = 0; i1 < m_reactions.size( ); ++i1 ) {
            Reaction const *reaction1 = m_reactions.get<Reaction>( i1 );

            if( !reaction1->active( ) ) continue;
            if( reaction1->hasFission( ) ) reaction1->delayedNeutronProducts( delayedNeutronProducts1 );
        }
    }

    return( delayedNeutronProducts1 );
}

/* *********************************************************************************************************//**
 * Write *this* to a file in GNDS/XML format.
 *
 * @param       a_fileName          [in]        Name of file to save XML lines to.
 ***********************************************************************************************************/

void ProtareSingle::saveAs( std::string const &a_fileName ) const {

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

void ProtareSingle::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );
    std::string header = GNDS_XML_verionEncoding;
    std::string attributes;

    a_writeInfo.push_back( header );

    attributes  = a_writeInfo.addAttribute( GIDI_projectileChars, projectile( ).ID( ) );
    attributes += a_writeInfo.addAttribute( GIDI_targetChars, GNDS_target( ).ID( ) );
    attributes += a_writeInfo.addAttribute( GIDI_evaluationChars, evaluation( ) );
    attributes += a_writeInfo.addAttribute( GIDI_formatChars, m_formatVersion.format( ) );
    attributes += a_writeInfo.addAttribute( GIDI_projectileFrameChars, frameToString( projectileFrame( ) ) );

    a_writeInfo.addNodeStarter( a_indent, moniker( ), attributes );

    m_externalFiles.toXMLList( a_writeInfo, indent2 );
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

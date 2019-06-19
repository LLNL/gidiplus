/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#ifndef GIDI_hpp_included
#define GIDI_hpp_included 1

#include <string>
#include <vector>
#include <set>
#include <map>
#include <list>
#include <iostream>
#include <stdexcept>

#include <pugixml.hpp>
#include <PoPs.hpp>

#include <nf_utilities.h>
#include <ptwXY.h>

#include "GIDI_data.hpp"

namespace GIDI {

class Form;
class Suite;
class OutputChannel;
class Protare;
class ProtareSingleton;
class ProtareBaseEntry;
class TNSLEntry;
class Map;
class Function2dForm;

typedef bool (*MapWalkCallBack)( ProtareBaseEntry const *a_protareEntry, void *a_userData, int a_level );

namespace Construction {

class Settings;

}                   // End of namespace Construction.

namespace Documentation {

class Suite;

}                   // End of namespace Documentation.

namespace Styles {

class Suite;

}                   // End of namespace Styles.

typedef Form *(*parseSuite)( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, 
        PoPs::Database const &a_pop, PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles );

enum formType { f_generic, f_group, f_groups, f_transportable, f_flux, f_fluxes, f_style,
                f_reaction, f_product, f_delayedNeutron, f_fissionFragmentData, f_rate,
                f_physicalQuantity, f_axisDomain, f_axis, f_grid, f_axes,
                f_flattenedArrayData, f_array3d,
                    // 1d functions.
                f_constant1d, f_XYs1d, f_Ys1d, f_polynomial1d, f_Legendre1d, f_gridded1d, f_reference1d, f_xs_pdf_cdf1d, f_regions1d, 
                f_resonancesWithBackground1d, f_resonanceBackground1d, f_resonanceBackgroundRegion1d, f_URR_probabilityTables1d,
                f_fissionEnergyRelease1d, f_branching1d, f_branching1dPids, f_thermalNeutronScatteringLaw1d, f_unspecified1d,
                    // 2d functions.
                f_XYs2d, f_gridded2d, f_recoil2d, f_isotropic2d, f_discreteGamma2d, f_primaryGamma2d, f_regions2d, 
                f_generalEvaporation2d, f_simpleMaxwellianFission2d, f_evaporation2d, f_Watt2d, f_MadlandNix2d, 
                f_weighted_function2d, f_weightedFunctionals2d, f_NBodyPhaseSpace2d,
                    // 3d functions.
                f_XYs3d, f_regions3d, f_gridded3d,
                    // distributions.
                f_angularTwoBody, f_KalbachMann, f_uncorrelated, f_unspecified, f_reference3d, f_multiGroup3d, 
                f_energyAngular, f_energyAngularMC, f_angularEnergy, f_angularEnergyMC, f_LLNL_angularEnergy,
                f_coherentPhotonScattering, f_incoherentPhotonScattering, f_thermalNeutronScatteringLaw, f_branching3d,
                f_coherentElastic, f_incoherentElastic, f_incoherentInelastic,
                    // Sums stuff.
                f_crossSectionSum, f_multiplicitySum, f_summands } ;

enum frame { lab, centerOfMass };

enum transportCorrectionType { transportCorrection_None, transportCorrection_Pendlebury, 
                               transportCorrection_LLNL, transportCorrection_Ferguson };

enum fileType { XML };
enum transportMode { multiGroup, SnElasticUpScatter, MonteCarlo };

#define GIDI_format "1.10"

#define GIDI_emptyFileName ""

#define GIDI_mapFormat "0.1"

#define GIDI_mapMoniker "map"
#define GIDI_importMoniker "import"
#define GIDI_protareMoniker "protare"
#define GIDI_TNSLMoniker "TNSL"

#define GIDI_topLevelMoniker "reactionSuite"

#define documentationsMoniker "documentations"
#define stylesMoniker "styles"
#define reactionsMoniker "reactions"
#define reactionMoniker "reaction"
#define orphanProductsMoniker "orphanProducts"
#define orphanProductMoniker "orphanProduct"
#define fissionComponentsMoniker "fissionComponents"
#define fissionComponentMoniker "fissionComponent"

#define sumsMoniker "sums"
#define sumsCrossSectionsMoniker "crossSections"
#define sumsMultiplicitiesMoniker "multiplicities"
#define sumsAddMoniker "add"
#define sumsSummandsMoniker "summands"
#define crossSectionSumMoniker "crossSectionSum"
#define multiplicitySumMoniker "multiplicitySum"

#define doubleDifferentialCrossSectionMoniker "doubleDifferentialCrossSection"
#define crossSectionMoniker "crossSection"
#define availableEnergyMoniker "availableEnergy"
#define availableMomentumMoniker "availableMomentum"

#define QMoniker "Q"
#define productsMoniker "products"
#define productMoniker "product"

#define multiplicityMoniker "multiplicity"
#define distributionMoniker "distribution"
#define averageEnergyMoniker "averageProductEnergy"
#define averageMomentumMoniker "averageProductMomentum"
#define outputChannelMoniker "outputChannel"

#define fissionFragmentDataMoniker "fissionFragmentData"
#define delayedNeutronsMoniker "delayedNeutrons"
#define delayedNeutronMoniker "delayedNeutron"
#define fissionEnergyReleasesMoniker "fissionEnergyReleases"
#define fissionEnergyReleaseMoniker "fissionEnergyRelease"
#define rateMoniker "rate"

#define groupsMoniker "groups"
#define groupMoniker "group"
#define fluxesMoniker "fluxes"

#define evaluatedStyleMoniker "evaluated"
#define crossSectionReconstructedStyleMoniker "crossSectionReconstructed"
#define angularDistributionReconstructedStyleMoniker "angularDistributionReconstructed"
#define CoulombPlusNuclearElasticMuCutoffStyleMoniker "CoulombPlusNuclearElasticMuCutoff"
#define TNSLStyleMoniker "thermalNeutronScatteringLaw"
#define averageProductDataStyleMoniker "averageProductData"
#define MonteCarlo_cdfStyleMoniker "MonteCarlo_cdf"
#define multiGroupStyleMoniker "multiGroup"
#define transportablesMoniker "transportables"
#define transportableMoniker "transportable"
#define heatedStyleMoniker "heated"
#define griddedCrossSectionStyleMoniker "griddedCrossSection"
#define URR_probabilityTablesStyleMoniker "URR_probabilityTables"
#define heatedMultiGroupStyleMoniker "heatedMultiGroup"
#define SnElasticUpScatterStyleMoniker "SnElasticUpScatter"

// 1d Function monikers.
#define constant1dMoniker "constant1d"
#define XYs1dMoniker "XYs1d"
#define Ys1dMoniker "Ys1d"
#define polynomial1dMoniker "polynomial1d"
#define LegendreMoniker "Legendre"
#define regions1dMoniker "regions1d"
#define gridded1dMoniker "gridded1d"
#define referenceMoniker "reference"
#define xs_pdf_cdf1dMoniker "xs_pdf_cdf1d"
#define unspecified1dMoniker "unspecified"
#define branching1dMoniker "branching1d"
#define TNSL1dMoniker "thermalNeutronScatteringLaw1d"

// 2d Function monikers.
#define XYs2dMoniker "XYs2d"
#define recoilMoniker "recoil"
#define isotropic2dMoniker "isotropic2d"
#define discreteGammaMoniker "discreteGamma"
#define primaryGammaMoniker "primaryGamma"
#define generalEvaporationMoniker "generalEvaporation"
#define simpleMaxwellianFissionMoniker "simpleMaxwellianFission"
#define evaporationMoniker "evaporation"
#define WattMoniker "Watt"
#define MadlandNixMoniker "MadlandNix"
#define weightedFunctionalsMoniker "weightedFunctionals"
#define NBodyPhaseSpaceMoniker "NBodyPhaseSpace"
#define regions2dMoniker "regions2d"

// 3d Function monikers.
#define XYs3dMoniker "XYs3d"
#define gridded3dMoniker "gridded3d"

// Distribution forms.
#define multiGroup3dMoniker "multiGroup3d"
#define angularTwoBodyMoniker "angularTwoBody"
#define uncorrelatedMoniker "uncorrelated"
#define KalbachMannMoniker "KalbachMann"
#define energyAngularMoniker "energyAngular"
#define energyAngularMCMoniker "energyAngularMC"
#define angularEnergyMoniker "angularEnergy"
#define angularEnergyMCMoniker "angularEnergyMC"
#define LLNLAngularEnergyMoniker "LLNLAngularEnergy"
#define coherentPhotonScatteringMoniker "coherentPhotonScattering"
#define incoherentPhotonScatteringMoniker "incoherentPhotonScattering"
#define TNSL_coherentElasticMoniker "thermalNeutronScatteringLaw_coherentElastic"
#define TNSL_incoherentElasticMoniker "thermalNeutronScatteringLaw_incoherentElastic"
#define TNSL_incoherentInelasticMoniker "thermalNeutronScatteringLaw_incoherentInelastic"
#define TNSLMoniker "thermalNeutronScatteringLaw"
#define branching3dMoniker "branching3d"
#define unspecifiedMoniker "unspecified"

#define scatteringAtomsMoniker "scatteringAtoms"
#define scatteringAtomMoniker "scatteringAtom"

#define resonancesWithBackgroundMoniker "resonancesWithBackground"
#define resonancesMoniker "resonances"
#define resonanceBackground1dMoniker   "background"
#define resolvedRegionMoniker "resolvedRegion"
#define unresolvedRegionMoniker "unresolvedRegion"
#define fastRegionMoniker "fastRegion"
#define CoulombPlusNuclearElasticMoniker "CoulombPlusNuclearElastic"
#define URR_probabilityTables1ddMoniker "URR_probabilityTables1d"
#define LLNLLegendreMoniker "LLNLLegendre"

#define pidsMoniker "pids"

#define hrefAttribute "href"
#define initialAttribute "initial"
#define finalAttribute "final"

typedef std::pair<std::string, double> stringAndDoublePair;
typedef std::vector<stringAndDoublePair> stringAndDoublePairs;

#ifdef _WIN32
#define FILE_SEPARATOR   "\\"
#else
#define FILE_SEPARATOR   "/"
#endif

std::vector<std::string> vectorOfStrings( std::string const &a_string );

namespace Construction {

/* *********************************************************************************************************
 * This enum allows a user to limit the data read in by various constructors. Limiting the data speeds up the reading
 * and parsing, and uses less memory.
 ***********************************************************************************************************/

enum ParseMode { e_all,                             /**< Read and parse all data. */
                 e_multiGroupOnly,                  /**< Only read and parse data needed for multi-group transport. */
                 e_MonteCarloContinuousEnergy,      /**< Only read and parse data needed for continuous energy Monte Carlo. */
                 e_excludeProductMatrices,          /**< Read and parse all data but multi-group product matrices. */
                 e_readOnly,                        /**< Only read and parse all the data but do no calculations. Useful for reading an incomplete GNDS file. */
                 e_outline                          /**< Does parse any component data (e.g., cross section, multiplicity, distribution). */ };

enum PhotoMode { e_nuclearAndAtomic,                /**< Instructs method Map::protare to create a Protare with both photo-nuclear and photo-atomic data when the projectile is photon. */
                 e_nuclearOnly,                     /**< Instructs method Map::protare to create a Protare with only photo-nuclear data when the projectile is photon. */
                 e_atomicOnly                       /**< Instructs method Map::protare to create a Protare with only photo-atomic data when the projectile is photon. */ };

/*
============================================================
========================= Settings =========================
============================================================
*/
class Settings {

    private:
        ParseMode m_parseMode;                                      /**< Parameter used by various constructors to limit data read into. */
        PhotoMode m_photoMode;                                      /**< Determines whether photo-nuclear and/or photo-atomic are included a Protare when the projectile is photon. */
        int m_useSystem_strtod;                                     /**< Flag passed to the function nfu_stringToListOfDoubles of the numericalFunctions library. */

    public:
        Settings( ParseMode a_parseMode, PhotoMode a_photoMode );

        ParseMode parseMode( ) const { return( m_parseMode ); }     /**< Returns the value of the **m_parseMode** member. */

        PhotoMode photoMode( ) const { return( m_photoMode ); }     /**< Returns the value of the **m_photoMode** member. */
        void photoMode( PhotoMode a_photoMode ) { m_photoMode = a_photoMode; }

        int useSystem_strtod( ) const { return( m_useSystem_strtod ); }
        void useSystem_strtod( bool a_useSystem_strtod ) { m_useSystem_strtod = a_useSystem_strtod ? 1 : 0; }
};

}               // End namespace Construction.

/*
============================================================
======================== WriteInfo =========================
============================================================
*/
class WriteInfo {

    public:
        std::list<std::string> m_lines;
        std::string m_incrementalIndent;
        int m_valuesPerLine;
        std::string m_sep;

        WriteInfo( std::string const &a_incrementalIndent = "  ", int a_valuesPerLine = 100, std::string const &a_sep = " " );

        std::string incrementalIndent( std::string const &indent ) { return( indent + m_incrementalIndent ); }
        void push_back( std::string const &a_line ) { m_lines.push_back( a_line ); }

        void addNodeStarter( std::string const &indent, std::string const &a_moniker, std::string const &a_attributes = "" ) {
                m_lines.push_back( indent + "<" + a_moniker + a_attributes + ">" ); }
        void addNodeStarterEnder( std::string const &indent, std::string const &a_moniker, std::string const &a_attributes = "" ) {
                m_lines.push_back( indent + "<" + a_moniker + a_attributes + "/>" ); }
        void addNodeEnder( std::string const &a_moniker ) { m_lines.back( ) += "</" + a_moniker + ">"; }
        std::string addAttribute( std::string const &a_name, std::string const &a_value ) const { return( " " + a_name + "=\"" + a_value + "\"" ); }

        std::string nodeStarter( std::string const &indent, std::string const &a_moniker, std::string const &a_attributes = "" ) { return( indent + "<" + a_moniker + a_attributes + ">" ); }
        std::string nodeEnder( std::string const &a_moniker ) { return( "</" + a_moniker + ">" ); }
};

/*
============================================================
========================= Ancestry =========================
============================================================
*/
class Ancestry {

    public:
            /* *********************************************************************************************************//**
             * Constructs and returns the key name/value for the *this* node.
             *
             * @return          The constructed key name/value.
             ***********************************************************************************************************/
        static std::string buildXLinkItemKey( std::string const &a_name, std::string const &a_key ) {

            if( a_key.size( ) == 0 ) return( "" );
            return( "[@" + a_name + "='" + a_key + "']" );
        }

    private:
        std::string m_moniker;                                  /**< The node's name (i.e., moniker). */
        Ancestry const *m_ancestor;                             /**< The parent node of *this*. */
        std::string m_attribute;                                /**< The name of the attribute in the node that uniquely identifies the node when the parent node contains other child nodes with the same moniker. */

    public:
        Ancestry( std::string const &a_moniker, std::string const &a_attribute = "" );
        virtual ~Ancestry( );

        std::string moniker( ) const { return( m_moniker ); }                           /**< Returns the value of the **m_moniker** member. */
        void moniker( std::string const &a_moniker ) { m_moniker = a_moniker; }         /**< Set the value of the **m_moniker** member to *a_moniker*. */
        Ancestry const *ancestor( ) const { return( m_ancestor ); }                     /**< Returns the value of the **m_ancestor** member. */
        void setAncestor( Ancestry const *a_ancestor ) { m_ancestor = a_ancestor; }     /**< Sets the **m_ancestor** member to *a_ancestor*. */
        std::string attribute( ) const { return( m_attribute ); }                       /**< Returns the value of the **m_attribute** member. */

        Ancestry const *root( ) const ;
        bool isChild( Ancestry const *a_instance ) const { return( this == a_instance->m_ancestor ); }  /**< Returns true if **a_instance** is a child of *this*. */
        bool isParent( Ancestry const *a_parent ) const { return( this->m_ancestor == a_parent ); }     /**< Returns true if **a_instance** is the parent of *this*. */
        bool isRoot( ) const { return( this->m_ancestor == NULL ); }                                    /**< Returns true if *this* is the root ancestor. */

        Ancestry const *findInAncestry( std::string const &a_href ) const ;
        Ancestry const *findInAncestry2( std::size_t a_index, std::vector<std::string> const &a_segments ) const ;

            /* *********************************************************************************************************//**
             * Used to tranverse **GNDS** nodes. This method returns a pointer to a derived class' *a_item* member or NULL if none exists.
             *
             * @param a_item    [in]    The name of the class member whose pointer is to be return.
             * @return                  The pointer to the class member or NULL if class does not have a member named a_item.
             ***********************************************************************************************************/
        virtual Ancestry const *findInAncestry3( std::string const &a_item ) const = 0;

        virtual std::string xlinkItemKey( ) const { return( "" ); }                                     /**< Returns the value of *this*'s key. */
        std::string toXLink( ) const ;

        virtual void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
        void printXML( ) const ;
};

/*
============================================================
=========================== Form ===========================
============================================================
*/
class Form : public Ancestry {

    private:
        Suite *m_parent;                        /**< The parent for the form. */
        formType m_type;                        /**< The type of the form. */
        std::string m_label;                    /**< The label for the form. */

    public:
        Form( formType a_type );
        Form( formType a_type, std::string const &a_label );
        Form( pugi::xml_node const &a_node, formType a_type, Suite *a_suite = NULL );
        Form( Form const &a_form );
        virtual ~Form( );

        Suite const *parent( ) const { return( m_parent ); }                                    /**< Returns the value of the **m_parent** member. */
        std::string const &label( ) const { return( m_label ); }                                /**< Returns the value of the **m_label** member. */
        void label( std::string const &a_label ) { m_label = a_label; }                         /**< Sets the **m_label** member to *a_label*. */
        formType type( ) const { return( m_type ); }                                            /**< Returns the value of the **m_type** member. */
        Form const *sibling( std::string a_label ) const ;

        Ancestry const *findInAncestry3( std::string const &a_item ) const { return( NULL ); }
        std::string xlinkItemKey( ) const {

            if( m_label == "" ) return( "" );
            return( buildXLinkItemKey( "label", m_label ) );
        }                                                                                       /**< Returns the value of *this*'s key. */
};

/*
============================================================
===================== PhysicalQuantity =====================
============================================================
*/
class PhysicalQuantity  : public Form {

    private:
        double m_value;                                                 /**< The value for the physical quantity. */
        std::string m_unit;                                             /**< The unit for the physical quantity. */

    public:
        PhysicalQuantity( pugi::xml_node const &a_node );
        PhysicalQuantity( double a_value, std::string a_unit );
        ~PhysicalQuantity( );

        double value( ) const { return( m_value ); }                    /**< Returns the value of the **m_value** member. */
        std::string const &unit( ) const { return( m_unit ); }          /**< Returns the value of the **m_unit** member. */

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
======================= ParticleInfo =======================
============================================================
*/
class ParticleInfo {

    public:
        static std::string const IDPortion( std::string const &a_id );
        static std::string const qualifierPortion( std::string const &a_id );

    private:
        std::string m_id;                                       /**< The particle's PoPs id. */
        std::string m_qualifier;                                /**< The particle's qualifier. For example the "1s1/2" in "Th{1s1/2}". */
        std::string m_pid;                                      /**< The same as *m_id* unless particle is an alias, then the final particle's id. */
        PhysicalQuantity m_mass;                                /**< The mass of the particle including nuclear excitation energy. */
        PhysicalQuantity m_excitationEnergy;                    /**< If the particle is a PoPs::Nuclide or PoPs::Nucleus, this is it nuclear excitation energy. Otherwise, it is 0. */

    public:
        ParticleInfo( std::string const &a_id, std::string const &a_pid, double a_mass, double a_excitationEnergy = 0.0 );
        ParticleInfo( std::string const &a_id, PoPs::Database const &a_globalPoPs, PoPs::Database const &a_internalPoPs, bool a_requiredInGlobalPoPs );

        std::string const &ID( ) const { return( m_id  ); }                     /**< Returns the value of the **m_id** member. */
        std::string const &pid( ) const { return( m_pid  ); }                   /**< Returns the value of the **m_pid** member. */
        bool isAlias( ) const { return( m_pid != "" ); }                        /**< Returns true if particle id is an alias and false otherwise. */

        PhysicalQuantity const &mass( ) const { return( m_mass ); }             /**< Returns the value of the **m_mass** member. */
        PhysicalQuantity const &excitationEnergy( ) const { return( m_excitationEnergy ); }     /**< Returns the value of the **m_excitationEnergy** member. */
        double mass( std::string const &a_unit ) const ;
};

/*
============================================================
======================== AxisDomain ========================
============================================================
*/
class AxisDomain : public Form {

    private:
        double m_minimum;                                           /**< The minimum value for the domain. */
        double m_maximum;                                           /**< The maximum value for the domain. */
        std::string m_unit;                                         /**< The unit for the domain. */

    public:
        AxisDomain( pugi::xml_node const &a_node );
        AxisDomain( double m_minimum, double m_maximum, std::string const &a_unit );
        ~AxisDomain( );

        double minimum( ) const { return( m_minimum ); }            /**< Returns the value of the **m_minimum** member. */
        double maximum( ) const { return( m_maximum ); }            /**< Returns the value of the **m_maximum** member. */
        std::string const &unit( ) const { return( m_unit ); }      /**< Returns the value of the **m_unit** member. */

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
=========================== Axis ===========================
============================================================
*/
class Axis : public Form {

    private:
        int m_index;                                                            /**< The index for the axis. */
        std::string m_label;                                                    /**< The label for the axis. */
        std::string m_unit;                                                     /**< The unit for the axis. */
        std::string m_href;                                                     /**< The **GNDS**'s href if instance points to another Axis or Grid instance. */

    public:
        Axis( pugi::xml_node const &a_node, formType a_type = f_axis );
        virtual ~Axis( );

        int index( ) const { return( m_index ); }                               /**< Returns the value of the **m_index** member. */
        std::string const &unit( ) const { return( m_unit ); }                  /**< Returns the value of the **m_unit** member. */
        std::string const &label( ) const { return( m_label ); }                /**< Returns the value of the **m_label** member. */

        std::string const &href( ) const { return( m_href ); }                  /**< Returns the value of the **m_href** member. */

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
=========================== Grid ===========================
============================================================
*/
class Grid : public Axis {

    private:
        std::string m_style;                                                        /**< The **GNDS grid**'s style. */
        std::string m_keyName;                                                      /**< **FIXME**. */
        std::string m_keyValue;                                                     /**< **FIXME**. */
        std::string m_valueType;                                                    /**< The type of data in m_values. Can be "Integer32". */
        std::vector<double> m_values;                                               /**< The **GNDS grid**'s values. */

    public:
        Grid( pugi::xml_node const &a_node, int a_useSystem_strtod );

        std::size_t size( ) const { return( m_values.size( ) ); }                   /**< Returns the number of values in the **m_values** member. */
        double &operator[]( std::size_t a_index ) { return( m_values[a_index] ); }  /**< Returns the value at m_values[a_index]. */

        std::string keyName( ) const { return( m_keyName ); }                       /**< Returns the value of the **m_keyName** member. */
        std::string keyValue( ) const { return( m_keyValue ); }                     /**< Returns the value of the **m_keyValue** member. */

        std::string const &style( ) const { return( m_style ); }                    /**< Returns the value of the **m_style** member. */
        std::vector<double> const &data( ) const { return( m_values ); }            /**< Returns the value of the **m_values** member. */

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
=========================== Axes ===========================
============================================================
*/
class Axes : public Form {

    private:
        std::vector<Axis *> m_axes;                                                 /**< Stores list axes Axis nodes. */

    public:
        Axes( pugi::xml_node const &a_node, int a_useSystem_strtod );
        Axes( );
        ~Axes( );

        std::size_t size( ) const { return( m_axes.size( ) ); }                     /**< Returns the number of *Axis* instances in *this*. */
        Axis const *operator[]( std::size_t a_index ) const { return( (m_axes[a_index]) ); }    /**< Returns m_axes[a_index]. */
        std::size_t dimension( ) const { return( m_axes.size( ) - 1 ); }            /**< Returns the dimension of the instance. */

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
====================== FlattenedArrayData ==================
============================================================
*/
class FlattenedArrayData : public Form {

    public:
        std::vector<int> m_shape;                                               /**< The shape of the flattened array. */
        std::size_t m_numberOfStarts;                                           /**< The number of start values. */
        std::size_t m_numberOfLengths;                                          /**< The number of length values. */
        int32_t *m_starts;                                                      /**< The start values. */
        int32_t *m_lengths;                                                     /**< The length values. */
        std::vector<double> m_dValues;                                          /**< The given array data. */

        FlattenedArrayData( pugi::xml_node const &a_node, int a_dimensions, int a_useSystem_strtod );
        ~FlattenedArrayData( );

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
========================= Array3d ==========================
============================================================
*/
class Array3d : public Form {

    private:
        FlattenedArrayData m_array;                                             /**< The 3d array as a FlattenedArrayData instance. */

    public:
        Array3d( pugi::xml_node const &a_node, int a_useSystem_strtod );
        ~Array3d( );

        std::size_t size( ) const { return( m_array.m_shape.back( ) ); }        /**< The length of the 3d diminsion. */

        Matrix matrix( std::size_t a_index ) const ;

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const { m_array.toXMLList( a_writeInfo, a_indent ); }
};

/*
============================================================
=========================== Rate ===========================
============================================================
*/
class Rate : public Form {

    private:
        double m_value;                                                 /**< The value for the physical quantity. */
        std::string m_unit;                                             /**< The unit for the physical quantity. */

    public:
        Rate( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~Rate( );

        double value( ) const { return( m_value ); }                    /**< Returns the value of the **m_value** member. */
        std::string const &unit( ) const { return( m_unit ); }          /**< Returns the value of the **m_unit** member. */

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
====================== FunctionForm ======================
============================================================
*/
class FunctionForm : public Form {

    private:
        int m_dimension;                                    /**< The dimension of the function (i.e., the number of independent axes. */
        std::string m_domainUnit;                           /**< The unit for the highest independent axis. */
        std::string m_rangeUnit;                            /**< The unit for the dependent axis. */
        ptwXY_interpolation m_interpolation;                /**< The interpolation for functions highest independent axis and its dependent axis. */
        std::string m_interpolationString;                  /**< The interpolation for functions highest independent axis and its dependent axis. */
        int m_index;                                        /**< Currently not used. */
        double m_outerDomainValue;                          /**< If function is part of a higher dimensional function, this is the next higher dimensions domain value. */
        Axes m_axes;                                        /**< The axes node for the function. */

    public:
        FunctionForm( std::string const &a_moniker, formType a_type, std::string const &a_label, int a_dimension, int a_index = 0, double a_outerDomainValue = 0.0 );
        FunctionForm( formType a_type, int a_dimension, std::string const &a_domainUnit, std::string const &a_rangeUnit,
                ptwXY_interpolation a_interpolation, int a_index, double a_outerDomainValue );
        FunctionForm( Construction::Settings const &a_construction, pugi::xml_node const &a_node, formType a_type, int a_dimension, Suite *a_suite = NULL );
        FunctionForm( FunctionForm const &a_form );
        ~FunctionForm( );

        int dimension( ) const { return( m_dimension ); }                                       /**< Returns the value of the **m_dimension** member. */

        std::string const &domainUnit( ) const { return( m_domainUnit ); }                      /**< Returns the value of the **m_domainUnit** member. */
        void domainUnit( std::string const &a_domainUnit ) { m_domainUnit = a_domainUnit; }     /**< Sets the **m_domainUnit** member to *a_domainUnit*. */
        std::string const &rangeUnit( ) const { return( m_rangeUnit ); }                        /**< Returns the value of the **m_rangeUnit** member. */
        void rangeUnit( std::string const &a_rangeUnit ) { m_rangeUnit = a_rangeUnit; }         /**< Sets the **m_rangeUnit** member to *a_rangeUnit*. */

        int index( ) const { return( m_index ); }                                               /**< Returns the value of the **m_index** member. */
        double outerDomainValue( ) const { return( m_outerDomainValue ); }                      /**< Returns the value of the **m_outerDomainValue** member. */
        void outerDomainValue( double a_outerDomainValue ) { m_outerDomainValue = a_outerDomainValue; }
        Axes const &axes( ) const { return( m_axes ); }                                         /**< Returns the value of the **m_axes** member. */

        ptwXY_interpolation interpolation( ) const { return( m_interpolation ); }               /**< Returns the value of the **m_interpolation** member. */
        void interpolation( ptwXY_interpolation a_interpolation ) { m_interpolation = a_interpolation; }    /**< Sets the **m_interpolation** member to **a_interpolation**. */
        std::string interpolationString( ) const { return( m_interpolationString ); }           /**< Returns the value of the **m_interpolationString** member. */

        virtual double domainMin( ) const = 0;
        virtual double domainMax( ) const = 0;

        virtual void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const { toXMLList_func( a_writeInfo, a_indent, false, false ); }
};

/*
============================================================
======================= Function1dForm =====================
============================================================
*/
class Function1dForm : public FunctionForm {

    public:
        Function1dForm( formType a_type, std::string const &a_domainUnit, std::string const &a_rangeUnit,
                ptwXY_interpolation a_interpolation, int a_index, double a_outerDomainValue );
        Function1dForm( Construction::Settings const &a_construction, pugi::xml_node const &a_node, formType a_type, Suite *a_suite = NULL );
        Function1dForm( Function1dForm const &a_form );
        ~Function1dForm( );

        virtual double evaluate( double a_x1 ) const = 0;
};

/*
============================================================
========================= Constant1d =======================
============================================================
*/
class Constant1d : public Function1dForm {

    private:
        double m_value;                                                 /**< The constant value of the function. */
        double m_domainMin;                                             /**< The minimum domain value the function is valid. */
        double m_domainMax;                                             /**< The maximum domain value the function is valid. */

    public:
        Constant1d( double value, double a_domainMin, double a_domainMax, std::string const &a_domainUnit = "", std::string const &a_rangeUnit = "" );
        Constant1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~Constant1d( );

        double value( ) const { return( m_value ); }                    /**< Returns the value of the **m_value** member. */
        double domainMin( ) const { return( m_domainMin ); }            /**< Returns the value of the **m_domainMin** member. */
        double domainMax( ) const { return( m_domainMax ); }            /**< Returns the value of the **m_domainMax** member. */

        double evaluate( double a_x1 ) const ;

        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
========================== XYs1d ===========================
============================================================
*/
class XYs1d : public Function1dForm {

    private:
        ptwXYPoints *m_ptwXY;                                               /**< The ptwXYPoints instance that stores points and is used to do calculations. */

    public:
        XYs1d( std::vector<double> const &a_values, std::string const &a_domainUnit = "", std::string const &a_rangeUnit = "",
                ptwXY_interpolation a_interpolation = ptwXY_interpolationLinLin, int a_index = 0, double a_outerDomainValue = 0.0 );
        XYs1d( std::string const &a_domainUnit = "", std::string const &a_rangeUnit = "",
                ptwXY_interpolation a_interpolation = ptwXY_interpolationLinLin, int a_index = 0, double a_outerDomainValue = 0.0 );
        XYs1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        XYs1d( XYs1d const &a_XYs1d );
        XYs1d( std::string const &a_domainUnit, std::string const &a_rangeUnit, ptwXYPoints *a_ptwXY );
        ~XYs1d( );

        std::size_t size( ) const { return( ptwXY_length( NULL, m_ptwXY ) ); }      /**< Returns the number of points (i.e., x,y pairs) in this. */
        ptwXYPoints const *ptwXY( ) const { return( m_ptwXY ); }                    /**< Returns the value of the **m_ptwXY** member. */

        std::pair<double, double> operator[]( std::size_t a_index ) const ;
        XYs1d operator+( XYs1d const &a_XYs1d ) const ;
        XYs1d &operator+=( XYs1d const &a_XYs1d );
        XYs1d operator-( XYs1d const &a_XYs1d ) const ;
        XYs1d &operator-=( XYs1d const &a_XYs1d );

        double domainMin( ) const { return( (*this)[0].first ); }                   /**< Returns first x1 value of this. */
        double domainMax( ) const { return( (*this)[size( )-1].first ); }           /**< Returns last x1 value of this. */
        std::vector<double> xs( ) const ;
        std::vector<double> ys( ) const ;
        std::vector<double> ysMappedToXs( std::vector<double> const &a_xs, std::size_t *a_offset ) const ;
        XYs1d domainSliceMax( double a_domainMax ) const ;

        double evaluate( double a_x1 ) const ;
        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
=========================== Ys1d ===========================
============================================================
*/
class Ys1d : public Function1dForm {

    private:
        std::size_t m_start;                                /**< The index in the grid for the x1 value for the first y value in **m_Ys**. */
        std::vector<double> m_Ys;                           /**< This list of y values. */

    public:
        Ys1d( std::string const &a_domainUnit = "", std::string const &a_rangeUnit = "",
                ptwXY_interpolation a_interpolation = ptwXY_interpolationLinLin, int a_index = 0, double a_outerDomainValue = 0.0 );
        Ys1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        Ys1d( Ys1d const &a_Ys1d );
        Ys1d( std::string const &a_domainUnit, std::string const &a_rangeUnit, std::size_t a_start, std::vector<double> const &a_Ys );
        ~Ys1d( );

        std::size_t size( ) const { return( m_Ys.size( ) ); }                           /**< Returns the number of values in **m_Ys**. */

        double operator[]( std::size_t a_index ) const { return( m_Ys[a_index] ); }     /**< Returns the y value at **m_Ys**[a_index]. */
        void push_back( double a_y ) { m_Ys.push_back( a_y ); }
        Ys1d operator+( Ys1d const &a_Ys1d ) const ;
        Ys1d &operator+=( Ys1d const &a_Ys1d );

        double domainMin( ) const ;
        double domainMax( ) const ;
        std::size_t start( ) const { return( m_start ); }                               /**< Returns the value of the **m_start** member. */
        void start( std::size_t a_start ) { m_start = a_start; }                        /**< Sets the **m_start** member to **a_start*. */
        std::size_t length( ) const { return( m_start + m_Ys.size( ) ); }               /**< Returns the sum of m_start and size( ). */
        std::vector<double> const &Ys( ) const { return( m_Ys ); }                      /**< Returns a reference to the list of y-values. */

        double evaluate( double a_x1 ) const ;
        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
======================= Polynomial1d =======================
============================================================
*/
class Polynomial1d : public Function1dForm {

    private:
        double m_domainMin;                                             /**< The minimum domain value the function is valid. */
        double m_domainMax;                                             /**< The maximum domain value the function is valid. */
        std::vector<double> m_coefficients;                             /**< The coefficients of the polynomial. */

    public:
        Polynomial1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~Polynomial1d( );

        double domainMin( ) const { return( m_domainMin ); }                                /**< Returns the value of the **m_domainMin** member. */
        double domainMax( ) const { return( m_domainMax ); }                                /**< Returns the value of the **m_domainMax** member. */

        std::vector<double> const &coefficients( ) const { return( m_coefficients ); }      /**< Returns the value of the **m_coefficients** member. */

        double evaluate( double a_x1 ) const ;
        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
========================= Legendre1d =======================
============================================================
*/
class Legendre1d : public Function1dForm {

    private:
        std::vector<double> m_coefficients;                                         /**< The Legendre coefficients. */

    public:
        Legendre1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~Legendre1d( );

        double domainMin( ) const { return( -1.0 ); }                               /**< Returns the value of the *domainMin* which is always -1.0. */
        double domainMax( ) const { return( 1.0 ); }                                /**< Returns the value of the *domainMax* which is always 1.0. */

        std::vector<double> const &coefficients( ) { return( m_coefficients ); }    /**< Returns the value of the **m_coefficients** member. */

        double evaluate( double a_x1 ) const ;
        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
========================== Gridded1d =======================
============================================================
*/
class Gridded1d : public Function1dForm {

    private:
        Vector m_grid;                                                          /**< The grid for the gridded 1d function. Can be a link. */
        Vector m_data;                                                          /**< The value of the function on the grid. */
// BRB should have <array compression="flattened"> ... instead of m_data.

    public:
        Gridded1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        Gridded1d( Vector const &a_grid, Vector const &a_data, Suite *a_parent );
        ~Gridded1d( );

        double domainMin( ) const { return( m_grid[0] ); }                      /**< Returns the value of the *domainMin*. */
        double domainMax( ) const { return( m_grid[m_grid.size( )-1] ); }       /**< Returns the value of the *domainMax*. */

        Vector const &grid( ) const { return( m_grid ); }                       /**< Returns the value of the **m_grid** member. */
        Vector const &data( ) const { return( m_data ); }                       /**< Returns the value of the **m_data** member. */
        void setData( Vector const &a_data ) { m_data = a_data; }               /**< Sets the **m_data** member to **a_data*. */
        double evaluate( double a_x1 ) const ;
        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
======================== Reference1d =======================
============================================================
*/
class Reference1d : public Function1dForm {

    private:
        std::string m_xlink;                                                    /**< Link to the other function. */

    public:
        Reference1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~Reference1d( );

        double domainMin( ) const ;
        double domainMax( ) const ;

        std::string const &xlink( ) const { return( m_xlink ); }                /**< Returns the value of the **m_xlink** member. */
        double evaluate( double a_x1 ) const ;
};

/*
============================================================
======================= Xs_pdf_cdf1d =======================
============================================================
*/
class Xs_pdf_cdf1d : public Function1dForm {

    private:
        std::vector<double> m_xs;                                               /**< List of x1 values. */
        std::vector<double> m_pdf;                                              /**< The pdf evaluated at the x1 values. */
        std::vector<double> m_cdf;                                              /**< The cdf evaluated at the x1 values. */
// BRB m_xs, m_pdf and m_cdf need to be a class like ListOfDoubles.

    public:
        Xs_pdf_cdf1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        Xs_pdf_cdf1d( std::string const &a_domainUnit, std::string const &a_rangeUnit, ptwXY_interpolation a_interpolation, 
            int a_index, double a_outerDomainValue, std::vector<double> const &a_Xs, std::vector<double> const &a_pdf, std::vector<double> const &a_cdf );
        ~Xs_pdf_cdf1d( );

        double domainMin( ) const { return( m_xs[0] ); }                        /**< Returns the value of the *domainMin*. */
        double domainMax( ) const { return( m_xs[m_xs.size( )-1] ); }           /**< Returns the value of the *domainMax*. */

        std::vector<double> const &Xs( ) const { return( m_xs ); }              /**< Returns the value of the **m_xs** member. */
        std::vector<double> const &pdf( ) const { return( m_pdf ); }            /**< Returns the value of the **m_pdf** member. */
        std::vector<double> const &cdf( ) const { return( m_cdf ); }            /**< Returns the value of the **m_cdf** member. */
        double evaluate( double a_x1 ) const ;
        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
========================== Regions1d =======================
============================================================
*/
class Regions1d : public Function1dForm {

    private:
        std::vector<double> m_Xs;                                               /**< List of *x1* domain values that bounds each region. */
        std::vector<Function1dForm *> m_functions1d;                            /**< List of regions. */

    public:
        Regions1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~Regions1d( );

        std::size_t size( ) const { return( m_functions1d.size( ) ); }                          /**< Returns number of regions. */
        Function1dForm const *operator[]( std::size_t a_index ) const { return( m_functions1d[a_index] ); } /**< Returns the region at index *a_index* - 1. */

        double domainMin( ) const ;
        double domainMax( ) const ;

        void append( Function1dForm *a_function );
        double evaluate( double a_x1 ) const ;

        std::vector<double> const &Xs( ) const { return( m_Xs ); }                              /**< Returns the value of the **m_Xs** member. */
        std::vector<Function1dForm *> const &functions1d( ) { return( m_functions1d ); }        /**< Returns the value of the **m_functions1d** member. */
        std::vector<Function1dForm *> &functions1d2( ) { return( m_functions1d ); }             /**< Returns the value of the **m_functions1d** member. */

        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
======================= Branching1dPids ====================
============================================================
*/
class Branching1dPids : public Form {

    private:
        std::string m_initial;                                          /**< The nuclide level that decays, emitting a photon. */
        std::string m_final;                                            /**< The nuclide level that the decay goes to. */

    public:
        Branching1dPids( pugi::xml_node const &a_node, Suite *a_parent );
        ~Branching1dPids( );

        std::string const &initial( ) const { return( m_initial ); }           /**< Returns the value of the **m_initial** member. */
        std::string const &final( ) const { return( m_final ); }               /**< Returns the value of the **m_final** member. */
};

/*
============================================================
========================= Branching1d ======================
============================================================
*/
class Branching1d : public Function1dForm {

    private:
        Branching1dPids m_pids;                                                 /**< The pids for *this*. */
        double m_multiplicity;                                                  /**< The photon multiplicity for transitioning from the initial to the final state. */

    public:
        Branching1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~Branching1d( );

        Branching1dPids const &pids( ) const { return( m_pids ); }              /**< Returns the value of the **m_pids** member. */

        double domainMin( ) const ;
        double domainMax( ) const ;

        double evaluate( double a_x1 ) const ;
};

/*
============================================================
================ ResonanceBackgroundRegion1d ===============
============================================================
*/

class ResonanceBackgroundRegion1d : public Function1dForm {

    private:
        Function1dForm *m_function1d;                                           /**< The 1-d function representing *this*. */

    public:
        ResonanceBackgroundRegion1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~ResonanceBackgroundRegion1d( );

        double domainMin( ) const ;
        double domainMax( ) const ;

        double evaluate( double a_x1 ) const ;

        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
=================== ResonanceBackground1d ==================
============================================================
*/
class ResonanceBackground1d : public Function1dForm {

    private:
        ResonanceBackgroundRegion1d *m_resolvedRegion;                      /**< The 1-d function for the resolved region. */
        ResonanceBackgroundRegion1d *m_unresolvedRegion;                    /**< The 1-d function for the unresolved region. */
        ResonanceBackgroundRegion1d *m_fastRegion;                          /**< The 1-d function for the fast region. */

    public:
        ResonanceBackground1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~ResonanceBackground1d( );

        Function1dForm const *resolvedRegion( ) const { return( m_resolvedRegion ); }       /**< Returns the value of the **m_resolvedRegion** member. */
        Function1dForm const *unresolvedRegion( ) const { return( m_unresolvedRegion ); }   /**< Returns the value of the **m_unresolvedRegion** member. */
        Function1dForm const *fastRegion( ) const { return( m_fastRegion ); }               /**< Returns the value of the **m_fastRegion** member. */

        double domainMin( ) const ;
        double domainMax( ) const ;

        double evaluate( double a_x1 ) const ;

        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
================= ResonancesWithBackground1d ===============
============================================================
*/
class ResonancesWithBackground1d : public Function1dForm {

    private:
        std::string m_resonances;                                                           /**< The reference to the resonance data for *this*.*/
        ResonanceBackground1d m_background;                                                 /**< The background .*/

    public:
        ResonancesWithBackground1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~ResonancesWithBackground1d( );

        double domainMin( ) const { return( m_background.domainMin( ) ); }                  /**< Returns *this* function's domain mimimun value. */
        double domainMax( ) const { return( m_background.domainMax( ) ); }                  /**< Returns *this* function's domain mimimun value. */

        double evaluate( double a_x1 ) const { return( m_background.evaluate( a_x1 ) ); }   /**< Returns the value *this* evaluated at *a_x1*. */

        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
================= URR_probabilityTables1d ==================
============================================================
*/
class URR_probabilityTables1d : public Function1dForm {

    private:
        Function2dForm *m_function2d;                                                       /**< The URR probability tables. */

    public:
        URR_probabilityTables1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~URR_probabilityTables1d( );

        Function2dForm const *function2d( ) const { return( m_function2d ); }               /**< Returns the pointer to the **m_function2d** member. */

        double domainMin( ) const ;
        double domainMax( ) const ;

        double evaluate( double a_x1 ) const ;

        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
=============== ThermalNeutronScatteringLaw1d ================
============================================================
*/
class ThermalNeutronScatteringLaw1d : public Function1dForm {

    private:
        std::string m_href;                                                 /**< xlink to the IncoherentPhotoAtomicScattering instance under the *m_doubleDifferentialCrossSection* node. */

    public:
        ThermalNeutronScatteringLaw1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~ThermalNeutronScatteringLaw1d( );

        std::string const &href( ) const { return( m_href ); }                          /**< Returns the value of the **m_href** member. */

        double domainMin( ) const ;
        double domainMax( ) const ;

        double evaluate( double a_x1 ) const ;
};

/*
============================================================
====================== Unspecified1d =======================
============================================================
*/
class Unspecified1d : public Function1dForm {

    public:
        Unspecified1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~Unspecified1d( );

        double domainMin( ) const ;
        double domainMax( ) const ;

        double evaluate( double a_x1 ) const ;

        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
======================= Function2dForm =====================
============================================================
*/
class Function2dForm : public FunctionForm {

    public:
        Function2dForm( std::string const &a_moniker, formType a_type, std::string const &a_label, int a_index = 0, double a_outerDomainValue = 0.0 );
        Function2dForm( Construction::Settings const &a_construction, pugi::xml_node const &a_node, formType a_type, Suite *a_suite = NULL );
        ~Function2dForm( );

        virtual double evaluate( double a_x2, double a_x1 ) const = 0;
};

/*
============================================================
=========================== XYs2d ==========================
============================================================
*/
class XYs2d : public Function2dForm {

    private:
        std::string m_interpolationQualifier;                       /**< The interpolation qualifier for *this*. */
        std::vector<double> m_Xs;                                   /**< The list of *x2* values for each function. */
        std::vector<Function1dForm *> m_function1ds;                /**< The list of 1d functions. */

    public:
        XYs2d( std::string const &a_label, std::string const &a_interpolationQualifier = "" );
        XYs2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~XYs2d( );

        std::string interpolationQualifier( ) const { return( m_interpolationQualifier ); }         /**< Returns the value of the **m_interpolationQualifier** member. */
        double domainMin( ) const ;
        double domainMax( ) const ;
        double evaluate( double a_x2, double a_x1 ) const ;

        std::vector<double> const &Xs( ) const { return( m_Xs ); }                                  /**< Returns the value of the **m_Xs** member. */
        std::vector<Function1dForm *> const &function1ds( ) const { return( m_function1ds ); }      /**< Returns the value of the **m_function1ds** member. */
        std::vector<Function1dForm *>       &function1ds( )       { return( m_function1ds ); }      /**< Returns the value of the **m_function1ds** member. */
        void append( Function1dForm *a_function1d );

        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
========================= Recoil2d =========================
============================================================
*/
class Recoil2d : public Function2dForm {

    private:
        std::string m_xlink;                                        /**< Link to the recoil product. */

    public:
        Recoil2d( std::string const &a_label, std::string const &a_href );
        Recoil2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~Recoil2d( );

        double domainMin( ) const ;
        double domainMax( ) const ;

        std::string const &xlink( ) const { return( m_xlink ); }    /**< Returns the value of the **m_xlink** member. */
        double evaluate( double a_x2, double a_x1 ) const ;
        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
========================= Isotropic2d ======================
============================================================
*/
class Isotropic2d : public Function2dForm {

    public:
        Isotropic2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~Isotropic2d( );

        double domainMin( ) const ;
        double domainMax( ) const ;

        double evaluate( double a_x2, double a_x1 ) const ;
        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const { 
                a_writeInfo.addNodeStarterEnder( a_indent, moniker( ) ); }
};

/*
============================================================
======================= DiscreteGamma2d ====================
============================================================
*/
class DiscreteGamma2d : public Function2dForm {

    private:
        double m_domainMin;                                             /**< The minimum domain value the function is valid. */
        double m_domainMax;                                             /**< The maximum domain value the function is valid. */
        double m_value;                                                 /**< The energy of the discrete gamma. */

    public:
        DiscreteGamma2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~DiscreteGamma2d( );

        double domainMin( ) const { return( m_domainMin ); }            /**< Returns the value of the **m_domainMin** member. */
        double domainMax( ) const { return( m_domainMax ); }            /**< Returns the value of the **m_domainMax** member. */
        double value( ) const { return( m_value ); }                    /**< Returns the value of the **m_value** member. */

        double evaluate( double a_x2, double a_x1 ) const ;
        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
======================= PrimaryGamma2d ====================
============================================================
*/
class PrimaryGamma2d : public Function2dForm {

    private:
        double m_domainMin;                                             /**< The minimum domain value the function is valid. */
        double m_domainMax;                                             /**< The maximum domain value the function is valid. */
        double m_value;                                                 /**< The binding energy needed to calculate the energy of the primary gamma. */

    public:
        PrimaryGamma2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~PrimaryGamma2d( );

        double domainMin( ) const { return( m_domainMin ); }            /**< Returns the value of the **m_domainMin** member. */
        double domainMax( ) const { return( m_domainMax ); }            /**< Returns the value of the **m_domainMax** member. */
        double value( ) const { return( m_value ); }                    /**< Returns the value of the **m_value** member. */

        double evaluate( double a_x2, double a_x1 ) const ;
        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
=================== GeneralEvaporation2d ===================
============================================================
*/
class GeneralEvaporation2d : public Function2dForm {

    private:
        PhysicalQuantity m_U;                                           /**< The *U* value for the general evaporation function. */
        Function1dForm *m_theta;                                        /**< The *theta* function for the general evaporation function. */
        Function1dForm *m_g;                                            /**< The *g* function for the general evaporation function. */

    public:
        GeneralEvaporation2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~GeneralEvaporation2d( );

        double U( ) const { return( m_U.value( ) ); }                   /**< Returns the GNDS *U* value for *this*. */
        Function1dForm const *theta( ) const { return( m_theta ); }     /**< Returns the value of the **m_theta** member. */
        Function1dForm const *g( ) const { return( m_g ); }             /**< Returns the value of the **m_g** member. */

        double domainMin( ) const ;
        double domainMax( ) const ;

        double evaluate( double a_x2, double a_x1 ) const ;
        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
================= SimpleMaxwellianFission2d ================
============================================================
*/
class SimpleMaxwellianFission2d : public Function2dForm {

    private:
        PhysicalQuantity m_U;                                           /**< The *U* value for the simple Maxwellian function. */
        Function1dForm *m_theta;                                        /**< The *theta* function for the simple Maxwellian function. */

    public:
        SimpleMaxwellianFission2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~SimpleMaxwellianFission2d( );

        double U( ) const { return( m_U.value( ) ); }                   /**< Returns the GNDS *U* value for *this*. */
        Function1dForm const *theta( ) const { return( m_theta ); }     /**< Returns the value of the **m_theta** member. */

        double domainMin( ) const ;
        double domainMax( ) const ;

        double evaluate( double a_x2, double a_x1 ) const ;
        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
====================== Evaporation2d =======================
============================================================
*/
class Evaporation2d : public Function2dForm {

    private:
        PhysicalQuantity m_U;                                           /**< The *U* value for the evaporation function. */
        Function1dForm *m_theta;                                        /**< The *theta* function for the evaporation function. */

    public:
        Evaporation2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~Evaporation2d( );

        double U( ) const { return( m_U.value( ) ); }                   /**< Returns the *m_U* value for *this*. */
        Function1dForm const *theta( ) const { return( m_theta ); }     /**< Returns the value of the **m_theta** member. */

        double domainMin( ) const ;
        double domainMax( ) const ;

        double evaluate( double a_x2, double a_x1 ) const ;
        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
========================== Watt2d ==========================
============================================================
*/
class Watt2d : public Function2dForm {

    private:
        PhysicalQuantity m_U;                                           /**< The *U* value for the Watt function. */
        Function1dForm *m_a;                                            /**< The *a* function for the Watt function. */
        Function1dForm *m_b;                                            /**< The *b* function for the Watt function. */

    public:
        Watt2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~Watt2d( );

        double U( ) const { return( m_U.value( ) ); }                   /**< Returns the GNDS *U* value for *this*. */
        Function1dForm const *a( ) const { return( m_a ); }             /**< Returns the value of the **m_a** member. */
        Function1dForm const *b( ) const { return( m_b ); }             /**< Returns the value of the **m_b** member. */

        double domainMin( ) const ;
        double domainMax( ) const ;

        double evaluate( double a_x2, double a_x1 ) const ;
        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
======================= MadlandNix2d =======================
============================================================
*/
class MadlandNix2d : public Function2dForm {

    private:
        PhysicalQuantity m_EFL;                                         /**< The *EFL* value for the Madland/Nix function. */
        PhysicalQuantity m_EFH;                                         /**< The *EFH* value for the Madland/Nix function. */
        Function1dForm *m_T_M;                                          /**< The *T_M* function for the Madland/Nix function. */

    public:
        MadlandNix2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~MadlandNix2d( );

        double EFL( ) const { return( m_EFL.value( ) ); }               /**< Returns the GNDS *EFL* value for *this*. */
        double EFH( ) const { return( m_EFH.value( ) ); }               /**< Returns the GNDS *EFH* value for *this*. */
        Function1dForm const *T_M( ) const { return( m_T_M ); }         /**< Returns the value of the **m_T_M** member. */


        double domainMin( ) const ;
        double domainMax( ) const ;

        double evaluate( double a_x2, double a_x1 ) const ;
        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
=================== Weighted_function2d ====================
============================================================
*/
class Weighted_function2d : public Function2dForm {

    private:
        Function1dForm *m_weight;                                       /**< The weight for this function. */
        Function2dForm *m_energy;                                       /**< The energy functional. */

    public:
        Weighted_function2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~Weighted_function2d( );

        double domainMin( ) const ;
        double domainMax( ) const ;

        Function1dForm const *weight( ) const { return( m_weight ); }       /**< Returns the value of the **m_weight** member. */
        Function2dForm const *energy( ) const { return( m_energy ); }       /**< Returns the value of the **m_energy** member. */
        double evaluate( double a_x2, double a_x1 ) const ;
};

/*
============================================================
================== WeightedFunctionals2d ===================
============================================================
*/
class WeightedFunctionals2d : public Function2dForm {

    private:
        std::vector<Weighted_function2d *> m_weighted_function2d;       /**< The list of Weighted_function2d. */

    public:
        WeightedFunctionals2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~WeightedFunctionals2d( );

        double domainMin( ) const ;
        double domainMax( ) const ;

        std::vector<Weighted_function2d *> const &weighted_function2d( ) const { return( m_weighted_function2d ); } /**< Returns the value of the **m_weighted_function2d** member. */
        double evaluate( double a_x2, double a_x1 ) const ;
};

/*
============================================================
==================== NBodyPhaseSpace2d =====================
============================================================
*/
class NBodyPhaseSpace2d : public Function2dForm {

    private:
        int m_numberOfProducts;                                         /**< The number of products for the NBodyPhaseSpace function. */
        PhysicalQuantity m_mass;                                        /**< The mass for the NBodyPhaseSpace function. */

    public:
        NBodyPhaseSpace2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~NBodyPhaseSpace2d( );

        double domainMin( ) const ;
        double domainMax( ) const ;

        int numberOfProducts( ) const { return( m_numberOfProducts ); }     /**< Returns the value of the **m_numberOfProducts** member. */
        PhysicalQuantity const &mass( ) const { return( m_mass ); }         /**< Returns the value of the **m_mass** member. */

        double evaluate( double a_x2, double a_x1 ) const ;
};

/*
============================================================
========================== Regions2d =======================
============================================================
*/
class Regions2d : public Function2dForm {

    private:
        std::vector<double> m_Xs;                                           /**< List of *x2* domain values that bounds each region. */
        std::vector<Function2dForm *> m_functions2d;                        /**< List of 2d regions. */

    public:
        Regions2d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~Regions2d( );

        double domainMin( ) const ;
        double domainMax( ) const ;

        void append( Function2dForm *a_function );
        double evaluate( double a_x2, double a_x1 ) const ;

        std::vector<double> const &Xs( ) const { return( m_Xs ); }                                  /**< Returns the value of the **m_Xs** member. */
        std::vector<Function2dForm *> const &functions2d( ) const { return( m_functions2d ); }      /**< Returns the value of the **m_functions2d** member. */
        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
======================= Function3dForm =====================
============================================================
*/
class Function3dForm : public FunctionForm {

    public:
        Function3dForm( Construction::Settings const &a_construction, pugi::xml_node const &a_node, formType a_type, Suite *a_suite = NULL );
        ~Function3dForm( );

        virtual double evaluate( double a_x3, double a_x2, double a_x1 ) const = 0;
};

/*
============================================================
=========================== XYs3d ==========================
============================================================
*/
class XYs3d : public Function3dForm {

    private:
        std::string m_interpolationQualifier;                       /**< The interpolation qualifier for *this*. */
        std::vector<double> m_Xs;                                   /**< The list of *x3* values for each function. */
        std::vector<Function2dForm *> m_function2ds;                /**< The list of 2d functions. */

    public:
        XYs3d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~XYs3d( );

        std::string interpolationQualifier( ) const { return( m_interpolationQualifier ); }         /**< Returns the value of the **m_interpolationQualifier** member. */
        double domainMin( ) const ;
        double domainMax( ) const ;
        double evaluate( double a_x3, double a_x2, double a_x1 ) const ;

        std::vector<double> const &Xs( ) const { return( m_Xs ); }                                  /**< Returns the value of the **m_Xs** member. */
        std::vector<Function2dForm *> const &function2ds( ) const { return( m_function2ds ); }      /**< Returns the value of the **m_function2ds** member. */
        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
========================== Gridded3d =======================
============================================================
*/
class Gridded3d : public Function3dForm {

    private:
        std::string m_domain1Unit;                                                      /**< The unit for the energy_in axis. */
        std::string m_domain2Unit;                                                      /**< The unit for the energy_out axis. */
        std::string m_domain3Unit;                                                      /**< The unit for the mu axis. */
        std::string m_rangeUnit;                                                        /**< The unit for the function axis. */
        Array3d m_data;                                                                 /**< The multi-group transfer matrix. */

    public:
        Gridded3d( Construction::Settings const &a_construction, pugi::xml_node const &a_node );
        ~Gridded3d( );

        double domainMin( ) const { return( 0.0 ); }                                        /**< Not properly implemented. */
        double domainMax( ) const { return( 0.0 ); }                                        /**< Not properly implemented. */
        double evaluate( double a_x3, double a_x2, double a_x1 ) const { return( 0.0 ); }   /**< Not properly implemented. */

        std::string const &domain1Unit( ) const { return( m_domain1Unit ); }            /**< Returns the value of the **m_domain1Unit** member. */
        std::string const &domain2Unit( ) const { return( m_domain2Unit ); }            /**< Returns the value of the **m_domain2Unit** member. */
        std::string const &domain3Unit( ) const { return( m_domain3Unit ); }            /**< Returns the value of the **m_domain3Unit** member. */
        std::string const &rangeUnit( ) const { return( m_rangeUnit ); }                /**< Returns the value of the **m_rangeUnit** member. */
        Array3d const &data( ) const { return( m_data ); }                              /**< Returns the value of the **m_data** member. */

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
=========== DoubleDifferentialCrossSection stuff ===========
============================================================
*/
namespace DoubleDifferentialCrossSection {

/*
============================================================
============================= Base =========================
============================================================
*/
class Base : public Form {

    public:
        Base( pugi::xml_node const &a_node, formType a_type, Suite *a_parent );
};

/*
============================================================
================ CoherentPhotoAtomicScattering =============
============================================================
*/
class CoherentPhotoAtomicScattering : public Base {

    private:
        Function1dForm *m_formFactor;                                   /**< The form factor for coherent photo-atomic scattering. */
        Function1dForm *m_realAnomalousFactor;                          /**< The real anomalous factor of coherent photo-atomic scattering. */
        Function1dForm *m_imaginaryAnomalousFactor;                     /**< The imaginary anomalous factor of coherent photo-atomic scattering. */

    public:
        CoherentPhotoAtomicScattering( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs, Suite *a_parent );
        ~CoherentPhotoAtomicScattering( );

        Function1dForm const *formFactor( ) const { return( m_formFactor ); }                               /**< Returns the value of the **m_formFactor** member. */
        Function1dForm const *realAnomalousFactor( ) const { return( m_realAnomalousFactor ); }             /**< Returns the value of the **m_realAnomalousFactor** member. */
        Function1dForm const *imaginaryAnomalousFactor( ) const { return( m_imaginaryAnomalousFactor ); }   /**< Returns the value of the **m_imaginaryAnomalousFactor** member. */
};

/*
============================================================
============== IncoherentPhotoAtomicScattering =============
============================================================
*/
class IncoherentPhotoAtomicScattering : public Base {

    private:
        Function1dForm *m_scatteringFunction;                       /**< The scattering factor for incoherent photo-atomic scattering. */

    public:
        IncoherentPhotoAtomicScattering( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs, Suite *a_parent );
        ~IncoherentPhotoAtomicScattering( );

        Function1dForm const *scatteringFunction( ) const { return( m_scatteringFunction ); }           /**< Returns the value of the **m_scatteringFunction** member. */
};

namespace n_ThermalNeutronScatteringLaw {

/*
============================================================
========================== S_table =========================
============================================================
*/
class S_table : public Form {


    private:
        Function2dForm *m_function2d;           /**< The cumulative scattering factor S(E,T). */

    public:
        S_table( Construction::Settings const &a_construction, pugi::xml_node const &a_node );
        ~S_table( );

        Function2dForm *function2d( ) { return( m_function2d ); }           /**< Returns the value of the **m_function2d** member. */
};

/*
============================================================
====================== CoherentElastic =====================
============================================================
*/
class CoherentElastic : public Base {
    
    private:
        S_table m_S_table;                                                  /**< The S(E,T). */

    public:
        CoherentElastic( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs, Suite *a_parent );
        ~CoherentElastic( );

        S_table const &s_table( ) { return( m_S_table ); }                  /**< Returns the value of the **m_S_table** member. */
};

/*
============================================================
======================== DebyeWaller =======================
============================================================
*/
class DebyeWaller : public Form {


    private:
        Function1dForm *m_function1d;                                   /**< The 1-d function representing the Debye-Waller function W(T). */

    public:
        DebyeWaller( Construction::Settings const &a_construction, pugi::xml_node const &a_node );
        ~DebyeWaller( );

        Function1dForm *function1d( ) { return( m_function1d ); }       /**< Returns the value of the **m_function1d** member. */
};

/*
============================================================
====================== IncoherentElastic =====================
============================================================
*/
class IncoherentElastic : public Base {
    
    private:
        PhysicalQuantity m_characteristicCrossSection;                  /**< The characteristic bound cross section. */
        DebyeWaller m_DebyeWaller;                                      /**< The Debye-Waller function. */

    public:
        IncoherentElastic( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs, Suite *a_parent );
        ~IncoherentElastic( );

        PhysicalQuantity const &characteristicCrossSection( ) { return( m_characteristicCrossSection ); }   /**< Returns the value of the **m_characteristicCrossSection** member. */
        DebyeWaller const &debyeWaller( ) { return( m_DebyeWaller ); }  /**< Returns the value of the **m_DebyeWaller** member. */
};

/*
============================================================
========================== Options =========================
============================================================
*/

class Options : public Form {

    private:
        bool m_calculatedAtThermal;                                     /**< If *true* calculate at 0.0253 eV/k. */
        bool m_asymmetric;                                              /**< If *true* S(alpha,beta) is asymmetric, otherwise it is symmetric. */

    public:
        Options( Construction::Settings const &a_construction, pugi::xml_node const &a_node );
        ~Options( );

        bool calculatedAtThermal( ) { return( m_calculatedAtThermal ); }    /**< Returns the value of the **m_calculatedAtThermal** member. */
        bool asymmetric( ) { return( m_asymmetric ); }                      /**< Returns the value of the **m_asymmetric** member. */
};

/*
============================================================
======================== T_effective =======================
============================================================
*/
class T_effective : public Form {

    private:
        Function1dForm *m_function1d;                               /**< The 1-d function representing effective temperature. */

    public:
        T_effective( Construction::Settings const &a_construction, pugi::xml_node const &a_node );
        ~T_effective( );

        Function1dForm const *function1d( ) const { return( m_function1d ); }   /**< Returns the value of the **m_function1d** member. */
};

/*
============================================================
====================== ScatteringAtom ======================
============================================================
*/
class ScatteringAtom : public Form {

    private:
        PhysicalQuantity m_mass;                                /**< The mass of the atom. */
        PhysicalQuantity m_freeAtomCrossSection;                /**< The free atom scattering cross section. */
        PhysicalQuantity m_e_critical;                          /**< The energy value above which the static model of elastic scattering is adequate. */
        PhysicalQuantity m_e_max;                               /**< The upper energy limit for the constant. */
        T_effective m_T_effective;                              /**< The effective temperatures for the shortcollision-time approximation given as a function of moderator temperature for the atom. */

    public:
        ScatteringAtom( Construction::Settings const &a_construction, pugi::xml_node const &a_node );
        ~ScatteringAtom( );

        PhysicalQuantity const &mass( ) const { return( m_mass ); }                                     /**< Returns the value of the **m_mass** member. */
        PhysicalQuantity const &freeAtomCrossSection( ) const { return( m_freeAtomCrossSection ); }     /**< Returns the value of the **m_freeAtomCrossSection** member. */
        PhysicalQuantity const &e_critical( ) const { return( m_e_critical ); }                         /**< Returns the value of the **m_e_critical** member. */
        PhysicalQuantity const &e_max( ) const { return( m_e_max ); }                                   /**< Returns the value of the **m_e_max** member. */
        T_effective const &t_effective( ) const { return( m_T_effective ); }                            /**< Returns the value of the **m_T_effective** member. */
};

/*
============================================================
======================= S_alpha_beta =======================
============================================================
*/
class S_alpha_beta : public Form {

    private:
        Function3dForm *m_function3d;                           /**< The S(alpha,beta,T) function. */

    public:
        S_alpha_beta( Construction::Settings const &a_construction, pugi::xml_node const &a_node );
        ~S_alpha_beta( );

        Function3dForm *function3d( ) { return( m_function3d ); }                   /**< Returns the value of the **m_function3d** member. */
};

}               // End namespace n_ThermalNeutronScatteringLaw.

}               // End namespace DoubleDifferentialCrossSection.

/*
============================================================
========================= Distribution =====================
============================================================
*/
class Distribution : public Form {

    private:
        frame m_productFrame;                                                   /**< The product frame for the distribution form. */

    public:
        Distribution( std::string const &a_moniker, formType a_type, std::string const &a_label, frame a_productFrame );
        Distribution( pugi::xml_node const &a_node, formType a_type, Suite *a_parent );

        frame productFrame( ) const { return( m_productFrame ); }               /**< Returns the value of the **m_productFrame** member. */
        void toXMLNodeStarter( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
======================= AngularTwoBody =====================
============================================================
*/
class AngularTwoBody : public Distribution {

    private:
        Function2dForm *m_angular;                                              /**< The P(mu|E) distribution as a Function2dForm. */

    public:
        AngularTwoBody( std::string const &a_label, frame a_productFrame, Function2dForm *a_angular = NULL );
        AngularTwoBody( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~AngularTwoBody( );

        Function2dForm const *angular( ) const { return( m_angular ); }         /**< Returns the value of the **m_angular** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
========================= KalbachMann ======================
============================================================
*/
class KalbachMann : public Distribution {

    private:
        Function2dForm *m_f;                                                    /**< The P(E'|E) distribution as a Function2dForm. */
        Function2dForm *m_r;                                                    /**< The Kalbach/Mann r(E,E') function as a Function2dForm. */
        Function2dForm *m_a;                                                    /**< The Kalbach/Mann a(E,E') function as a Function2dForm. */

    public:
        KalbachMann( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~KalbachMann( );

        Function2dForm const *f( ) const { return( m_f ); }                     /**< Returns the value of the **m_f** member. */
        Function2dForm const *r( ) const { return( m_r ); }                     /**< Returns the value of the **m_r** member. */
        Function2dForm const *a( ) const { return( m_a ); }                     /**< Returns the value of the **m_a** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
======================== EnergyAngular =====================
============================================================
*/
class EnergyAngular : public Distribution {

    private:
        Function3dForm *m_energyAngular;                                                /**< The P(E',mu|E) distribution as a Function3dForm. */

    public:
        EnergyAngular( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~EnergyAngular( );

        Function3dForm const *energyAngular( ) const { return( m_energyAngular ); }     /**< Returns the value of the **m_energyAngular** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
======================= EnergyAngularMC ====================
============================================================
*/
class EnergyAngularMC : public Distribution {

    private:
        Function2dForm *m_energy;                                               /**< The P(E'|E) distribution as a Function2dForm. */
        Function3dForm *m_energyAngular;                                        /**< The P(mu|E,E') distribution as a Function3dForm. */

    public:
        EnergyAngularMC( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~EnergyAngularMC( );

        Function2dForm const *energy( ) const { return( m_energy ); }                   /**< Returns the value of the **m_energy** member. */
        Function3dForm const *energyAngular( ) const { return( m_energyAngular ); }     /**< Returns the value of the **m_energyAngular** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
======================== AngularEnergy =====================
============================================================
*/
class AngularEnergy : public Distribution {

    private:
        Function3dForm *m_angularEnergy;                                                /**< The P(mu,E'|E) distribution as a Function3dForm. */

    public:
        AngularEnergy( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~AngularEnergy( );

        Function3dForm const *angularEnergy( ) const { return( m_angularEnergy ); }     /**< Returns the value of the **m_angularEnergy** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
======================= AngularEnergyMC ====================
============================================================
*/
class AngularEnergyMC : public Distribution {

    private:
        Function2dForm *m_angular;                                                      /**< The P(mu|E) distribution as a Function2dForm. */
        Function3dForm *m_angularEnergy;                                                /**< The P(E'|E,mu) distribution as a Function3dForm. */

    public:
        AngularEnergyMC( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~AngularEnergyMC( );

        Function2dForm const *angular( ) const { return( m_angular ); }                 /**< Returns the value of the **m_angular** member. */
        Function3dForm const *angularEnergy( ) const { return( m_angularEnergy ); }     /**< Returns the value of the **m_angularEnergy** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
========================= Uncorrelated =====================
============================================================
*/
class Uncorrelated : public Distribution {

    private:
        Function2dForm *m_angular;                                              /**< The P(mu|E) distribution as a Function2dForm. */
        Function2dForm *m_energy;                                               /**< The P(E'|E) distribution as a Function2dForm. */

    public:
        Uncorrelated( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~Uncorrelated( );

        Function2dForm const *angular( ) const { return( m_angular ); }         /**< Returns the value of the **m_angular** member. */
        Function2dForm const *energy( ) const { return( m_energy ); }           /**< Returns the value of the **m_energy** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
========================= MultiGroup3d =====================
============================================================
*/
class MultiGroup3d : public Distribution {

    private:
        Gridded3d m_gridded3d;                                              /**< The multi-group Legendre distribution as a Gridded3d instance. */

    public:
        MultiGroup3d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );

        Gridded3d const &data( ) const { return( m_gridded3d ); }           /**< Returns the value of the **m_gridded3d** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
====================== LLNLAngularEnergy ===================
============================================================
*/
class LLNLAngularEnergy : public Distribution {

    private:
        Function2dForm *m_angular;                                          /**< The P(mu|E) distribution as a Function2dForm. */
        Function3dForm *m_angularEnergy;                                    /**< The P(E'|E,mu) distribution as a Function3dForm. */

    public:
        LLNLAngularEnergy( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~LLNLAngularEnergy( );

        Function2dForm const *angular( ) const { return( m_angular ); }                 /**< Returns the value of the **m_angular** member. */
        Function3dForm const *angularEnergy( ) const { return( m_angularEnergy ); }     /**< Returns the value of the **m_angularEnergy** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
============== CoherentPhotoAtomicScattering ===============
============================================================
*/
class CoherentPhotoAtomicScattering : public Distribution {

    private:
        std::string m_href;                                                 /**< xlink to the IncoherentPhotoAtomicScattering instance under the *m_doubleDifferentialCrossSection* node. */

    public:
        CoherentPhotoAtomicScattering( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );

        std::string const &href( ) const { return( m_href ); }                          /**< Returns the value of the **m_href** member. */
};

/*
============================================================
============== IncoherentPhotoAtomicScattering =============
============================================================
*/
class IncoherentPhotoAtomicScattering : public Distribution {

    private:
        std::string m_href;                                                 /**< xlink to the IncoherentPhotoAtomicScattering instance under the *m_doubleDifferentialCrossSection* node. */

    public:
        IncoherentPhotoAtomicScattering( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );

        std::string const &href( ) const { return( m_href ); }                          /**< Returns the value of the **m_href** member. */
};

/*
============================================================
=============== ThermalNeutronScatteringLaw ================
============================================================
*/
class ThermalNeutronScatteringLaw : public Distribution {

    private:
        std::string m_href;                                                 /**< xlink to the IncoherentPhotoAtomicScattering instance under the *m_doubleDifferentialCrossSection* node. */

    public:
        ThermalNeutronScatteringLaw( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );

        std::string const &href( ) const { return( m_href ); }                          /**< Returns the value of the **m_href** member. */
};

/*
============================================================
======================== Branching3d =======================
============================================================
*/
class Branching3d : public Distribution {

    public:
        Branching3d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );

};

/*
============================================================
======================= Reference3d ========================
============================================================
*/
class Reference3d : public Distribution {

    private:
        std::string m_href;                                                     /**< Link to the other function. */

    public:
        Reference3d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );

        std::string const &href( ) const { return( m_href ); }                  /**< Returns the value of the **m_xlink** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
========================= Unspecified ======================
============================================================
*/
class Unspecified : public Distribution {

    public:
        Unspecified( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
=========================== Suite ==========================
============================================================
*/
class Suite : public Ancestry {

    public:
        typedef std::vector<Form *> forms;                              /**< The typedef the the *m_forms* member. */

    private:
        forms m_forms;                                                  /**< The list of nodes stored within *this*. */
        std::map<std::string,int> m_map;                                /**< A map of *this* node labels to their index in *m_forms*. */
        Styles::Suite const *m_styles;                                  /**< The Styles::Suite for the Protare that *this* resides in. */
        // FIXME should we make public or private copy constructor?

    public:
        Suite( std::string const &a_moniker );
        Suite( Construction::Settings const &a_construction, std::string const &a_moniker, pugi::xml_node const &a_node, PoPs::Database const &a_pops, 
                        PoPs::Database const &a_internalPoPs, parseSuite a_parseSuite, Styles::Suite const *a_styles );
        ~Suite( );

        std::size_t size( ) const { return( m_forms.size( ) ); }                            /**< Returns the number of node contained by *this*. */
        typedef forms::iterator iterator;
        typedef forms::const_iterator const_iterator;
        iterator begin( ) { return m_forms.begin( ); }                                      /**< The C++ *begin iterator* for *this*. */
        const_iterator begin( ) const { return m_forms.begin( ); }                          /**< The C++ const *begin iterator* for *this*. */
        iterator end( ) { return m_forms.end( ); }                                          /**< The C++ *end iterator* for *this*. */
        const_iterator end( ) const { return m_forms.end( ); }                              /**< The C++ const *end iterator* for *this*. */
        int operator[]( std::string const &a_label ) const ;
        template<typename T> T const *get( std::size_t a_Index ) const ;
        template<typename T> T const *get( std::string const &a_label ) const ;
        template<typename T> T *getNonConst( std::size_t a_Index ) const ;
        template<typename T> T const *getViaLineage( std::string const &a_label ) const ;

        Styles::Suite const *styles( ) const { return( m_styles ); }                            /**< Returns the value of the **m_styles** member. */

        void parse( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs, 
                        parseSuite a_parseSuite, Styles::Suite const *a_styles );
        void add( Form *a_form );
        const_iterator find( std::string const &a_label ) const ;
        bool has( std::string const &a_label ) const { return( find( a_label ) != end( ) ); }

        Ancestry const *findInAncestry3( std::string const &a_item ) const ;
        std::vector<const_iterator> findAllOfMoniker( std::string const &a_moniker ) const ;

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
        void printFormLabels( std::string const &a_header ) const ;
};

/* *********************************************************************************************************//**
 * Returns the node at index *a_index*.
 *
 * @param a_index               [in]    The index of the node to return.
 *
 * @return                              The node at index *a_index*.
 ***********************************************************************************************************/

template<typename T> T const *Suite::get( std::size_t a_index ) const {

    Form const *__form = m_forms[a_index];
    T const *object = dynamic_cast<T const *>( __form );

    if( object == NULL ) throw std::runtime_error( "Suite::get( std::size_t ): invalid cast" );

    return( object );
}

/* *********************************************************************************************************//**
 * Returns the node with label *a_label*.
 *
 * @param a_label               [in]    The label of the node to return.
 *
 * @return                              The node with label *a_label*.
 ***********************************************************************************************************/

template<typename T> T const *Suite::get( std::string const &a_label ) const {

    int index = (*this)[a_label];
    Form *__form = m_forms[index];
    T const *object = dynamic_cast<T const *>( __form );

    if( object == NULL ) throw std::runtime_error( "Suite::get( std::string const & ): invalid cast" );

    return( object );
}

/* *********************************************************************************************************//**
 * Returns the node at index *a_index*.
 *
 * @param a_index               [in]    The index of the node to return.
 *
 * @return                              The node at index *a_index*.
 ***********************************************************************************************************/

template<typename T> T *Suite::getNonConst( std::size_t a_index ) const {

    Form *__form = m_forms[a_index];
    T *object = dynamic_cast<T *>( __form );

    if( object == NULL ) throw std::runtime_error( "invalid cast" );

    return( object );
}

/*
============================================================
=========================== Flux ===========================
============================================================
*/
class Flux : public Form {

    private:
        Function2dForm *m_flux;                                          /**< The flux f(E,mu). */

    public:
        Flux( Construction::Settings const &a_construction, pugi::xml_node const &a_node );
        ~Flux( );

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
========================== Group ===========================
============================================================
*/
class Group : public Form {

    private:
        Grid m_grid;                                                /**< Multi-group boundaries for this Group. */

    public:
        Group( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops );

        std::size_t size( ) const { return( m_grid.size( ) ); }                             /**< Returns the number of multi-group boundaries. */
        inline double &operator[]( std::size_t a_index ) { return( m_grid[a_index] ); }     /**< Returns the multi-group boundary at index *a_index*. */
        std::vector<double> const &data( ) const { return( m_grid.data( ) ); }              /**< Returns the multi-group boundaries. */
        Grid const &grid( ) const { return( m_grid ); }                                     /**< Returns the value of the **m_grid** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
====================== Transportable =======================
============================================================
*/
class Transportable : public Form {

    private:
        std::string m_conserve;                                     /**< Conservation flag for the transfer matrices for this particle. Currently, only "*number*" is allowed. */
        Group m_group;                                              /**< Multi-group boundaries for this Transportable. */

    public:
        Transportable( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, Suite *a_parent );

        std::string pid( ) const { return( label( ) ); }                                    /**< Returns the value of the particle id for the **Transportable**. */
        Group const &group( ) const { return( m_group ); }                                  /**< Returns the value of the **m_group** member. */
        std::vector<double> const &groupBoundaries( ) const { return( m_group.data( ) ); }  /**< Returns the multi-group boundaries for this transportable particle. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
======================= Documentations =====================
============================================================
*/

namespace Documentation {

class Documentation : public Form {

    private:
        std::string m_label;
        std::string m_text;

    public:
        Documentation( pugi::xml_node const &a_node, GIDI::Suite *a_parent );
        ~Documentation( ) { };

        std::string label( ) const { return m_label; }
        std::string text( ) const { return m_text; }

};

class Suite : public GIDI::Suite {

    public:
        Suite( );
        void parse(pugi::xml_node const &a_node);

};

}                     // End of namespace Documentation.

/*
============================================================
========================= Styles stuff =====================
============================================================
*/

namespace Styles {

/*
============================================================
========================== Base ============================
============================================================
*/
class Base : public Form {

    private:
        std::string m_date;                     /**< The GNDS <**date**> attribute. */
        std::string m_derivedStyle;             /**< The GNDS <**derivedFrom**> attribute. */

    public:
        Base( pugi::xml_node const &a_node, GIDI::Suite *a_parent );

        std::string const &date( ) const { return( m_date ); }                      /**< Returns the value of the **m_date** member. */
        std::string const &derivedStyle( ) const { return( m_derivedStyle ); }      /**< Returns the value of the **m_derivedStyle** member. */
        virtual PhysicalQuantity const &temperature( ) const = 0;
        Base const *getDerivedStyle( ) const ;
        Base const *getDerivedStyle( std::string const &a_moniker ) const ;

        std::string baseXMLAttributes( WriteInfo &a_writeInfo ) const ;
};

/*
============================================================
======================== Evaluated =========================
============================================================
*/
class Evaluated : public Base {

    private:
        std::string m_library;                      /**< The GNDS <**library**> attribute. */
        std::string m_version;                      /**< The GNDS <**version**> attribute. */
        PhysicalQuantity m_temperature;             /**< The GNDS <**temperature**> node data. */
        AxisDomain m_projectileEnergyDomain;        /**< The GNDS <**projectileEnergyDomain**> node data. */

    public:
        Evaluated( pugi::xml_node const &a_node, GIDI::Suite *a_parent );

        PhysicalQuantity const &temperature( ) const { return( m_temperature ); }   /**< Returns the value of the **m_temperature** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
================ CrossSectionReconstructed =================
============================================================
*/
class CrossSectionReconstructed : public Base {

    public:
        CrossSectionReconstructed( pugi::xml_node const &a_node, GIDI::Suite *a_parent );

        PhysicalQuantity const &temperature( ) const ;
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
============= CoulombPlusNuclearElasticMuCutoff ============
============================================================
*/
class CoulombPlusNuclearElasticMuCutoff : public Base {

    private:
        double m_muCutoff;                      /**< The GNDS <**muCutoff**> attribute. */

    public:
        CoulombPlusNuclearElasticMuCutoff( pugi::xml_node const &a_node, GIDI::Suite *a_parent );

        PhysicalQuantity const &temperature( ) const ;
        double muCutoff( ) const { return( m_muCutoff ); }          /**< Returns the value of the **m_muCutoff** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
=========================== TNSL ===========================
============================================================
*/
class TNSL : public Base {

    private:
        PhysicalQuantity m_temperature;                                 /**< The GNDS <**temperature**> node data. */

    public:
        TNSL( pugi::xml_node const &a_node, GIDI::Suite *a_parent );
        PhysicalQuantity const & temperature( ) const { return( m_temperature ); }  /**< Returns the value of the **m_temperature** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
=================== AverageProductData =====================
============================================================
*/
class AverageProductData : public Base {

    public:
        AverageProductData( pugi::xml_node const &a_node, GIDI::Suite *a_parent );

        PhysicalQuantity const &temperature( ) const ;
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
===================== MonteCarlo_cdf =======================
============================================================
*/
class MonteCarlo_cdf : public Base {

    public:
        MonteCarlo_cdf( pugi::xml_node const &a_node, GIDI::Suite *a_parent );

        PhysicalQuantity const &temperature( ) const ;
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
======================== MultiGroup ========================
============================================================
*/
class MultiGroup : public Base {

    private:
        int m_maximumLegendreOrder;         /**< The GNDS <**lMax**> attribute. */
        GIDI::Suite m_transportables;       /**< The GNDS <**transportables**> node. */

    public:
        MultiGroup( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs, GIDI::Suite *a_parent );
        ~MultiGroup( );

        int maximumLegendreOrder( ) const { return( m_maximumLegendreOrder ); }     /**< Returns the value of the **m_maximumLegendreOrder** member. */
        PhysicalQuantity const &temperature( ) const ;

        std::vector<double> const &groupBoundaries( std::string const &a_productID ) const ;
        GIDI::Suite const &transportables( ) const { return( m_transportables ); }  /**< Returns the value of the **m_transportables** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
========================= Heated ===========================
============================================================
*/
class Heated : public Base {

    private:
        PhysicalQuantity m_temperature;                                 /**< The GNDS <**temperature**> node data. */

    public:
        Heated( pugi::xml_node const &a_node, GIDI::Suite *a_parent );
        PhysicalQuantity const & temperature( ) const { return( m_temperature ); }  /**< Returns the value of the **m_temperature** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
===================== HeatedMultiGroup =====================
============================================================
*/
class HeatedMultiGroup : public Base {

    private:
        std::string m_parameters;               /**< The GNDS <**parameters**> attribute. */
        Flux m_flux;                            /**< The GNDS <**flux**> node. */
        Gridded1d m_inverseSpeed;               /**< The GNDS <**inverseSpeed**> node data. */

    public:
        HeatedMultiGroup( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, GIDI::Suite *a_parent );
        ~HeatedMultiGroup( );

        std::string const &parameters( ) const { return( m_parameters ); }      /**< Returns the value of the **m_parameters** member. */
        MultiGroup const &multiGroup( ) const ;
        int maximumLegendreOrder( ) const ;
        PhysicalQuantity const &temperature( ) const ;

        std::vector<double> const &groupBoundaries( std::string const &a_productID ) const ;
        Vector inverseSpeed( ) const { return( m_inverseSpeed.data( ) ); }      /**< Returns the value of the **m_inverseSpeed** data. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
// Need methods for m_flux.
};

/*
============================================================
==================== SnElasticUpScatter ====================
============================================================
*/
class SnElasticUpScatter : public Base {

    private:
        int m_upperCalculatedGroup;             /**< The GNDS <**upperCalculatedGroup**> attribute. */

    public:
        SnElasticUpScatter( pugi::xml_node const &a_node, PoPs::Database const &a_pops, GIDI::Suite *a_parent );
        ~SnElasticUpScatter( );

        PhysicalQuantity const &temperature( ) const ;
        int upperCalculatedGroup( ) const { return( m_upperCalculatedGroup ); }     /**< Returns the value of the **m_upperCalculatedGroup** data. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
==================== GriddedCrossSection ===================
============================================================
*/
class GriddedCrossSection : public Base {

    private:
        Grid m_grid;                        /**< The GNDS <**grid**> node. */

    public:
        GriddedCrossSection( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, GIDI::Suite *a_parent );
        ~GriddedCrossSection( );

        PhysicalQuantity const &temperature( ) const ;
        Grid const &grid( ) const { return( m_grid ); }     /**< Returns the value of the **m_grid**. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
=================== URR_probabilityTables ==================
============================================================
*/
class URR_probabilityTables : public Base {

    public:
        URR_probabilityTables( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, GIDI::Suite *a_parent );
        ~URR_probabilityTables( );

        PhysicalQuantity const &temperature( ) const ;
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
========================== Suite ===========================
============================================================
*/
class Suite : public GIDI::Suite {

    public:
        Suite( );

        std::string const *findLabelInLineage( GIDI::Suite const &a_suite, std::string const &a_label ) const ;
        MultiGroup const *multiGroup( std::string const &a_label ) const ;
};

/*
============================================================
===================== TemperatureInfo ======================
============================================================
*/

class TemperatureInfo {

    private:
        PhysicalQuantity m_temperature;                 /**< The temperature for this TemperatureInfo. */
        std::string m_heatedCrossSection;               /**< The label for the **heatedCrossSection** data for this temperature. */
        std::string m_griddedCrossSection;              /**< The label for the **griddedCrossSection** data for this temperature. */
        std::string m_URR_probabilityTables;            /**< The label for the **URR_probabilityTables** data for this temperature. */
        std::string m_heatedMultiGroup;                 /**< The label for the **heatedMultiGroup** data for this temperature. */
        std::string m_SnElasticUpScatter;               /**< The label for the **SnElasticUpScatter** data for this temperature. */

    public:
        TemperatureInfo( );
        TemperatureInfo( PhysicalQuantity const &a_temperature, std::string const &a_heatedCrossSection, std::string const &a_griddedCrossSection,
                std::string const &a_URR_probabilityTables, std::string const &a_heatedMultiGroup, std::string const &a_SnElasticUpScatter );

        PhysicalQuantity const &temperature( ) const { return( m_temperature ); }                   /**< Returns the value of the **m_temperature**. */
        std::string const &heatedCrossSection( ) const { return( m_heatedCrossSection ); }          /**< Returns the value of the **m_heatedCrossSection**. */
        std::string const &griddedCrossSection( ) const { return( m_griddedCrossSection ); }        /**< Returns the value of the **m_griddedCrossSection**. */
        std::string const &URR_probabilityTables( ) const { return( m_URR_probabilityTables ); }    /**< Returns the value of the **m_URR_probabilityTables**. */
        std::string const &heatedMultiGroup( ) const { return( m_heatedMultiGroup ); }              /**< Returns the value of the **m_heatedMultiGroup**. */
        std::string const &SnElasticUpScatter( ) const { return( m_SnElasticUpScatter ); }          /**< Returns the value of the **m_SnElasticUpScatter**. */

        void print( ) const ;
};

typedef std::vector<Styles::TemperatureInfo> TemperatureInfos;

}               // End of namespace Styles.

/*
=========================================================
*/
template<typename T> T const *Suite::getViaLineage( std::string const &a_label ) const {

    std::string const *label = m_styles->findLabelInLineage( (Styles::Suite &) *this, a_label );

    return( get<T>( *label ) );
}

/*
============================================================
======================== Settings stuff ====================
============================================================
*/

namespace Settings {

class ProcessedFlux;
class Settings;

/*
============================================================
========================== MultiGroup ======================
============================================================
*/
class MultiGroup {

    private:
        std::string m_label;                                        /**< The label for the multi-group. */
        std::vector<double> m_boundaries;                           /**< The list of boundaries for the multi-group. */

    public:
        MultiGroup( );
        MultiGroup( std::string const &a_label, int a_length, double const *a_values );
        MultiGroup( std::string const &a_label, std::vector<double> const &a_boundaries );
        MultiGroup( Group const &a_group );
        MultiGroup( MultiGroup const &a_multiGroup );
        ~MultiGroup( );

        double operator[]( int const a_index ) const { return( m_boundaries[a_index] ); }           /**< Returns the multi-group boundary at index *a_index*. */
        std::size_t size( ) const { return( m_boundaries.size( ) ); }                               /**< Returns the number of multi-group boundaries. */
        int numberOfGroups( ) const { return( (int) ( m_boundaries.size( ) - 1 ) ); }               /**< Returns the number of multi-group groups. */
        std::vector<double> const &boundaries( ) const { return( m_boundaries ); }                  /**< Returns the value of the **m_boundaries** member. */
        double const *pointer( ) const { return( &(m_boundaries[0]) ); }                            /**< Returns a pointer to the beginning of the multi-group boundaries. */

        void set( std::string const &a_label, std::vector<double> const &a_boundaries );
        std::string const &label( ) const { return( m_label ); }                                    /**< Returns the value of the **m_label** member. */
        int multiGroupIndexFromEnergy( double a_energy, bool a_encloseOutOfRange ) const ;
        void print( std::string const &a_indent, bool a_outline = false, int a_valuesPerLine = 10 ) const ;
};

/*
============================================================
==================== Groups_from_bdfls =====================
============================================================
*/
class Groups_from_bdfls {

    private:
        std::vector<MultiGroup> m_multiGroups;                                          /**< List of MultiGroup's read in from the bdfls file. */

    public:
        Groups_from_bdfls( std::string const &a_fileName );
        Groups_from_bdfls( char const *a_fileName );
        ~Groups_from_bdfls( );

        MultiGroup viaLabel( std::string const &a_label ) const ;
        MultiGroup getViaGID( int a_gid ) const;
        std::vector<std::string> labels( ) const;
        std::vector<int> GIDs( ) const;
        void print( bool a_outline = true, int a_valuesPerLine = 10 ) const;

    private:
        void initialize( char const *a_fileName );
};

/*
============================================================
========================= Flux_order =======================
============================================================
*/
class Flux_order {

    private:
        int m_order;                        /**< The Legendre order of the flux. */
        std::vector<double> m_energies;     /**< List of flux energies. */
        std::vector<double> m_fluxes;       /**< List of flux values - one for each element of m_energies. */

    public:
        Flux_order( int a_order, int a_length, double const *a_energies, double const *a_fluxes );
        Flux_order( int a_order, std::vector<double> const &a_energies, std::vector<double> const &a_fluxes );
        Flux_order( Flux_order const  &a_fluxOrder  );
        ~Flux_order( );

        int order( ) const { return( m_order ); }                                   /**< Returns the value of the **m_order** member. */
        int size( ) const { return( (int) m_energies.size( ) ); }                   /**< Returns the number of energy, flux pairs. */
        double const *energies( ) const { return( &(m_energies[0]) ); }             /**< Returns a pointer to the beginning of the energy data. */
        std::vector<double> const &v_energies( ) const { return( m_energies ); }    /**< Returns the value of the **m_energies** member. */
        double const *fluxes( ) const { return( &(m_fluxes[0]) ); }                 /**< Returns a pointer to the beginning of the flux data. */
        std::vector<double> const &v_fluxes( ) const { return( m_fluxes ); }        /**< Returns the value of the **m_fluxes** member. */
        void print( int a_valuesPerLine = 10 ) const;
};

/*
============================================================
============================ Flux ==========================
============================================================
*/
class Flux {

    private:
        std::string m_label;                        /**< Label for the flux. */
        double m_temperature;                       /**< Temperature of the material that produced this flux. */
        std::vector<Flux_order> m_fluxOrders;       /**< List of fluxes for each Legendre order, *l*, sorted by Legendre order starting with *l* = 0. */

    public:
        Flux( std::string const &a_label, double a_temperature_MeV );
        Flux( char const *a_label, double a_temperature_MeV );
        Flux( Flux const &a_flux );
        ~Flux( );

        Flux_order const &operator[]( int a_order ) const { return( m_fluxOrders[a_order] ); }  /**< Returns the Flux_order for Legendre order *a_order*. */
        int maxOrder( ) const { return( (int) m_fluxOrders.size( ) - 1 ); }                     /**< Returns the maximum number of Legendre orders for *this*. */
        int size( ) const { return( (int) m_fluxOrders.size( ) ); }                             /**< Returns the number of stored Legendre orders. */

        std::string const &label( ) const { return( m_label ); }                                /**< Returns the value of the **m_label** member. */
        double temperature( ) const { return( m_temperature ); }                                /**< Returns the value of the **m_temperature** member. */
        void addFluxOrder( Flux_order const &a_fluxOrder );
        ProcessedFlux process( std::vector<double> const &a_multiGroup ) const ;
        void print( std::string const &a_indent, bool a_outline = true, int a_valuesPerLine = 10 ) const ;
};

/*
============================================================
===================== Fluxes_from_bdfls ====================
============================================================
*/
class Fluxes_from_bdfls {

    private:
        std::vector<Flux> m_fluxes;                     /**< The list of Flux read in from the *bdfls* file. */

    public:
        Fluxes_from_bdfls( std::string const &a_fileName, double a_temperature_MeV );
        Fluxes_from_bdfls( char const *a_fileName, double a_temperature_MeV );
        ~Fluxes_from_bdfls( );

        Flux getViaFID( int a_fid ) const ;
        std::vector<std::string> labels( ) const ;
        std::vector<int> FIDs( ) const ;
        void print( bool a_outline = true, int a_valuesPerLine = 10 ) const ;

    private:
        void initialize( char const *a_fileName, double a_temperature_MeV );
};

/*
============================================================
======================= ProcessedFlux ======================
============================================================
*/
class ProcessedFlux {

    private:
        double m_temperature;                                           /**< The temperature of the material that produced the flux. */
        std::vector<double> m_multiGroupFlux;                           /**< The Legendre order = 0 multi-grouped flux. */

    public:
        ProcessedFlux( double a_temperature, std::vector<double> const &a_multiGroupFlux );
        ProcessedFlux( ProcessedFlux const &a_processedFlux );
        ~ProcessedFlux( );

        double temperature( ) const { return( m_temperature ); }                            /**< Returns the value of the **m_temperature** member. */
        std::vector<double> const &multiGroupFlux( ) const { return( m_multiGroupFlux ); }  /**< Returns the value of the **m_multiGroupFlux** member. */
};

/*
============================================================
========================= Particle =========================
============================================================
*/
class Particle {

    private:
        std::string m_pid;                                                  /**< The PoPs id for the particle. */
        MultiGroup m_multiGroup;                                            /**< Coarse multi-group to collapse to. */
        MultiGroup m_fineMultiGroup;                                        /**< Fine multi-group to collapse to. */
        std::vector<int> m_collapseIndices;                                 /**< Indices for collapsing to m_multiGroup. */
        std::vector<Flux> m_fluxes;                                         /**< One flux for each temperature. */
        std::vector<ProcessedFlux> m_processedFluxes;                       /**< One processed flux for each temperature. */

    public:
        Particle( std::string const &m_pid, MultiGroup const &a_multiGroup );
        Particle( Particle const &a_particle );
        ~Particle( );

        std::string const &pid( ) const { return( m_pid ); }                                 /**< Returns the value of the **m_pid** member. */
        int multiGroupIndexFromEnergy( double a_e_in, bool a_encloseOutOfRange ) const { return( m_multiGroup.multiGroupIndexFromEnergy( a_e_in, a_encloseOutOfRange ) ); };
                                                                                            /**< Returns the coarse multi-group index corresponding to energy *a_e_in*. See MultiGroup::multiGroupIndexFromEnergy. */
        int numberOfGroups( ) const { return( m_multiGroup.numberOfGroups( ) ); };          /**< Returns the number of coarse multi-group groups. */
        MultiGroup multiGroup( ) const { return( m_multiGroup ); }                          /**< Returns the value of the **m_multiGroup** member. */
        MultiGroup fineMultiGroup( ) const { return( m_fineMultiGroup ); }                  /**< Returns the value of the **m_fineMultiGroup** member. */
        int appendFlux( Flux const &a_flux );
        ProcessedFlux const *nearestProcessedFluxToTemperature( double a_temperature ) const;
        std::vector<int> const &collapseIndices( ) const { return( m_collapseIndices ); }   /**< Returns the value of the **m_collapseIndices** member. */

        void process( std::vector<double> const &a_boundaries, double a_epsilon = 1e-6 );
        void print( std::string const &a_indent ) const ;
};

/*
============================================================
======================== Particles =========================
============================================================
*/
class Particles {

    private:
        std::map<std::string, Particle> m_particles;

    public:
        Particles( );
        ~Particles( );

        std::map<std::string, Particle> &particles( ) { return( m_particles ); }                /**< Returns the value of the **m_particles** member. */
        std::map<std::string, Particle> const &particles( ) const { return( m_particles ); }    /**< Returns the value of the **m_particles** member. */
        Particle const *particle( std::string const &a_particleID ) const;
        bool add( Particle const &a_particle );
        bool remove( std::string const &a_particleID );
        void clear( ) { m_particles.clear( ); }
        bool hasParticle( std::string const &a_id ) const ;

        void process( Protare const &a_protare, std::string const &a_label );

        std::vector<std::string> sortedIDs( bool a_orderIsAscending = true ) const ;

        void print( ) const ;
};

/*
============================================================
========================= Settings =========================
============================================================
*/
class Settings {

    private:
        std::string m_projectileID;                                 /**< The PoPs id of the projectile. */
        std::string m_label;                                        /**< The label in the styles to get data from. */
        bool m_delayedNeutrons;                                     /**< If true, include delayed neutrons when returning or setting up data. */

    public:
        Settings( std::string const &a_projectileID, std::string const &a_label, bool a_delayedNeutrons );
        ~Settings( );

        std::string const &projectileID( ) const { return( m_projectileID ); }              /**< Returns the value of the **m_projectileID** member. */

        std::string const &label( ) const { return( m_label ); }                            /**< Returns the value of the **m_label** member. */
        void label( std::string const &a_label ) { m_label = a_label; }                     /**< Sets the **m_label** member to **a_label*. */

        bool delayedNeutrons( ) const { return( m_delayedNeutrons ); }                      /**< Returns the value of the **m_delayedNeutrons** member. */
        void delayedNeutrons( bool a_delayedNeutrons ) { m_delayedNeutrons = a_delayedNeutrons; }    /**< Sets the **m_delayedNeutrons** member to **a_delayedNeutrons*. */

        Vector multiGroupZeroVector( Particles const &a_particles, bool a_collapse = true ) const ;
        Matrix multiGroupZeroMatrix( Particles const &a_particles, std::string const &a_particleID, bool a_collapse = true ) const ;

        void print( ) const ;
};

/*
============================================================
============================ MG ============================
============================================================
*/
class MG : public Settings {

    public:
        MG( std::string const &a_projectileID, std::string const &a_label, bool a_delayedNeutrons );

};

}           // End of namespace Settings.

/*
============================================================
========================= Product ==========================
============================================================
*/
class Product : public Form {

    private:
        ParticleInfo m_particle;                    /**< The products **ParticleInfo** data. */

        int m_productMultiplicity;                  /**< Product multiplicity (e.g., 0, 1, 2, ...) or -1 if energy dependent or not an integer for particle with id *a_id*. */
        Suite m_multiplicity;                       /**< The GNDS <**multiplicity**> node. */
        Suite m_distribution;                       /**< The GNDS <**distribution**> node. */
        Suite m_averageEnergy;                      /**< The GNDS <**averageEnergy**> node. */
        Suite m_averageMomentum;                    /**< The GNDS <**averageMomentum**> node. */
        OutputChannel *m_outputChannel;             /**< The GNDS <**outputChannel**> node if present. */

    public:
        Product( PoPs::Database const &a_pops, std::string const &a_productID, std::string const &a_label );
        Product( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs, Suite *a_parent, Styles::Suite const *a_styles );
        ~Product( );

        ParticleInfo const &particle( ) const { return( m_particle ); }                 /**< Returns the value of the **m_particle** member. */
        void particle( ParticleInfo const &a_particle ) { m_particle = a_particle; }    /**< Sets **m_particle** to *a_particle*. */
        int depth( ) const ;

        Suite const &multiplicity( ) const { return( m_multiplicity ); }                /**< Returns a *const* reference to the **m_multiplicity** member. */
        Suite &multiplicity( ) { return( m_multiplicity ); }                            /**< Returns a reference to the **m_multiplicity** member. */
        Suite const &distribution( ) const { return( m_distribution ); }                /**< Returns a *const* reference to the **m_distribution** member. */
        Suite &distribution( ) { return( m_distribution ); }                            /**< Returns a reference to the **m_distribution** member. */
        Suite const &averageEnergy( ) const { return( m_averageEnergy ); }              /**< Returns a *const* reference to the **m_averageEnergy** member. */
        Suite const &averageMomentum( ) const { return( m_averageMomentum ); }          /**< Returns a *const* reference to the **m_averageMomentum** member. */
        OutputChannel const *outputChannel( ) const { return( m_outputChannel ); }      /**< Returns a *const* reference to the **m_outputChannel** member. */

        bool hasFission( ) const ;
        Ancestry const *findInAncestry3( std::string const &a_item ) const ;
        void productIDs( std::set<std::string> &a_ids, Settings::Particles const &a_particles, bool a_transportablesOnly ) const ;
        int productMultiplicity( std::string const &a_id ) const ;
        int maximumLegendreOrder( Settings::MG const &a_settings, std::string const &a_productID ) const ;

        Vector multiGroupQ( Settings::MG const &a_settings, Settings::Particles const &a_particles, bool a_final ) const ;
        Vector multiGroupMultiplicity( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;
        Matrix multiGroupProductMatrix( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID, int a_order ) const ;

        Vector multiGroupAverageEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;
        Vector multiGroupAverageMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;

        void continuousEnergyProductData( std::string const &a_particleID, double a_energy, double &a_productEnergy, double &a_productMomentum, double &a_productGain ) const ;

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
====================== DelayedNeutron ======================
============================================================
*/
class DelayedNeutron : public Form {

    private:
        int m_delayedNeutronIndex;                  /**< If this is a delayed fission neutron, this is its index. */
        Suite m_rate;                               /**< The GNDS <**rate**> node. */
        Product m_product;                          /**< The GNDS <**product**> node. */

    public:
        DelayedNeutron( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs, Suite *a_parent, Styles::Suite const *a_styles );
        ~DelayedNeutron( );

        int delayedNeutronIndex( ) const { return( m_delayedNeutronIndex ); };
        void delayedNeutronIndex( int a_delayedNeutronIndex ) { m_delayedNeutronIndex = a_delayedNeutronIndex; }
        Suite const &rate( ) const { return( m_rate ); }
        Product const &product( ) const { return( m_product ); }

        Ancestry const *findInAncestry3( std::string const &a_item ) const ;

        void productIDs( std::set<std::string> &a_indices, Settings::Particles const &a_particles, bool a_transportablesOnly ) const ;
        int productMultiplicity( std::string const &a_id ) const ;
        int maximumLegendreOrder( Settings::MG const &a_settings, std::string const &a_productID ) const ;
        Vector multiGroupQ( Settings::MG const &a_settings, Settings::Particles const &a_particles, bool a_final ) const ;
        Vector multiGroupMultiplicity( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;
        Matrix multiGroupProductMatrix( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID, int a_order ) const ;
        Vector multiGroupAverageEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;
        Vector multiGroupAverageMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
==================== FissionFragmentData ===================
============================================================
*/
class FissionFragmentData : public Ancestry {

    private:
        Suite m_delayedNeutrons;                            /**< The GNDS <**delayedNeutrons**> node. */
        Suite m_fissionEnergyReleases;                      /**< The GNDS <**fissionEnergyReleases**> node. */

    public:
        FissionFragmentData( );
        FissionFragmentData( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs, Styles::Suite const *a_styles );
        ~FissionFragmentData( );

        Suite const &delayedNeutrons( ) const { return( m_delayedNeutrons ); }
        Suite const &fissionEnergyReleases( ) const { return( m_fissionEnergyReleases ); }

        Ancestry const *findInAncestry3( std::string const &a_item ) const ;

        void productIDs( std::set<std::string> &a_indices, Settings::Particles const &a_particles, bool a_transportablesOnly ) const ;
        int productMultiplicity( std::string const &a_id ) const ;
        int maximumLegendreOrder( Settings::MG const &a_settings, std::string const &a_productID ) const ;
        Vector multiGroupQ( Settings::MG const &a_settings, Settings::Particles const &a_particles, bool a_final ) const ;
        Vector multiGroupMultiplicity( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;
        Matrix multiGroupProductMatrix( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID, int a_order ) const ;
        Vector multiGroupAverageEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;
        Vector multiGroupAverageMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
======================= OutputChannel ======================
============================================================
*/
class OutputChannel : public Ancestry {

    private:
        bool m_twoBody;                                     /**< true if the output channel is two-body and false otherwise. */
        bool m_fissions;                                    /**< true if the output channel is a fission channel and false otherwise. */
        std::string m_process;                              /**< The GNDS **process** attribute for the channel. */

        Suite m_Q;                                          /**< The GNDS <**Q**> node. */
        Suite m_products;                                   /**< The GNDS <**products**> node. */
        FissionFragmentData m_fissionFragmentData;          /**< The GNDS <**fissionFragmentData**> node. */

    public:
        OutputChannel( bool a_twoBody, bool a_fissions, std::string a_process );
        OutputChannel( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs, Styles::Suite const *a_styles );
        ~OutputChannel( );

        bool twoBody( ) const { return( m_twoBody ); }                              /**< Returns the value of the **m_twoBody** member. */
        std::string process( ) const { return( m_process ); }
        int depth( ) const ;

        Suite const &Q( ) const { return( m_Q ); }                                  /**< Returns a *const* reference to the **m_Q** member. */
        Suite &Q( ) { return( m_Q ); }                                              /**< Returns a *const* reference to the **m_Q** member. */
        Suite const &products( ) const { return( m_products ); }                    /**< Returns a *const* reference to the **m_products** member. */
        Suite &products( ) { return( m_products ); }                    /**< Returns a *const* reference to the **m_products** member. */
        FissionFragmentData const &fissionFragmentData( ) const { return( m_fissionFragmentData ); }

        Ancestry const *findInAncestry3( std::string const &a_item ) const ;

        bool isFission( ) const { return( m_fissions ); }                           /**< Returns true if the output channel is a fission output channel. */
        bool hasFission( ) const ;
        void productIDs( std::set<std::string> &a_ids, Settings::Particles const &a_particles, bool a_transportablesOnly ) const ;
        int productMultiplicity( std::string const &a_id ) const ;
        int maximumLegendreOrder( Settings::MG const &a_settings, std::string const &a_productID ) const ;

        Vector multiGroupQ( Settings::MG const &a_settings, Settings::Particles const &a_particles, bool a_final ) const ;
        Vector multiGroupMultiplicity( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;
        Matrix multiGroupProductMatrix( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID, int order ) const ;
        Vector multiGroupAverageEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;
        Vector multiGroupAverageMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;

        void continuousEnergyProductData( std::string const &a_particleID, double a_energy, double &a_productEnergy, double &a_productMomentum, double &a_productGain ) const ;

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

namespace DoubleDifferentialCrossSection {

namespace n_ThermalNeutronScatteringLaw {

/*
============================================================
==================== IncoherentInelastic ===================
============================================================
*/
class IncoherentInelastic : public Base {
    
    private:
        Options m_options;                              /**< Options for *this*. */
        Suite m_scatteringAtoms;                        /**< The list of atoms and there information. */
        S_alpha_beta m_S_alpha_beta;                    /**< The S(alpha,beta,T) function. */
    
    public:
        IncoherentInelastic( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs, Suite *a_parent );
        ~IncoherentInelastic( );
        
        Options const &options( ) const { return( m_options ); }                        /**< Returns the value of the **m_options** */
        Suite const &scatteringAtoms( ) const { return( m_scatteringAtoms ); }          /**< Returns the value of the **m_scatteringAtoms** */
        S_alpha_beta const &s_alpha_beta( ) const { return( m_S_alpha_beta ); }         /**< Returns the value of the **m_S_alpha_beta** */
};

}               // End namespace n_ThermalNeutronScatteringLaw.

}               // End namespace DoubleDifferentialCrossSection.

/*
============================================================
========================= Reaction =========================
============================================================
*/
class Reaction : public Form {

    private:
        int m_ENDF_MT;                                  /**< The ENDF MT value for the reaction. */
        int m_ENDL_C;                                   /**< The ENDL C value for the reaction. */
        int m_ENDL_S;                                   /**< The ENDL S value for the reaction. */
        std::string m_fissionGenre;                     /**< If the reaction is fission, this is its genre. */
        double m_QThreshold;                            /**< Threshold value calculated from the Q and the protare's m_thresholdFactor. */
        double m_crossSectionThreshold;                 /**< Threshold value derived from cross section data via *evaluated* or *griddedCrossSection*. */
        bool m_isPairProduction;                        /**< Kludge! Currently needed because GNDS specification unclear about how to specify photo-atomic pair production reaction. */

        Suite m_doubleDifferentialCrossSection;         /**< The GNDS <**doubleDifferentialCrossSection**> node. */
        Suite m_crossSection;                           /**< The GNDS <**crossSection**> node. */
        Suite m_availableEnergy;                        /**< The GNDS <**availableEnergy**> node. */
        Suite m_availableMomentum;                      /**< The GNDS <**availableMomentum**> node. */
        OutputChannel *m_outputChannel;                 /**< The reaction's output channel. */

    public:
        Reaction( int a_ENDF_MT, std::string a_fissionGenre );
        Reaction( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs, Protare const &a_protare,
                        Styles::Suite const *a_styles );
        ~Reaction( );

        int depth( ) const { return( m_outputChannel->depth( ) ); }                     /**< Returns the maximum product depth for this reaction. */
        int ENDF_MT( ) const { return( m_ENDF_MT ); }                                   /**< Returns the value of the **m_ENDF_MT** member. */
        int ENDL_C( ) const { return( m_ENDL_C ); }                                     /**< Returns the value of the **m_ENDL_C** member. */
        int ENDL_S( ) const { return( m_ENDL_S ); }                                     /**< Returns the value of the **m_ENDL_S** member. */
        bool isPairProduction( ) const { return( m_isPairProduction ); }                /**< Returns the value of the **m_isPairProduction** member. */

        Suite const &availableEnergy( ) const { return( m_availableEnergy ); }     /**< Returns a *const* reference to the **m_availableEnergy** member. */
        Suite const &availableMomentum( ) const { return( m_availableMomentum ); } /**< Returns a *const* reference to the **m_availableMomentum** member. */

        Suite const &doubleDifferentialCrossSection( ) const { return( m_doubleDifferentialCrossSection ); }    /**< Returns a *const* reference to the **m_doubleDifferentialCrossSection** member. */
        Suite const &crossSection( ) const { return( m_crossSection ); }           /**< Returns a *const* reference to the **m_crossSection** member. */
        Suite &crossSection( ) { return( m_crossSection ); }                        /**< Returns a reference to the **m_crossSection** member. */
        OutputChannel const *outputChannel( ) const { return( m_outputChannel ); }      /**< Returns a *const* reference to the **m_outputChannel** member. */
        void outputChannel( OutputChannel *a_outputChannel );

        Ancestry const *findInAncestry3( std::string const &a_item ) const ;
        std::string xlinkItemKey( ) const { return( Ancestry::buildXLinkItemKey( "label", label( ) ) ); }   /**< Returns the value of the **** member. */

        bool hasFission( ) const ;
        void productIDs( std::set<std::string> &a_ids, Settings::Particles const &a_particles, bool a_transportablesOnly ) const ;
        int productMultiplicity( std::string const &a_id ) const {
                return( m_outputChannel->productMultiplicity( a_id ) ); }               /**< Returns the product multiplicity (e.g., 0, 1, 2, ...) or -1 if energy dependent or not an integer. */
        int maximumLegendreOrder( Settings::MG const &a_settings, std::string const &a_productID ) const ;

        double threshold( ) const { return( m_QThreshold ); }                           /**< Returns the value of the **m_QThreshold** member. */
        double crossSectionThreshold( ) const { return( m_crossSectionThreshold ); }    /**< Returns the value of the **m_crossSectionThreshold** member. */

        Vector multiGroupCrossSection( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;
        Vector multiGroupQ( Settings::MG const &a_settings, Settings::Particles const &a_particles, bool a_final ) const {
                return( m_outputChannel->multiGroupQ( a_settings, a_particles, a_final ) ); }                  /**< Returns the multi-group, total Q for the requested label. This is a cross section weighted Q summed over all reactions. */
        Vector multiGroupMultiplicity( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;

        Matrix multiGroupProductMatrix( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID, int a_order ) const ;
        Matrix multiGroupFissionMatrix( Settings::MG const &a_settings, Settings::Particles const &a_particles, int a_order ) const ;

        Vector multiGroupAvailableEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;
        Vector multiGroupAverageEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;
        Vector multiGroupDepositionEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;

        Vector multiGroupAvailableMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;
        Vector multiGroupAverageMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;
        Vector multiGroupDepositionMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;

        Vector multiGroupGain( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID, std::string const &a_projectileID ) const ;

        void continuousEnergyProductData( std::string const &a_particleID, double a_energy, double &a_productEnergy, double &a_productMomentum, double &a_productGain ) const ;

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

namespace Sums {

namespace Summand {

/*
============================================================
=========================== Base ===========================
============================================================
*/
class Base : public Ancestry {

    private:
        std::string m_href;                                                     /**< xlink for the summand. */

    public:
        Base( Construction::Settings const &a_construction, pugi::xml_node const &a_node );
        ~Base( );

        std::string const &href( ) const { return( m_href ); }                  /**< Returns the value of the **m_href** member. */
        Ancestry const *findInAncestry3( std::string const &a_item ) const { return( NULL ); }

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
=========================== Add ============================
============================================================
*/
class Add : public Base {

    public:
        Add( Construction::Settings const &a_construction, pugi::xml_node const &a_node );
};

}           // End of namespace Summand.

/*
============================================================
========================= Summands =========================
============================================================
*/
class Summands : public Form {

    private:
        std::vector<Summand::Base *> m_summands;                            /**< List of summand for *this*. */

    public:
        Summands( Construction::Settings const &a_construction, pugi::xml_node const &a_node );
        ~Summands( );

        std::size_t size( ) const { return( m_summands.size( ) ); }         /**< Returns the number of summands in *this*. */
        Summand::Base const *operator[]( std::size_t a_index ) const { return( m_summands[a_index] ); } /**< Returns the summand at index *a_index*. */

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
=========================== Base ===========================
============================================================
*/
class Base : public Form {

    private:
        int m_ENDF_MT;                                                      /**< ENDF MT value for the sum. */
        Summands m_summands;                                                /**< List of Summands for *this*. */

    public:
        Base( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs,
                formType a_type );

        int ENDF_MT( ) const { return( m_ENDF_MT ); }                       /**< Returns the value of the **m_ENDF_MT** member. */
        Summands const &summands( ) const { return( m_summands ); }         /**< Returns the value of the **m_summands** member. */
};

/*          
============================================================
====================== CrossSectionSum =====================
============================================================
*/
class CrossSectionSum : public Base {

    private:
        Suite m_Q;                                                          /**< The GNDS <**Q**> node. */
        Suite m_crossSection;                                               /**< The GNDS <**crossSection**> node. */

    public:
        CrossSectionSum( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs );
        Ancestry const *findInAncestry3( std::string const &a_item ) const ;

        Suite const &Q( ) const { return( m_Q ); }                         /**< Returns a *const* reference to the **m_Q** member. */
        Suite const &crossSection( ) const { return( m_crossSection ); }   /**< Returns a *const* reference to the **m_crossSection** member. */

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*          
============================================================
====================== MultiplicitySum =====================
============================================================
*/
class MultiplicitySum : public Base {

    private:
        Suite m_multiplicity;                                               /**< The GNDS <**multiplicity**> node. */

    public:
        MultiplicitySum( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs );

        Suite const &multiplicity( ) const { return( m_multiplicity ); }   /**< Returns a *const* reference to the **m_multiplicity** member. */

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
=========================== Sums ===========================
============================================================
*/
class Sums : public Ancestry {

    private:
        Suite m_crossSections;                                              /**< The GNDS <**crossSections**> node. */
        Suite m_multiplicities;                                             /**< The GNDS <**multiplicities**> node. */

    public:
        Sums( );
        ~Sums( );

        Suite const &crossSections( ) const { return( m_crossSections ); }      /**< Returns the value of the **m_crossSections** member. */
        Suite const &multiplicities( ) const { return( m_multiplicities ); }    /**< Returns the value of the **m_multiplicities** member. */

        void parse( Construction::Settings const &a_construction, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs );
        Ancestry const *findInAncestry3( std::string const &a_item ) const ;

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

}           // End of namespace Sums.

/*
============================================================
========================== Protare =========================
============================================================
*/
class Protare : public Ancestry {

    private:
        ParticleInfo m_projectile;              /**< Information about the projectile. */
        ParticleInfo m_target;                  /**< Information about the target. */

    protected:
        void initialize( pugi::xml_node const &a_protare, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs, bool a_targetRequiredInGlobalPoPs, 
                        bool a_requiredInPoPs = true );

    public:
        Protare( );
        ~Protare( );

        ParticleInfo const &projectile( ) const { return( m_projectile ); }         /**< Returns the value of the **m_projectile** member. */
        void projectile( ParticleInfo const &a_projectile ) { m_projectile = a_projectile; }    /**< Sets **m_projectile** to *a_projectile*. */
        ParticleInfo const &target( ) const { return( m_target ); }                 /**< Returns the value of the **m_target** member. */
        void target( ParticleInfo const &a_target ) { m_target = a_target; }        /**< Sets **m_target** to *a_target*. */

        virtual bool isSingleton( ) const { return( false ); }                      /**< Returns *true* if the instance is a ProtareSingleton instance and *false* otherwise. */
        virtual bool isComposite( ) const { return( false ); }                      /**< Returns *true* if the instance is a ProtareComposite instance and *false* otherwise. */
        virtual bool isTNSL( ) const { return( false ); }                           /**< Returns *true* if the instance is a ProtareTNSL instance and *false* otherwise. */
        virtual bool isTNSL_ProtareSingleton( ) const { return( false ); }          /**< Returns *true* if the instance is a ProtareSingleton instance with TNSL data and *false* otherwise. */
        virtual std::size_t numberOfProtares( ) const = 0;                          /**< Returns the number of protares contained in *this*. */
        virtual ProtareSingleton const *protare( std::size_t a_index ) const = 0;   /**< Returns the **a_index** - 1 Protare contained in *this*. */

        virtual std::string const &formatVersion( std::size_t a_index = 0 ) const = 0;
        virtual std::string const &fileName( std::size_t a_index = 0 ) const = 0;
        virtual std::string const &realFileName( std::size_t a_index = 0 ) const = 0;

        virtual std::vector<std::string> libraries( std::size_t a_index = 0 ) const = 0;
        virtual std::string const &evaluation( std::size_t a_index = 0 ) const = 0;
        virtual frame projectileFrame( std::size_t a_index = 0 ) const = 0;
        virtual double thresholdFactor( ) const = 0;

        virtual Documentation::Suite const &documentations( ) const = 0;

        virtual Styles::Base const &style( std::string const a_label ) const = 0;
        virtual Styles::Suite const &styles( ) const = 0;

        virtual void productIDs( std::set<std::string> &a_ids, Settings::Particles const &a_particles, bool a_transportablesOnly ) const = 0;
        virtual int maximumLegendreOrder( Settings::MG const &a_settings, std::string const &a_productID ) const = 0;

        virtual Styles::TemperatureInfos temperatures( ) const  = 0;

        virtual std::size_t numberOfReactions( ) const = 0;
        virtual Reaction const *reaction( std::size_t a_index ) const = 0;
        virtual std::size_t numberOfOrphanProducts( ) const = 0;
        virtual Reaction const *orphanProduct( std::size_t a_index ) const = 0;

        virtual bool hasFission( ) const = 0;

        virtual Ancestry const *findInAncestry3( std::string const &a_item ) const = 0;

        virtual Styles::MultiGroup const *multiGroup( std::string const &a_label ) const = 0;
        virtual std::vector<double> const &groupBoundaries( Settings::MG const &a_settings, std::string const &a_productID ) const = 0;
        virtual Vector multiGroupInverseSpeed( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const = 0;

        virtual Vector multiGroupCrossSection( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const = 0;
        virtual Vector multiGroupQ( Settings::MG const &a_settings, Settings::Particles const &a_particles, bool a_final ) const = 0;

        virtual Vector multiGroupMultiplicity( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const = 0;
        virtual Vector multiGroupFissionNeutronMultiplicity( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const = 0;

        virtual Matrix multiGroupProductMatrix( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID, int a_order ) const = 0;
        virtual Matrix multiGroupFissionMatrix( Settings::MG const &a_settings, Settings::Particles const &a_particles, int a_order ) const = 0;
        virtual Vector multiGroupTransportCorrection( Settings::MG const &a_settings, Settings::Particles const &a_particles, int a_order, transportCorrectionType a_transportCorrectionType, double a_temperature ) const = 0;

        virtual Vector multiGroupAvailableEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const = 0;
        virtual Vector multiGroupAverageEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const = 0;
        virtual Vector multiGroupDepositionEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const = 0;

        virtual Vector multiGroupAvailableMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const = 0;
        virtual Vector multiGroupAverageMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const = 0;
        virtual Vector multiGroupDepositionMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const = 0;

        virtual Vector multiGroupGain( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const = 0;

        virtual stringAndDoublePairs muCutoffForCoulombPlusNuclearElastic( ) const = 0;
};

/*
============================================================
===================== ProtareSingleton =====================
============================================================
*/
class ProtareSingleton : public Protare {

    private:
        std::string m_formatVersion;            /**< The GNDS format version. */
        PoPs::Database m_internalPoPs;          /**< The **PoPs** specified under the protare (e.g., reactionSuite) node. */

        std::vector<std::string> m_libraries;   /**< The list of libraries *this* was found in. */
        std::string m_evaluation;               /**< The protare's evaluation string. */
        std::string m_fileName;                 /**< The path to the protare's file. May be relative. */
        std::string m_realFileName;             /**< The real path to the protare's file. Equivalent to the value returned by the C-function *realpath( m_fileName )* on Unix systems. */
        frame m_projectileFrame;                /**< The frame the projectile data are given in. */
        bool m_isTNSL_ProtareSingleton;         /**< If *this* is a ProtareSingleton instance with TNSL data *true* and otherwise *false*. */

        double m_thresholdFactor;               /**< The non-relativistic factor that converts a Q-value into a threshold. */

        PoPs::NuclideGammaBranchStateInfos m_nuclideGammaBranchStateInfos;  /**< Simplified list of gamma branching data from nuclide level decays derived from the internal PoPs::Database. */

        Documentation::Suite m_documentations;  /**< The GNDS <**documentations**> node. */
        Styles::Suite m_styles;                 /**< The GNDS <**styles**> node. */
        Suite m_reactions;                      /**< The GNDS <**reactions**> node. */
        Suite m_orphanProducts;                 /**< The GNDS <**orphanProducts**> node. */

        Sums::Sums m_sums;                      /**< The GNDS <**sums**> node. */
        Suite m_fissionComponents;              /**< The GNDS <**fissionComponents**> node. */

        void initialize( Construction::Settings const &a_construction, pugi::xml_node const &a_protare, PoPs::Database const &a_pops, bool a_targetRequiredInGlobalPoPs,
                        bool a_requiredInPoPs = true );

    public:
        ProtareSingleton( );
        ProtareSingleton( PoPs::Database const &a_pops, std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_evaluation );
        ProtareSingleton( Construction::Settings const &a_construction, std::string const &a_fileName, fileType a_fileType, PoPs::Database const &a_pops, 
                std::vector<std::string> const &a_libraries, bool a_targetRequiredInGlobalPoPs = true, bool a_requiredInPoPs = true );
        ProtareSingleton( Construction::Settings const &a_construction, pugi::xml_node const &a_protare, PoPs::Database const &a_pops, 
                std::vector<std::string> const &a_libraries, bool a_targetRequiredInGlobalPoPs = true, bool a_requiredInPoPs = true );
        ~ProtareSingleton( );

        PoPs::NuclideGammaBranchStateInfos const &nuclideGammaBranchStateInfos( ) const { return( m_nuclideGammaBranchStateInfos ); }
                                                                                    /**< Returns the value of the **m_nuclideGammaBranchStateInfos** member. */

        Suite &reactions( ) { return( m_reactions ); }                              /**< Returns the value of the **m_reactions** member. */
        Suite const &reactions( ) const { return( m_reactions ); }                  /**< Returns the value of the **m_reactions** member. */
        Suite const &orphanProducts( ) const { return( m_orphanProducts ); }        /**< Returns the value of the **m_orphanProducts** member. */
        Sums::Sums const &sums( ) const { return( m_sums ); }                       /**< Returns the value of the **m_sums** member. */
        Suite const &fissionComponents( ) const { return( m_fissionComponents ); }  /**< Returns the value of the **m_fissionComponents** member. */

// The rest are virtual methods defined in the Protare class.

        bool isSingleton( ) const { return( true ); }
        bool isTNSL_ProtareSingleton( ) const { return( m_isTNSL_ProtareSingleton ); }                      /**< Returns the value of the **m_isTNSL_ProtareSingleton** member. */
        std::size_t numberOfProtares( ) const { return( 1 ); }                                              /**< Returns 1. */
        ProtareSingleton const *protare( std::size_t a_index ) const ;

        std::string const &formatVersion( std::size_t a_index = 0 ) const { return( m_formatVersion ); }    /**< Returns the value of the **m_formatVersion** member. */
        std::string const &fileName( std::size_t a_index = 0 ) const { return( m_fileName ); }              /**< Returns the value of the **m_fileName** member. */
        std::string const &realFileName( std::size_t a_index = 0 ) const { return( m_realFileName ); }      /**< Returns the value of the **m_realFileName** member. */

        std::vector<std::string> libraries( std::size_t a_index = 0 ) const { return( m_libraries ); }      /**< Returns the libraries that *this* resided in. */
        std::string const &evaluation( std::size_t a_index = 0 ) const { return( m_evaluation ); }          /**< Returns the value of the **m_evaluation** member. */
        frame projectileFrame( std::size_t a_index = 0 ) const { return( m_projectileFrame ); }             /**< Returns the value of the **m_projectileFrame** member. */
        double thresholdFactor( ) const { return( m_thresholdFactor ); }            /**< Returns the value of the **m_thresholdFactor** member. */

        Documentation::Suite const &documentations( ) const { return( m_documentations ); }                /**< Returns the value of the **m_documentations** member. */

        Styles::Base const &style( std::string const a_label ) const { return( *m_styles.get<Styles::Base>( a_label ) ); }      /**< Returns the style with label **a_label**. */
        Styles::Suite const &styles( ) const { return( m_styles ); }                /**< Returns the value of the **m_styles** member. */

        void productIDs( std::set<std::string> &a_ids, Settings::Particles const &a_particles, bool a_transportablesOnly ) const ;
        int maximumLegendreOrder( Settings::MG const &a_settings, std::string const &a_productID ) const ;

        Styles::TemperatureInfos temperatures( ) const ;

        std::size_t numberOfReactions( ) const { return( m_reactions.size( ) ); }       /**< Returns the number of reactions in the **Protare**. */
        Reaction const *reaction( std::size_t a_index ) const { return( m_reactions.get<Reaction>( a_index ) ); }               /**< Returns the **a_index** - 1 reaction. */       /**< Returns the **a_index** - 1 reaction. */
        std::size_t numberOfOrphanProducts( ) const { return( m_orphanProducts.size( ) ); };
        Reaction const *orphanProduct( std::size_t a_index ) const { return( m_orphanProducts.get<Reaction>( a_index ) ); }     /**< Returns the **a_index** - 1 orphan product. */

        bool hasFission( ) const ;

        Ancestry const *findInAncestry3( std::string const &a_item ) const ;

        Styles::MultiGroup const *multiGroup( std::string const &a_label ) const ;
        std::vector<double> const &groupBoundaries( Settings::MG const &a_settings, std::string const &a_productID ) const ;
        Vector multiGroupInverseSpeed( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;

        Vector multiGroupCrossSection( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;
        Vector multiGroupQ( Settings::MG const &a_settings, Settings::Particles const &a_particles, bool a_final ) const ;

        Vector multiGroupMultiplicity( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;
        Vector multiGroupFissionNeutronMultiplicity( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;

        Matrix multiGroupProductMatrix( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID, int a_order ) const ;
        Matrix multiGroupFissionMatrix( Settings::MG const &a_settings, Settings::Particles const &a_particles, int a_order ) const ;
        Vector multiGroupTransportCorrection( Settings::MG const &a_settings, Settings::Particles const &a_particles, int a_order, transportCorrectionType a_transportCorrectionType, double a_temperature ) const ;

        Vector multiGroupAvailableEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;
        Vector multiGroupAverageEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;
        Vector multiGroupDepositionEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;

        Vector multiGroupAvailableMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;
        Vector multiGroupAverageMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;
        Vector multiGroupDepositionMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;

        Vector multiGroupGain( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;

        stringAndDoublePairs muCutoffForCoulombPlusNuclearElastic( ) const ;

        void saveAs( std::string const &a_fileName ) const ;
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
===================== ProtareComposite =====================
============================================================
*/
class ProtareComposite : public Protare {

    private:
        std::vector<Protare *> m_protares;                                      /**< List of protares added to *this* instance. */

    public:
        ProtareComposite( Construction::Settings const &a_construction );
        ~ProtareComposite( );

        std::vector<Protare *> &protares( ) { return( m_protares ); }           /**< Returns the value of the **m_protares** member. */
        void append( Protare *a_protare );

// The rest are virtual methods defined in the Protare class.

        bool isComposite( ) const { return( true ); }                           /**< Always returns *true*. */
        std::size_t numberOfProtares( ) const ;
        ProtareSingleton const *protare( std::size_t a_index ) const ;

        std::string const &formatVersion( std::size_t a_index = 0 ) const ;
        std::string const &fileName( std::size_t a_index = 0 ) const ;
        std::string const &realFileName( std::size_t a_index = 0 ) const ;

        std::vector<std::string> libraries( std::size_t a_index = 0 ) const ;
        std::string const &evaluation( std::size_t a_index = 0 ) const ;
        frame projectileFrame( std::size_t a_index = 0 ) const ;
        double thresholdFactor( ) const ;

        Documentation::Suite const &documentations( ) const;

        Styles::Base const &style( std::string const a_label ) const ;
        Styles::Suite const &styles( ) const ;

        void productIDs( std::set<std::string> &a_ids, Settings::Particles const &a_particles, bool a_transportablesOnly ) const ;
        int maximumLegendreOrder( Settings::MG const &a_settings, std::string const &a_productID ) const ;

        Styles::TemperatureInfos temperatures( ) const ;

        std::size_t numberOfReactions( ) const ;
        Reaction const *reaction( std::size_t a_index ) const ;
        std::size_t numberOfOrphanProducts( ) const ;
        Reaction const *orphanProduct( std::size_t a_index ) const ;

        bool hasFission( ) const ;

        Ancestry const *findInAncestry3( std::string const &a_item ) const { return( NULL ); }  /**< Always returns *NULL*. */

        Styles::MultiGroup const *multiGroup( std::string const &a_label ) const ;
        std::vector<double> const &groupBoundaries( Settings::MG const &a_settings, std::string const &a_productID ) const ;
        Vector multiGroupInverseSpeed( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;

        Vector multiGroupCrossSection( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;
        Vector multiGroupQ( Settings::MG const &a_settings, Settings::Particles const &a_particles, bool a_final ) const ;

        Vector multiGroupMultiplicity( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;
        Vector multiGroupFissionNeutronMultiplicity( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;

        Matrix multiGroupProductMatrix( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID, int a_order ) const ;
        Matrix multiGroupFissionMatrix( Settings::MG const &a_settings, Settings::Particles const &a_particles, int a_order ) const ;
        Vector multiGroupTransportCorrection( Settings::MG const &a_settings, Settings::Particles const &a_particles, int a_order, transportCorrectionType a_transportCorrectionType, double a_temperature ) const ;

        Vector multiGroupAvailableEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;
        Vector multiGroupAverageEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;
        Vector multiGroupDepositionEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;

        Vector multiGroupAvailableMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;
        Vector multiGroupAverageMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;
        Vector multiGroupDepositionMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;

        Vector multiGroupGain( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;

        stringAndDoublePairs muCutoffForCoulombPlusNuclearElastic( ) const ;
};

/*
============================================================
======================= ProtareTNSL ========================
============================================================
*/
class ProtareTNSL : public Protare {

    private:
        ProtareSingleton *m_protare;                                        /**< Protare with non thermal neutron scattering law data. */
        ProtareSingleton *m_TNSL;                                           /**< Protare with thermal neutron scattering law data. */
        Reaction const *m_elasticReaction;                                  /**< The elastic reaction from the non TNSL protare. */
//        std::map<std::string,std::size_t> m_maximumMultiGroupIndices;

    public:
        ProtareTNSL( Construction::Settings const &a_construction, ProtareSingleton *a_protare, ProtareSingleton *a_TNSL );
        ~ProtareTNSL( );

        ProtareSingleton const *protare( ) const { return( m_protare ); }   /**< Returns the **m_protare** member. */
        ProtareSingleton const *TNSL( ) const { return( m_TNSL ); }         /**< Returns the **m_TNSL** member. */
        Reaction const *elasticReaction( ) const { return( m_elasticReaction ); } /**< Returns the **m_elasticReaction** member. */
        std::size_t maximumMultiGroupIndices( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;
        void combineVectors( Settings::MG const &a_settings, Settings::Particles const &a_particles, Vector &a_vector, Vector const &a_vectorElastic, Vector const &a_vectorTNSL ) const ;
        void combineMatrices( Settings::MG const &a_settings, Settings::Particles const &a_particles, Matrix &a_matrix, Matrix const &a_matrixElastic, Matrix const &a_matrixTNSL ) const ;

// The rest are virtual methods defined in the Protare class.

        bool isTNSL( ) const { return( true ); }                            /**< Returns *true* if the instance is a ProtareTNSL instance and *false* otherwise. */
        std::size_t numberOfProtares( ) const { return( 2 ); }              /**< Always returns 2. */
        ProtareSingleton const *protare( std::size_t a_index ) const ;

        std::string const &formatVersion( std::size_t a_index = 0 ) const ;
        std::string const &fileName( std::size_t a_index = 0 ) const ;
        std::string const &realFileName( std::size_t a_index = 0 ) const ;

        std::vector<std::string> libraries( std::size_t a_index = 0 ) const ;
        std::string const &evaluation( std::size_t a_index = 0 ) const ;
        frame projectileFrame( std::size_t a_index = 0 ) const ;
        double thresholdFactor( ) const ;

        Documentation::Suite const &documentations( ) const ;

        Styles::Base const &style( std::string const a_label ) const ;
        Styles::Suite const &styles( ) const ;

        void productIDs( std::set<std::string> &a_ids, Settings::Particles const &a_particles, bool a_transportablesOnly ) const ;
        int maximumLegendreOrder( Settings::MG const &a_settings, std::string const &a_productID ) const ;

        Styles::TemperatureInfos temperatures( ) const ;

        std::size_t numberOfReactions( ) const ;
        Reaction const *reaction( std::size_t a_index ) const ;
        std::size_t numberOfOrphanProducts( ) const ;
        Reaction const *orphanProduct( std::size_t a_index ) const ;

        bool hasFission( ) const ;

        Ancestry const *findInAncestry3( std::string const &a_item ) const { return( NULL ); }          /**< Always returns *NULL*. */

        Styles::MultiGroup const *multiGroup( std::string const &a_label ) const ;
        std::vector<double> const &groupBoundaries( Settings::MG const &a_settings, std::string const &a_productID ) const ;
        Vector multiGroupInverseSpeed( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;

        Vector multiGroupCrossSection( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;
        Vector multiGroupQ( Settings::MG const &a_settings, Settings::Particles const &a_particles, bool a_final ) const ;

        Vector multiGroupMultiplicity( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;
        Vector multiGroupFissionNeutronMultiplicity( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;

        Matrix multiGroupProductMatrix( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID, int a_order ) const ;
        Matrix multiGroupFissionMatrix( Settings::MG const &a_settings, Settings::Particles const &a_particles, int a_order ) const ;
        Vector multiGroupTransportCorrection( Settings::MG const &a_settings, Settings::Particles const &a_particles, int a_order, transportCorrectionType a_transportCorrectionType, double a_temperature ) const ;

        Vector multiGroupAvailableEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;
        Vector multiGroupAverageEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;
        Vector multiGroupDepositionEnergy( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;

        Vector multiGroupAvailableMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;
        Vector multiGroupAverageMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;
        Vector multiGroupDepositionMomentum( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;

        Vector multiGroupGain( Settings::MG const &a_settings, Settings::Particles const &a_particles, std::string const &a_productID ) const ;

        stringAndDoublePairs muCutoffForCoulombPlusNuclearElastic( ) const ;
};

/*
============================================================
======================= MapBaseEntry =======================
============================================================
*/
class MapBaseEntry {

    public:
        enum pathForm { e_entered, e_cumulative, e_real };

    private:
        std::string m_name;                                 /**< Designates the entry as either a protare or a map. */
        Map const *m_parent;                                /**< Pointer to map containing *this*. */
        std::string m_path;                                 /**< Absolute or relative (to map file) path of the protare or map file. */
        std::string m_cumulativePath;                       /**< Currently not used. */
        std::string m_realPath;                             /**< Absolute path of the protare or map file. */

    public:
        MapBaseEntry( pugi::xml_node const &a_node, std::string const &a_basePath, Map const *a_parent );
        virtual ~MapBaseEntry( ) = 0;

        std::string const &name( ) const { return( m_name ); }              /**< Returns the value of the **m_name** member. */
        Map const *parent( ) const { return( m_parent ); }                  /**< Returns the value of the **m_parent** member. */
        std::string const &path( pathForm a_form = e_real ) const ;
        void libraries( std::vector<std::string> &a_libraries ) const ;
        virtual ProtareBaseEntry const *findProtareEntry( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_evaluation = "" ) const = 0 ;
};

/*
============================================================
======================== MapEntry ==========================
============================================================
*/
class MapEntry : public MapBaseEntry {

    private:
        Map *m_map;                                         /**< Map instance for this MapEntry. */

    public:
        MapEntry( pugi::xml_node const &a_node, PoPs::Database const &a_pops, std::string const &a_basePath, Map const *a_parent );
        ~MapEntry( );

        Map const *map( ) const { return( m_map ); }                    /**< Returns the value of the **m_map** member. */

        ProtareBaseEntry const *findProtareEntry( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_evaluation = "" ) const ;
        std::string protareFilename( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_evaluation = "",
                pathForm a_form = e_real ) const ;
        bool isProtareAvailable( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_evaluation = "" ) const {
                return( protareFilename( a_projectileID, a_targetID, a_evaluation ) != GIDI_emptyFileName ); }
                                                                        /**< Returns the value of the **m_map** member. */
        std::vector<std::string> availableEvaluations( std::string const &a_projectileID, std::string const &a_targetID ) const ;
};

/*
============================================================
======================= IDBaseEntry ========================
============================================================
*/
class IDBaseEntry {

    private:
        std::string m_projectileID;             /**< Projectile id for protare. */
        std::string m_targetID;                 /**< Target id for protare. */
        std::string m_evaluation;               /**< Evaluation string for protare. */

    public:
        IDBaseEntry( pugi::xml_node const &a_node );
        virtual ~IDBaseEntry( ) = 0;

        std::string const &projectileID( ) const { return( m_projectileID ); }      /**< Returns the value of the **m_projectileID** member. */
        std::string const &targetID( ) const { return( m_targetID ); }              /**< Returns the value of the **m_targetID** member. */
        std::string const &evaluation( ) const { return( m_evaluation ); }          /**< Returns the value of the **m_evaluation** member. */

        bool isMatch( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_evaluation = "" ) const ;
};

/*
============================================================
==================== ProtareBaseEntry ======================
============================================================
*/
class ProtareBaseEntry : public MapBaseEntry, public IDBaseEntry {

    public:
        ProtareBaseEntry( pugi::xml_node const &a_node, std::string const &a_basePath, Map const *const a_map );
        ~ProtareBaseEntry( );

        std::string const &library( ) const ;
        std::string const &resolvedLibrary( ) const ;

        ProtareBaseEntry const *findProtareEntry( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_evaluation = "" ) const ;
        virtual Protare *protare( Construction::Settings const &a_construction, PoPs::Database const &a_pops ) const = 0 ;
};

/*
============================================================
====================== ProtareEntry ========================
============================================================
*/
class ProtareEntry : public ProtareBaseEntry {

    private:
        bool m_isPhotoAtomic;                   /**< true if photo-atomic protare and false otherwise. */

    public:
        ProtareEntry( pugi::xml_node const &a_node, PoPs::Database const &a_pops, std::string const &a_basePath, Map const *const a_parent );
        ~ProtareEntry( );

        bool isPhotoAtomic( ) const { return( m_isPhotoAtomic ); }                  /**< Returns the value of the **m_isPhotoAtomic** member. */
        Protare *protare( Construction::Settings const &a_construction, PoPs::Database const &a_pops ) const ;
};

/*
============================================================
======================== TNSLsProtare =======================
============================================================
*/
class TNSLsProtare : public IDBaseEntry {

    public:
        TNSLsProtare( pugi::xml_node const &a_node, PoPs::Database const &a_pops, std::string const &a_basePath, Map const *const a_parent );
        ~TNSLsProtare( );
};

/*
============================================================
======================== TNSLEntry =========================
============================================================
*/
class TNSLEntry : public ProtareBaseEntry {

    private:
        TNSLsProtare m_TNSLsProtare;            /**< The non-TNSL protare. */

    public:
        TNSLEntry( pugi::xml_node const &a_node, PoPs::Database const &a_pops, std::string const &a_basePath, Map const *const a_parent );
        ~TNSLEntry( );

        TNSLsProtare const &TNSLs_protare( ) const { return( m_TNSLsProtare ); }    /**< Returns the value of the **m_TNSLsProtare** member. */
        Protare *protare( Construction::Settings const &a_construction, PoPs::Database const &a_pops ) const ;
};

/*
============================================================
=========================== Map ============================
============================================================
*/
class Map {

    private:
        Map const *m_parent;                            /**< Pointer to map containing *this*. */
        std::string m_fileName;                         /**< Specified path to Map. */
        std::string m_realFileName;                     /**< Absolute, read path to Map. */
        std::string m_library;                          /**< The name of the library. */
        std::vector<MapBaseEntry *> m_entries;          /**< List of Map entries. */

        void initialize( std::string const &a_fileName, PoPs::Database const &a_pops, Map const *a_parent );
        void initialize( pugi::xml_node const &a_node, std::string const &a_fileName, PoPs::Database const &a_pops, Map const *a_parent );

    public:
        Map( std::string const &a_fileName, PoPs::Database const &a_pops, Map const *a_parent = NULL );
        Map( char const *a_fileName, PoPs::Database const &a_pops, Map const *a_parent = NULL );
        Map( pugi::xml_node const &a_node, std::string const &a_fileName, PoPs::Database const &a_pops, Map const *a_parent = NULL );
        ~Map( );

        Map const *parent( ) const { return( m_parent ); }                      /**< Returns the value of the **m_parent** member. */
        std::string const &fileName( ) const { return( m_fileName ); }          /**< Returns the value of the **m_fileName** member. */
        std::string const &realFileName( ) const { return( m_realFileName ); }  /**< Returns the value of the **m_realFileName** member. */

        std::string const &library( ) const { return( m_library ); }            /**< Returns the value of the **m_library** member. */
        std::string const &resolvedLibrary( ) const ;
        void libraries( std::vector<std::string> &a_libraries ) const ;

        std::size_t size( ) const { return( m_entries.size( ) ); }              /**< Returns the number of entries in *this*. Does not descend map entries. */
        MapBaseEntry const *operator[]( std::size_t a_index ) const { return( m_entries[a_index] ); }
                                                                                /**< Returns the map entry at index *a_index*. */

        ProtareBaseEntry const *findProtareEntry( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_evaluation = "" ) const ;
        std::string protareFilename( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_evaluation = "",
                MapBaseEntry::pathForm a_form = MapBaseEntry::e_real ) const ;

        bool isProtareAvailable( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_evaluation = "" ) const {
                return( protareFilename( a_projectileID, a_targetID, a_evaluation ) != GIDI_emptyFileName ); }
                                                                            /**< Returns true if the map contains a Protare matching *a_projectileID*, *a_targetID* and *a_evaluation* and false otherwise. */
        std::vector<std::string> availableEvaluations( std::string const &a_projectileID, std::string const &a_targetID ) const ;

        std::vector<ProtareBaseEntry const *> directory( std::string const &a_projectileID = "", std::string const &a_targetID = "", std::string const &a_evaluation = "" ) const ;
        bool walk( MapWalkCallBack a_mapWalkCallBack, void *a_userData, int a_level = 0 ) const ;

        Protare *protare( Construction::Settings const &a_construction, PoPs::Database const &a_pops, std::string const &a_projectileID, std::string const &a_targetID, 
                std::string const &a_evaluation = "", bool a_targetRequiredInGlobalPoPs = true, bool a_requiredInPoPs = true ) const ;
};

/*
============================================================
================= FissionEnergyRelease ==================
============================================================
*/
class FissionEnergyRelease : public Function1dForm {

    private:
        Function1dForm *m_promptProductKE;                  /**< The **ENDF** prompt total product kinetic energy released. */
        Function1dForm *m_promptNeutronKE;                  /**< The **ENDF** prompt neutron kinetic energy released. */
        Function1dForm *m_delayedNeutronKE;                 /**< The **ENDF** delayed neutron kinetic energy released. */
        Function1dForm *m_promptGammaEnergy;                /**< The **ENDF** prompt gamma energy released. */
        Function1dForm *m_delayedGammaEnergy;               /**< The **ENDF** delayed gamma energy released. */
        Function1dForm *m_delayedBetaEnergy;                /**< The **ENDF** delayed beta kinetic energy released. */
        Function1dForm *m_neutrinoEnergy;                   /**< The **ENDF** neutrino energy released. */
        Function1dForm *m_nonNeutrinoEnergy;                /**< The **ENDF** non neutrino energy released. */
        Function1dForm *m_totalEnergy;                      /**< The **ENDF** total energy released. */

        void energyReleaseToXMLList( WriteInfo &a_writeInfo, std::string const &a_moniker, std::string const &a_indent, Function1dForm *a_function1d ) const ;

    public:
        FissionEnergyRelease( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
        ~FissionEnergyRelease( );

        double domainMin( ) const { return( m_nonNeutrinoEnergy->domainMin( ) ); }  /**< Returns the minimum domain value for the energy released. */
        double domainMax( ) const { return( m_nonNeutrinoEnergy->domainMax( ) ); }  /**< Returns the maximum domain value for the energy released. */
        double evaluate( double a_x1 ) const { return( m_nonNeutrinoEnergy->evaluate( a_x1 ) ); }   /**< Returns the value of **m_nonNeutrinoEnergy** evaluated at *a_x1*. */
        Vector multiGroupQ( Settings::MG const &a_settings, Settings::Particles const &a_particles ) const ;

        Function1dForm const *promptProductKE( ) const { return( m_promptProductKE ); }             /**< Returns the value of the **m_promptProductKE** member. */
        Function1dForm const *promptNeutronKE( ) const { return( m_promptNeutronKE ); }             /**< Returns the value of the **m_promptNeutronKE** member. */
        Function1dForm const *delayedNeutronKE( ) const { return( m_delayedNeutronKE ); }           /**< Returns the value of the **m_delayedNeutronKE** member. */
        Function1dForm const *promptGammaEnergy( ) const { return( m_promptGammaEnergy ); }         /**< Returns the value of the **m_promptGammaEnergy** member. */
        Function1dForm const *delayedGammaEnergy( ) const { return( m_delayedGammaEnergy ); }       /**< Returns the value of the **m_delayedGammaEnergy** member. */
        Function1dForm const *delayedBetaEnergy( ) const { return( m_delayedBetaEnergy ); }         /**< Returns the value of the **m_delayedBetaEnergy** member. */
        Function1dForm const *neutrinoEnergy( ) const { return( m_neutrinoEnergy ); }               /**< Returns the value of the **m_neutrinoEnergy** member. */
        Function1dForm const *nonNeutrinoEnergy( ) const { return( m_nonNeutrinoEnergy ); }         /**< Returns the value of the **m_neutrinoEnergy** member. */
        Function1dForm const *totalEnergy( ) const { return( m_totalEnergy ); }                     /**< Returns the value of the **m_totalEnergy** member. */

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
========================== Groups ==========================
============================================================
*/
class Groups : public Suite {

    public:
        Groups( std::string const &a_fileName );
};

/*
============================================================
========================== Groups ==========================
============================================================
*/
class Fluxes : public Suite {

    public:
        Fluxes( std::string const &a_fileName );
};

/*
============================================================
========================== others ==========================
============================================================
*/
Form *parseStylesSuite( Construction::Settings const &a_construction, Suite *parent, pugi::xml_node const &a_node, PoPs::Database const &a_pop, PoPs::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseTransportablesSuite( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseReaction( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseOrphanProduct( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseFissionComponent( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops,
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles );
Form *parseReactionType( std::string const &a_moniker, Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseSumsCrossSectionsSuite( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops,
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles );
Form *parseSumsMultiplicitiesSuite( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops,
                PoPs::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles );
Form *parseDoubleDifferentialCrossSectionSuite( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseScatteringAtom( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseCrossSectionSuite( Construction::Settings const &a_construction, Suite *parent, pugi::xml_node const &a_node, PoPs::Database const &a_pop, PoPs::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseDelayedNeutronsSuite( Construction::Settings const &a_construction, Suite *parent, pugi::xml_node const &a_node, PoPs::Database const &a_pop, PoPs::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseFissionEnergyReleasesSuite( Construction::Settings const &a_construction, Suite *parent, pugi::xml_node const &a_node, PoPs::Database const &a_pop, PoPs::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseRateSuite( Construction::Settings const &a_construction, Suite *parent, pugi::xml_node const &a_node, PoPs::Database const &a_pop, PoPs::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseAvailableSuite( Construction::Settings const &a_construction, Suite *a_parent, pugi::xml_node const &a_node, PoPs::Database const &a_pops, PoPs::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseQSuite( Construction::Settings const &a_construction, Suite *parent, pugi::xml_node const &a_node, PoPs::Database const &a_pop, PoPs::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseProductSuite( Construction::Settings const &a_construction, Suite *parent, pugi::xml_node const &a_node, PoPs::Database const &a_pop, PoPs::Database const &a_internalPoPs, 
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseMultiplicitySuite( Construction::Settings const &a_construction, Suite *parent, pugi::xml_node const &a_node, PoPs::Database const &a_pop, PoPs::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseDistributionSuite( Construction::Settings const &a_construction, Suite *parent, pugi::xml_node const &a_node, PoPs::Database const &a_pop, PoPs::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseAverageEnergySuite( Construction::Settings const &a_construction, Suite *parent, pugi::xml_node const &a_node, PoPs::Database const &a_pop, PoPs::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseAverageMomentumSuite( Construction::Settings const &a_construction, Suite *parent, pugi::xml_node const &a_node, PoPs::Database const &a_pop, PoPs::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Function1dForm *data1dParse( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *parent );
Function1dForm *data1dParseAllowEmpty( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *a_parent );
Function2dForm *data2dParse( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *parent );
Function3dForm *data3dParse( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Suite *parent );

int parseFlattened1d( Construction::Settings const &a_construction, pugi::xml_node const &a_node, Vector &data );

Vector collapse( Vector const &a_vector, Settings::Settings const &a_settings, Settings::Particles const &a_particles, double a_temperature );
Matrix collapse( Matrix const &a_matrix, Settings::Settings const &a_settings, Settings::Particles const &a_particles, double a_temperature, std::string const &a_productID );

Vector transportCorrect( Vector const &a_vector, Vector const &a_transportCorrection );
Matrix transportCorrect( Matrix const &a_matrix, Vector const &a_transportCorrection );

Vector multiGroupXYs1d( Settings::MultiGroup const &a_boundaries, XYs1d const &a_function, Settings::Flux const &a_flux );

int ENDL_CFromENDF_MT( int ENDF_MT, int *ENDL_C, int *ENDL_S );

/*
*   The following are in the file GIDI_misc.cpp.
*/
std::string realPath( char const *a_path );
std::string realPath( std::string const &a_path );
std::vector<std::string> splitString( std::string const &a_string, char a_delimiter );
long binarySearchVector( double a_x, std::vector<double> const &a_Xs );
void intsToXMLList( WriteInfo &a_writeInfo, std::string const &a_indent, std::vector<int> a_values, std::string const &a_attributes );
void parseValuesOfDoubles( Construction::Settings const &a_construction, pugi::xml_node const &a_node, std::vector<double> &a_vector );
void parseValuesOfDoubles( pugi::xml_node const &a_node, std::vector<double> &a_vector, int a_useSystem_strtod );
void doublesToXMLList( WriteInfo &a_writeInfo, std::string const &a_indent, std::vector<double> a_values, std::size_t a_start = 0, bool a_newLine = true,
        std::string const &a_valueType = "" );
frame parseFrame( pugi::xml_node const &a_node, std::string const &a_name );
std::string frameToString( frame a_frame );
std::string intToString( int a_value );
std::string size_t_ToString( std::size_t a_value );
std::string nodeWithValuesToDoubles( WriteInfo &a_writeInfo, std::string const &a_nodeName, std::vector<double> const &a_values );
std::string doubleToShortestString( double a_value, int a_significantDigits = 15, int a_favorEFormBy = 0 );

Ys1d gridded1d2GIDI_Ys1d( Function1dForm const &a_function1d );
Ys1d vector2GIDI_Ys1d( Vector const &a_vector );

std::string LLNL_gidToLabel( int a_gid );
std::string LLNL_fidToLabel( int a_fid );

std::vector<std::string> sortedListOfStrings( std::vector<std::string> const &a_strings, bool a_orderIsAscending = true );

void energy2dToXMLList( WriteInfo &a_writeInfo, std::string const &a_moniker, std::string const &a_indent, Function1dForm *a_function );


}           // End of namespace GIDI.

#endif      // End of GIDI_hpp_included

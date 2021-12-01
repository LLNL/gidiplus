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

#include <HAPI.hpp>
#include <PoPI.hpp>

#include <nf_utilities.h>
#include <nf_buffer.h>
#include <ptwXY.h>

#include "GIDI_data.hpp"

namespace GIDI {

class SetupInfo;
class Form;
class Suite;
class OutputChannel;
class Protare;
class ProtareSingle;
class ParticleInfo;

namespace Functions {
    class Function2dForm;
}                   // End namespace Functions.

namespace Map {
    class ProtareBase;
    class TNSL;
    class Map;
}                   // End of namespace Map.

typedef bool (*MapWalkCallBack)( Map::ProtareBase const *a_protareEntry, std::string const &a_library, void *a_userData, int a_level );

namespace Construction {
    class Settings;
}                   // End of namespace Construction.

namespace Documentation_1_10 {
    class Suite;
}                   // End of namespace Documentation_1_10.

namespace ExternalFiles {

class Suite;

}                   // End of namespace ExternalFiles

namespace Styles {
    class Suite;
    class MultiGroup;
}                   // End of namespace Styles.

typedef Form *(*parseSuite)( Construction::Settings const &a_construction, Suite *a_parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo,
        PoPI::Database const &a_pop, PoPI::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles );

enum class GNDS_FileType { uninitialized, unknown, pops, protare, covarianceSuite, map };

class GNDS_FileTypeInfo {

    private:
        GNDS_FileType m_GNDS_fileType;
        std::string m_projectileID;
        std::string m_targetID;
        std::string m_evaluation;
        std::string m_interaction;

    public:
        GNDS_FileTypeInfo( );
        GNDS_FileTypeInfo( GNDS_FileType a_GNDS_fileType, std::string a_projectileID = "", std::string a_targetID = "", std::string a_evaluation = "",
                        std::string a_interaction = "" );
        GNDS_FileTypeInfo( GNDS_FileTypeInfo const &a_GNDS_fileTypeInfo );

        GNDS_FileType GNDS_fileType( ) const { return( m_GNDS_fileType ); }
        void setGNDS_fileType( GNDS_FileType a_GNDS_fileType ) { m_GNDS_fileType = a_GNDS_fileType; }
        std::string const &projectileID( ) const { return( m_projectileID ); }
        std::string const &targetID( ) const { return( m_targetID ); }
        std::string const &evaluation( ) const { return( m_evaluation ); }
        std::string const &interaction( ) const { return( m_interaction ); }
};

enum class ProtareType { single, composite, TNSL };

enum class FormType { generic, group, groups, transportable, flux, fluxes, externalFile, style,
                reaction, product, delayedNeutron, fissionFragmentData, rate,
                physicalQuantity, axisDomain, axis, grid, axes,
                flattenedArrayData, array3d,
                    // 1d functions.
                constant1d, XYs1d, Ys1d, polynomial1d, Legendre1d, gridded1d, reference1d, xs_pdf_cdf1d, regions1d, 
                resonancesWithBackground1d, resonanceBackground1d, resonanceBackgroundRegion1d, URR_probabilityTables1d,
                fissionEnergyRelease1d, branching1d, branching1dPids, thermalNeutronScatteringLaw1d, unspecified1d,
                    // 2d functions.
                XYs2d, gridded2d, recoil2d, isotropic2d, discreteGamma2d, primaryGamma2d, regions2d, 
                generalEvaporation2d, simpleMaxwellianFission2d, evaporation2d, Watt2d, MadlandNix2d, 
                weighted_function2d, weightedFunctionals2d, NBodyPhaseSpace2d,
                    // 3d functions.
                XYs3d, regions3d, gridded3d,
                    // distributions.
                angularTwoBody, KalbachMann, uncorrelated, unspecified, reference3d, multiGroup3d, 
                energyAngular, energyAngularMC, angularEnergy, angularEnergyMC, LLNL_angularEnergy,
                coherentPhotonScattering, incoherentPhotonScattering, thermalNeutronScatteringLaw, branching3d,
                coherentElastic, incoherentElastic, incoherentInelastic,
                    // Sums stuff.
                crossSectionSum, multiplicitySum, summands };

enum class Frame { lab, centerOfMass };
enum class TransportCorrectionType { None, Pendlebury, LLNL, Ferguson };
enum class FileType { XML, HDF };

#define GNDS_XML_verionEncoding "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"

#define GNDS_formatVersion_1_10Chars "1.10"
#define GNDS_formatVersion_2_0Chars "2.0"
#define GNDS_formatVersion_2_0_LLNL_4Chars "2.0.LLNL_4"

#define GIDI_emptyFileNameChars ""

#define GIDI_mapFormatVersion_0_1Chars "0.1"
#define GIDI_mapFormatVersion_0_2Chars "0.2"

#define GIDI_LLNL_Chars "LLNL"

#define GIDI_mapChars "map"
#define GIDI_importChars "import"
#define GIDI_protareChars "protare"
#define GIDI_TNSLChars "TNSL"

#define GIDI_formatChars "format"

#define GIDI_topLevelChars "reactionSuite"
#define GIDI_covarianceSuiteChars "covarianceSuite"

#define GIDI_externalFilesChars "externalFiles"
#define GIDI_externalFileChars "externalFile"

#define GIDI_documentations_1_10_Chars "documentations"
#define GIDI_documentation_1_10_Chars "documentation"
#define GIDI_stylesChars "styles"
#define GIDI_PoPsChars "PoPs"
#define GIDI_reactionsChars "reactions"
#define GIDI_reactionChars "reaction"
#define GIDI_orphanProductsChars "orphanProducts"
#define GIDI_orphanProductChars "orphanProduct"
#define GIDI_incompleteReactionsChars "incompleteReactions"
#define GIDI_fissionComponentsChars "fissionComponents"
#define GIDI_fissionComponentChars "fissionComponent"

#define GIDI_applicationDataChars "applicationData"
#define GIDI_institutionChars "institution"
#define GIDI_nuclearPlusCoulombInterferenceChars "nuclearPlusCoulombInterference"

#define GIDI_sumsChars "sums"
#define GIDI_sumsCrossSectionsChars "crossSections"
#define GIDI_sumsMultiplicitiesChars "multiplicities"
#define GIDI_sumsAddChars "add"
#define GIDI_sumsSummandsChars "summands"
#define GIDI_crossSectionSumsChars "crossSectionSums"
#define GIDI_crossSectionSumChars "crossSectionSum"
#define GIDI_multiplicitySumsChars "multiplicitySums"
#define GIDI_multiplicitySumChars "multiplicitySum"

#define GIDI_doubleDifferentialCrossSectionChars "doubleDifferentialCrossSection"
#define GIDI_crossSectionChars "crossSection"
#define GIDI_availableEnergyChars "availableEnergy"
#define GIDI_availableMomentumChars "availableMomentum"

#define GIDI_QChars "Q"
#define GIDI_productsChars "products"
#define GIDI_productChars "product"

#define GIDI_multiplicityChars "multiplicity"
#define GIDI_distributionChars "distribution"
#define GIDI_averageEnergyChars "averageProductEnergy"
#define GIDI_averageMomentumChars "averageProductMomentum"
#define GIDI_outputChannelChars "outputChannel"

#define GIDI_fissionFragmentDataChars "fissionFragmentData"
#define GIDI_delayedNeutronsChars "delayedNeutrons"
#define GIDI_delayedNeutronChars "delayedNeutron"
#define GIDI_fissionEnergyReleasesChars "fissionEnergyReleases"
#define GIDI_fissionEnergyReleaseChars "fissionEnergyRelease"
#define GIDI_rateChars "rate"

#define GIDI_groupsChars "groups"
#define GIDI_groupChars "group"
#define GIDI_fluxesChars "fluxes"

#define GIDI_evaluatedStyleChars "evaluated"
#define GIDI_crossSectionReconstructedStyleChars "crossSectionReconstructed"
#define GIDI_angularDistributionReconstructedStyleChars "angularDistributionReconstructed"
#define GIDI_CoulombPlusNuclearElasticMuCutoffStyleChars "CoulombPlusNuclearElasticMuCutoff"
#define GIDI_averageProductDataStyleChars "averageProductData"
#define GIDI_MonteCarlo_cdfStyleChars "MonteCarlo_cdf"
#define GIDI_multiGroupStyleChars "multiGroup"
#define GIDI_transportablesChars "transportables"
#define GIDI_transportableChars "transportable"
#define GIDI_realizationChars "realization"
#define GIDI_heatedStyleChars "heated"
#define GIDI_griddedCrossSectionStyleChars "griddedCrossSection"
#define GIDI_URR_probabilityTablesStyleChars "URR_probabilityTables"
#define GIDI_heatedMultiGroupStyleChars "heatedMultiGroup"
#define GIDI_SnElasticUpScatterStyleChars "SnElasticUpScatter"
#define GIDI_projectileEnergyDomainChars "projectileEnergyDomain"

// 1d Function monikers.
#define GIDI_constant1dChars "constant1d"
#define GIDI_XYs1dChars "XYs1d"
#define GIDI_Ys1dChars "Ys1d"
#define GIDI_polynomial1dChars "polynomial1d"
#define GIDI_LegendreChars "Legendre"
#define GIDI_regions1dChars "regions1d"
#define GIDI_gridded1dChars "gridded1d"
#define GIDI_referenceChars "reference"
#define GIDI_xs_pdf_cdf1dChars "xs_pdf_cdf1d"
#define GIDI_branching1dChars "branching1d"
#define GIDI_TNSL1dChars "thermalNeutronScatteringLaw1d"

// 2d Function monikers.
#define GIDI_XYs2dChars "XYs2d"
#define GIDI_recoilChars "recoil"
#define GIDI_isotropic2dChars "isotropic2d"
#define GIDI_discreteGammaChars "discreteGamma"
#define GIDI_primaryGammaChars "primaryGamma"
#define GIDI_generalEvaporationChars "generalEvaporation"
#define GIDI_simpleMaxwellianFissionChars "simpleMaxwellianFission"
#define GIDI_evaporationChars "evaporation"
#define GIDI_WattChars "Watt"
#define GIDI_MadlandNixChars "MadlandNix"
#define GIDI_weightedFunctionalsChars "weightedFunctionals"
#define GIDI_NBodyPhaseSpaceChars "NBodyPhaseSpace"
#define GIDI_regions2dChars "regions2d"

// 3d Function monikers.
#define GIDI_XYs3dChars "XYs3d"
#define GIDI_gridded3dChars "gridded3d"

// Double differentials
#define GIDI_optionsChars "options"
#define GIDI_S_alpha_betaChars "S_alpha_beta"
#define GIDI_S_tableChars "S_table"
#define GIDI_formFactorChars "formFactor"
#define GIDI_realAnomalousFactorChars "realAnomalousFactor"
#define GIDI_imaginaryAnomalousFactorChars "imaginaryAnomalousFactor"
#define GIDI_characteristicCrossSectionChars "characteristicCrossSection"
#define GIDI_DebyeWallerChars "DebyeWaller"
#define GIDI_massChars "mass"
#define GIDI_freeAtomCrossSectionChars "freeAtomCrossSection"
#define GIDI_e_criticalChars "e_critical"
#define GIDI_e_maxChars "e_max"
#define GIDI_T_effectiveChars "T_effective"
#define GIDI_UChars "U"
#define GIDI_thetaChars "theta"
#define GIDI_gChars "g"

// Distribution forms.
#define GIDI_multiGroup3dChars "multiGroup3d"
#define GIDI_angularTwoBodyChars "angularTwoBody"
#define GIDI_uncorrelatedChars "uncorrelated"
#define GIDI_angularChars "angular"
#define GIDI_energyChars "energy"
#define GIDI_KalbachMannChars "KalbachMann"
#define GIDI_energyAngularChars "energyAngular"
#define GIDI_energyAngularMCChars "energyAngularMC"
#define GIDI_angularEnergyChars "angularEnergy"
#define GIDI_angularEnergyMCChars "angularEnergyMC"
#define GIDI_LLNLAngularEnergyChars "LLNLAngularEnergy"
#define GIDI_LLNLAngularOfAngularEnergyChars "LLNLAngularOfAngularEnergy"
#define GIDI_LLNLAngularEnergyOfAngularEnergyChars "LLNLAngularEnergyOfAngularEnergy"
#define GIDI_coherentPhotonScatteringChars "coherentPhotonScattering"
#define GIDI_incoherentPhotonScatteringChars "incoherentPhotonScattering"
#define GIDI_TNSL_coherentElasticChars "thermalNeutronScatteringLaw_coherentElastic"
#define GIDI_TNSL_incoherentElasticChars "thermalNeutronScatteringLaw_incoherentElastic"
#define GIDI_TNSL_incoherentInelasticChars "thermalNeutronScatteringLaw_incoherentInelastic"
#define GIDI_thermalNeutronScatteringLawChars "thermalNeutronScatteringLaw"
#define GIDI_branching3dChars "branching3d"
#define GIDI_unspecifiedChars "unspecified"

#define GIDI_scatteringAtomsChars "scatteringAtoms"
#define GIDI_scatteringAtomChars "scatteringAtom"

#define GIDI_resonancesWithBackgroundChars "resonancesWithBackground"
#define GIDI_resonancesChars "resonances"
#define GIDI_resonanceBackground1dChars   "background"
#define GIDI_resolvedRegionChars "resolvedRegion"
#define GIDI_unresolvedRegionChars "unresolvedRegion"
#define GIDI_fastRegionChars "fastRegion"

#define GIDI_CoulombPlusNuclearElasticChars "CoulombPlusNuclearElastic"
#define GIDI_RutherfordScatteringChars "RutherfordScattering"

#define GIDI_URR_probabilityTables1ddChars "URR_probabilityTables1d"
#define GIDI_LLNLLegendreChars "LLNLLegendre"

#define GIDI_pidsChars "pids"

#define GIDI_axesChars "axes"
#define GIDI_axisChars "axis"
#define GIDI_gridChars "grid"
#define GIDI_fluxNodeChars "flux"

#define GIDI_function1dsChars "function1ds"
#define GIDI_function2dsChars "function2ds"
#define GIDI_uncertaintyChars "uncertainty"
#define GIDI_valuesChars "values"
#define GIDI_startsChars "starts"
#define GIDI_lengthsChars "lengths"
#define GIDI_fChars "f"
#define GIDI_rChars "r"
#define GIDI_aChars "a"
#define GIDI_bChars "b"
#define GIDI_EFL_Chars "EFL"
#define GIDI_EFH_Chars "EFH"
#define GIDI_T_M_Chars "T_M"
#define GIDI_weightedChars "weighted"
#define GIDI_arrayChars "array"

#define GIDI_promptProductKEChars "promptProductKE"
#define GIDI_promptNeutronKEChars "promptNeutronKE"
#define GIDI_delayedNeutronKEChars  "delayedNeutronKE"
#define GIDI_promptGammaEnergyChars "promptGammaEnergy"
#define GIDI_delayedGammaEnergyChars "delayedGammaEnergy"
#define GIDI_delayedBetaEnergyChars "delayedBetaEnergy"
#define GIDI_neutrinoEnergyChars "neutrinoEnergy"
#define GIDI_nonNeutrinoEnergyChars "nonNeutrinoEnergy"
#define GIDI_totalEnergyChars "totalEnergy"

#define GIDI_trueChars "true"
#define GIDI_fissionGenreChars "fissionGenre"
#define GIDI_libraryChars "library"
#define GIDI_startChars "start"
#define GIDI_projectileChars "projectile"
#define GIDI_targetChars "target"
#define GIDI_evaluationChars "evaluation"
#define GIDI_interactionChars "interaction"
#define GIDI_standardTargetChars "standardTarget"
#define GIDI_standardEvaluationChars "standardEvaluation"
#define GIDI_projectileFrameChars "projectileFrame"
#define GIDI_ENDF_MT_Chars "ENDF_MT"
#define GIDI_dateChars "date"
#define GIDI_derivedFromChars "derivedFrom"
#define GIDI_versionChars "version"
#define GIDI_temperatureChars "temperature"
#define GIDI_muCutoffChars "muCutoff"
#define GIDI_lMaxChars "lMax"
#define GIDI_parametersChars "parameters"
#define GIDI_upperCalculatedGroupChars "upperCalculatedGroup"
#define GIDI_calculatedAtThermalChars "calculatedAtThermal"
#define GIDI_asymmetricChars "asymmetric"
#define GIDI_valueTypeChars "valueType"

#define GIDI_shapeChars "shape"
#define GIDI_productFrameChars "productFrame"
#define GIDI_interpolationChars "interpolation"
#define GIDI_interpolationQualifierChars "interpolationQualifier"
#define GIDI_outerDomainValueChars "outerDomainValue"
#define GIDI_indexChars "index"
#define GIDI_labelChars "label"
#define GIDI_unitChars "unit"
#define GIDI_hrefChars "href"
#define GIDI_initialChars "initial"
#define GIDI_finalChars "final"
#define GIDI_minChars "min"
#define GIDI_maxChars "max"
#define GIDI_valueChars "value"
#define GIDI_domainMinChars "minDomain"
#define GIDI_domainMaxChars "maxDomain"
#define GIDI_numberOfProductsChars "numberOfProducts"
#define GIDI_pathChars "path"
#define GIDI_styleChars "style"
#define GIDI_genreChars "genre"
#define GIDI_processChars "process"
#define GIDI_pidChars "pid"
#define GIDI_offsetChars "offset"
#define GIDI_countChars "count"

#define GIDI_inverseSpeedChars "inverseSpeed"

#define GIDI_centerOfMassChars "centerOfMass"
#define GIDI_labChars "lab"
#define GIDI_twoBodyChars "twoBody"
#define GIDI_NBodyChars "NBody"

typedef std::pair<std::string, double> stringAndDoublePair;
typedef std::vector<stringAndDoublePair> stringAndDoublePairs;
typedef std::map<std::string, ParticleInfo> ParticleSubstitution;

#ifdef _WIN32
#define GIDI_FILE_SEPARATOR   "\\"
#else
#define GIDI_FILE_SEPARATOR   "/"
#endif

std::vector<std::string> vectorOfStrings( std::string const &a_string );

/*
============================================================
========================= Exception ========================
============================================================
*/
class Exception : public std::runtime_error {

    public :
        explicit Exception( std::string const &a_message );
};

/*
============================================================
====================== FormatVersion =======================
============================================================
*/
class FormatVersion {

    private:
        std::string m_format;               /**< The GNDS format version. */
        int m_major;                        /**< The GNDS format major value as an integer. */
        int m_minor;                        /**< The GNDS format minor value as an integer. */
        std::string m_patch;                /**< The GNDS format patch string. This will be an empty string except for unofficial formats. */

    public:
        FormatVersion( );
        FormatVersion( std::string const &a_formatVersion );

        std::string const &format( ) const { return( m_format ); }
        int major( ) const { return( m_major ); }
        int minor( ) const { return( m_minor ); }
        std::string const &patch( ) const { return( m_patch ); }

        bool setFormat( std::string const &a_formatVersion );
        bool supported( );
};

namespace Construction {

/* *********************************************************************************************************
 * This enum allows a user to limit the data read in by various constructors. Limiting the data speeds up the reading
 * and parsing, and uses less memory.
 ***********************************************************************************************************/

enum class ParseMode : int { all,                             /**< Read and parse all data. */
                             multiGroupOnly,                  /**< Only read and parse data needed for multi-group transport. */
                             MonteCarloContinuousEnergy,      /**< Only read and parse data needed for continuous energy Monte Carlo. */
                             excludeProductMatrices,          /**< Read and parse all data but multi-group product matrices. */
                             readOnly,                        /**< Only read and parse all the data but do no calculations. Useful for reading an incomplete GNDS file. */
                             outline                          /**< Does parse any component data (e.g., cross section, multiplicity, distribution). */ };

enum class PhotoMode : int { nuclearAndAtomic,                /**< Instructs method Map::protare to create a Protare with both photo-nuclear and photo-atomic data when the projectile is photon. */
                             nuclearOnly,                     /**< Instructs method Map::protare to create a Protare with only photo-nuclear data when the projectile is photon. */
                             atomicOnly                       /**< Instructs method Map::protare to create a Protare with only photo-atomic data when the projectile is photon. */ };

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
        void setPhotoMode( PhotoMode a_photoMode ) { m_photoMode = a_photoMode; }

        int useSystem_strtod( ) const { return( m_useSystem_strtod ); }
        void setUseSystem_strtod( bool a_useSystem_strtod ) { m_useSystem_strtod = a_useSystem_strtod ? 1 : 0; }
};

}               // End namespace Construction.

/*
============================================================
========================= SetupInfo ========================
============================================================
*/
class SetupInfo {

    public:
        Protare *m_protare;
        HAPI::DataManager *m_dataManager;
        ParticleSubstitution *m_particleSubstitution;
        FormatVersion m_formatVersion;
        Styles::MultiGroup *m_multiGroup;

        SetupInfo( Protare *a_protare ) :
                m_protare( a_protare ),
                m_dataManager ( nullptr ),
                m_particleSubstitution( nullptr ),
                m_formatVersion( ),
                m_multiGroup( nullptr ) {

        }
        ~SetupInfo(){
          delete m_dataManager;
        }
};

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

        void print( );
        void clear( ) { m_lines.clear( ); }      /**< Clears the contents of *m_lines*. */
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
        Ancestry *m_ancestor;                                   /**< The parent node of *this*. */
        std::string m_attribute;                                /**< The name of the attribute in the node that uniquely identifies the node when the parent node contains other child nodes with the same moniker. */

        Ancestry *findInAncestry2( std::size_t a_index, std::vector<std::string> const &a_segments );
        Ancestry const *findInAncestry2( std::size_t a_index, std::vector<std::string> const &a_segments ) const ;

    public:
        Ancestry( std::string const &a_moniker, std::string const &a_attribute = "" );
        virtual ~Ancestry( );

        std::string moniker( ) const { return( m_moniker ); }                               /**< Returns the value of the **m_moniker** member. */
        void setMoniker( std::string const &a_moniker ) { m_moniker = a_moniker; }          /**< Set the value of the **m_moniker** member to *a_moniker*. */
        Ancestry *ancestor( ) { return( m_ancestor ); }                                     /**< Returns the value of the **m_ancestor** member. */
        Ancestry const *ancestor( ) const { return( m_ancestor ); }                                     /**< Returns the value of the **m_ancestor** member. */
        void setAncestor( Ancestry *a_ancestor ) { m_ancestor = a_ancestor; }               /**< Sets the **m_ancestor** member to *a_ancestor*. */
        std::string attribute( ) const { return( m_attribute ); }                           /**< Returns the value of the **m_attribute** member. */

        Ancestry *root( );
        Ancestry const *root( ) const ;
        bool isChild( Ancestry *a_instance ) { return( this == a_instance->m_ancestor ); }  /**< Returns true if **a_instance** is a child of *this*. */
        bool isParent( Ancestry *a_parent ) { return( this->m_ancestor == a_parent ); }     /**< Returns true if **a_instance** is the parent of *this*. */
        bool isRoot( ) const { return( this->m_ancestor == nullptr ); }                     /**< Returns true if *this* is the root ancestor. */

        Ancestry *findInAncestry( std::string const &a_href );
        Ancestry const *findInAncestry( std::string const &a_href ) const ;

            /* *********************************************************************************************************//**
             * Used to tranverse **GNDS** nodes. This method returns a pointer to a derived class' *a_item* member or nullptr if none exists.
             *
             * @param a_item    [in]    The name of the class member whose pointer is to be return.
             * @return                  The pointer to the class member or nullptr if class does not have a member named a_item.
             ***********************************************************************************************************/
        virtual Ancestry *findInAncestry3( std::string const &a_item ) = 0;
        virtual Ancestry const *findInAncestry3( std::string const &a_item ) const = 0;

        virtual std::string xlinkItemKey( ) const { return( "" ); }                         /**< Returns the value of *this*'s key. */
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
        FormType m_type;                        /**< The type of the form. */
        std::string m_label;                    /**< The label for the form. */

    public:
        Form( FormType a_type );
        Form( std::string const &a_moniker, FormType a_type, std::string const &a_label );
        Form( HAPI::Node const &a_node, SetupInfo &a_setupInfo, FormType a_type, Suite *a_suite = nullptr );
        Form( Form const &a_form );
        virtual ~Form( );

        Suite *parent( ) const { return( m_parent ); }                                          /**< Returns the value of the **m_parent** member. */
        std::string const &label( ) const { return( m_label ); }                                /**< Returns the value of the **m_label** member. */
        void setLabel( std::string const &a_label ) { m_label = a_label; }                      /**< Sets the **m_label** member to *a_label*. */
        FormType type( ) const { return( m_type ); }                                            /**< Returns the value of the **m_type** member. */
        Form const *sibling( std::string a_label ) const ;

        Ancestry *findInAncestry3( std::string const &a_item ) { return( nullptr ); }
        Ancestry const *findInAncestry3( std::string const &a_item ) const { return( nullptr ); }
        std::string xlinkItemKey( ) const {

            if( m_label == "" ) return( "" );
            return( buildXLinkItemKey( GIDI_labelChars, m_label ) );
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
        PhysicalQuantity( HAPI::Node const &a_node, SetupInfo &a_setupInfo );
        PhysicalQuantity( double a_value, std::string a_unit );
        PhysicalQuantity( PhysicalQuantity const &a_physicalQuantity ) : 
                Form( FormType::physicalQuantity ),
                m_value( a_physicalQuantity.value( ) ),
                m_unit( a_physicalQuantity.unit( ) ) { }
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
        PhysicalQuantity m_excitationEnergy;                    /**< If the particle is a PoPI::Nuclide or PoPI::Nucleus, this is it nuclear excitation energy. Otherwise, it is 0. */

    public:
        ParticleInfo( std::string const &a_id, std::string const &a_pid, double a_mass, double a_excitationEnergy = 0.0 );
        ParticleInfo( std::string const &a_id, PoPI::Database const &a_globalPoPs, PoPI::Database const &a_internalPoPs, bool a_requiredInGlobalPoPs );
        ParticleInfo( ParticleInfo const &a_particleInfo );

        std::string const &ID( ) const { return( m_id  ); }                     /**< Returns a const reference to**m_id** member. */
        std::string const &qualifier( ) const { return( m_qualifier ); }        /**< Returns a const reference to **m_qualifier***. */
        std::string const &pid( ) const { return( m_pid  ); }                   /**< Returns a const reference to **m_pid** member. */
        bool isAlias( ) const { return( m_pid != "" ); }                        /**< Returns true if particle id is an alias and false otherwise. */

        PhysicalQuantity const &mass( ) const { return( m_mass ); }             /**< Returns a const reference to **m_mass** member. */
        PhysicalQuantity const &excitationEnergy( ) const { return( m_excitationEnergy ); }     /**< Returns a const reference to **m_excitationEnergy** member. */
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
        AxisDomain( HAPI::Node const &a_node, SetupInfo &a_setupInfo );
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
        std::string m_unit;                                                     /**< The unit for the axis. */
        std::string m_href;                                                     /**< The **GNDS**'s href if instance points to another Axis or Grid instance. */

    public:
        Axis( HAPI::Node const &a_node, SetupInfo &a_setupInfo, FormType a_type = FormType::axis );
        Axis( int a_index, std::string a_label, std::string a_unit, FormType a_type = FormType::axis );
        Axis( Axis const &a_axis );
        virtual ~Axis( );

        int index( ) const { return( m_index ); }                               /**< Returns the value of the **m_index** member. */
        std::string const &unit( ) const { return( m_unit ); }                  /**< Returns the value of the **m_unit** member. */

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
        nf_Buffer<double> m_values;                                                 /**< The **GNDS grid**'s values. */

    public:
        Grid( HAPI::Node const &a_node, SetupInfo &a_setupInfo, int a_useSystem_strtod );
        Grid( Grid const &a_grid );

        std::size_t size( ) const { return( m_values.size( ) ); }                   /**< Returns the number of values in the **m_values** member. */
        inline double &operator[]( std::size_t a_index ) noexcept { return( m_values[a_index] ); }  /**< Returns the value at m_values[a_index]. */

        std::string const &style( ) const { return( m_style ); }                    /**< Returns the value of the **m_style** member. */
        std::string keyName( ) const { return( m_keyName ); }                       /**< Returns the value of the **m_keyName** member. */
        std::string keyValue( ) const { return( m_keyValue ); }                     /**< Returns the value of the **m_keyValue** member. */
        std::string valueType( ) const { return( m_valueType ); }                     /**< Returns the value of the **m_valueType** member. */

        nf_Buffer<double> const &values( ) const { return( m_values ); }          /**< Returns the value of the **m_values** member. */
        nf_Buffer<double> const &data( ) const { return( m_values ); }            /**< Returns the value of the **m_values** member. */

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
=========================== Axes ===========================
============================================================
*/
class Axes : public Form {

    private:
        std::vector<Axis *> m_axes;                                                 /**< Stores the list of Axis nodes. */

    public:
        Axes( );
        Axes( HAPI::Node const &a_node, SetupInfo &a_setupInfo, int a_useSystem_strtod );
        Axes( Axes const &a_axes );
        ~Axes( );

        std::size_t size( ) const { return( m_axes.size( ) ); }                     /**< Returns the number of *Axis* instances in *this*. */
        Axis const *operator[]( std::size_t a_index ) const { return( (m_axes[a_index]) ); }    /**< Returns m_axes[a_index]. */
        std::size_t dimension( ) const { return( m_axes.size( ) - 1 ); }            /**< Returns the dimension of the instance. */

        void append( Axis *a_axis ) { m_axes.push_back( a_axis ); }                 /**< Appends *a_axis* to the* list of *Axis* nodes. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;

        static Axes makeAxes( std::vector<std::pair<std::string, std::string>> const &a_labelsAndUnits );
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
        nf_Buffer<int> m_starts;                                                /**< The start values. */
        nf_Buffer<int> m_lengths;                                               /**< The length values. */
        nf_Buffer<double> m_dValues;                                            /**< The given array data. */
//        int32_t *m_starts;                                                      /**< The start values. */
//        int32_t *m_lengths;                                                     /**< The length values. */
//        std::vector<double> m_dValues;                                          /**< The given array data. */

        FlattenedArrayData( HAPI::Node const &a_node, SetupInfo &a_setupInfo, int a_dimensions, int a_useSystem_strtod );
        ~FlattenedArrayData( );

        std::vector<int> const &shape( ) const { return( m_shape ); }
        void setToValueInFlatRange( int a_start, int a_end, double a_value );
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
        Array3d( HAPI::Node const &a_node, SetupInfo &a_setupInfo, int a_useSystem_strtod );
        ~Array3d( );

        std::size_t size( ) const { return( m_array.m_shape.back( ) ); }        /**< The length of the 3d diminsion. */

        Matrix matrix( std::size_t a_index ) const ;

        void modifiedMultiGroupElasticForTNSL( int maxTNSL_index );
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const { m_array.toXMLList( a_writeInfo, a_indent ); }
};

namespace Functions {

/*
============================================================
====================== FunctionForm ======================
============================================================
*/
class FunctionForm : public Form {

    private:
        int m_dimension;                                    /**< The dimension of the function (i.e., the number of independent axes. */
        Axes m_axes;                                        /**< The axes node for the function. */
        ptwXY_interpolation m_interpolation;                /**< The interpolation for functions highest independent axis and its dependent axis. */
        std::string m_interpolationString;                  /**< The interpolation for functions highest independent axis and its dependent axis. */
        int m_index;                                        /**< Currently not used. */
        double m_outerDomainValue;                          /**< If function is part of a higher dimensional function, this is the next higher dimensions domain value. */

    public:
        FunctionForm( std::string const &a_moniker, FormType a_type, int a_dimension, ptwXY_interpolation a_interpolation, int a_index, double a_outerDomainValue );
        FunctionForm( std::string const &a_moniker, FormType a_type, int a_dimension, Axes const &a_axes, ptwXY_interpolation a_interpolation, int a_index, double a_outerDomainValue );
        FunctionForm( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, FormType a_type, int a_dimension, Suite *a_suite = nullptr );
        FunctionForm( FunctionForm const &a_form );
        ~FunctionForm( );

        int dimension( ) const { return( m_dimension ); }                                       /**< Returns the value of the **m_dimension** member. */

        int index( ) const { return( m_index ); }                                               /**< Returns the value of the **m_index** member. */
        double outerDomainValue( ) const { return( m_outerDomainValue ); }                      /**< Returns the value of the **m_outerDomainValue** member. */
        void setOuterDomainValue( double a_outerDomainValue ) { m_outerDomainValue = a_outerDomainValue; }
        Axes const &axes( ) const { return( m_axes ); }                                         /**< Returns a const reference to the **m_axes** member. */
        Axes &axes( ) { return( m_axes ); }                                                     /**< Returns a reference to the **m_axes** member. */

        ptwXY_interpolation interpolation( ) const { return( m_interpolation ); }               /**< Returns the value of the **m_interpolation** member. */
        void setInterpolation( ptwXY_interpolation a_interpolation ) { m_interpolation = a_interpolation; }    /**< Sets the **m_interpolation** member to **a_interpolation**. */
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
        Function1dForm( std::string const &a_moniker, FormType a_type, ptwXY_interpolation a_interpolation, int a_index, double a_outerDomainValue );
        Function1dForm( std::string const &a_moniker, FormType a_type, Axes const &a_axes, ptwXY_interpolation a_interpolation, int a_index, double a_outerDomainValue );
        Function1dForm( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, FormType a_type, Suite *a_suite = nullptr );
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
        Constant1d( Axes const &a_axes, double value, double a_domainMin, double a_domainMax, int a_index = 0, double a_outerDomainValue = 0.0 );
        Constant1d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
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
        XYs1d( );
        XYs1d( Axes const &a_axes, ptwXY_interpolation m_interpolation, int a_index = 0, double a_outerDomainValue = 0.0 );
        XYs1d( Axes const &a_axes, ptwXY_interpolation m_interpolation, std::vector<double> const &a_values, int a_index = 0, double a_outerDomainValue = 0.0 );
        XYs1d( Axes const &a_axes, ptwXYPoints *a_ptwXY, int a_index = 0, double a_outerDomainValue = 0.0 );
        XYs1d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
        XYs1d( XYs1d const &a_XYs1d );
        ~XYs1d( );

        std::size_t size( ) const { return( ptwXY_length( nullptr, m_ptwXY ) ); }   /**< Returns the number of points (i.e., x,y pairs) in this. */
        ptwXYPoints const *ptwXY( ) const { return( m_ptwXY ); }                    /**< Returns the value of the **m_ptwXY** member. */
        ptwXYPoints *ptwXY( ) { return( m_ptwXY ); }                                /**< Returns the value of the **m_ptwXY** member. */

        std::pair<double, double> operator[]( std::size_t a_index ) const ;
        XYs1d operator+( XYs1d const &a_XYs1d ) const ;
        XYs1d &operator+=( XYs1d const &a_XYs1d );
        XYs1d operator-( XYs1d const &a_XYs1d ) const ;
        XYs1d &operator-=( XYs1d const &a_XYs1d );
        XYs1d operator*( XYs1d const &a_XYs1d ) const ;
        XYs1d &operator*=( XYs1d const &a_XYs1d );

        double domainMin( ) const { return( (*this)[0].first ); }                   /**< Returns first x1 value of this. */
        double domainMax( ) const { return( (*this)[size( )-1].first ); }           /**< Returns last x1 value of this. */
        std::vector<double> xs( ) const ;
        std::vector<double> ys( ) const ;
        std::vector<double> ysMappedToXs( std::vector<double> const &a_xs, std::size_t *a_offset ) const ;
        XYs1d domainSlice( double a_domainMin, double a_domainMax, bool a_fill ) const ;
        XYs1d domainSliceMax( double a_domainMax ) const ;

        double evaluate( double a_x1 ) const ;
        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;

        void print( char const *a_format );
        void print( std::string const &a_format );

        static XYs1d *makeConstantXYs1d( Axes const &a_axes, double a_domainMin, double a_domainMax, double a_value );
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
        Ys1d( Axes const &a_axes, ptwXY_interpolation a_interpolation, int a_index = 0, double a_outerDomainValue = 0.0 );
        Ys1d( Axes const &a_axes, ptwXY_interpolation a_interpolation, std::size_t a_start, std::vector<double> const &a_Ys, int a_index = 0, double a_outerDomainValue = 0.0 );
        Ys1d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
        Ys1d( Ys1d const &a_Ys1d );
        ~Ys1d( );

        std::size_t size( ) const { return( m_Ys.size( ) ); }                           /**< Returns the number of values in **m_Ys**. */

        double operator[]( std::size_t a_index ) const { return( m_Ys[a_index] ); }     /**< Returns the y value at **m_Ys**[a_index]. */
        void push_back( double a_y ) { m_Ys.push_back( a_y ); }
        Ys1d operator+( Ys1d const &a_Ys1d ) const ;
        Ys1d &operator+=( Ys1d const &a_Ys1d );

        double domainMin( ) const ;
        double domainMax( ) const ;
        std::size_t start( ) const { return( m_start ); }                               /**< Returns the value of the **m_start** member. */
        void setStart( std::size_t a_start ) { m_start = a_start; }                     /**< Sets the **m_start** member to **a_start*. */
        std::size_t length( ) const { return( m_start + m_Ys.size( ) ); }               /**< Returns the sum of m_start and size( ). */
        std::vector<double> const &Ys( ) const { return( m_Ys ); }                      /**< Returns a reference to the list of y-values. */
        std::vector<double> &Ys( ) { return( m_Ys ); }                                  /**< Returns a reference to the list of y-values. */

        double evaluate( double a_x1 ) const ;
        void set( std::size_t a_index, double a_value ) { m_Ys[a_index] = a_value; }    /**< Set the value at **m_Ys**[a_index] to a_value. */
        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;

        void print( char const *a_format );
        void print( std::string const &a_format );
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
        Polynomial1d( Axes const &a_axes, double a_domainMin, double a_domainMax, std::vector<double> const &a_coefficients, int a_index = 0, double a_outerDomainValue = 0.0 );
        Polynomial1d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
        Polynomial1d( Polynomial1d const &a_polynomial1d );
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
        std::vector<double> m_coefficients;                                             /**< The Legendre coefficients. */

    public:
        Legendre1d( Axes const &a_axes, int a_index = 0, double a_outerDomainValue = 0.0 );
        Legendre1d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
        Legendre1d( Legendre1d const &a_Legendre1d );
        ~Legendre1d( );

        double domainMin( ) const { return( -1.0 ); }                                   /**< Returns the value of the *domainMin* which is always -1.0. */
        double domainMax( ) const { return( 1.0 ); }                                    /**< Returns the value of the *domainMax* which is always 1.0. */

        std::vector<double> const &coefficients( ) const { return( m_coefficients ); }  /**< Returns the value of the **m_coefficients** member. */
        std::vector<double> &coefficients( ) { return( m_coefficients ); }              /**< Returns the value of the **m_coefficients** member. */

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
        Gridded1d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
        Gridded1d( Vector const &a_grid, Vector const &a_data, Suite *a_parent );
        ~Gridded1d( );

        double domainMin( ) const { return( m_grid[0] ); }                      /**< Returns the value of the *domainMin*. */
        double domainMax( ) const { return( m_grid[m_grid.size( )-1] ); }       /**< Returns the value of the *domainMax*. */

        Vector const &grid( ) const { return( m_grid ); }                       /**< Returns the value of the **m_grid** member. */
        Vector const &data( ) const { return( m_data ); }                       /**< Returns the value of the **m_data** member. */
        void setData( Vector const &a_data ) { m_data = a_data; }               /**< Sets the **m_data** member to **a_data*. */

        void modifiedMultiGroupElasticForTNSL( int a_maxTNSL_index );
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
        Reference1d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
        ~Reference1d( );

        double domainMin( ) const ;
        double domainMax( ) const ;

        std::string const &xlink( ) const { return( m_xlink ); }                /**< Returns the value of the **m_xlink** member. */
        double evaluate( double a_x1 ) const ;
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
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
        Xs_pdf_cdf1d( Axes const &a_axes, ptwXY_interpolation a_interpolation, std::vector<double> const &a_Xs, 
                std::vector<double> const &a_pdf, std::vector<double> const &a_cdf, int a_index = 0, double a_outerDomainValue = 0.0 );
        Xs_pdf_cdf1d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
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
        std::vector<Function1dForm *> m_function1ds;                            /**< List of regions. */

    public:
        Regions1d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
        ~Regions1d( );

        std::size_t size( ) const { return( m_function1ds.size( ) ); }                          /**< Returns number of regions. */
        Function1dForm const *operator[]( std::size_t a_index ) const { return( m_function1ds[a_index] ); } /**< Returns the region at index *a_index* - 1. */

        double domainMin( ) const ;
        double domainMax( ) const ;

        void append( Function1dForm *a_function );
        double evaluate( double a_x1 ) const ;

        std::vector<double> const &Xs( ) const { return( m_Xs ); }                              /**< Returns the value of the **m_Xs** member. */
        std::vector<Function1dForm *> const &function1ds( ) const { return( m_function1ds ); }        /**< Returns the value of the **m_function1ds** member. */
        std::vector<Function1dForm *> &function1ds( ) { return( m_function1ds ); }             /**< Returns the value of the **m_function1ds** member. */

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
        Branching1dPids( HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
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
        Branching1d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
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
        ResonanceBackgroundRegion1d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
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
        ResonanceBackground1d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
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
        ResonancesWithBackground1d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
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
        URR_probabilityTables1d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
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
        ThermalNeutronScatteringLaw1d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
        ~ThermalNeutronScatteringLaw1d( );

        std::string const &href( ) const { return( m_href ); }              /**< Returns the value of the **m_href** member. */

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
        Unspecified1d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
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
        Function2dForm( std::string const &a_moniker, FormType a_type, ptwXY_interpolation a_interpolation, int a_index, double a_outerDomainValue );
        Function2dForm( std::string const &a_moniker, FormType a_type, Axes const &a_axes, ptwXY_interpolation a_interpolation, int a_index, double a_outerDomainValue );
        Function2dForm( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, FormType a_type, Suite *a_suite = nullptr );
        Function2dForm( Function2dForm const &a_form );
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
        XYs2d( Axes const &a_axes, ptwXY_interpolation a_interpolation, int a_index = 0, double a_outerDomainValue = 0.0 );
        XYs2d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
        ~XYs2d( );

        std::string interpolationQualifier( ) const { return( m_interpolationQualifier ); }         /**< Returns the value of the **m_interpolationQualifier** member. */
        void setInterpolationQualifier( std::string a_interpolationQualifier ) { m_interpolationQualifier = a_interpolationQualifier; };
                                                                                                    /**< Sets the **m_interpolationQualifier** member to *a_interpolationQualifier*. */

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
        Recoil2d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
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
        Isotropic2d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
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
        DiscreteGamma2d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
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
        PrimaryGamma2d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
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
        GeneralEvaporation2d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
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
        SimpleMaxwellianFission2d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
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
        Evaporation2d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
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
        Watt2d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
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
        MadlandNix2d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
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
        Weighted_function2d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
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
        WeightedFunctionals2d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
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
        NBodyPhaseSpace2d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
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
        std::vector<Function2dForm *> m_function2ds;                        /**< List of 2d regions. */

    public:
        Regions2d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
        ~Regions2d( );

        double domainMin( ) const ;
        double domainMax( ) const ;

        void append( Function2dForm *a_function );
        double evaluate( double a_x2, double a_x1 ) const ;

        std::vector<double> const &Xs( ) const { return( m_Xs ); }                                  /**< Returns the value of the **m_Xs** member. */
        std::vector<Function2dForm *> const &function2ds( ) const { return( m_function2ds ); }      /**< Returns the value of the **m_function2ds** member. */
        void toXMLList_func( WriteInfo &a_writeInfo, std::string const &a_indent, bool a_embedded, bool a_inRegions ) const ;
};

/*
============================================================
======================= Function3dForm =====================
============================================================
*/
class Function3dForm : public FunctionForm {

    public:
        Function3dForm( std::string const &a_moniker, FormType a_type, ptwXY_interpolation a_interpolation, int a_index, double a_outerDomainValue );
        Function3dForm( std::string const &a_moniker, FormType a_type, Axes const &a_axes, ptwXY_interpolation a_interpolation, int a_index, double a_outerDomainValue );
        Function3dForm( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, FormType a_type, Suite *a_suite = nullptr );
        Function3dForm( Function3dForm const &a_form );
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
        XYs3d( Axes const &a_axes, ptwXY_interpolation a_interpolation = ptwXY_interpolationLinLin, int a_index = 0, double a_outerDomainValue = 0.0 );
        XYs3d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
        ~XYs3d( );

        std::string interpolationQualifier( ) const { return( m_interpolationQualifier ); }         /**< Returns the value of the **m_interpolationQualifier** member. */
        void setInterpolationQualifier( std::string a_interpolationQualifier ) { m_interpolationQualifier = a_interpolationQualifier; };
                                                                                                    /**< Sets the **m_interpolationQualifier** member to *a_interpolationQualifier*. */

        double domainMin( ) const ;
        double domainMax( ) const ;
        double evaluate( double a_x3, double a_x2, double a_x1 ) const ;

        std::vector<double> const &Xs( ) const { return( m_Xs ); }                                  /**< Returns the value of the **m_Xs** member. */
        std::vector<Function2dForm *> const &function2ds( ) const { return( m_function2ds ); }      /**< Returns a const reference to the **m_function2ds** member. */

        void append( Function2dForm *a_function2d );                                                /**< Appends the 2d function *a_function2d* to the end the *this*. */
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
        Gridded3d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo );
        ~Gridded3d( );

        double domainMin( ) const { return( 0.0 ); }                                        /**< Not properly implemented. */
        double domainMax( ) const { return( 0.0 ); }                                        /**< Not properly implemented. */
        double evaluate( double a_x3, double a_x2, double a_x1 ) const { return( 0.0 ); }   /**< Not properly implemented. */

        std::string const &domain1Unit( ) const { return( m_domain1Unit ); }                /**< Returns the value of the **m_domain1Unit** member. */
        std::string const &domain2Unit( ) const { return( m_domain2Unit ); }                /**< Returns the value of the **m_domain2Unit** member. */
        std::string const &domain3Unit( ) const { return( m_domain3Unit ); }                /**< Returns the value of the **m_domain3Unit** member. */
        std::string const &rangeUnit( ) const { return( m_rangeUnit ); }                    /**< Returns the value of the **m_rangeUnit** member. */
        Array3d const &data( ) const { return( m_data ); }                                  /**< Returns the value of the **m_data** member. */

        void modifiedMultiGroupElasticForTNSL( int maxTNSL_index );
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

}               // End namespace Functions.

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
        Base( HAPI::Node const &a_node, SetupInfo &a_setupInfo, FormType a_type, Suite *a_parent );
};

/*
============================================================
================ CoherentPhotoAtomicScattering =============
============================================================
*/
class CoherentPhotoAtomicScattering : public Base {

    private:
        Functions::Function1dForm *m_formFactor;                                   /**< The form factor for coherent photo-atomic scattering. */
        Functions::Function1dForm *m_realAnomalousFactor;                          /**< The real anomalous factor of coherent photo-atomic scattering. */
        Functions::Function1dForm *m_imaginaryAnomalousFactor;                     /**< The imaginary anomalous factor of coherent photo-atomic scattering. */

    public:
        CoherentPhotoAtomicScattering( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs, Suite *a_parent );
        ~CoherentPhotoAtomicScattering( );

        Functions::Function1dForm *formFactor( ) { return( m_formFactor ); }                                            /**< Returns the value of the **m_formFactor** member. */
        Functions::Function1dForm const *formFactor( ) const { return( m_formFactor ); }                                /**< Returns the value of the **m_formFactor** member. */
        Functions::Function1dForm *realAnomalousFactor( ) { return( m_realAnomalousFactor ); }                          /**< Returns the value of the **m_realAnomalousFactor** member. */
        Functions::Function1dForm const *realAnomalousFactor( ) const { return( m_realAnomalousFactor ); }              /**< Returns the value of the **m_realAnomalousFactor** member. */
        Functions::Function1dForm *imaginaryAnomalousFactor( ) { return( m_imaginaryAnomalousFactor ); }                /**< Returns the value of the **m_imaginaryAnomalousFactor** member. */
        Functions::Function1dForm const *imaginaryAnomalousFactor( ) const { return( m_imaginaryAnomalousFactor ); }    /**< Returns the value of the **m_imaginaryAnomalousFactor** member. */
};

/*
============================================================
============== IncoherentPhotoAtomicScattering =============
============================================================
*/
class IncoherentPhotoAtomicScattering : public Base {

    private:
        Functions::Function1dForm *m_scatteringFunction;                       /**< The scattering factor for incoherent photo-atomic scattering. */

    public:
        IncoherentPhotoAtomicScattering( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs, Suite *a_parent );
        ~IncoherentPhotoAtomicScattering( );

        Functions::Function1dForm const *scatteringFunction( ) const { return( m_scatteringFunction ); }           /**< Returns the value of the **m_scatteringFunction** member. */
};

namespace n_ThermalNeutronScatteringLaw {

/*
============================================================
========================== S_table =========================
============================================================
*/
class S_table : public Form {


    private:
        Functions::Function2dForm *m_function2d;           /**< The cumulative scattering factor S(E,T). */

    public:
        S_table( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo );
        ~S_table( );

        Functions::Function2dForm *function2d( ) { return( m_function2d ); }           /**< Returns the value of the **m_function2d** member. */
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
        CoherentElastic( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs, Suite *a_parent );
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
        Functions::Function1dForm *m_function1d;                                   /**< The 1-d function representing the Debye-Waller function W(T). */

    public:
        DebyeWaller( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo );
        ~DebyeWaller( );

        Functions::Function1dForm *function1d( ) { return( m_function1d ); }       /**< Returns the value of the **m_function1d** member. */
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
        IncoherentElastic( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs, Suite *a_parent );
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
        Options( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo );
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
        Functions::Function1dForm *m_function1d;                               /**< The 1-d function representing effective temperature. */

    public:
        T_effective( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo );
        ~T_effective( );

        Functions::Function1dForm const *function1d( ) const { return( m_function1d ); }   /**< Returns the value of the **m_function1d** member. */
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
        ScatteringAtom( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo );
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
        Functions::Function3dForm *m_function3d;                           /**< The S(alpha,beta,T) function. */

    public:
        S_alpha_beta( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo );
        ~S_alpha_beta( );

        Functions::Function3dForm *function3d( ) { return( m_function3d ); }                   /**< Returns the value of the **m_function3d** member. */
};

}               // End namespace n_ThermalNeutronScatteringLaw.

}               // End namespace DoubleDifferentialCrossSection.

namespace Distributions {

/*
============================================================
========================= Distribution =====================
============================================================
*/
class Distribution : public Form {

    private:
        Frame m_productFrame;                                                   /**< The product frame for the distribution form. */

    public:
        Distribution( std::string const &a_moniker, FormType a_type, std::string const &a_label, Frame a_productFrame );
        Distribution( HAPI::Node const &a_node, SetupInfo &a_setupInfo, FormType a_type, Suite *a_parent );

        Frame productFrame( ) const { return( m_productFrame ); }               /**< Returns the value of the **m_productFrame** member. */
        void toXMLNodeStarter( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
======================= AngularTwoBody =====================
============================================================
*/
class AngularTwoBody : public Distribution {

    private:
        Functions::Function2dForm *m_angular;                                              /**< The P(mu|E) distribution as a Function2dForm. */

    public:
        AngularTwoBody( std::string const &a_label, Frame a_productFrame, Functions::Function2dForm *a_angular = nullptr );
        AngularTwoBody( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
        ~AngularTwoBody( );

        Functions::Function2dForm const *angular( ) const { return( m_angular ); }         /**< Returns the value of the **m_angular** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
========================= KalbachMann ======================
============================================================
*/
class KalbachMann : public Distribution {

    private:
        Functions::Function2dForm *m_f;                                                    /**< The P(E'|E) distribution as a Function2dForm. */
        Functions::Function2dForm *m_r;                                                    /**< The Kalbach/Mann r(E,E') function as a Function2dForm. */
        Functions::Function2dForm *m_a;                                                    /**< The Kalbach/Mann a(E,E') function as a Function2dForm. */

    public:
        KalbachMann( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
        ~KalbachMann( );

        Functions::Function2dForm const *f( ) const { return( m_f ); }                     /**< Returns the value of the **m_f** member. */
        Functions::Function2dForm const *r( ) const { return( m_r ); }                     /**< Returns the value of the **m_r** member. */
        Functions::Function2dForm const *a( ) const { return( m_a ); }                     /**< Returns the value of the **m_a** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
======================== EnergyAngular =====================
============================================================
*/
class EnergyAngular : public Distribution {

    private:
        Functions::Function3dForm *m_energyAngular;                                                /**< The P(E',mu|E) distribution as a Function3dForm. */

    public:
        EnergyAngular( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
        ~EnergyAngular( );

        Functions::Function3dForm const *energyAngular( ) const { return( m_energyAngular ); }     /**< Returns the value of the **m_energyAngular** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
======================= EnergyAngularMC ====================
============================================================
*/
class EnergyAngularMC : public Distribution {

    private:
        Functions::Function2dForm *m_energy;                                               /**< The P(E'|E) distribution as a Function2dForm. */
        Functions::Function3dForm *m_energyAngular;                                        /**< The P(mu|E,E') distribution as a Function3dForm. */

    public:
        EnergyAngularMC( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
        ~EnergyAngularMC( );

        Functions::Function2dForm const *energy( ) const { return( m_energy ); }                   /**< Returns the value of the **m_energy** member. */
        Functions::Function3dForm const *energyAngular( ) const { return( m_energyAngular ); }     /**< Returns the value of the **m_energyAngular** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
======================== AngularEnergy =====================
============================================================
*/
class AngularEnergy : public Distribution {

    private:
        Functions::Function3dForm *m_angularEnergy;                                                /**< The P(mu,E'|E) distribution as a Function3dForm. */

    public:
        AngularEnergy( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
        ~AngularEnergy( );

        Functions::Function3dForm const *angularEnergy( ) const { return( m_angularEnergy ); }     /**< Returns the value of the **m_angularEnergy** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
======================= AngularEnergyMC ====================
============================================================
*/
class AngularEnergyMC : public Distribution {

    private:
        Functions::Function2dForm *m_angular;                                                      /**< The P(mu|E) distribution as a Function2dForm. */
        Functions::Function3dForm *m_angularEnergy;                                                /**< The P(E'|E,mu) distribution as a Function3dForm. */

    public:
        AngularEnergyMC( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
        ~AngularEnergyMC( );

        Functions::Function2dForm const *angular( ) const { return( m_angular ); }                 /**< Returns the value of the **m_angular** member. */
        Functions::Function3dForm const *angularEnergy( ) const { return( m_angularEnergy ); }     /**< Returns the value of the **m_angularEnergy** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
========================= Uncorrelated =====================
============================================================
*/
class Uncorrelated : public Distribution {

    private:
        Functions::Function2dForm *m_angular;                                              /**< The P(mu|E) distribution as a Function2dForm. */
        Functions::Function2dForm *m_energy;                                               /**< The P(E'|E) distribution as a Function2dForm. */

    public:
        Uncorrelated( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
        ~Uncorrelated( );

        Functions::Function2dForm const *angular( ) const { return( m_angular ); }         /**< Returns the value of the **m_angular** member. */
        Functions::Function2dForm const *energy( ) const { return( m_energy ); }           /**< Returns the value of the **m_energy** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
========================= MultiGroup3d =====================
============================================================
*/
class MultiGroup3d : public Distribution {

    private:
        Functions::Gridded3d m_gridded3d;                                              /**< The multi-group Legendre distribution as a Gridded3d instance. */

    public:
        MultiGroup3d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );

        Functions::Gridded3d const &data( ) const { return( m_gridded3d ); }           /**< Returns the value of the **m_gridded3d** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
====================== LLNLAngularEnergy ===================
============================================================
*/
class LLNLAngularEnergy : public Distribution {

    private:
        Functions::Function2dForm *m_angular;                                          /**< The P(mu|E) distribution as a Function2dForm. */
        Functions::Function3dForm *m_angularEnergy;                                    /**< The P(E'|E,mu) distribution as a Function3dForm. */

    public:
        LLNLAngularEnergy( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
        ~LLNLAngularEnergy( );

        Functions::Function2dForm const *angular( ) const { return( m_angular ); }                 /**< Returns the value of the **m_angular** member. */
        Functions::Function3dForm const *angularEnergy( ) const { return( m_angularEnergy ); }     /**< Returns the value of the **m_angularEnergy** member. */
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
        CoherentPhotoAtomicScattering( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );

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
        IncoherentPhotoAtomicScattering( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );

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
        ThermalNeutronScatteringLaw( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );

        std::string const &href( ) const { return( m_href ); }                          /**< Returns the value of the **m_href** member. */
};

/*
============================================================
======================== Branching3d =======================
============================================================
*/
class Branching3d : public Distribution {

    public:
        Branching3d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );

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
        Reference3d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );

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
        Unspecified( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

}                     // End of namespace Distributions.

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
        Suite( );
        Suite( std::string const &a_moniker );
        Suite( Construction::Settings const &a_construction, std::string const &a_moniker, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, 
                        PoPI::Database const &a_internalPoPs, parseSuite a_parseSuite, Styles::Suite const *a_styles );
        ~Suite( );

        std::size_t size( ) const { return( m_forms.size( ) ); }                            /**< Returns the number of node contained by *this*. */
        typedef forms::iterator iterator;
        typedef forms::const_iterator const_iterator;
        iterator begin( ) { return m_forms.begin( ); }                                      /**< The C++ *begin iterator* for *this*. */
        const_iterator begin( ) const { return m_forms.begin( ); }                          /**< The C++ const *begin iterator* for *this*. */
        iterator end( ) { return m_forms.end( ); }                                          /**< The C++ *end iterator* for *this*. */
        const_iterator end( ) const { return m_forms.end( ); }                              /**< The C++ const *end iterator* for *this*. */
        int operator[]( std::string const &a_label ) const ;
        template<typename T> T       *get( std::size_t a_Index );
        template<typename T> T const *get( std::size_t a_Index ) const ;
        template<typename T> T       *get( std::string const &a_label );
        template<typename T> T const *get( std::string const &a_label ) const ;
        template<typename T> T *getViaLineage( std::string const &a_label );

        Styles::Suite const *styles( ) { return( m_styles ); }                              /**< Returns the value of the **m_styles** member. */

        void parse( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs, 
                        parseSuite a_parseSuite, Styles::Suite const *a_styles );
        void add( Form *a_form );
        iterator find( std::string const &a_label );
        const_iterator find( std::string const &a_label ) const ;
        bool has( std::string const &a_label ) const { return( find( a_label ) != end( ) ); }

        void modifiedMultiGroupElasticForTNSL( std::map<std::string,std::size_t> a_maximumTNSL_MultiGroupIndex );
        Ancestry *findInAncestry3( std::string const &a_item );
        Ancestry const *findInAncestry3( std::string const &a_item ) const ;
        std::vector<iterator> findAllOfMoniker( std::string const &a_moniker ) ;
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

template<typename T> T *Suite::get( std::size_t a_index ) {

    Form *__form = m_forms[a_index];
    T *object = dynamic_cast<T *>( __form );

    if( object == nullptr ) throw Exception( "Suite::get( std::size_t ): invalid cast" );

    return( object );
}

/* *********************************************************************************************************//**
 * Returns the node at index *a_index*.
 *
 * @param a_index               [in]    The index of the node to return.
 *
 * @return                              The node at index *a_index*.
 ***********************************************************************************************************/

template<typename T> T const *Suite::get( std::size_t a_index ) const {

    Form *__form = m_forms[a_index];
    T *object = dynamic_cast<T *>( __form );

    if( object == nullptr ) throw Exception( "Suite::get( std::size_t ): invalid cast" );

    return( object );
}

/* *********************************************************************************************************//**
 * Returns the node with label *a_label*.
 *
 * @param a_label               [in]    The label of the node to return.
 *
 * @return                              The node with label *a_label*.
 ***********************************************************************************************************/

template<typename T> T *Suite::get( std::string const &a_label ) {

    int index = (*this)[a_label];
    Form *__form = m_forms[index];
    T *object = dynamic_cast<T *>( __form );

    if( object == nullptr ) throw Exception( "Suite::get( std::string const & ): invalid cast" );

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
    T *object = dynamic_cast<T *>( __form );

    if( object == nullptr ) throw Exception( "Suite::get( std::string const & ): invalid cast" );

    return( object );
}

/*
============================================================
=========================== Flux ===========================
============================================================
*/
class Flux : public Form {

    private:
        Functions::Function2dForm *m_flux;                                          /**< The flux f(E,mu). */

    public:
        Flux( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo );
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
        Group( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops );
        Group( Group const &a_group );

        std::size_t size( ) const { return( m_grid.size( ) ); }                             /**< Returns the number of multi-group boundaries. */
        inline double &operator[]( std::size_t a_index ) { return( m_grid[a_index] ); }     /**< Returns the multi-group boundary at index *a_index*. */
        std::vector<double> data( ) const { return( m_grid.data().vector() ); }              /**< Returns the multi-group boundaries. */
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
        Transportable( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, Suite *a_parent );
        Transportable( Transportable const &a_transportable );

        std::string pid( ) const { return( label( ) ); }                                    /**< Returns the value of the particle id for the **Transportable**. */
        std::string const &conserve( ) const { return( m_conserve ); }                      /**< Returns a const reference to member **m_conserve**. */
        Group const &group( ) const { return( m_group ); }                                  /**< Returns the value of the **m_group** member. */
        std::vector<double> groupBoundaries( ) const { return( m_group.data( ) ); }  /**< Returns the multi-group boundaries for this transportable particle. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
======================= ExternalFile =======================
============================================================
*/

class ExternalFile : public Form {

    private:
        std::string m_path;                         /**< The path to the external file. */

    public:
        ExternalFile( std::string const &a_label, std::string const &a_path );
        ExternalFile( HAPI::Node const &a_node, SetupInfo &a_setupInfo, GIDI::Suite *a_parent );
        ~ExternalFile( );

        std::string const &path( ) const { return( m_path ); }

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
==================== Documentation_1_10 ====================
============================================================
*/

namespace Documentation_1_10 {

class Documentation : public Form {

    private:
        std::string m_label;
        std::string m_text;

    public:
        Documentation( HAPI::Node const &a_node, SetupInfo &a_setupInfo, GIDI::Suite *a_parent );
        ~Documentation( ) { };

        std::string label( ) const { return m_label; }
        std::string text( ) const { return m_text; }

};

class Suite : public GIDI::Suite {

    public:
        Suite( );
        void parse( HAPI::Node const &a_node, SetupInfo &a_setupInfo );

};

}                     // End of namespace Documentation_1_10.

/*
============================================================
===================== ExternalFiles stuff ==================
============================================================
*/

namespace ExternalFiles {

/*
============================================================
========================== Suite ===========================
============================================================
*/
class Suite : public GIDI::Suite {

    public:
        void registerBinaryFiles(std::string a_parentDir, SetupInfo &a_setupInfo);

};

}                     // End of namespace ExternalFiles.

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
        Base( HAPI::Node const &a_node, SetupInfo &a_setupInfo, GIDI::Suite *a_parent );

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
        Evaluated( HAPI::Node const &a_node, SetupInfo &a_setupInfo, GIDI::Suite *a_parent );

        PhysicalQuantity const &temperature( ) const { return( m_temperature ); }   /**< Returns the value of the **m_temperature** member. */
        AxisDomain const &projectileEnergyDomain( ) const { return( m_projectileEnergyDomain ); }
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
================ CrossSectionReconstructed =================
============================================================
*/
class CrossSectionReconstructed : public Base {

    public:
        CrossSectionReconstructed( HAPI::Node const &a_node, SetupInfo &a_setupInfo, GIDI::Suite *a_parent );

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
        CoulombPlusNuclearElasticMuCutoff( HAPI::Node const &a_node, SetupInfo &a_setupInfo, GIDI::Suite *a_parent );

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

// FIXME this class does not seem to be used.

    private:
        PhysicalQuantity m_temperature;                                 /**< The GNDS <**temperature**> node data. */

    public:
        TNSL( HAPI::Node const &a_node, SetupInfo &a_setupInfo, GIDI::Suite *a_parent );
        PhysicalQuantity const & temperature( ) const { return( m_temperature ); }  /**< Returns the value of the **m_temperature** member. */
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
======================= Realization ========================
============================================================
*/
class Realization : public Base {

    public:
        Realization( HAPI::Node const &a_node, SetupInfo &a_setupInfo, GIDI::Suite *a_parent );

        PhysicalQuantity const & temperature( ) const ;
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
};

/*
============================================================
=================== AverageProductData =====================
============================================================
*/
class AverageProductData : public Base {

    public:
        AverageProductData( HAPI::Node const &a_node, SetupInfo &a_setupInfo, GIDI::Suite *a_parent );

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
        MonteCarlo_cdf( HAPI::Node const &a_node, SetupInfo &a_setupInfo, GIDI::Suite *a_parent );

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
        MultiGroup( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs, GIDI::Suite *a_parent );
        ~MultiGroup( );

        int maximumLegendreOrder( ) const { return( m_maximumLegendreOrder ); }     /**< Returns the value of the **m_maximumLegendreOrder** member. */
        PhysicalQuantity const &temperature( ) const ;

        std::vector<double> const groupBoundaries( std::string const &a_productID ) const ;
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
        Heated( HAPI::Node const &a_node, SetupInfo &a_setupInfo, GIDI::Suite *a_parent );
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
        std::string m_href;                             /**< The GNDS <**transportables**> href value if present. */
        GIDI::Suite m_transportables;                   /**< The GNDS <**transportables**> node. For GNDS 2.0 and above. */
        Flux m_flux;                                    /**< The GNDS <**flux**> node. */
        Functions::Gridded1d m_inverseSpeed;            /**< The GNDS <**inverseSpeed**> node data. */
        std::string m_parameters;                       /**< The GNDS <**parameters**> attribute. Only used for GNDS 1.10. */

    public:
        HeatedMultiGroup( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, GIDI::Suite *a_parent );
        ~HeatedMultiGroup( );

        PhysicalQuantity const &temperature( ) const ;

        std::string const &href( ) const { return( m_href ); }                  /**< Returns a const reference to member **m_href**. */
        void set_href( std::string const &a_href );
        GIDI::Suite const &transportables( ) const { return( m_transportables ); }   /**< Returns a const reference to **m_transportables**. */
        std::vector<double> const groupBoundaries( std::string const &a_productID ) const ;
        Flux const &flux( ) const { return( m_flux ); }                         /**< Returns a const reference to member **m_flux**. */
        std::string const &parameters( ) const { return( m_parameters ); }      /**< Returns a const reference to member **m_parameters**. Only used for GNDS 1.10. */

        Vector inverseSpeedData( ) const { return( m_inverseSpeed.data( ) ); }      /**< Returns the value of the **m_inverseSpeed** data. */

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const ;
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
        SnElasticUpScatter( HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, GIDI::Suite *a_parent );
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
        GriddedCrossSection( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, GIDI::Suite *a_parent );
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
        URR_probabilityTables( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, GIDI::Suite *a_parent );
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
template<typename T> T *Suite::getViaLineage( std::string const &a_label ) {

    std::string const *label = m_styles->findLabelInLineage( (Styles::Suite &) *this, a_label );

    return( get<T>( *label ) );
}

/*
============================================================
==================== Transporting stuff ====================
============================================================
*/

namespace Transporting {

class ProcessedFlux;

enum class Mode { multiGroup, multiGroupWithSnElasticUpScatter, MonteCarloContinuousEnergy };
enum class DelayedNeutrons { off, on };

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
        Functions::XYs3d *get3dViaFID( int a_fid ) const ;
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
        Transporting::Mode m_mode;                                          /**< Indicates the type of transport the user is likely, but not guaranteed, to do. */
        MultiGroup m_multiGroup;                                            /**< Coarse multi-group to collapse to. */
        MultiGroup m_fineMultiGroup;                                        /**< Fine multi-group to collapse from. For internal use only. */
        std::vector<int> m_collapseIndices;                                 /**< Indices for collapsing to m_multiGroup. */
        std::vector<Flux> m_fluxes;                                         /**< One flux for each temperature. */
        std::vector<ProcessedFlux> m_processedFluxes;                       /**< One processed flux for each temperature. */

    public:
        Particle( std::string const &a_pid, MultiGroup const &a_multiGroup, Functions::Function3dForm const &a_fluxes, Transporting::Mode a_mode = Transporting::Mode::multiGroup );
        Particle( std::string const &a_pid, Transporting::Mode a_mode = Transporting::Mode::multiGroup );
        Particle( std::string const &a_pid, MultiGroup const &a_multiGroup, Transporting::Mode a_mode = Transporting::Mode::multiGroup );
        Particle( Particle const &a_particle );
        ~Particle( );

        std::string const &pid( ) const { return( m_pid ); }                                /**< Returns the value of the **m_pid** member. */
        Transporting::Mode mode( ) const { return( m_mode ); }                              /**< Returns the value of the **m_mode** member. */
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
        DelayedNeutrons m_delayedNeutrons;                          /**< If true, include delayed neutrons when returning or setting up data. */
        bool m_nuclearPlusCoulombInterferenceOnly;                  /**< If true, for charge particle as projectile and elastic scattering, the Rutherford term is excluded from the elastic reaction. */

    public:
        Settings( std::string const &a_projectileID, DelayedNeutrons a_delayedNeutrons );
        ~Settings( );

        std::string const &projectileID( ) const { return( m_projectileID ); }                                      /**< Returns the value of the **m_projectileID** member. */

        DelayedNeutrons delayedNeutrons( ) const { return( m_delayedNeutrons ); }                                   /**< Returns the value of the **m_delayedNeutrons** member. */
        void setDelayedNeutrons( DelayedNeutrons a_delayedNeutrons ) { m_delayedNeutrons = a_delayedNeutrons; }     /**< Sets the **m_delayedNeutrons** member to **a_delayedNeutrons*. */

        bool nuclearPlusCoulombInterferenceOnly( ) const { return( m_nuclearPlusCoulombInterferenceOnly ); }        /**< Returns the value of the **m_nuclearPlusCoulombInterferenceOnly** member. */
        void setNuclearPlusCoulombInterferenceOnly( bool a_nuclearPlusCoulombInterferenceOnly )
            { m_nuclearPlusCoulombInterferenceOnly = a_nuclearPlusCoulombInterferenceOnly; }                        /**< Sets the **m_nuclearPlusCoulombInterferenceOnly** to **a_nuclearPlusCoulombInterferenceOnly**. */

        Vector multiGroupZeroVector( Particles const &a_particles, bool a_collapse = true ) const ;
        Matrix multiGroupZeroMatrix( Particles const &a_particles, std::string const &a_particleID, bool a_collapse = true ) const ;

//        void print( ) const ;
};

/*
============================================================
============================ MG ============================
============================================================
*/
class MG : public Settings {

    private:
        Mode m_mode;                                    /**< Specifies the type of data to use or retrieve for transport codes. */

    public:
        MG( std::string const &a_projectileID, Mode a_mode, DelayedNeutrons a_delayedNeutrons );

        Mode mode( ) const { return( m_mode ); }                /**< Returns the value of the **m_mode** member. */
        void setMode( Mode a_mode ) { m_mode = a_mode; }        /**< Sets the **m_mode** member to **a_mode*. */

        Form const *form( GIDI::Suite const &a_suite, Styles::TemperatureInfo const &a_temperatureInfo, bool a_throwOnError = true ) const ;
};

}           // End of namespace Transporting.

/*
============================================================
========================= Product ==========================
============================================================
*/
class Product : public Form {

    private:
        ParticleInfo m_particle;                    /**< The products **ParticleInfo** data. */
        ParticleInfo m_GNDS_particle;               /**< The products **ParticleInfo** data. This is the product's equivalent of the Protare::m_GNDS_target member. */

        int m_productMultiplicity;                  /**< Product multiplicity (e.g., 0, 1, 2, ...) or -1 if energy dependent or not an integer for particle with id *a_id*. */
        Suite m_multiplicity;                       /**< The GNDS <**multiplicity**> node. */
        Suite m_distribution;                       /**< The GNDS <**distribution**> node. */
        Suite m_averageEnergy;                      /**< The GNDS <**averageEnergy**> node. */
        Suite m_averageMomentum;                    /**< The GNDS <**averageMomentum**> node. */
        OutputChannel *m_outputChannel;             /**< The GNDS <**outputChannel**> node if present. */

    public:
        Product( PoPI::Database const &a_pops, std::string const &a_productID, std::string const &a_label );
        Product( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs, Suite *a_parent, Styles::Suite const *a_styles );
        ~Product( );

        ParticleInfo const &particle( ) const { return( m_particle ); }                     /**< Returns the value of the **m_particle** member. */
        void setParticle( ParticleInfo const &a_particle ) { m_particle = a_particle; }     /**< Sets **m_particle** to *a_particle*. */
        ParticleInfo const &GNDS_particle( ) const { return( m_GNDS_particle ); }           /**< Returns a const reference to the **m_GNDS_particle** member. */
        ParticleInfo &GNDS_particle( ) { return( m_GNDS_particle ); }                       /**< Returns the value of the **m_GNDS_particle** member. */
        int depth( ) const ;

        Suite &multiplicity( ) { return( m_multiplicity ); }                                /**< Returns a reference to the **m_multiplicity** member. */
        Suite const &multiplicity( ) const { return( m_multiplicity ); }                    /**< Returns a const reference to the **m_multiplicity** member. */
        Suite &distribution( ) { return( m_distribution ); }                                /**< Returns a reference to the **m_distribution** member. */
        Suite const &distribution( ) const { return( m_distribution ); }                                /**< Returns a reference to the **m_distribution** member. */
        Suite &averageEnergy( ) { return( m_averageEnergy ); }                              /**< Returns a reference to the **m_averageEnergy** member. */
        Suite const &averageEnergy( ) const { return( m_averageEnergy ); }                  /**< Returns a const reference to the **m_averageEnergy** member. */
        Suite &averageMomentum( ) { return( m_averageMomentum ); }                          /**< Returns a reference to the **m_averageMomentum** member. */
        Suite const &averageMomentum( ) const { return( m_averageMomentum ); }              /**< Returns a const reference to the **m_averageMomentum** member. */
        OutputChannel *outputChannel( ) const { return( m_outputChannel ); }                /**< Returns a reference to the **m_outputChannel** member. */

        void modifiedMultiGroupElasticForTNSL( std::map<std::string,std::size_t> a_maximumTNSL_MultiGroupIndex );

        bool hasFission( ) const ;
        Ancestry *findInAncestry3( std::string const &a_item );
        Ancestry const *findInAncestry3( std::string const &a_item ) const ;
        void productIDs( std::set<std::string> &a_ids, Transporting::Particles const &a_particles, bool a_transportablesOnly ) const ;
        int productMultiplicity( std::string const &a_id ) const ;
        int maximumLegendreOrder( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;

        Vector multiGroupQ( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, bool a_final ) const ;
        Vector multiGroupMultiplicity( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Matrix multiGroupProductMatrix( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, std::string const &a_productID, int a_order ) const ;

        Vector multiGroupAverageEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Vector multiGroupAverageMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;

        void continuousEnergyProductData( std::string const &a_particleID, double a_energy, double &a_productEnergy, double &a_productMomentum, double &a_productGain ) const ;

        void incompleteParticles( Transporting::Settings const &a_settings, std::set<std::string> &a_incompleteParticles ) const ;

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
        DelayedNeutron( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs, Suite *a_parent, Styles::Suite const *a_styles );
        ~DelayedNeutron( );

        int delayedNeutronIndex( ) const { return( m_delayedNeutronIndex ); };
        void setDelayedNeutronIndex( int a_delayedNeutronIndex ) { m_delayedNeutronIndex = a_delayedNeutronIndex; }
        Suite &rate( ) { return( m_rate ); }
        Suite const &rate( ) const { return( m_rate ); }
        Product &product( ) { return( m_product ); }
        Product const &product( ) const { return( m_product ); }

        Ancestry *findInAncestry3( std::string const &a_item );
        Ancestry const *findInAncestry3( std::string const &a_item ) const ;

        void productIDs( std::set<std::string> &a_indices, Transporting::Particles const &a_particles, bool a_transportablesOnly ) const ;
        int productMultiplicity( std::string const &a_id ) const ;
        int maximumLegendreOrder( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Vector multiGroupQ( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, bool a_final ) const ;
        Vector multiGroupMultiplicity( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Matrix multiGroupProductMatrix( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, std::string const &a_productID, int a_order ) const ;
        Vector multiGroupAverageEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Vector multiGroupAverageMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;

        void incompleteParticles( Transporting::Settings const &a_settings, std::set<std::string> &a_incompleteParticles ) const ;

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
=================== DelayedNeutronProduct ==================
============================================================
*/
class DelayedNeutronProduct {

    private:
        int m_delayedNeutronIndex;                  /**< If this is a delayed fission neutron, this is its index. */
        PhysicalQuantity m_rate;
        Product const *m_product;

    public:
        DelayedNeutronProduct( int a_delayedNeutronIndex, PhysicalQuantity a_rate, Product const *a_product ) : 
                m_delayedNeutronIndex( a_delayedNeutronIndex ),
                m_rate( a_rate ),
                m_product( a_product ) {
        }
        DelayedNeutronProduct( DelayedNeutronProduct const &a_delayedNeutronProduct ) :
                m_delayedNeutronIndex( a_delayedNeutronProduct.delayedNeutronIndex( ) ),
                m_rate( a_delayedNeutronProduct.rate( ) ),
                m_product( a_delayedNeutronProduct.product( ) ) {
        }
        ~DelayedNeutronProduct( ) {}

        int delayedNeutronIndex( ) const { return( m_delayedNeutronIndex ); };
        PhysicalQuantity rate( ) const { return( m_rate ); }
        Product const *product( ) const { return( m_product ); }
};

typedef std::vector<DelayedNeutronProduct> DelayedNeutronProducts;

/*
============================================================
==================== FissionFragmentData ===================
============================================================
*/
class FissionFragmentData : public Ancestry {

    private:
        Suite m_delayedNeutrons;                            /**< The GNDS <**delayedNeutrons**> node. This members stores a list of DelayedNeutron instances. */
        Suite m_fissionEnergyReleases;                      /**< The GNDS <**fissionEnergyReleases**> node. */

    public:
        FissionFragmentData( );
        FissionFragmentData( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs, Styles::Suite const *a_styles );
        ~FissionFragmentData( );

        Suite &delayedNeutrons( ) { return( m_delayedNeutrons ); }
        Suite const &delayedNeutrons( ) const { return( m_delayedNeutrons ); }
        Suite &fissionEnergyReleases( ) { return( m_fissionEnergyReleases ); }
        Suite const &fissionEnergyReleases( ) const { return( m_fissionEnergyReleases ); }

        Ancestry *findInAncestry3( std::string const &a_item );
        Ancestry const *findInAncestry3( std::string const &a_item ) const ;

        void productIDs( std::set<std::string> &a_indices, Transporting::Particles const &a_particles, bool a_transportablesOnly ) const ;
        int productMultiplicity( std::string const &a_id ) const ;
        int maximumLegendreOrder( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Vector multiGroupQ( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, bool a_final ) const ;
        Vector multiGroupMultiplicity( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Matrix multiGroupProductMatrix( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, std::string const &a_productID, int a_order ) const ;
        Vector multiGroupAverageEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Vector multiGroupAverageMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;

        void delayedNeutronProducts( DelayedNeutronProducts &a_delayedNeutronProducts ) const ;
        void incompleteParticles( Transporting::Settings const &a_settings, std::set<std::string> &a_incompleteParticles ) const ;

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
        OutputChannel( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs, 
                Styles::Suite const *a_styles, bool a_isFission );
        ~OutputChannel( );

        bool twoBody( ) const { return( m_twoBody ); }                              /**< Returns the value of the **m_twoBody** member. */
        std::string process( ) const { return( m_process ); }
        int depth( ) const ;

        Suite &Q( ) { return( m_Q ); }                                              /**< Returns a reference to the **m_Q** member. */
        Suite const &Q( ) const { return( m_Q ); }                                  /**< Returns a reference to the **m_Q** member. */
        Suite &products( ) { return( m_products ); }                                /**< Returns a reference to the **m_products** member. */
        Suite const &products( ) const { return( m_products ); }                    /**< Returns a reference to the **m_products** member. */
        FissionFragmentData &fissionFragmentData( ) { return( m_fissionFragmentData ); }
        FissionFragmentData const &fissionFragmentData( ) const { return( m_fissionFragmentData ); }

        void modifiedMultiGroupElasticForTNSL( std::map<std::string,std::size_t> a_maximumTNSL_MultiGroupIndex );

        Ancestry *findInAncestry3( std::string const &a_item );
        Ancestry const *findInAncestry3( std::string const &a_item ) const ;

        bool isFission( ) const { return( m_fissions ); }                           /**< Returns true if the output channel is a fission output channel. */
        bool hasFission( ) const ;
        void productIDs( std::set<std::string> &a_ids, Transporting::Particles const &a_particles, bool a_transportablesOnly ) const ;
        int productMultiplicity( std::string const &a_id ) const ;
        int maximumLegendreOrder( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;

        Vector multiGroupQ( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, bool a_final ) const ;
        Vector multiGroupMultiplicity( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Matrix multiGroupProductMatrix( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, std::string const &a_productID, int a_order ) const ;
        Vector multiGroupAverageEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Vector multiGroupAverageMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;

        void delayedNeutronProducts( DelayedNeutronProducts &a_delayedNeutronProducts ) const { m_fissionFragmentData.delayedNeutronProducts( a_delayedNeutronProducts ); }
        void incompleteParticles( Transporting::Settings const &a_settings, std::set<std::string> &a_incompleteParticles ) const ;
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
        IncoherentInelastic( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs, Suite *a_parent );
        ~IncoherentInelastic( );
        
        Options &options( ) { return( m_options ); }                                    /**< Returns the value of the **m_options** */
        Suite &scatteringAtoms( ) { return( m_scatteringAtoms ); }                      /**< Returns the value of the **m_scatteringAtoms** */
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
        bool m_active;                                  /**< If true, this reaction is used for calcualtion (e.g., its cross section is added to the total for its protare), otherwise, this reaction is ignored. */
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
        Reaction( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs, Protare const &a_protare,
                        Styles::Suite const *a_styles );
        ~Reaction( );

        bool active( ) const { return( m_active ); }                                    /**< Returns the value of the **m_active** member. */
        void setActive( bool a_active ) { m_active = a_active; }                        /**< Sets **m_active** to *a_active*. */
        int depth( ) const { return( m_outputChannel->depth( ) ); }                     /**< Returns the maximum product depth for this reaction. */
        int ENDF_MT( ) const { return( m_ENDF_MT ); }                                   /**< Returns the value of the **m_ENDF_MT** member. */
        int ENDL_C( ) const { return( m_ENDL_C ); }                                     /**< Returns the value of the **m_ENDL_C** member. */
        int ENDL_S( ) const { return( m_ENDL_S ); }                                     /**< Returns the value of the **m_ENDL_S** member. */
        bool isPairProduction( ) const { return( m_isPairProduction ); }                /**< Returns the value of the **m_isPairProduction** member. */

        Suite &availableEnergy( ) { return( m_availableEnergy ); }                      /**< Returns a reference to the **m_availableEnergy** member. */
        Suite const &availableEnergy( ) const { return( m_availableEnergy ); }          /**< Returns a reference to the **m_availableEnergy** member. */
        Suite &availableMomentum( ) { return( m_availableMomentum ); }                  /**< Returns a reference to the **m_availableMomentum** member. */
        Suite const &availableMomentum( ) const { return( m_availableMomentum ); }      /**< Returns a reference to the **m_availableMomentum** member. */

        Suite &doubleDifferentialCrossSection( ) { return( m_doubleDifferentialCrossSection ); }    /**< Returns a reference to the **m_doubleDifferentialCrossSection** member. */
        Suite const &doubleDifferentialCrossSection( ) const { return( m_doubleDifferentialCrossSection ); }    /**< Returns a reference to the **m_doubleDifferentialCrossSection** member. */
        Suite &crossSection( ) { return( m_crossSection ); }                            /**< Returns a reference to the **m_crossSection** member. */
        Suite const &crossSection( ) const { return( m_crossSection ); }                            /**< Returns a reference to the **m_crossSection** member. */
        OutputChannel *outputChannel( ) const { return( m_outputChannel ); }            /**< Returns a reference to the **m_outputChannel** member. */
        void outputChannel( OutputChannel *a_outputChannel );

        void modifiedMultiGroupElasticForTNSL( std::map<std::string,std::size_t> a_maximumTNSL_MultiGroupIndex );

        Ancestry *findInAncestry3( std::string const &a_item );
        Ancestry const *findInAncestry3( std::string const &a_item ) const ;
        std::string xlinkItemKey( ) const { return( Ancestry::buildXLinkItemKey( GIDI_labelChars, label( ) ) ); }   /**< Returns the value of the **** member. */

        bool hasFission( ) const ;
        void productIDs( std::set<std::string> &a_ids, Transporting::Particles const &a_particles, bool a_transportablesOnly ) const ;
        int productMultiplicity( std::string const &a_id ) const {
                return( m_outputChannel->productMultiplicity( a_id ) ); }               /**< Returns the product multiplicity (e.g., 0, 1, 2, ...) or -1 if energy dependent or not an integer. */
        int maximumLegendreOrder( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;

        double threshold( ) const { return( m_QThreshold ); }                           /**< Returns the value of the **m_QThreshold** member. */
        double crossSectionThreshold( ) const { return( m_crossSectionThreshold ); }    /**< Returns the value of the **m_crossSectionThreshold** member. */

        Vector multiGroupCrossSection( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const ;
        Vector multiGroupQ( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, bool a_final ) const {
                return( m_outputChannel->multiGroupQ( a_settings, a_temperatureInfo, a_final ) ); }                  /**< Returns the multi-group, total Q for the requested label. This is a cross section weighted Q summed over all reactions. */
        Vector multiGroupMultiplicity( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;

        Matrix multiGroupProductMatrix( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, std::string const &a_productID, int a_order ) const ;
        Matrix multiGroupFissionMatrix( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, int a_order ) const ;

        Vector multiGroupAvailableEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const ;
        Vector multiGroupAverageEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Vector multiGroupDepositionEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles ) const ;

        Vector multiGroupAvailableMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const ;
        Vector multiGroupAverageMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Vector multiGroupDepositionMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles ) const ;

        Vector multiGroupGain( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID, std::string const &a_projectileID ) const ;

        void delayedNeutronProducts( DelayedNeutronProducts &a_delayedNeutronProducts ) const ;
        void incompleteParticles( Transporting::Settings const &a_settings, std::set<std::string> &a_incompleteParticles ) const ;
        void continuousEnergyProductData( std::string const &a_particleID, double a_energy, double &a_productEnergy, double &a_productMomentum, double &a_productGain ) const ;

        void modifiedCrossSection( Functions::XYs1d const &a_offset, Functions::XYs1d const &a_slope );

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
        Base( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo );
        ~Base( );

        std::string const &href( ) const { return( m_href ); }                  /**< Returns the value of the **m_href** member. */
        Ancestry *findInAncestry3( std::string const &a_item ) { return( nullptr ); }
        Ancestry const *findInAncestry3( std::string const &a_item ) const { return( nullptr ); }

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
=========================== Add ============================
============================================================
*/
class Add : public Base {

    public:
        Add( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo );
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
        Summands( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo );
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
        Base( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs,
                FormType a_type );

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
        CrossSectionSum( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs );
        Ancestry *findInAncestry3( std::string const &a_item );
        Ancestry const *findInAncestry3( std::string const &a_item ) const ;

        Suite &Q( ) { return( m_Q ); }                                      /**< Returns a reference to the **m_Q** member. */
        Suite &crossSection( ) { return( m_crossSection ); }                /**< Returns a reference to the **m_crossSection** member. */

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
        MultiplicitySum( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs );

        Suite &multiplicity( ) { return( m_multiplicity ); }                /**< Returns a reference to the **m_multiplicity** member. */

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
=========================== Sums ===========================
============================================================
*/
class Sums : public Ancestry {

    private:
        Suite m_crossSectionSums;                                               /**< The GNDS <**crossSectionSums**> node. */
        Suite m_multiplicitySums;                                               /**< The GNDS <**multiplicitySums**> node. */

    public:
        Sums( );
        ~Sums( );

        Suite &crossSectionSums( ) { return( m_crossSectionSums ); }                /**< Returns the value of the **m_crossSectionSums** member. */
        Suite const &crossSectionSums( ) const { return( m_crossSectionSums ); }    /**< Returns the value of the **m_crossSectionSums** member. */
        Suite &multiplicitySums( ) { return( m_multiplicitySums ); }                /**< Returns the value of the **m_multiplicitySums** member. */
        Suite const &multiplicitySums( ) const { return( m_multiplicitySums ); }    /**< Returns the value of the **m_multiplicitySums** member. */

        void parse( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs );
        Ancestry *findInAncestry3( std::string const &a_item );
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
        ParticleInfo m_GNDS_target;             /**< Information about the target as specified in the GNDS file. For example, for requested target 'H1' for a photo-atomic GNDS file, the GNDS target will be 'H'. */

    protected:
        void initialize( HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs, bool a_targetRequiredInGlobalPoPs, 
                        bool a_requiredInPoPs = true );

    public:
        Protare( );
        ~Protare( );

        ParticleInfo const &projectile( ) const { return( m_projectile ); }         /**< Returns the value of the **m_projectile** member. */
        void setProjectile( ParticleInfo const &a_projectile ) { m_projectile = a_projectile; }    /**< Sets **m_projectile** to *a_projectile*. */
        ParticleInfo const &target( ) const { return( m_target ); }                 /**< Returns the value of the **m_target** member. */
        void setTarget( ParticleInfo const &a_target ) { 
                m_target = a_target;
                if( m_GNDS_target.ID( ) == "" ) m_GNDS_target = a_target; }         /**< Sets **m_target** to *a_target* and m_GNDS_target if it is an empty string. */
        ParticleInfo const &GNDS_target( ) const { return( m_GNDS_target ); }       /**< Returns the value of the **m_GNDS_target** member. */

        virtual ProtareType protareType( ) const = 0;                               /**< Returns the type of the protare. */
        virtual bool isTNSL_ProtareSingle( ) const { return( false ); }             /**< Returns *true* if the instance is a ProtareSingle instance with only TNSL data and *false* otherwise. */
        virtual std::size_t numberOfProtares( ) const = 0;                          /**< Returns the number of protares contained in *this*. */
        virtual ProtareSingle *protare( std::size_t a_index ) = 0;                  /**< Returns the **a_index** - 1 Protare contained in *this*. */
        virtual ProtareSingle const *protare( std::size_t a_index ) const = 0;      /**< Returns the **a_index** - 1 Protare contained in *this*. */

        virtual FormatVersion const &formatVersion( std::size_t a_index = 0 ) const = 0;
        virtual std::string const &fileName( std::size_t a_index = 0 ) const = 0;
        virtual std::string const &realFileName( std::size_t a_index = 0 ) const = 0;

        virtual std::vector<std::string> libraries( std::size_t a_index = 0 ) const = 0;
        virtual std::string const &evaluation( std::size_t a_index = 0 ) const = 0;
        virtual Frame projectileFrame( std::size_t a_index = 0 ) const = 0;
        virtual double thresholdFactor( ) const = 0;

        virtual Documentation_1_10::Suite &documentations( ) = 0;

        virtual Styles::Base &style( std::string const a_label ) = 0;
        virtual Styles::Suite &styles( ) = 0;
        virtual Styles::Suite const &styles( ) const = 0;

        virtual void productIDs( std::set<std::string> &a_ids, Transporting::Particles const &a_particles, bool a_transportablesOnly ) const = 0;
        virtual int maximumLegendreOrder( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const = 0;

        virtual Styles::TemperatureInfos temperatures( ) const  = 0;

        virtual std::size_t numberOfReactions( ) const = 0;
        virtual Reaction *reaction( std::size_t a_index ) = 0;
        virtual Reaction const *reaction( std::size_t a_index ) const = 0;
        virtual std::size_t numberOfOrphanProducts( ) const = 0;
        virtual Reaction *orphanProduct( std::size_t a_index ) = 0;
        virtual Reaction const *orphanProduct( std::size_t a_index ) const = 0;

        virtual bool hasFission( ) const = 0;

        virtual Ancestry *findInAncestry3( std::string const &a_item ) = 0;
        virtual Ancestry const *findInAncestry3( std::string const &a_item ) const = 0;

        virtual std::vector<double> const groupBoundaries( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const = 0;
        virtual Vector multiGroupInverseSpeed( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const = 0;

        virtual Vector multiGroupCrossSection( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const = 0;
        virtual Vector multiGroupQ( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, bool a_final ) const = 0;

        virtual Vector multiGroupMultiplicity( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const = 0;
        virtual Vector multiGroupFissionNeutronMultiplicity( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const = 0;

        virtual Matrix multiGroupProductMatrix( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, std::string const &a_productID, int a_order ) const = 0;
        virtual Matrix multiGroupFissionMatrix( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, int a_order ) const = 0;
        virtual Vector multiGroupTransportCorrection( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, int a_order, TransportCorrectionType a_transportCorrectionType, double a_temperature ) const = 0;

        virtual Vector multiGroupAvailableEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const = 0;
        virtual Vector multiGroupAverageEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const = 0;
        virtual Vector multiGroupDepositionEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles ) const = 0;

        virtual Vector multiGroupAvailableMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const = 0;
        virtual Vector multiGroupAverageMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const = 0;
        virtual Vector multiGroupDepositionMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles ) const = 0;

        virtual Vector multiGroupGain( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const = 0;

        virtual void TNSL_crossSectionSumCorrection( std::string const &a_label, Functions::XYs1d &a_crossSectionSum );
        virtual void TNSL_crossSectionSumCorrection( std::string const &a_label, Functions::Ys1d &a_crossSectionSum );
        virtual void TNSL_crossSectionSumCorrection( std::string const &a_label, Vector &a_crossSectionSum );

        virtual stringAndDoublePairs muCutoffForCoulombPlusNuclearElastic( ) const = 0;
        virtual DelayedNeutronProducts delayedNeutronProducts( ) const = 0;
        virtual void incompleteParticles( Transporting::Settings const &a_settings, std::set<std::string> &a_incompleteParticles ) const = 0;
        std::set<int> reactionIndicesMatchingENDLCValues( std::set<int> const &a_CValues, bool a_checkActiveState = true );
};

/*
============================================================
====================== ProtareSingle =======================
============================================================
*/
class ProtareSingle : public Protare {

    private:
        FormatVersion m_formatVersion;          /**< Store the GNDS format version. */
        PoPI::Database m_internalPoPs;          /**< The **PoPs** specified under the protare (e.g., reactionSuite) node. */

        std::vector<std::string> m_libraries;   /**< The list of libraries *this* was found in. */
        std::string m_evaluation;               /**< The protare's evaluation string. */
        std::string m_interaction;              /**< The protare's interaction string. */
        std::string m_fileName;                 /**< The path to the protare's file. May be relative. */
        std::string m_realFileName;             /**< The real path to the protare's file. Equivalent to the value returned by the C-function *realpath( m_fileName )* on Unix systems. */
        Frame m_projectileFrame;                /**< The frame the projectile data are given in. */
        double m_projectileEnergyMin;           /**< The projectile's minimum energy for which data are complete as specified in the evaluated style. */
        double m_projectileEnergyMax;           /**< The projectile's maximum energy for which data are complete as specified in the evaluated style. */
        bool m_isTNSL_ProtareSingle;            /**< If *this* is a ProtareSingle instance with TNSL data *true* and otherwise *false*. */
        bool m_isPhotoAtomic;                   /**< true if photo-atomic protare and false otherwise. */

        double m_thresholdFactor;               /**< The non-relativistic factor that converts a Q-value into a threshold. */

        PoPI::NuclideGammaBranchStateInfos m_nuclideGammaBranchStateInfos;  /**< Simplified list of gamma branching data from nuclide level decays derived from the internal PoPI::Database. */

        ExternalFiles::Suite m_externalFiles;   /**< The GNDS <**externalFiles**> node. */
        Styles::Suite m_styles;                 /**< The GNDS <**styles**> node. */
        Documentation_1_10::Suite m_documentations;  /**< The GNDS <**documentations**> node. */
        Suite m_reactions;                      /**< The GNDS <**reactions**> node. */
        Suite m_orphanProducts;                 /**< The GNDS <**orphanProducts**> node. */
        Suite m_incompleteReactions;            /**< The GNDS <**incompleteReactions**> node. */

        Sums::Sums m_sums;                      /**< The GNDS <**sums**> node. */
        Suite m_fissionComponents;              /**< The GNDS <**fissionComponents**> node. */

        bool m_onlyRutherfordScatteringPresent; /**> For charged particle elastic scattering, this member of *true* if only Rutherford scattering is present and *false* otherwise. */
        Reaction *m_nuclearPlusCoulombInterferenceOnlyReaction;    /**< The nuclear + interference (ENDL C=9) reaction in the applicationData node. */

        void initialize( );
        void initialize( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, bool a_targetRequiredInGlobalPoPs,
                        bool a_requiredInPoPs = true );

    public:
        ProtareSingle( PoPI::Database const &a_pops, std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_evaluation,
                std::string const &a_interaction, std::string const &a_formatVersion = GNDS_formatVersion_1_10Chars );
        ProtareSingle( Construction::Settings const &a_construction, std::string const &a_fileName, FileType a_fileType, PoPI::Database const &a_pops, 
                ParticleSubstitution const &a_particleSubstitution, std::vector<std::string> const &a_libraries, std::string const &a_interaction,
                bool a_targetRequiredInGlobalPoPs = true, bool a_requiredInPoPs = true );
        ProtareSingle( Construction::Settings const &a_construction, HAPI::Node const &a_protare, PoPI::Database const &a_pops, 
                ParticleSubstitution const &a_particleSubstitution, std::vector<std::string> const &a_libraries, std::string const &a_interaction,
                bool a_targetRequiredInGlobalPoPs = true, bool a_requiredInPoPs = true );
        ~ProtareSingle( );

        PoPI::NuclideGammaBranchStateInfos const &nuclideGammaBranchStateInfos( ) const { return( m_nuclideGammaBranchStateInfos ); }
                                                                                    /**< Returns the value of the **m_nuclideGammaBranchStateInfos** member. */

        double projectileEnergyMin( ) const { return( m_projectileEnergyMin ); }
        double projectileEnergyMax( ) const { return( m_projectileEnergyMax ); }
        bool isTNSL_ProtareSingle( ) const { return( m_isTNSL_ProtareSingle ); }    /**< Returns *true* if the instance is a ProtareSingle instance with only TNSL data and *false* otherwise. */
        bool isPhotoAtomic( ) const { return( m_isPhotoAtomic ); }                  /**< Returns the value of the **m_isPhotoAtomic** member. */

        Suite &reactions( ) { return( m_reactions ); }                              /**< Returns a reference to the **m_reactions** member. */
        Suite const &reactions( ) const { return( m_reactions ); }                  /**< Returns a *const* reference to the **m_reactions** member. */
        Suite &orphanProducts( ) { return( m_orphanProducts ); }                    /**< Returns a reference to the **m_orphanProducts** member. */
        Suite const &orphanProducts( ) const { return( m_orphanProducts ); }        /**< Returns a *const* reference to the **m_orphanProducts** member. */
        Suite &incompleteReactions( ) { return( m_incompleteReactions ); }          /**< Returns a reference to the **m_incompleteReactions** member. */
        Suite const &incompleteReactions( ) const { return( m_incompleteReactions ); }  /**< Returns a *const* reference to the **m_incompleteReactions** member. */

        Sums::Sums &sums( ) { return( m_sums ); }                                   /**< Returns a reference to the **m_sums** member. */
        Sums::Sums const &sums( ) const { return( m_sums ); }                       /**< Returns a reference to the **m_sums** member. */
        Suite &fissionComponents( ) { return( m_fissionComponents ); }              /**< Returns a reference to the **m_fissionComponents** member. */

        bool onlyRutherfordScatteringPresent( ) const { return( m_onlyRutherfordScatteringPresent ); } /**< Returns the value of **m_onlyRutherfordScatteringPresent**. */
        Reaction const *nuclearPlusCoulombInterferenceOnlyReaction( ) const { return( m_nuclearPlusCoulombInterferenceOnlyReaction ); } /**< Returns a reference to the **m_nuclearPlusCoulombInterferenceOnlyReaction** member. */

// The rest are virtual methods defined in the Protare class.

        ProtareType protareType( ) const { return( ProtareType::single ); }                                 /**< Returns the type of the protare. */
        std::size_t numberOfProtares( ) const { return( 1 ); }                                              /**< Returns 1. */
        ProtareSingle *protare( std::size_t a_index );
        ProtareSingle const *protare( std::size_t a_index ) const ;

        FormatVersion const &formatVersion( std::size_t a_index = 0 ) const { return( m_formatVersion ); }  /**< Returns the value of the **m_formatVersion** member. */
        std::string const &fileName( std::size_t a_index = 0 ) const { return( m_fileName ); }              /**< Returns the value of the **m_fileName** member. */
        std::string const &realFileName( std::size_t a_index = 0 ) const { return( m_realFileName ); }      /**< Returns the value of the **m_realFileName** member. */

        std::vector<std::string> libraries( std::size_t a_index = 0 ) const { return( m_libraries ); }      /**< Returns the libraries that *this* resided in. */
        std::string const &evaluation( std::size_t a_index = 0 ) const { return( m_evaluation ); }          /**< Returns the value of the **m_evaluation** member. */
        std::string const &interaction( std::size_t a_index = 0 ) const { return( m_interaction ); }        /**< Returns the value of the **m_interaction** member. */
        Frame projectileFrame( std::size_t a_index = 0 ) const { return( m_projectileFrame ); }             /**< Returns the value of the **m_projectileFrame** member. */
        double thresholdFactor( ) const { return( m_thresholdFactor ); }                                    /**< Returns the value of the **m_thresholdFactor** member. */

        Documentation_1_10::Suite &documentations( ) { return( m_documentations ); }                             /**< Returns the value of the **m_documentations** member. */

        ExternalFile const &externalFile( std::string const a_label ) const { return( *m_externalFiles.get<ExternalFile>( a_label ) ); }      /**< Returns the external file with label **a_label**. */
        ExternalFiles::Suite const &externalFiles( ) const { return( m_externalFiles ); }                /**< Returns the value of the **m_externalFiles** member. */

        Styles::Base &style( std::string const a_label ) { return( *m_styles.get<Styles::Base>( a_label ) ); }      /**< Returns the style with label **a_label**. */
        Styles::Suite &styles( ) { return( m_styles ); }                                                    /**< Returns the value of the **m_styles** member. */
        Styles::Suite const &styles( ) const { return( m_styles ); }                                        /**< Returns a *const* reference to the **m_styles** member. */

        PoPI::Database const &internalPoPs( ) { return( m_internalPoPs ); }                                        /**< Returns a *const* reference to the **m_internalPoPs** member. */

        void productIDs( std::set<std::string> &a_ids, Transporting::Particles const &a_particles, bool a_transportablesOnly ) const ;
        int maximumLegendreOrder( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;

        Styles::TemperatureInfos temperatures( ) const ;

        std::size_t numberOfReactions( ) const { return( m_reactions.size( ) ); }                                   /**< Returns the number of reactions in the **Protare**. */
        Reaction *reaction( std::size_t a_index ) { return( m_reactions.get<Reaction>( a_index ) ); }               /**< Returns the **a_index** - 1 reaction. */
        Reaction const *reaction( std::size_t a_index ) const { return( m_reactions.get<Reaction>( a_index ) ); }   /**< Returns the **a_index** - 1 reaction. */

        std::size_t numberOfOrphanProducts( ) const { return( m_orphanProducts.size( ) ); };                        /**< Returns the number of orphan product reactions in the **Protare**. */
        Reaction *orphanProduct( std::size_t a_index ) { return( m_orphanProducts.get<Reaction>( a_index ) ); }     /**< Returns the **a_index** - 1 orphan product reaction. */
        Reaction const *orphanProduct( std::size_t a_index ) const { return( m_orphanProducts.get<Reaction>( a_index ) ); }     /**< Returns the **a_index** - 1 orphan product reaction. */

        std::size_t numberOfIncompleteReactions( ) const { return( m_incompleteReactions.size( ) ); }                                   /**< Returns the number of incomplete reactions in the **Protare**. */
        Reaction *incompleteReaction( std::size_t a_index ) { return( m_incompleteReactions.get<Reaction>( a_index ) ); }               /**< Returns the **a_index** - 1 reaction. */
        Reaction const *incompleteReaction( std::size_t a_index ) const { return( m_incompleteReactions.get<Reaction>( a_index ) ); }   /**< Returns the **a_index** - 1 reaction. */

        bool hasFission( ) const ;

        Ancestry *findInAncestry3( std::string const &a_item );
        Ancestry const *findInAncestry3( std::string const &a_item ) const ;

        std::vector<double> const groupBoundaries( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Vector multiGroupInverseSpeed( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const ;

        Vector multiGroupCrossSection( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const ;
        Vector multiGroupQ( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, bool a_final ) const ;

        Vector multiGroupMultiplicity( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Vector multiGroupFissionNeutronMultiplicity( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const ;

        Matrix multiGroupProductMatrix( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, std::string const &a_productID, int a_order ) const ;
        Matrix multiGroupFissionMatrix( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, int a_order ) const ;
        Vector multiGroupTransportCorrection( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, int a_order, TransportCorrectionType a_transportCorrectionType, double a_temperature ) const ;

        Vector multiGroupAvailableEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const ;
        Vector multiGroupAverageEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Vector multiGroupDepositionEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles ) const ;

        Vector multiGroupAvailableMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const ;
        Vector multiGroupAverageMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Vector multiGroupDepositionMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles ) const ;

        Vector multiGroupGain( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;

        stringAndDoublePairs muCutoffForCoulombPlusNuclearElastic( ) const ;
        DelayedNeutronProducts delayedNeutronProducts( ) const ;
        void incompleteParticles( Transporting::Settings const &a_settings, std::set<std::string> &a_incompleteParticles ) const ;

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

        ProtareType protareType( ) const { return( ProtareType::composite ); }  /**< Returns the type of the protare. */
        std::size_t numberOfProtares( ) const ;
        ProtareSingle *protare( std::size_t a_index );
        ProtareSingle const *protare( std::size_t a_index ) const ;

        FormatVersion const &formatVersion( std::size_t a_index = 0 ) const ;
        std::string const &fileName( std::size_t a_index = 0 ) const ;
        std::string const &realFileName( std::size_t a_index = 0 ) const ;

        std::vector<std::string> libraries( std::size_t a_index = 0 ) const ;
        std::string const &evaluation( std::size_t a_index = 0 ) const ;
        Frame projectileFrame( std::size_t a_index = 0 ) const ;
        double thresholdFactor( ) const ;

        Documentation_1_10::Suite &documentations( );

        Styles::Base &style( std::string const a_label );
        Styles::Suite &styles( );
        Styles::Suite const &styles( ) const ;

        void productIDs( std::set<std::string> &a_ids, Transporting::Particles const &a_particles, bool a_transportablesOnly ) const ;
        int maximumLegendreOrder( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;

        Styles::TemperatureInfos temperatures( ) const ;

        std::size_t numberOfReactions( ) const ;
        Reaction *reaction( std::size_t a_index );
        Reaction const *reaction( std::size_t a_index ) const ;
        std::size_t numberOfOrphanProducts( ) const ;
        Reaction *orphanProduct( std::size_t a_index );
        Reaction const *orphanProduct( std::size_t a_index ) const ;

        bool hasFission( ) const ;

        Ancestry *findInAncestry3( std::string const &a_item ) { return( nullptr ); }  /**< Always returns *nullptr*. */
        Ancestry const *findInAncestry3( std::string const &a_item ) const { return( nullptr ); }  /**< Always returns *nullptr*. */

        std::vector<double> const groupBoundaries( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Vector multiGroupInverseSpeed( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const ;

        Vector multiGroupCrossSection( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const ;
        Vector multiGroupQ( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, bool a_final ) const ;

        Vector multiGroupMultiplicity( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Vector multiGroupFissionNeutronMultiplicity( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const ;

        Matrix multiGroupProductMatrix( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, std::string const &a_productID, int a_order ) const ;
        Matrix multiGroupFissionMatrix( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, int a_order ) const ;
        Vector multiGroupTransportCorrection( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, int a_order, TransportCorrectionType a_transportCorrectionType, double a_temperature ) const ;

        Vector multiGroupAvailableEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const ;
        Vector multiGroupAverageEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Vector multiGroupDepositionEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles ) const ;

        Vector multiGroupAvailableMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const ;
        Vector multiGroupAverageMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Vector multiGroupDepositionMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles ) const ;

        Vector multiGroupGain( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;

        stringAndDoublePairs muCutoffForCoulombPlusNuclearElastic( ) const ;
        DelayedNeutronProducts delayedNeutronProducts( ) const ;
        void incompleteParticles( Transporting::Settings const &a_settings, std::set<std::string> &a_incompleteParticles ) const ;
};

/*
============================================================
======================= ProtareTNSL ========================
============================================================
*/
class ProtareTNSL : public Protare {

    private:
        ProtareSingle *m_protare;                                           /**< Protare with non thermal neutron scattering law data. */
        ProtareSingle *m_TNSL;                                              /**< Protare with thermal neutron scattering law data. */
        Reaction *m_elasticReaction;                                        /**< The elastic reaction from the non TNSL protare. */
        std::map<std::string,std::size_t> m_maximumTNSL_MultiGroupIndex;    /**< For each neutron multi-group data, this the number of valid groups for the TNSL data. */

    public:
        ProtareTNSL( Construction::Settings const &a_construction, ProtareSingle *a_protare, ProtareSingle *a_TNSL );
        ~ProtareTNSL( );

        ProtareSingle *TNSL( ) { return( m_TNSL ); }                        /**< Returns the **m_TNSL** member. */
        ProtareSingle const *TNSL( ) const { return( m_TNSL ); }            /**< Returns the **m_TNSL** member. */
        Reaction *elasticReaction( ) { return( m_elasticReaction ); }       /**< Returns the **m_elasticReaction** member. */
        std::size_t maximumTNSL_MultiGroupIndex( Styles::TemperatureInfo const &a_temperatureInfo ) const ;
        void combineVectors( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Vector &a_vector, Vector const &a_vectorElastic, Vector const &a_vectorTNSL ) const ;
        void combineMatrices( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Matrix &a_matrix, Matrix const &a_matrixElastic, Matrix const &a_matrixTNSL ) const ;

// The rest are virtual methods defined in the Protare class.

        ProtareType protareType( ) const { return( ProtareType::TNSL ); }   /**< Returns the type of the protare. */
        std::size_t numberOfProtares( ) const { return( 2 ); }              /**< Always returns 2. */
        ProtareSingle *protare( std::size_t a_index = 0 );
        ProtareSingle const *protare( std::size_t a_index = 0 ) const ;

        FormatVersion const &formatVersion( std::size_t a_index = 0 ) const ;
        std::string const &fileName( std::size_t a_index = 0 ) const ;
        std::string const &realFileName( std::size_t a_index = 0 ) const ;

        std::vector<std::string> libraries( std::size_t a_index = 0 ) const ;
        std::string const &evaluation( std::size_t a_index = 0 ) const ;
        Frame projectileFrame( std::size_t a_index = 0 ) const ;
        double thresholdFactor( ) const ;

        Documentation_1_10::Suite &documentations( );

        Styles::Base &style( std::string const a_label );
        Styles::Suite &styles( );
        Styles::Suite const &styles( ) const ;

        void productIDs( std::set<std::string> &a_ids, Transporting::Particles const &a_particles, bool a_transportablesOnly ) const ;
        int maximumLegendreOrder( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;

        Styles::TemperatureInfos temperatures( ) const ;

        std::size_t numberOfReactions( ) const ;
        Reaction *reaction( std::size_t a_index );
        Reaction const *reaction( std::size_t a_index ) const ;
        std::size_t numberOfOrphanProducts( ) const ;
        Reaction *orphanProduct( std::size_t a_index );
        Reaction const *orphanProduct( std::size_t a_index ) const ;

        bool hasFission( ) const ;

        Ancestry *findInAncestry3( std::string const &a_item ) { return( nullptr ); }                      /**< Always returns *nullptr*. */
        Ancestry const *findInAncestry3( std::string const &a_item ) const { return( nullptr ); }          /**< Always returns *nullptr*. */

        std::vector<double> const groupBoundaries( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Vector multiGroupInverseSpeed( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const ;

        Vector multiGroupCrossSection( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const ;
        Vector multiGroupQ( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, bool a_final ) const ;

        Vector multiGroupMultiplicity( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Vector multiGroupFissionNeutronMultiplicity( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const ;

        Matrix multiGroupProductMatrix( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, std::string const &a_productID, int a_order ) const ;
        Matrix multiGroupFissionMatrix( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, int a_order ) const ;
        Vector multiGroupTransportCorrection( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles, int a_order, TransportCorrectionType a_transportCorrectionType, double a_temperature ) const ;

        Vector multiGroupAvailableEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const ;
        Vector multiGroupAverageEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Vector multiGroupDepositionEnergy( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles ) const ;

        Vector multiGroupAvailableMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const ;
        Vector multiGroupAverageMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;
        Vector multiGroupDepositionMomentum( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, Transporting::Particles const &a_particles ) const ;

        Vector multiGroupGain( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo, std::string const &a_productID ) const ;

        void TNSL_crossSectionSumCorrection( std::string const &a_label, Functions::XYs1d &a_crossSectionSum );
        void TNSL_crossSectionSumCorrection( std::string const &a_label, Functions::Ys1d &a_crossSectionSum );
        void TNSL_crossSectionSumCorrection( std::string const &a_label, Vector &a_crossSectionSum ) {
            return( Protare::TNSL_crossSectionSumCorrection( a_label, a_crossSectionSum ) );
        }

        stringAndDoublePairs muCutoffForCoulombPlusNuclearElastic( ) const ;
        DelayedNeutronProducts delayedNeutronProducts( ) const { return( m_protare->delayedNeutronProducts( ) ); }
        void incompleteParticles( Transporting::Settings const &a_settings, std::set<std::string> &a_incompleteParticles ) const ;
};

namespace Map {

enum class EntryType { import, protare, TNSL };
#define GIDI_MapInteractionNuclearChars "nuclear"
#define GIDI_MapInteractionAtomicChars "atomic"

/*
============================================================
========================= BaseEntry ========================
============================================================
*/
class BaseEntry : public Ancestry {

    public:
        enum class PathForm { entered, cumulative, real };

    private:
        std::string m_name;                                 /**< Designates the entry as either a protare or a map. */
        Map const *m_parent;                                /**< Pointer to map containing *this*. */
        std::string m_path;                                 /**< Absolute or relative (to map file) path of the protare or map file. */
        std::string m_cumulativePath;                       /**< Currently not used. */

    public:
        BaseEntry( HAPI::Node const &a_node, std::string const &a_basePath, Map const *a_parent );
        virtual ~BaseEntry( ) = 0;

        std::string const &name( ) const { return( m_name ); }              /**< Returns the value of the **m_name** member. */
        Map const *parent( ) const { return( m_parent ); }                  /**< Returns the value of the **m_parent** member. */
        std::string path( PathForm a_form = PathForm::real ) const ;

        virtual EntryType entryType( ) const = 0;

        void libraries( std::vector<std::string> &a_libraries ) const ;
        virtual ProtareBase const *findProtareEntry( std::string const &a_projectileID, std::string const &a_targetID, 
                std::string const &a_library = "", std::string const &a_evaluation = "" ) const = 0 ;

        virtual void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const = 0;
};

/*
============================================================
========================== Import ==========================
============================================================
*/
class Import : public BaseEntry {

    private:
        Map *m_map;                                         /**< Map instance for this Import. */

    public:
        Import( HAPI::Node const &a_node, PoPI::Database const &a_pops, std::string const &a_basePath, Map const *a_parent );
        ~Import( );

        EntryType entryType( ) const { return( EntryType::import ); }   /**< Returns EntryType::import. */

        Map const *map( ) const { return( m_map ); }                    /**< Returns the value of the **m_map** member. */

        ProtareBase const *findProtareEntry( std::string const &a_projectileID, std::string const &a_targetID,
                std::string const &a_library = "", std::string const &a_evaluation = "" ) const ;
        std::string protareFilename( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_library = "",
                std::string const &a_evaluation = "", PathForm a_form = PathForm::real ) const ;
        bool isProtareAvailable( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_library = "",
                std::string const &a_evaluation = "" ) const {
            return( protareFilename( a_projectileID, a_targetID, a_library, a_evaluation ) != GIDI_emptyFileNameChars ); }
                                                                        /**< Returns the value of the **m_map** member. */
        std::vector<std::string> availableEvaluations( std::string const &a_projectileID, std::string const &a_targetID ) const ;

        Ancestry *findInAncestry3( std::string const &a_item ) { return( nullptr ); }                  /**< Always returns *nullptr*. */
        Ancestry const *findInAncestry3( std::string const &a_item ) const { return( nullptr ); }      /**< Always returns *nullptr*. */

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
======================= ProtareBase ========================
============================================================
*/
class ProtareBase : public BaseEntry {

    private:
        std::string m_projectileID;             /**< Projectile id for protare. */
        std::string m_targetID;                 /**< Target id for protare. */
        std::string m_evaluation;               /**< Evaluation string for protare. */
        std::string m_interaction;              /**< The interaction type for the protare. */

    public:
        ProtareBase( HAPI::Node const &a_node, std::string const &a_basePath, Map const *const a_map );
        ~ProtareBase( );

        std::string const &interaction( ) const { return( m_interaction ); }        /**< Returns the value of the **m_interaction** member. */
        void setInteraction( std::string const &a_interaction ) { m_interaction = a_interaction; }  /**< Set the **m_interaction** member to *a_interaction*. */
        std::string const &projectileID( ) const { return( m_projectileID ); }      /**< Returns the value of the **m_projectileID** member. */
        std::string const &targetID( ) const { return( m_targetID ); }              /**< Returns the value of the **m_targetID** member. */
        std::string const &evaluation( ) const { return( m_evaluation ); }          /**< Returns the value of the **m_evaluation** member. */

        bool isMatch( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_evaluation = "" ) const ;
        std::string const &library( ) const ;
        std::string const &resolvedLibrary( ) const ;

        ProtareBase const *findProtareEntry( std::string const &a_projectileID, std::string const &a_targetID,
                std::string const &a_library = "", std::string const &a_evaluation = "" ) const ;
        virtual GIDI::Protare *protare( Construction::Settings const &a_construction, PoPI::Database const &a_pops, ParticleSubstitution const &a_particleSubstitution ) const = 0 ;

        Ancestry *findInAncestry3( std::string const &a_item ) { return( nullptr ); }                  /**< Always returns *nullptr*. */
        Ancestry const *findInAncestry3( std::string const &a_item ) const { return( nullptr ); }      /**< Always returns *nullptr*. */
};

/*
============================================================
========================= Protare ==========================
============================================================
*/
class Protare : public ProtareBase {

    private:
        bool m_isPhotoAtomic;                   /**< true if photo-atomic protare and false otherwise. */

    public:
        Protare( HAPI::Node const &a_node, PoPI::Database const &a_pops, std::string const &a_basePath, Map const *const a_parent );
        ~Protare( );

        EntryType entryType( ) const { return( EntryType::protare ); }              /**< Returns EntryType::protare. */

        bool isPhotoAtomic( ) const { return( m_isPhotoAtomic ); }                  /**< Returns the value of the **m_isPhotoAtomic** member. */
        GIDI::Protare *protare( Construction::Settings const &a_construction, PoPI::Database const &a_pops, ParticleSubstitution const &a_particleSubstitution ) const ;

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
=========================== TNSL ===========================
============================================================
*/
class TNSL : public ProtareBase {

    private:
        std::string m_standardTarget;                           /**< The non-TNSL target. */
        std::string m_standardEvaluation;                       /**< The non-TNSL evaluation. */

    public:
        TNSL( HAPI::Node const &a_node, PoPI::Database const &a_pops, std::string const &a_basePath, Map const *const a_parent );
        ~TNSL( );

        EntryType entryType( ) const { return( EntryType::TNSL ); }                 /**< Returns EntryType::TNSL. */

        std::string const &standardTarget( ) const { return( m_standardTarget ); }
        std::string const &standardEvaluation( ) const { return( m_standardEvaluation ); }
        GIDI::Protare *protare( Construction::Settings const &a_construction, PoPI::Database const &a_pops, ParticleSubstitution const &a_particleSubstitution ) const ;

        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
=========================== Map ============================
============================================================
*/
class Map : public Ancestry {

    private:
        Map const *m_parent;                            /**< Pointer to map containing *this*. */
        std::string m_fileName;                         /**< Specified path to Map. */
        std::string m_realFileName;                     /**< Absolute, read path to Map. */
        std::string m_library;                          /**< The name of the library. */
        std::vector<BaseEntry *> m_entries;             /**< List of Map entries. */

        void initialize( std::string const &a_fileName, PoPI::Database const &a_pops, Map const *a_parent );
        void initialize( HAPI::Node const &a_node, std::string const &a_fileName, PoPI::Database const &a_pops, Map const *a_parent );

    public:
        Map( std::string const &a_fileName, PoPI::Database const &a_pops, Map const *a_parent = nullptr );
        Map( HAPI::Node const &a_node, std::string const &a_fileName, PoPI::Database const &a_pops, Map const *a_parent = nullptr );
        ~Map( );

        Map const *parent( ) const { return( m_parent ); }                      /**< Returns the value of the **m_parent** member. */
        std::string const &fileName( ) const { return( m_fileName ); }          /**< Returns the value of the **m_fileName** member. */
        std::string const &realFileName( ) const { return( m_realFileName ); }  /**< Returns the value of the **m_realFileName** member. */

        std::string const &library( ) const { return( m_library ); }            /**< Returns the value of the **m_library** member. */
        std::string const &resolvedLibrary( ) const ;
        void libraries( std::vector<std::string> &a_libraries ) const ;

        std::size_t size( ) const { return( m_entries.size( ) ); }              /**< Returns the number of entries in *this*. Does not descend map entries. */
        BaseEntry const *operator[]( std::size_t a_index ) const { return( m_entries[a_index] ); }
                                                                                /**< Returns the map entry at index *a_index*. */

        ProtareBase const *findProtareEntry( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_library = "",
                std::string const &a_evaluation = "" ) const ;
        std::string protareFilename( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_library = "",
                std::string const &a_evaluation = "", BaseEntry::PathForm a_form = BaseEntry::PathForm::real ) const ;

        bool isProtareAvailable( std::string const &a_projectileID, std::string const &a_targetID, std::string const &a_library = "",
                std::string const &a_evaluation = "" ) const {
            return( protareFilename( a_projectileID, a_targetID, a_library, a_evaluation, BaseEntry::PathForm::entered ) != GIDI_emptyFileNameChars ); }
                                /**< Returns true if the map contains a Protare matching *a_projectileID*, *a_targetID*, *a_library* and *a_evaluation*, and false otherwise. */
        bool isTNSL_target( std::string const &a_targetID ) const ;
        std::vector<std::string> availableEvaluations( std::string const &a_projectileID, std::string const &a_targetID ) const ;

        std::vector<ProtareBase const *> directory( std::string const &a_projectileID = "", std::string const &a_targetID = "", std::string const &a_library = "",
                std::string const &a_evaluation = "" ) const ;
        bool walk( MapWalkCallBack a_mapWalkCallBack, void *a_userData, int a_level = 0 ) const ;

        GIDI::Protare *protare( Construction::Settings const &a_construction, PoPI::Database const &a_pops, std::string const &a_projectileID, std::string const &a_targetID, 
                std::string const &a_library = "", std::string const &a_evaluation = "", bool a_targetRequiredInGlobalPoPs = true, bool a_requiredInPoPs = true ) const ;


        Ancestry *findInAncestry3( std::string const &a_item ) { return( nullptr ); }                  /**< Always returns *nullptr*. */
        Ancestry const *findInAncestry3( std::string const &a_item ) const { return( nullptr ); }      /**< Always returns *nullptr*. */

        void saveAs( std::string const &a_fileName ) const ;
        void toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

}           // End of namespace Map.

namespace Functions {
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
        FissionEnergyRelease( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
        ~FissionEnergyRelease( );

        double domainMin( ) const { return( m_nonNeutrinoEnergy->domainMin( ) ); }                  /**< Returns the minimum domain value for the energy released. */
        double domainMax( ) const { return( m_nonNeutrinoEnergy->domainMax( ) ); }                  /**< Returns the maximum domain value for the energy released. */
        double evaluate( double a_x1 ) const { return( m_nonNeutrinoEnergy->evaluate( a_x1 ) ); }   /**< Returns the value of **m_nonNeutrinoEnergy** evaluated at *a_x1*. */
        Vector multiGroupQ( Transporting::MG const &a_settings, Styles::TemperatureInfo const &a_temperatureInfo ) const ;

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

}           // End of namespace Functions.

/*
============================================================
========================== Groups ==========================
============================================================
*/
class Groups : public Suite {

    public:
        Groups( );
        Groups( std::string const &a_fileName );

        void addFile( std::string const &a_fileName );
};

/*
============================================================
========================== Fluxes ==========================
============================================================
*/
class Fluxes : public Suite {

    public:
        Fluxes( );
        Fluxes( std::string const &a_fileName );

        void addFile( std::string const &a_fileName );
};

/*
============================================================
========================== others ==========================
============================================================
*/
Form *parseExternalFilesSuite( Construction::Settings const &a_construction, Suite *parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pop, PoPI::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseStylesSuite( Construction::Settings const &a_construction, Suite *parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pop, PoPI::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseTransportablesSuite( Construction::Settings const &a_construction, Suite *a_parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseReaction( Construction::Settings const &a_construction, Suite *a_parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseOrphanProduct( Construction::Settings const &a_construction, Suite *a_parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseFissionComponent( Construction::Settings const &a_construction, Suite *a_parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops,
                PoPI::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles );
Form *parseReactionType( std::string const &a_moniker, Construction::Settings const &a_construction, Suite *a_parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseSumsCrossSectionsSuite( Construction::Settings const &a_construction, Suite *a_parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops,
                PoPI::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles );
Form *parseSumsMultiplicitiesSuite( Construction::Settings const &a_construction, Suite *a_parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops,
                PoPI::Database const &a_internalPoPs, std::string const &a_name, Styles::Suite const *a_styles );
Form *parseDoubleDifferentialCrossSectionSuite( Construction::Settings const &a_construction, Suite *a_parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseScatteringAtom( Construction::Settings const &a_construction, Suite *a_parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseCrossSectionSuite( Construction::Settings const &a_construction, Suite *parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pop, PoPI::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseDelayedNeutronsSuite( Construction::Settings const &a_construction, Suite *parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pop, PoPI::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseFissionEnergyReleasesSuite( Construction::Settings const &a_construction, Suite *parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pop, PoPI::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parsePhysicalQuantitySuite( Construction::Settings const &a_construction, Suite *parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pop, PoPI::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseAvailableSuite( Construction::Settings const &a_construction, Suite *a_parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseQSuite( Construction::Settings const &a_construction, Suite *parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pop, PoPI::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseProductSuite( Construction::Settings const &a_construction, Suite *parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pop, PoPI::Database const &a_internalPoPs, 
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseMultiplicitySuite( Construction::Settings const &a_construction, Suite *parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pop, PoPI::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseDistributionSuite( Construction::Settings const &a_construction, Suite *parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pop, PoPI::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseAverageEnergySuite( Construction::Settings const &a_construction, Suite *parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pop, PoPI::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Form *parseAverageMomentumSuite( Construction::Settings const &a_construction, Suite *parent, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pop, PoPI::Database const &a_internalPoPs,
                std::string const &a_name, Styles::Suite const *a_styles );
Functions::Function1dForm *data1dParse( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *parent );
Functions::Function1dForm *data1dParseAllowEmpty( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent );
void data1dListParse( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, std::vector<Functions::Function1dForm *> &a_function1ds );
Functions::Function2dForm *data2dParse( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *parent );
void data2dListParse( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, std::vector<Functions::Function2dForm *> &a_function2ds );
Functions::Function3dForm *data3dParse( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *parent );
void checkOuterDomainValues1d( std::vector<Functions::Function1dForm *> &a_functions, std::vector<double> &a_Xs );
void checkOuterDomainValues2d( std::vector<Functions::Function2dForm *> &a_functions, std::vector<double> &a_Xs );
void checkSequentialDomainLimits1d( std::vector<Functions::Function1dForm *> &a_functions, std::vector<double> &a_Xs );
void checkSequentialDomainLimits2d( std::vector<Functions::Function2dForm *> &a_functions, std::vector<double> &a_Xs );

int parseFlattened1d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Vector &data );

Vector collapse( Vector const &a_vector, Transporting::Settings const &a_settings, Transporting::Particles const &a_particles, double a_temperature );
Matrix collapse( Matrix const &a_matrix, Transporting::Settings const &a_settings, Transporting::Particles const &a_particles, double a_temperature, std::string const &a_productID );

Vector transportCorrect( Vector const &a_vector, Vector const &a_transportCorrection );
Matrix transportCorrect( Matrix const &a_matrix, Vector const &a_transportCorrection );

Vector multiGroupXYs1d( Transporting::MultiGroup const &a_boundaries, Functions::XYs1d const &a_function, Transporting::Flux const &a_flux );

int ENDL_CFromENDF_MT( int ENDF_MT, int *ENDL_C, int *ENDL_S );

GNDS_FileType GNDS_fileType( std::string const &a_fileName, GNDS_FileTypeInfo &a_GNDS_fileTypeInfo );

/*
*   The following are in the file GIDI_misc.cpp.
*/
std::string realPath( char const *a_path );
std::string realPath( std::string const &a_path );
std::vector<std::string> splitString( std::string const &a_string, char a_delimiter );
long binarySearchVector( double a_x, std::vector<double> const &a_Xs );
void intsToXMLList( WriteInfo &a_writeInfo, std::string const &a_indent, std::vector<int> a_values, std::string const &a_attributes );
void parseValuesOfDoubles( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, nf_Buffer<double> &a_vector );
void parseValuesOfDoubles( HAPI::Node const &a_node, SetupInfo &a_setupInfo, nf_Buffer<double> &a_vector, int a_useSystem_strtod );
void parseValuesOfInts( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, std::vector<int> &a_vector );
void parseValuesOfInts( HAPI::Node const &a_node, SetupInfo &a_setupInfo, nf_Buffer<int> &a_vector );
void doublesToXMLList( WriteInfo &a_writeInfo, std::string const &a_indent, std::vector<double> a_values, std::size_t a_start = 0, bool a_newLine = true,
        std::string const &a_valueType = "" );
Frame parseFrame( HAPI::Node const &a_node, SetupInfo &a_setupInfo, std::string const &a_name );
std::string frameToString( Frame a_frame );
std::string intToString( int a_value );
std::string size_t_ToString( std::size_t a_value );
std::string nodeWithValuesToDoubles( WriteInfo &a_writeInfo, std::string const &a_nodeName, std::vector<double> const &a_values );
std::string doubleToShortestString( double a_value, int a_significantDigits = 15, int a_favorEFormBy = 0 );

Functions::Ys1d gridded1d2GIDI_Ys1d( Functions::Function1dForm const &a_function1d );
Functions::Ys1d vector2GIDI_Ys1d( Axes const &a_axes, Vector const &a_vector );

std::string LLNL_gidToLabel( int a_gid );
std::string LLNL_fidToLabel( int a_fid );

std::vector<std::string> sortedListOfStrings( std::vector<std::string> const &a_strings, bool a_orderIsAscending = true );

void energy2dToXMLList( WriteInfo &a_writeInfo, std::string const &a_moniker, std::string const &a_indent, Functions::Function1dForm *a_function );

std::vector<Transporting::Flux> settingsFluxesFromFunction3d( Functions::Function3dForm const &a_function3d );

}           // End of namespace GIDI.

#endif      // End of GIDI_hpp_included

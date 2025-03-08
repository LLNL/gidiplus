/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#ifndef CADI_hpp_included
#define CADI_hpp_included 1

#include <GUPI.hpp>
#include <PoPI.hpp>

#define CADI_formatChars "format"
#define CADI_libraryChars "library"
#define CADI_evaluationChars "evaluation"
#define CADI_isotopicAbundancesByChemicalElementChars "isotopicAbundancesByChemicalElement"
#define CADI_chemicalElementsChars "chemicalElements"
#define CADI_chemicalElementChars "chemicalElement"
#define CADI_symbolChars "symbol"
#define CADI_isotopesChars "isotopes"
#define CADI_isotopeChars "isotope"
#define CADI_idChars "id"
#define CADI_atomFractionChars "atomFraction"
#define CADI_uncertaintyChars "uncertainty"
#define CADI_iamChars "iam"
#define CADI_importChars "import"
#define CADI_checksumChars "checksum"
#define CADI_algorithmChars "algorithm"
#define CADI_pathChars "path"

namespace CADI {

namespace Map {

class IAM;

}           // End of namespace Map.

/*
============================================================
========================== Isotope =========================
============================================================
*/

class Isotope : public GUPI::Entry {

    private:
        double m_atomFraction;                              /**< The atom fraction for *this* isotope. */
        double m_uncertainty;                               /**< The uncertainty of the atom fraction. */

    public:
        Isotope( std::string a_id, double a_atomFraction, double a_uncertainty );
        Isotope( HAPI::Node const &a_node );
        ~Isotope( );

        std::string const &id( ) const { return( keyValue( ) ); }       /**< Returns a const reference to the results of the call to the *keyValue()* method. */
        double atomFraction( ) const { return( m_atomFraction ); }      /**< Returns the value of the *m_atomFraction* member. */
        double uncertainty( ) const { return( m_uncertainty ); }        /**< Returns the value of the *m_uncertainty* member. */

        GUPI::Ancestry *findInAncestry3( LUPI_maybeUnused std::string const &a_item ) { return( nullptr ); }
        GUPI::Ancestry const *findInAncestry3( LUPI_maybeUnused std::string const &a_item ) const { return( nullptr ); }

        LUPI_HOST void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
        void toXMLList( GUPI::WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
===================== ChemicalElement =====================
============================================================
*/

class ChemicalElement : public GUPI::Entry {

    private:
        GUPI::Suite m_isotopes;                                     /**< The list of isotopes for the chemical element. */

    public:
        ChemicalElement( std::string const &a_symbol );
        ChemicalElement( HAPI::Node const &a_node );
        ~ChemicalElement( );

        std::string const &symbol( ) const { return( keyValue( ) ); }
        GUPI::Suite &isotopes( ) { return( m_isotopes ); }
        GUPI::Suite const &isotopes( ) const { return( m_isotopes ); }

        Isotope const *operator[]( std::string const &a_id ) const ;
        GUPI::Ancestry *findInAncestry3( LUPI_maybeUnused std::string const &a_item ) { return( nullptr ); }
        GUPI::Ancestry const *findInAncestry3( LUPI_maybeUnused std::string const &a_item ) const { return( nullptr ); }

        double mass( PoPI::Database const &a_pops, std::string const &a_unit = "amu" ) const ;

        LUPI_HOST void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
        void toXMLList( GUPI::WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
=========== IsotopicAbundancesByChemicalElement ============
============================================================
*/

class IsotopicAbundancesByChemicalElement : public GUPI::Ancestry {

    private:
        std::string m_format;
        std::string m_evaluation;
        GUPI::Suite m_chemicalElements;    

    public:
        IsotopicAbundancesByChemicalElement( );
        IsotopicAbundancesByChemicalElement( std::string const &a_format, std::string const &a_evaluation );
        IsotopicAbundancesByChemicalElement( std::string const &a_fileName );
        ~IsotopicAbundancesByChemicalElement( );

        std::string const &format( ) const { return( m_format ); }                              /**< Returns a const reference to the *m_format* member. */
        std::string const &evaluation( ) const { return( m_evaluation ); }                      /**< Returns a const reference to the *m_evaluation* member. */
        GUPI::Suite &chemicalElements( ) { return( m_chemicalElements ); }                      /**< Returns a reference to the *m_chemicalElements* member. */
        GUPI::Suite const &chemicalElements( ) const { return( m_chemicalElements ); }          /**< Returns a const reference to the *m_chemicalElements* member. */
        ChemicalElement const *operator[]( std::string const &a_symbol ) const ;
        ChemicalElement const *find( std::string const &a_symbol, std::string const &a_evaluation = "" ) const ;

        GUPI::Ancestry *findInAncestry3( LUPI_maybeUnused std::string const &a_item ) { return( nullptr ); }
        GUPI::Ancestry const *findInAncestry3( LUPI_maybeUnused std::string const &a_item ) const { return( nullptr ); }

        LUPI_HOST void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
        void toXMLList( GUPI::WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

namespace Map {

/*
============================================================
=========================== Base ===========================
============================================================
*/

class Base : public GUPI::Ancestry {

    private:
        std::string m_sourcePath;                   /**< If an **IAM** instance then this is its source path; otherwise, it is the source path of the parent **IAM** instance. */
        std::string m_checksum;                     /**< Check sum for the included data. */
        std::string m_algorithm;                    /**< The algorithm used to calculate the checksum. */

    public:
        Base( std::string const &a_moniker, std::string const &a_sourcePath );
        Base( HAPI::Node const &a_node, std::string const &a_sourcePath );
        ~Base( );

        std::string const &sourcePath( ) const { return( m_sourcePath ); }                      /**< Returns a reference to *m_sourcePath*. */
        void setSourcePath( std::string const &a_sourcePath ) { m_sourcePath = a_sourcePath; };  /**< Sets member *m_sourcePath* to *a_sourcePath*. */
        std::string const &checksum( ) const { return( m_checksum ); }                          /**< Returns a reference to *m_sourcePath*. */
        void setChecksum( std::string const &a_checksum ) { m_checksum = a_checksum; }          /**< Sets member *m_checksum* to *a_checksum*. */
        std::string const algorithm( ) const ;
        void setAlgorithm( std::string const &a_algorithm ) { m_algorithm = a_algorithm; }      /**< Sets member *m_algorithm* to *a_algorithm*. */

        GUPI::Ancestry *findInAncestry3( LUPI_maybeUnused std::string const &a_item ) { return( nullptr ); }
        GUPI::Ancestry const *findInAncestry3( LUPI_maybeUnused std::string const &a_item ) const { return( nullptr ); }

        LUPI_HOST void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
        virtual std::string standardXML_attributes( bool a_checkAncestor = true ) const ;
};

/*
============================================================
========================= EntryBase ========================
============================================================
*/

class EntryBase : public Base {

    private:
        std::string m_path;

    public:
        EntryBase( std::string const &a_moniker );
        EntryBase( HAPI::Node const &a_node, std::string const &a_sourcePath );
        ~EntryBase( );

        std::string const &path( ) const { return( m_path ); }
        std::string fileName( ) const ;

        virtual ChemicalElement const *find( std::string const &a_symbol, std::string const &a_evaluation = "" ) const = 0;
        virtual CADI::IsotopicAbundancesByChemicalElement const *findEvaluation( std::string const &a_evaluation ) const = 0;

        virtual LUPI_HOST void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode ) = 0;
        std::string standardXML_attributes( bool a_checkAncestor = true ) const ;
        void toXMLList( GUPI::WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

/*
============================================================
========================== Import ==========================
============================================================
*/

class Import : public EntryBase {

    private:
        IAM *m_IAM;                                 /**< The IAM instance of referenced by *this*. */

    public:
        Import( );
        Import( HAPI::Node const &a_node, std::string const &a_sourcePath );
        ~Import( );

        LUPI_HOST void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
        ChemicalElement const *find( std::string const &a_symbol, std::string const &a_evaluation = "" ) const ;
        CADI::IsotopicAbundancesByChemicalElement const *findEvaluation( std::string const &a_evaluation ) const ;
};

/*
============================================================
============ IsotopicAbundancesByChemicalElement ===========
============================================================
*/

class IsotopicAbundancesByChemicalElement : public EntryBase {

    private:
        std::string m_evaluation;                                                           /**< The evalaution string for *this*. */
        CADI::IsotopicAbundancesByChemicalElement *m_isotopicAbundancesByChemicalElement;   /**< The isotopicAbundancesByChemicalElement instance of referenced by *this*. */

    public:
        IsotopicAbundancesByChemicalElement( );
        IsotopicAbundancesByChemicalElement( HAPI::Node const &a_node, std::string const &a_sourcePath );
        ~IsotopicAbundancesByChemicalElement( );

        std::string const &evaluation( ) const { return( m_evaluation ); }
        ChemicalElement const *find( std::string const &a_symbol, std::string const &a_evaluation = "" ) const ;
        CADI::IsotopicAbundancesByChemicalElement const *findEvaluation( std::string const &a_evaluation ) const ;

        LUPI_HOST void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
        std::string standardXML_attributes( bool a_checkAncestor = true ) const ;
};

/*
============================================================
============================ IAM ===========================
============================================================
*/

class IAM : public Base {

    private:
        std::string m_library;                                          /**< The name of the iam library. */
        std::string m_format;                                           /**< The format used to store the data. */
        std::vector<EntryBase *> m_entries;                             /**< The list of entries in *this*. */

    public:
        IAM( );
        IAM( std::string const &a_sourcePath );
        ~IAM( );

        std::string const &library( ) const { return( m_library ); }
        std::string const &format( ) const { return( m_format ); }

        void addFile( std::string const &a_sourcePath );
        ChemicalElement const *find( std::string const &a_symbol, std::string const &a_evaluation = "" ) const ;
        CADI::IsotopicAbundancesByChemicalElement const *findEvaluation( std::string const &a_evaluation ) const ;

        LUPI_HOST void serialize( LUPI::DataBuffer &a_buffer, LUPI::DataBuffer::Mode a_mode );
        void toXMLList( GUPI::WriteInfo &a_writeInfo, std::string const &a_indent = "" ) const ;
};

}           // End of namespace Map.

}           // End of namespace CADI.

#endif      // End of CADI_hpp_included

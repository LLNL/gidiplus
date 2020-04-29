/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#ifndef PoPI_hpp_included
#define PoPI_hpp_included 1

#include <string>
#include <map>
#include <vector>
#include <list>
#include <iostream>
#include <stdexcept>
#include <typeinfo>
#include <fstream>
#include <exception>

#include "pugixml.hpp"

namespace PoPI {

#define FORMAT "0.1"

#define PoPI_AMU2MeV_c2 931.494028
#define PoPI_electronMass_MeV_c2 0.5109989461

#define family_gaugeBoson "gaugeBoson"
#define family_lepton "lepton"
#define family_baryon "baryon"
#define family_nuclide "nuclide"
#define family_nucleus "nucleus"
#define family_unorthodox "unorthodox"

enum class Particle_class { gaugeBoson, lepton, baryon, unorthodox, nuclide, nucleus, chemicalElement, isotope, alias, metaStable };

#define pq_massTag "mass"
#define pq_spinTag "spin"
#define pq_chargeTag "charge"
#define pq_parityTag "parity"
#define pq_halflifeTag "halflife"
#define pq_energyTag "energy"

#define pq_doubleTag "double"
#define pq_integerTag "integer"
#define pq_fractionTag "fraction"
#define pq_stringTag "string"
#define pq_shellTag "shell"

#define PoPI_decayMode_electroMagnetic "electroMagnetic"

enum class PQ_class { Double, integer, fraction, string, shell };

class NuclideGammaBranchStateInfos;
class Base;
class SymbolBase;
class Decay;
class DecayMode;
class DecayData;
class Particle;
class Nuclide;
class Isotope;
class ChemicalElement;
class Database;

void appendXMLEnd( std::vector<std::string> &a_XMLList, std::string const &a_label );

int particleZ( Base const &a_particle, bool isNeutronProtonANucleon = false );
int particleZ( Database const &a_pops, int a_index, bool isNeutronProtonANucleon = false );
int particleZ( Database const &a_pops, std::string const &a_id, bool isNeutronProtonANucleon = false );

int particleA( Base const &a_particle, bool isNeutronProtonANucleon = false );
int particleA( Database const &a_pops, int a_index, bool isNeutronProtonANucleon = false );
int particleA( Database const &a_pops, std::string const &a_id, bool isNeutronProtonANucleon = false );

int particleZA( Base const &a_particle, bool isNeutronProtonANucleon = false );
int particleZA( Database const &a_pops, int a_index, bool isNeutronProtonANucleon = false );
int particleZA( Database const &a_pops, std::string const &a_id, bool isNeutronProtonANucleon = false );

struct IDs {
    static std::string const photon;
    static std::string const neutron;
    static std::string const proton;
};

typedef std::vector<Base *> ParticleList;
typedef std::vector<SymbolBase *> SymbolList;

/*
============================================================
======================== Exception =========================
============================================================
*/
class Exception : public std::runtime_error {

    public :
        explicit Exception( std::string const &a_message );

};

/*
============================================================
========================== Suite ===========================
============================================================
*/
template <class T, class T2>
class Suite {

    private:
        std::string m_label;
        std::vector<T *> m_items;

    public:
        Suite( std::string const &a_label ) : m_label( a_label ) { };
        ~Suite( );
        void appendFromParentNode( pugi::xml_node const &a_node, Database *a_DB, T2 *a_parent );
        void appendFromParentNode2( pugi::xml_node const &a_node, T2 *a_parent );

        std::string::size_type size( void ) const { return( m_items.size( ) ); }
        T &operator[]( int a_index ) const { return( *m_items[a_index] ); }
        std::string &label( void ) { return( m_label ); }

        void toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const ;
};
/*
=========================================================
*/
template <class T, class T2>
Suite<T, T2>::~Suite( ) {

    std::string::size_type i1, _size = m_items.size( );

    for( i1 = 0; i1 < _size; ++i1 ) delete m_items[i1];

// Ask Adam why next line does not work.
//    for( std::vector<T *>::iterator iter = m_items.begin( ); iter != m_items.end( ); ++iter ) delete *iter;
}
/*
=========================================================
*/
template <class T, class T2>
void Suite<T, T2>::appendFromParentNode( pugi::xml_node const &a_node, Database *a_DB, T2 *a_parent ) {

    for( pugi::xml_node child = a_node.first_child( ); child; child = child.next_sibling( ) ) {
        T *item = new T( child, a_DB, a_parent );
        m_items.push_back( item );
    }
}
/*
=========================================================
*/
template <class T, class T2>
void Suite<T, T2>::appendFromParentNode2( pugi::xml_node const &a_node, T2 *a_parent ) {

    for( pugi::xml_node child = a_node.first_child( ); child; child = child.next_sibling( ) ) {
        T *item = new T( child, a_parent );
        m_items.push_back( item );
    }
}
/*
=========================================================
*/
template <class T, class T2>
void Suite<T, T2>::toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const {

    std::string::size_type _size = m_items.size( );
    std::string indent2 = a_indent1 + "  ";

    if( _size == 0 ) return;

    std::string header = a_indent1 + "<" + m_label + ">";
    a_XMLList.push_back( header );
    for( std::string::size_type i1 = 0; i1 < _size; ++i1 ) m_items[i1]->toXMLList( a_XMLList, indent2 );

    appendXMLEnd( a_XMLList, m_label );
}

/*
============================================================
===================== PhysicalQuantity =====================
============================================================
*/
class PhysicalQuantity {

    private:
        PQ_class m_class;
        std::string m_tag;
        std::string m_label;
        std::string m_valueString;
        std::string m_unit;

    public:
        PhysicalQuantity( pugi::xml_node const &a_node, PQ_class a_class );
        virtual ~PhysicalQuantity( );

        PQ_class Class( void ) const { return( m_class ); }
        std::string const &tag( void ) const { return( m_tag ); }
        std::string const &label( void ) const { return( m_label ); }
        std::string const &valueString( void ) const { return( m_valueString ); }
        std::string const &unit( void ) const { return( m_unit ); }

        void toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const ;
        virtual std::string valueToString( void ) const = 0;
};

/*
============================================================
========================= PQ_double ========================
============================================================
*/
class PQ_double : public PhysicalQuantity {

    private:
        double m_value;

    public:
        PQ_double( pugi::xml_node const &a_node );
        PQ_double( pugi::xml_node const &a_node, PQ_class a_class );
        void initialize( );
        virtual ~PQ_double( );

        double value( void ) const { return( m_value ); }
        double value( char const *a_unit ) const ;
        double value( std::string const &a_unit ) const { return( value( a_unit.c_str( ) ) ); }
        virtual std::string valueToString( void ) const ;
};

/*
============================================================
========================= PQ_integer =======================
============================================================
*/
class PQ_integer : public PhysicalQuantity {

    private:
        int m_value;

    public:
        PQ_integer ( pugi::xml_node const &a_node );
        virtual ~PQ_integer( );

        int value( void ) const { return( m_value ); }
        int value( char const *a_unit ) const ;
        int value( std::string const &a_unit ) const { return( value( a_unit.c_str( ) ) ); }
        virtual std::string valueToString( void ) const ;
};

/*
============================================================
========================= PQ_fraction ========================
============================================================
*/
class PQ_fraction : public PhysicalQuantity {

    public:
        PQ_fraction( pugi::xml_node const &a_node );
        virtual ~PQ_fraction( );

        std::string value( void ) const ;
        std::string value( char const *a_unit ) const ;
        std::string value( std::string const &a_unit ) const { return( value( a_unit.c_str( ) ) ); }
        virtual std::string valueToString( void ) const ;
};

/*
============================================================
========================= PQ_string ========================
============================================================
*/
class PQ_string : public PhysicalQuantity {

    public:
        PQ_string( pugi::xml_node const &a_node );
        virtual ~PQ_string( );

        std::string value( void ) const { return( valueString( ) ); }
        std::string value( char const *a_unit ) const ;
        std::string value( std::string const &a_unit ) const { return( value( a_unit.c_str( ) ) ); }
        virtual std::string valueToString( void ) const ;
};

/*
============================================================
========================= PQ_shell =========================
============================================================
*/
class PQ_shell : public PQ_double {

    public:
        PQ_shell( pugi::xml_node const &a_node );
        ~PQ_shell( );
};

/*
============================================================
========================= PQ_suite =========================
============================================================
*/
class PQ_suite : public std::vector<PhysicalQuantity *> {

    private:
        std::string m_label;

    public:
        PQ_suite( pugi::xml_node const &a_node );
        ~PQ_suite( );

        std::string &label( void ) { return( m_label ); }

        void toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const ;
};

/*
============================================================
================== NuclideGammaBranchInfo ==================
============================================================
*/
class NuclideGammaBranchInfo {

    private:
        double m_probability;
        double m_photonEmissionProbability;
        double m_gammaEnergy;
        std::string m_residualState;

    public:
        NuclideGammaBranchInfo( double a_probability, double a_photonEmissionProbability, double a_gammaEnergy, std::string const &a_residualState );
        NuclideGammaBranchInfo( NuclideGammaBranchInfo const &a_nuclideGammaBranchInfo );

        double probability( ) const { return( m_probability ); }
        double photonEmissionProbability( ) const { return( m_photonEmissionProbability ); }
        double gammaEnergy( ) const { return( m_gammaEnergy ); }
        std::string const &residualState( ) const { return( m_residualState ); }
};

/*
============================================================
================= NuclideGammaBranchStateInfo ================
============================================================
*/
class NuclideGammaBranchStateInfo {

    private:
        std::string m_state;
        bool m_derivedCalculated;
        double m_multiplicity;                                  /* Data derived from m_branches data. */
        double m_averageGammaEnergy;                            /* Data derived from m_branches data. */
        std::vector<NuclideGammaBranchInfo> m_branches;

    public:
        NuclideGammaBranchStateInfo( std::string a_state );

        std::string const &state( ) const { return( m_state ); }
        bool derivedCalculated( ) const { return( m_derivedCalculated ); }
        double multiplicity( ) const { return( m_multiplicity ); }
        double averageGammaEnergy( ) const { return( m_averageGammaEnergy ); }
        std::vector<NuclideGammaBranchInfo> const &branches( ) const { return( m_branches ); }

        void add( NuclideGammaBranchInfo const &a_nuclideGammaBranchInfo );
        void calculateDerivedData( NuclideGammaBranchStateInfos &a_nuclideGammaBranchStateInfos );
};

/*
============================================================
================ NuclideGammaBranchStateInfos ================
============================================================
*/
class NuclideGammaBranchStateInfos {

    private:
        std::vector<NuclideGammaBranchStateInfo *> m_nuclideGammaBranchStateInfos;

    public:
        NuclideGammaBranchStateInfos( );
        ~NuclideGammaBranchStateInfos( );

        std::size_t size( ) const { return( m_nuclideGammaBranchStateInfos.size( ) ); }
        NuclideGammaBranchStateInfo const *operator[]( std::size_t a_index ) const { return( m_nuclideGammaBranchStateInfos[a_index] ); }
        std::vector<NuclideGammaBranchStateInfo *> &nuclideGammaBranchStateInfos( ) { return( m_nuclideGammaBranchStateInfos ); }
        void add( NuclideGammaBranchStateInfo *a_nuclideGammaBranchStateInfo );
        NuclideGammaBranchStateInfo *find( std::string const &a_state );
};

/*
============================================================
=========================== Base ===========================
============================================================
*/
class Base {

    private:
        std::string m_id;
        Particle_class m_class;
        int m_index;

    public:
        Base( std::string const &a_id, Particle_class a_class );
        Base( pugi::xml_node const &a_node, std::string const &a_label, Particle_class a_class );
        virtual ~Base( );

        std::string const &ID( void ) const { return( m_id ); }
        int index( void ) const { return( m_index ); }
        void setIndex( int a_index ) { m_index = a_index; }
        Particle_class Class( void ) const { return( m_class ); }
        virtual bool isParticle( ) const { return( true ); }
        bool isAlias( void ) const { return( ( m_class == Particle_class::alias ) || isMetaStableAlias( ) ); }
        bool isMetaStableAlias( void ) const { return( m_class == Particle_class::metaStable ); }

        bool isGaugeBoson( ) const { return( m_class == Particle_class::gaugeBoson ); }
        bool isLepton( ) const { return( m_class == Particle_class::lepton ); }
        bool isBaryon( ) const { return( m_class == Particle_class::baryon ); };
        bool isUnorthodox( ) const { return( m_class == Particle_class::unorthodox ); }
        bool isNucleus( ) const { return( m_class == Particle_class::nucleus ); }
        bool isNuclide( ) const { return( m_class == Particle_class::nuclide ); }
        bool isIsotope( ) const { return( m_class == Particle_class::isotope ); }
        bool isChemicalElement( ) const { return( m_class == Particle_class::chemicalElement ); }
};

/*
============================================================
========================== IDBase ==========================
============================================================
*/
class IDBase : public Base {

    public:
        IDBase( std::string const &a_id, Particle_class a_class );
        IDBase( pugi::xml_node const &a_node, Particle_class a_class );
        virtual ~IDBase( );       // BRB This should be virtual but I cannot get it to work without crashing.

        int addToDatabase( Database *a_DB );
};

/*
============================================================
======================== SymbolBase ========================
============================================================
*/
class SymbolBase : public Base {

    public:
        SymbolBase( pugi::xml_node const &a_node, Particle_class a_class );
        ~SymbolBase( );

        std::string const &symbol( ) const { return( ID( ) ); }

        int addToSymbols( Database *a_DB );
        bool isParticle( ) const { return( false ); }
};

/*
============================================================
========================= Product ==========================
============================================================
*/
class Product {

    private:
        int m_id;
        std::string m_pid;
        std::string m_label;

    public:
        Product( pugi::xml_node const &a_node, Decay *a_DB );
        ~Product( );

        int ID( ) const { return( m_id ); }
        std::string const &pid( ) const { return( m_pid ); }
        std::string const &label( ) const { return( m_label ); }

        void toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const ;
};

/*
============================================================
========================== Decay ===========================
============================================================
*/
class Decay {

    private:
        int m_index;
        Suite<Product, Decay> m_products;

    public:
        Decay( pugi::xml_node const &a_node, DecayMode const *a_decayMode );
        ~Decay( );

        int index( void ) const { return( m_index ); }
        Suite<Product, Decay> const &products( void ) const { return( m_products ); }
        void toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const ;
};

/*
============================================================
======================== DecayMode =========================
============================================================
*/
class DecayMode {

    private:
        std::string m_label;
        std::string m_mode;
        PQ_suite m_probability;
        PQ_suite m_photonEmissionProbabilities;
        Suite<Decay, DecayMode> m_decayPath;

    public:
        DecayMode( pugi::xml_node const &a_node, DecayData const *a_decayData );
        ~DecayMode( );

        std::string const &label( ) const { return( m_label ); }
        std::string const &mode( ) const { return( m_mode ); }
        PQ_suite const &probability( ) const { return( m_probability ); }
        PQ_suite const &photonEmissionProbabilities( ) const { return( m_photonEmissionProbabilities ); }
        Suite<Decay, DecayMode> const &decayPath( ) const { return( m_decayPath ); }

        void calculateNuclideGammaBranchStateInfo( PoPI::Database const &a_pops, NuclideGammaBranchStateInfo &a_nuclideGammaBranchStateInfo ) const ;
        void toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const ;
};

/*
============================================================
======================== DecayData =========================
============================================================
*/
class DecayData {

    private:
        Suite<DecayMode, DecayData> m_decayModes;

    public:
        DecayData( pugi::xml_node const &a_node );
        ~DecayData( );

        Suite<DecayMode, DecayData> const &decayModes( void ) const { return( m_decayModes ); }

        void calculateNuclideGammaBranchStateInfo( PoPI::Database const &a_pops, NuclideGammaBranchStateInfo &a_nuclideGammaBranchStateInfo ) const ;
        void toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const ;
};

/*
============================================================
========================= Particle =========================
============================================================
*/
class Particle : public IDBase {

    private:
        std::string m_family;
        int m_hasNucleus;           // 0 = no, -1 = yes and 1 = is nucleus
        PQ_suite m_mass;
        PQ_suite m_spin;
        PQ_suite m_parity;
        PQ_suite m_charge;
        PQ_suite m_halflife;
        DecayData m_decayData;

    public:
        Particle( pugi::xml_node const &a_node, Particle_class a_class, std::string const &a_family, int a_hasNucleus = 0 );
        virtual ~Particle( );

        std::string const &family( void ) const { return( m_family ); }
        int hasNucleus( void ) const { return( m_hasNucleus ); }

        virtual PQ_suite const &mass( void ) const { return( m_mass ); }
        virtual double massValue( char const *a_unit ) const ;
        double massValue( std::string const &a_unit ) const { return( massValue( a_unit.c_str( ) ) ); }

        PQ_suite const &spin( ) const { return( m_spin ); }
        PQ_suite const &parity( ) const { return( m_parity ); }
        PQ_suite const &charge( ) const { return( m_charge ); }
        PQ_suite const &halflife( ) const { return( m_halflife ); }
        DecayData const &decayData( ) const { return( m_decayData ); }

        void toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const ;
        virtual std::string toXMLListExtraAttributes( void ) const ;
        virtual void toXMLListExtraElements( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const ;
};

/*
============================================================
======================== GaugeBoson ========================
============================================================
*/
class GaugeBoson : public Particle {

    public:
        GaugeBoson( pugi::xml_node const &a_node, Database *a_DB, Database *a_parent );
        virtual ~GaugeBoson( );
};

/*
============================================================
========================== Lepton ==========================
============================================================
*/
class Lepton : public Particle {

    private:
        std::string m_generation;

    public:
        Lepton( pugi::xml_node const &a_node, Database *a_DB, Database *a_parent );
        virtual ~Lepton( );

        std::string const &generation( void ) const { return( m_generation ); }
        virtual std::string toXMLListExtraAttributes( void ) const ;
};

/*
============================================================
========================== Baryon ==========================
============================================================
*/
class Baryon : public Particle {

    private:

    public:
        Baryon( pugi::xml_node const &a_node, Database *a_DB, Database *a_parent );
        virtual ~Baryon( );
};

/*
============================================================
======================== Unorthodox ========================
============================================================
*/
class Unorthodox : public Particle {

    private:

    public:
        Unorthodox( pugi::xml_node const &a_node, Database *a_DB, Database *a_parent );
        virtual ~Unorthodox( );
};

/*
============================================================
========================== Nucleus =========================
============================================================
*/
class Nucleus : public Particle {

    private:
        Nuclide *m_nuclide;
        int m_Z;
        int m_A;
        std::string m_levelName;
        int m_levelIndex;
        PQ_suite m_energy;

    public:
        Nucleus( pugi::xml_node const &node, Database *a_DB, Nuclide *a_parent );
        virtual ~Nucleus( );

        int Z( void ) const { return( m_Z ); }
        int A( void ) const { return( m_A ); }
        std::string const &levelName( ) const { return( m_levelName ); }
        int levelIndex( void ) const { return( m_levelIndex ); }
        std::string const &atomsID( void ) const ;

        double massValue( char const *a_unit ) const ;
        PQ_suite const &energy( void ) const { return( m_energy ); }
        double energy( std::string const &a_unit ) const ;
        virtual std::string toXMLListExtraAttributes( void ) const ;
        virtual void toXMLListExtraElements( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const ;
};

/*
============================================================
========================== Nuclide =========================
============================================================
*/
class Nuclide : public Particle {

    private:
        Isotope *m_isotope;
        Nucleus m_nucleus;

    public:
        Nuclide( pugi::xml_node const &a_node, Database *a_DB, Isotope *a_parent );
        virtual ~Nuclide( );

        int Z( void ) const;
        int A( void ) const;
        std::string const &levelName( void ) const { return( m_nucleus.levelName( ) ); }
        int levelIndex( void ) const { return( m_nucleus.levelIndex( ) ); }
        std::string const &atomsID( void ) const ;

        Isotope const *isotope( ) const { return( m_isotope ); }

        Nucleus const &nucleus( ) const { return( m_nucleus ); }

        PQ_suite const &baseMass( void ) const ;
        double massValue( char const *a_unit ) const ;
        double levelEnergy( std::string const &a_unit ) const { return( m_nucleus.energy( a_unit ) ); }

        void calculateNuclideGammaBranchStateInfos( PoPI::Database const &a_pops, NuclideGammaBranchStateInfos &a_nuclideGammaBranchStateInfos ) const ;
        virtual void toXMLListExtraElements( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const ;
};

/*
============================================================
========================= Isotope ==========================
============================================================
*/
class Isotope : public SymbolBase {

    private:
        ChemicalElement *m_chemicalElement;
        int m_Z;
        int m_A;
        Suite<Nuclide, Isotope> m_nuclides;

    public:
        Isotope( pugi::xml_node const &a_node, Database *a_DB, ChemicalElement *a_parent );
        virtual ~Isotope( );

        ChemicalElement const *chemicalElement( ) const { return( m_chemicalElement ); }
        int Z( void ) const { return( m_Z ); }
        int A( void ) const { return( m_A ); }
        Suite<Nuclide, Isotope> const &nuclides( ) const { return( m_nuclides ); }

        void calculateNuclideGammaBranchStateInfos( PoPI::Database const &a_pops, NuclideGammaBranchStateInfos &a_nuclideGammaBranchStateInfos ) const ;
        void toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const ;
};

/*
============================================================
===================== ChemicalElement ======================
============================================================
*/
class ChemicalElement : public SymbolBase {

    private:
        int m_Z;
        std::string m_name;
        Suite<Isotope, ChemicalElement> m_isotopes;

    public:
       ChemicalElement( pugi::xml_node const &a_node, Database *a_DB, Database *a_parent );
        virtual ~ChemicalElement( );

        int Z( void ) const { return( m_Z ); }
        std::string const &name( void ) const { return( m_name ); }

        Suite<Isotope, ChemicalElement> const &isotopes( ) const { return( m_isotopes ); }

        void calculateNuclideGammaBranchStateInfos( PoPI::Database const &a_pops, NuclideGammaBranchStateInfos &a_nuclideGammaBranchStateInfos ) const ;
        void toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const ;
};

/*
============================================================
=========================== Alias ==========================
============================================================
*/
class Alias : public IDBase {

    private:
        std::string m_pid;
        int m_pidIndex;

    public:
        Alias( pugi::xml_node const &a_node, Database *a_DB, Particle_class a_class = Particle_class::alias );
        virtual ~Alias( );

        std::string const &pid( void ) const { return( m_pid ); }
        int pidIndex( void ) const { return( m_pidIndex ); }
        void setPidIndex( int a_index ) { m_pidIndex = a_index; }
        void toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const ;
};

/*
============================================================
======================== MetaStable ========================
============================================================
*/
class MetaStable : public Alias {

    private:
        int m_metaStableIndex;

    public:
        MetaStable( pugi::xml_node const &a_node, Database *a_DB );
        virtual ~MetaStable( );

        int metaStableIndex( void ) const { return( m_metaStableIndex ); }
        void toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const ;
};

/*
============================================================
========================= Database =========================
============================================================
*/
class Database {

    private:
        std::string m_name;
        std::string m_version;
        std::string m_format;
        ParticleList m_list;
        std::map<std::string,int> m_map;               // Be careful with this as a map[key] will add key if it is not in the map.
        SymbolList m_symbolList;
        std::map<std::string,int> m_symbolMap;         // Be careful with this as a map[key] will add key if it is not in the map.

        std::vector<Alias *> m_unresolvedAliases;
        std::vector<Alias *> m_aliases;

        Suite<GaugeBoson, Database> m_gaugeBosons;
        Suite<Lepton, Database> m_leptons;
        Suite<Baryon, Database> m_baryons;
        Suite<ChemicalElement, Database> m_chemicalElements;
        Suite<Unorthodox, Database> m_unorthodoxes;

    public:
        Database( );
        Database( std::string const &a_fileName );
        Database( pugi::xml_node const &a_database );
        ~Database( );

        std::string const &name( void ) const { return( m_name ); }
        std::string const &version( void ) const { return( m_version ); }
        std::string const &format( void ) const { return( m_format ); }

        void addFile( char const *a_fileName, bool a_warnIfDuplicate );
        void addFile( std::string const &a_fileName, bool a_warnIfDuplicate );
        void addDatabase( std::string const &a_string, bool a_warnIfDuplicate );
        void addDatabase( pugi::xml_node const &a_database, bool a_warnIfDuplicate );
        void addAlias( Alias *a_alias ) { m_aliases.push_back( a_alias ); }

        std::string::size_type size( void ) const { return( m_list.size( ) ); }
        int operator[]( std::string const &a_id ) const ;
        template<typename T> T const &get( std::string const &a_id ) const ;
        template<typename T> T const &get( int a_index ) const ;
        Particle const &particle( std::string const &a_id ) const { return( get<Particle>( a_id ) ); }
        Particle const &particle( int &a_index ) const { return( get<Particle>( a_index ) ); }
        ParticleList const &particleList( ) const { return( m_list ); }
        SymbolList symbolList( ) const { return( m_symbolList ); }

        bool exists( std::string const &a_id ) const ;
        bool exists( int a_index ) const ;

        Suite<ChemicalElement, Database> const &chemicalElements( ) const { return( m_chemicalElements ); }

        bool isParticle( std::string const &a_id ) const { return( get<Base>( a_id ).isParticle( ) ); }
        bool isParticle( int a_index ) const { return( m_list[a_index]->isParticle( ) ); }
        bool isAlias( std::string const &a_id ) const { return( get<Base>( a_id ).isAlias( ) ); }
        bool isAlias( int a_index ) const { return( m_list[a_index]->isAlias( ) ); }
        bool isMetaStableAlias( std::string const &a_id ) const { return( get<Base>( a_id ).isMetaStableAlias( ) ); }
        bool isMetaStableAlias( int a_index ) const { return( m_list[a_index]->isMetaStableAlias( ) ); }

        std::string final( std::string const &a_id, bool returnAtMetaStableAlias = false ) const ;
        int final( int a_index, bool returnAtMetaStableAlias = false ) const ;

        int add( Base *a_item );
        int addSymbol( SymbolBase *a_item );

        void calculateNuclideGammaBranchStateInfos( NuclideGammaBranchStateInfos &a_nuclideGammaBranchStateInfos ) const ;

        void saveAs( std::string const &a_fileName ) const ;
        void toXMLList( std::vector<std::string> &a_XMLList, std::string const &a_indent1 ) const ;
        void print( void );
};

/*
=========================================================
*/
template<typename T> T const &Database::get( int a_index ) const {

    Base *particle = m_list[a_index];
    if( particle == NULL ) throw std::range_error( "particle not in database" );
    T const *object = dynamic_cast<T const *>( particle );
    if( object == NULL ) throw std::bad_cast( );

    return( *object );
}
/*
=========================================================
*/
template<typename T> T const &Database::get( std::string const &a_id ) const {

    int index = (*this)[a_id];
    Base *particle = m_list[index];
    T const *object = dynamic_cast<T const *>( particle );
    if( object == NULL ) throw std::bad_cast( );

    return( *object );
}

double getPhysicalQuantityAsDouble( PhysicalQuantity const &a_physicalQuantity );
double getPhysicalQuantityOfSuiteAsDouble( PQ_suite const &a_suite, bool a_allowEmpty = false, double a_emptyValue = 0.0 );

}

#endif      // End of PoPI_hpp_included

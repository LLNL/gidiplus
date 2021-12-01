/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#ifndef LUPI_hpp_included
#define LUPI_hpp_included 1

#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <string>
#include <vector>
#include <map>

#include <statusMessageReporting.h>

namespace LUPI {

#ifdef _WIN32
#define LUPI_FILE_SEPARATOR   "\\"
#else
#define LUPI_FILE_SEPARATOR   "/"
#endif

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
================== StatusMessageReporting ==================
============================================================
*/

class StatusMessageReporting {

    private:
        statusMessageReporting m_smr;

    public:
        StatusMessageReporting( );
        ~StatusMessageReporting( );
        
        statusMessageReporting *smr( ) { return( &m_smr ); }
        bool isOk( ) { return( smr_isOk( &m_smr ) ); }
        void clear( ) { smr_release( &m_smr ); }
        std::string constructMessage( std::string a_prefix, int a_reports = 1, bool a_clear = false );
};

/*
============================================================
====================== ArgumentParser ======================
============================================================
*/

enum class ArgumentType { Boolean, Count, Store, Append, Positional };

class ArgumentBase;

class ArgumentParser {

    private:
        std::string m_codeName;
        std::string m_descriptor;
        std::vector<ArgumentBase *> m_arguments;
        std::map<std::string,int> m_indices;

        void add2( ArgumentBase *a_argumentBase );

    public:
        ArgumentParser( std::string const &a_codeName, std::string const &a_descriptor = "" );
        ~ArgumentParser( );

        std::string const &codeName( ) const { return( m_codeName ); }
        std::string const &descriptor( ) const { return( m_descriptor ); }
        template<typename T> T *add( std::string const &a_name, std::string const &a_descriptor, int a_minimumNeeded = 1, int a_maximumNeeded = 1 );
        void addAlias( std::string const &a_name, std::string const &a_alias );
        void addAlias( ArgumentBase const * const a_argumentBase, std::string const &a_alias );
        bool hasName( std::string const &a_name ) const ;
        void parse( int a_argc, char **a_argv );
        template<typename T> T *get( std::size_t a_name );
        void help( ) const ;
        void usage( ) const ;
        virtual void printStatus( std::string a_indent ) const ;
};

/* *********************************************************************************************************//**
 * Creates a new argument, adds the argument to *this* and returns a pointer the the newly created argument.
 *
 * @param a_name                [in]    The name of the argument.
 * @param a_descriptor          [in]    The argument's description, displayed when the help option is enetered.
 * @param a_minimumNeeded       [in]    The minimum number of required time *this* argument must be entered.
 * @param a_maximumNeeded       [in]    The maximum number of required time *this* argument must be entered.
 *
 * @return                              A pointer to the created argument.
 ***********************************************************************************************************/

template<typename T> T *ArgumentParser::add( std::string const &a_name, std::string const &a_descriptor, int a_minimumNeeded, int a_maximumNeeded ) {

    T *argument = new T( a_name, a_descriptor, a_minimumNeeded, a_maximumNeeded );
    add2( argument );

    return( argument );
}

/*
============================================================
======================= ArgumentBase =======================
============================================================
*/

class ArgumentBase {

    private:
        ArgumentType m_argumentType;                        /**< The enum for arguent type of *this*. */
        std::vector<std::string> m_names;                   /**< The allowed names for *this*. */
        std::string m_descriptor;                           /**< The desciption printed help. */
        int m_minimumNeeded;                                /**< Minimum number of times *this* argument is required on the command line. */
        int m_maximumNeeded;                                /**< Maximum number of times *this* argument is required on the command line. */
        int m_numberEntered;                                /**< The number of time this argument was entered on the command line. */

        void addAlias( std::string const &a_name );
        virtual std::string printStatus2( ) const ;
        virtual void printStatus3( std::string const &a_indent ) const ;

        friend void ArgumentParser::addAlias( std::string const &a_name, std::string const &a_alias );

    public:
        ArgumentBase( ArgumentType a_argumentType, std::string const &a_name, std::string const &a_descriptor, int a_minimumNeeded, int a_maximumNeeded );
        virtual ~ArgumentBase( ) = 0 ;

        ArgumentType argumentType( ) const { return( m_argumentType ); }
        std::string const &name( ) const { return( m_names[0] ); }
        std::vector<std::string> const &names( ) { return( m_names ); }
        bool hasName( std::string const &a_name ) const ;
        std::string const &descriptor( ) const { return( m_descriptor ); }
        int minimumNeeded( ) const { return( m_minimumNeeded ); }
        int maximumNeeded( ) const { return( m_maximumNeeded ); }
        int numberEntered( ) const { return( m_numberEntered ); }

        virtual bool isOptionalArgument( ) const { return( true ); }
        virtual bool requiresAValue( ) const { return( false ); }
        virtual int parse( ArgumentParser const &a_argumentParser, int a_index, int a_argc, char **a_argv );
        std::string usage( bool a_requiredOption ) const ;
        void printStatus( std::string a_indent ) const ;
};

/*
============================================================
======================= OptionBoolean ======================
============================================================
*/

class OptionBoolean : public ArgumentBase {

    private:
        bool m_default;

    public:
        OptionBoolean( std::string const &a_name, std::string const &a_descriptor, bool a_default ) :
            ArgumentBase( ArgumentType::Boolean, a_name, a_descriptor, 0, -1 ),
            m_default( a_default ) {

        }
        virtual ~OptionBoolean( ) = 0 ;

        bool _default( ) const { return( m_default ); }
        bool value( ) const ;
        std::string printStatus2( ) const ;
};

/*
============================================================
======================== OptionTrue ========================
============================================================
*/

class OptionTrue : public OptionBoolean {

    public:
        OptionTrue( std::string const &a_name, std::string const &a_descriptor = "", int a_minimumNeeded = 0, int a_maximumNeeded = -1 ) :
            OptionBoolean( a_name, a_descriptor, false ) {
        }
        ~OptionTrue( ) {}
};

/*
============================================================
======================= OptionFalse ========================
============================================================
*/

class OptionFalse : public OptionBoolean {

    public:
        OptionFalse( std::string const &a_name, std::string const &a_descriptor = "", int a_minimumNeeded = 0, int a_maximumNeeded = -1 ) :
                OptionBoolean( a_name, a_descriptor, true ) {
        }
        ~OptionFalse( ) {}
};

/*
============================================================
====================== OptionCounter =======================
============================================================
*/

class OptionCounter : public ArgumentBase {

    public:
        OptionCounter( std::string const &a_name, std::string const &a_descriptor = "", int a_minimumNeeded = 0, int a_maximumNeeded = -1 ) :
                ArgumentBase( ArgumentType::Count, a_name, a_descriptor, 0, -1 ) {

        }
        ~OptionCounter( ) {}

        int counts( ) const { return( numberEntered( ) ); }
        std::string printStatus2( ) const ;
};

/*
============================================================
======================= OptionStore ========================
============================================================
*/

class OptionStore : public ArgumentBase {

    private:
        std::string m_value;

    public:
        OptionStore( std::string const &a_name, std::string const &a_descriptor = "", int a_minimumNeeded = 0, int a_maximumNeeded = -1 ) :
                ArgumentBase( ArgumentType::Store, a_name, a_descriptor, 0, -1 ),
                m_value( "" ) {

        }
        ~OptionStore( ) {}

        virtual bool requiresAValue( ) const { return( true ); }
        int parse( ArgumentParser const &a_argumentParser, int a_index, int a_argc, char **a_argv );

        std::string const &value( ) const { return( m_value ); }
        void printStatus3( std::string const &a_indent ) const ;
};

/*
============================================================
======================= OptionAppend =======================
============================================================
*/

class OptionAppend : public ArgumentBase {

    private:
        std::vector<std::string> m_values;

    public:
        OptionAppend( std::string const &a_name, std::string const &a_descriptor = "", int a_minimumNeeded = 0, int a_maximumNeeded = -1 ) :
                ArgumentBase( ArgumentType::Append, a_name, a_descriptor, a_minimumNeeded, a_maximumNeeded ) {

        }
        ~OptionAppend( ) {}

        virtual bool requiresAValue( ) const { return( true ); }
        int parse( ArgumentParser const &a_argumentParser, int a_index, int a_argc, char **a_argv );

        std::vector<std::string> const &values( ) const { return( m_values ); }
        std::string const &value( int a_index ) const { return( m_values[a_index] ); }
        void addValue( std::string a_value ) { m_values.push_back( a_value ); }
        void printStatus3( std::string const &a_indent ) const ;
};

/*
============================================================
======================== Positional ========================
============================================================
*/

class Positional : public ArgumentBase {

    private:
        std::vector<std::string> m_values;

    public:
        Positional( std::string const &a_name, std::string const &a_descriptor = "", int a_minimumNeeded = 1, int a_maximumNeeded = 1 ) :
            ArgumentBase( ArgumentType::Positional, a_name, a_descriptor, a_minimumNeeded, a_maximumNeeded ) {

        }
        ~Positional( ) { }

        bool isOptionalArgument( ) const { return( false ); }
        virtual bool requiresAValue( ) const { return( true ); }
        int parse( ArgumentParser const &a_argumentParser, int a_index, int a_argc, char **a_argv );

        std::vector<std::string> const &values( ) { return( m_values ); }
        std::string const &value( int a_index ) const { return( m_values[a_index] ); }
        void addValue( std::string a_value ) { m_values.push_back( a_value ); }
        void printStatus3( std::string const &a_indent ) const ;
};

/*
============================================================
======================== DeltaTime =========================
============================================================
*/

#define LUPI_DeltaTime_toStringFormatIncremental "incremental: CPU %8.3fs, wall %8.3fs"
#define LUPI_DeltaTime_toStringFormatTotal "total: CPU %8.3fs, wall %8.3fs"

class DeltaTime {

    private:
        double m_CPU_time;
        double m_wallTime;
        double m_CPU_timeIncremental;
        double m_wallTimeIncremental;

    public:
        DeltaTime( );
        DeltaTime( double a_CPU_time, double a_wallTime, double a_CPU_timeIncremental, double a_wallTimeIncremental );
        DeltaTime( DeltaTime const &deltaTime );
        ~DeltaTime( ) {}

        double CPU_time( ) const { return( m_CPU_time ); }
        double wallTime( ) const { return( m_wallTime ); }
        double CPU_timeIncremental( ) const { return( m_CPU_timeIncremental ); }
        double wallTimeIncremental( ) const { return( m_wallTimeIncremental ); }
        std::string toString( std::string a_formatIncremental = LUPI_DeltaTime_toStringFormatIncremental,
                std::string a_format = LUPI_DeltaTime_toStringFormatTotal, std::string a_sep = "; " );
};

/*
============================================================
========================== Timer ===========================
============================================================
*/

class Timer {

    private:
        clock_t m_CPU_time;
        struct timeval m_wallTime;
        clock_t m_CPU_timeIncremental;
        struct timeval m_wallTimeIncremental;

    public:
        Timer( );
        ~Timer( ) {}

        DeltaTime deltaTime( );
        DeltaTime deltaTimeAndReset( );
        void reset( );
};

namespace FileInfo {        // Should be using std::filesystem stuff but this requires C++ 17.

std::string _basename( std::string const &a_path );
std::string basenameWithoutExtension( std::string const &a_path );
std::string _dirname( std::string const &a_path );
bool exists( std::string const &a_path );
bool isDirectory( std::string const &a_path );
bool createDirectories( std::string const &a_path );

/*
============================================================
========================= FileStat =========================
============================================================
*/
class FileStat {

    private:
        std::string m_path;             /**< The path that is stat-ed. */
        struct stat m_stat;             /**< The stat for the path. */

    public:
        FileStat( std::string const &a_path );

        std::string const &path( ) const { return( m_path ); }                              /**< Returns a reference to the **m_path** member. */
        struct stat const &statRef( ) const { return( m_stat ); }                           /**< Returns a reference to the **m_stat** member. */

        bool exists( );
        bool isDirectory( ) const { return( ( m_stat.st_mode & S_IFMT ) == S_IFDIR ); }           /**< Returns *true* if the path is a directory and *false* otherwise. */
        bool isRegularFile( ) const { return( ( m_stat.st_mode & S_IFMT ) == S_IFREG ); }   /**< Returns *true* if the path is a regular file and *false* otherwise. */
};

}               // End of namespace FileInfo.

// Miscellaneous functions

namespace Misc {

std::string argumentsToString( char const *a_format, ... );
std::string doubleToString3( char const *a_format, double a_value, bool a_reduceBits = false );
void printCommand( std::string const &a_indent, int a_argc, char **a_argv );

}               // End of namespace Misc.

}               // End of namespace LUPI.

#endif          // LUPI_hpp_included

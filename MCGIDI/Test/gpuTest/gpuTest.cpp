/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <iostream>

#ifdef __CUDACC__

#include <stdio.h>
#include <cuda.h>  
#include <math.h>
#include "MCGIDI.hpp"
#include <sys/time.h>

#define numberOfTallies 10
#define numberOfTalliesMinus1 ( numberOfTallies - 1 )
#define numberOfTalliesMinus2 ( numberOfTallies - 2 )

int main2( int argc, char *argv[] );
MCGIDI_HOST_DEVICE double myRNG( uint64_t *state );

/*
=========================================================
*/
class TallyProductHandler : public MCGIDI::Sampling::ProductHandler {

    public:
        int *m_tally;

        MCGIDI_HOST_DEVICE TallyProductHandler( ) : m_tally( nullptr ) {}
        MCGIDI_HOST_DEVICE ~TallyProductHandler( ) {}

        MCGIDI_HOST_DEVICE std::size_t size( ) { return 0; }
        MCGIDI_HOST_DEVICE void clear( ) {}

        MCGIDI_HOST_DEVICE void setTally( int *a_tally ) { m_tally = a_tally; }

        MCGIDI_HOST_DEVICE void push_back( MCGIDI::Sampling::Product &a_product ) {

            int index = static_cast<int>( log10( a_product.m_kineticEnergy ) ) + numberOfTalliesMinus2;
            if( index < 0 ) index = 0;
            if( index > numberOfTalliesMinus1 ) index = numberOfTalliesMinus1;
            #ifdef __CUDA_ARCH__
            atomicAdd( &m_tally[index], 1 );
            #else
            m_tally[index]++;
            #endif
        }
};

/*
=========================================================
*/
__global__ void sample( MCGIDI::ProtareSingle *a_MCProtare, int a_numCollisions, int *a_tally ) {

    MCGIDI::DomainHash domainHash( 4000, 1e-8, 10 );
    double temperature = 2.58522e-8;
    TallyProductHandler products;

    products.setTally( a_tally );

    int collisionIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if( collisionIndex >= a_numCollisions ) return;

    uint64_t seed = collisionIndex + 1;
    double energy = pow( 10.0, myRNG( &seed ) * 1.3 );

    MCGIDI::Sampling::Input input( true, MCGIDI::Sampling::Upscatter::Model::B );

    int hashIndex = domainHash.index( energy );

    MCGIDI::URR_protareInfos urr;
// The next 4 lines cause a "nvlink warning".
    double crossSection = a_MCProtare->crossSection( urr, hashIndex, temperature, energy );
    int reactionIndex = a_MCProtare->sampleReaction( urr, hashIndex, temperature, energy, crossSection, (double (*)(void *)) myRNG, &seed );

    MCGIDI::Reaction const *reaction = a_MCProtare->reaction( reactionIndex );
    reaction->sampleProducts( a_MCProtare, energy, input, (double (*)( void * )) myRNG, &seed, products );
}

/*
=========================================================
*/
__global__ void setUp( int a_numIsotopes, MCGIDI::DataBuffer **a_buf ) {  // Call this each isotope per block and one warp only (i.e. <<< number_isotopes, 32>>>)

    int isotopeIndex = blockIdx.x;

    MCGIDI::DataBuffer *buf = a_buf[isotopeIndex];
    MCGIDI::ProtareSingle *MCProtare = new(buf->m_placementStart) MCGIDI::ProtareSingle( );

    buf->zeroIndexes( );
    buf->m_placement = buf->m_placementStart + sizeof( MCGIDI::ProtareSingle );
    buf->m_maxPlacementSize = sizeof( *a_buf[isotopeIndex] ) + sizeof( MCGIDI::ProtareSingle );

    MCProtare->serialize( *buf, MCGIDI::DataBuffer::Mode::Unpack );                 // This line causes a "nvlink warning".
    buf->m_placement = buf->m_placementStart + sizeof( MCGIDI::ProtareSingle );
}

/*
=========================================================
*/
__global__ void printData( MCGIDI::ProtareSingle *MCProtare ) {       // Called on blocks only.

    int numberOfReactions = MCProtare->reactions( ).size( );

    MCGIDI::Sampling::Input input( true, MCGIDI::Sampling::Upscatter::Model::B );
    MCGIDI::Sampling::MCGIDIVectorProductHandler products;

    for( int i1 = 0; i1 < numberOfReactions; ++i1 ) {
        MCGIDI::Reaction const *reaction = MCProtare->reaction( i1 );               // This line causes a "nvlink warning".
        double threshold = MCProtare->threshold( i1 );
        printf( "D: reaction(%d) = %s threshold = %g\n", i1, reaction->label( ).c_str( ), threshold );
    }
}

/*
=========================================================
*/
int main( int argc, char **argv ) {

    int status;

    std::cerr << "    gpuTest - on CUDA" << std::endl;

    try {
        status = main2( argc, argv ); }
    catch (std::exception &exception) {
        std::cerr << exception.what( ) << std::endl;
        exit( EXIT_FAILURE ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE ); }
    catch (std::string &str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    exit( status );
}
 
/*
=========================================================
*/
int main2( int argc, char *argv[] ) {                                   // main routine that executes on the host

    const char *isotopeNames[] = {
            "U235",  "H1",    "U238",  "He4",   "Pu239", "C12",   "Pb209", "Hg200", "W185", "Gd156",
            "Sm148", "Nd145", "Cs135", "Xe128", "As73",  "Zn69",  "Br80",  "Fe56",  "Cr51", "Sc46",
            "Ar40",  "Al27",  "O16",   "Li6",   "Li7",   "Be9",   "H3",    "He3",   "Na23", "Mg25",
            "Am242", "Cf250", "Np238", "Er166", "Pm147", "Ce142", "Ba133", "Xe136", "I126", "Cd106",
            "Mo95",  "Zr90",  "Kr80",  "Sr84",  "Ni62",  "Co58",  "V51",   "Ca44",  "Ti45", "Bk248",
            "U235",  "H1",    "U238",  "He4",   "Pu239", "C12",   "Pb209", "Hg200", "W185", "Gd156",
            "Sm148", "Nd145", "Cs135", "Xe128", "As73",  "Zn69",  "Br80",  "Fe56",  "Cr51", "Sc46",
            "Ar40",  "Al27",  "O16",   "Li6",   "Li7",   "Be9",   "H3",    "He3",   "Na23", "Mg25",
            "Am242", "Cf250", "Np238", "Er166", "Pm147", "Ce142", "Ba133", "Xe136", "I126", "Cd106",
            "Mo95",  "Zr90",  "Kr80",  "Sr84",  "Ni62",  "Co58",  "V51",   "Ca44",  "Ti45", "Bk248" };

    int numberOfIsotopes = sizeof( isotopeNames ) / sizeof( isotopeNames[0] );

    size_t my_size;
    cudaDeviceSetLimit( cudaLimitStackSize, 80 * 1024 );
    cudaDeviceGetLimit( &my_size, cudaLimitStackSize ) ;
    printf( "cudaLimitStackSize =  %luk\n", my_size / 1024 );
    cudaDeviceSetLimit( cudaLimitMallocHeapSize, 100 * 1024 * 1024 );
    cudaDeviceGetLimit( &my_size, cudaLimitMallocHeapSize ) ;
    printf( "cudaLimitMallocHeapSize =  %luM\n", my_size / ( 1024 * 1024 ) );
    cudaDeviceSetLimit( cudaLimitPrintfFifoSize, 40 * 1024 * 1024 );
    cudaDeviceGetLimit( &my_size, cudaLimitPrintfFifoSize );
    printf( "cudaLimitPrintfFifoSize =  %luM\n", my_size / ( 1024 * 1024 ) );

    int doPrint = 0;                    // doPrint == 0 means do not print out results from unpacked data
    int numCollisions = 100 * 1000;     // Number of sample reactions
    int numIsotopes = 1;                // Number of isotopes
    int doCompare = 0;                  // Compare the bytes of gidi data. 0 - no comparison, 1 - no compare, write out data, 2 - Read in data and compare

    if( argc > 1 ) doPrint = atoi( argv[1] );
    if( argc > 2 ) numCollisions = atol( argv[2] );
    if( argc > 3 ) numIsotopes = atoi( argv[3] );
    if( numIsotopes > numberOfIsotopes ) numIsotopes = numberOfIsotopes;
    if( argc > 4 ) doCompare = atoi( argv[4] );

    printf( "doPrint = %d, numCollisions = %g, numIsotopes = %d, doCompare = %d\n", doPrint, static_cast<double>( numCollisions ), numIsotopes, doCompare );

    std::vector<MCGIDI::Protare *>protares( numIsotopes );
    std::string mapFilename( "/usr/gapps/Mercury/data/nuclear/endl/2009.3_gp3.17/gnd/all.map" );
    PoPI::Database pops( "/usr/gapps/Mercury/data/nuclear/endl/2009.3/gnd/pops.xml" );

    std::ifstream meta_stream( "/usr/gapps/data/nuclear/development/GIDI3/Versions/V10/metastables_alias.xml" );
    std::string metastable_string( ( std::istreambuf_iterator<char>( meta_stream ) ), std::istreambuf_iterator<char>( ) );
    pops.addDatabase( metastable_string, false );

    GIDI::Map::Map map( mapFilename, pops );
    MCGIDI::DomainHash domainHash( 4000, 1e-8, 10 );
    for( int isoIndex = 0; isoIndex < numIsotopes; isoIndex++ ) {
        std::string protareFilename( map.protareFilename( PoPI::IDs::neutron, isotopeNames[isoIndex] ) );
        GIDI::Protare *protare;

        GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::excludeProductMatrices, GIDI::Construction::PhotoMode::nuclearAndAtomic );
        protare = map.protare( construction, pops, PoPI::IDs::neutron, isotopeNames[isoIndex] );

        GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
        std::string label( temperatures[0].griddedCrossSection( ) );
        MCGIDI::Transporting::MC MC( pops, PoPI::IDs::neutron, &protare->styles( ), label, GIDI::Transporting::DelayedNeutrons::on, 20.0 );
        GIDI::Transporting::Particles particleList;
        GIDI::Transporting::MultiGroup continuous_energy_multigroup;
        GIDI::Transporting::Particle projectile( "n", continuous_energy_multigroup );
        particleList.add( projectile );
        std::set<int> exclusionSet;

        protares[isoIndex] = MCGIDI::protareFromGIDIProtare( *protare, pops, MC, particleList, domainHash, temperatures, exclusionSet );
    }

    MCGIDI::Protare *MCProtare = protares[numIsotopes-1];
    int numberOfReactions = MCProtare->numberOfReactions( );

    MCGIDI::Sampling::Input input( true, MCGIDI::Sampling::Upscatter::Model::B );
    MCGIDI::Sampling::MCGIDIVectorProductHandler products;
    printf( "CPU OUTPUT\n" );
    if( doPrint ) {
        for( int i1 = 0; i1 < numberOfReactions; ++i1 ) {
            MCGIDI::Reaction const *reaction = MCProtare->reaction( i1 );
            double threshold = MCProtare->threshold( i1 );
    
            printf( "HO: reaction(%d) = %s threshold = %g\n", i1, reaction->label( ).c_str( ), threshold );
        }
    }

    std::vector<MCGIDI::DataBuffer *>deviceBuffers_h( numIsotopes );
    std::vector<char *>deviceProtares( numIsotopes );
    for( int isoIndex = 0; isoIndex < numIsotopes; isoIndex++ ) {
        MCGIDI::DataBuffer buf_h;

        protares[isoIndex]->serialize( buf_h, MCGIDI::DataBuffer::Mode::Count );

        buf_h.allocateBuffers( );
        buf_h.zeroIndexes( );
        protares[isoIndex]->serialize( buf_h, MCGIDI::DataBuffer::Mode::Pack );

        size_t cpuSize = protares[isoIndex]->memorySize( );
        deviceBuffers_h[isoIndex] = buf_h.copyToDevice( cpuSize, deviceProtares[isoIndex] );
    }

    MCGIDI::DataBuffer **deviceBuffers_d = nullptr;
    cudaMalloc( (void **) &deviceBuffers_d, sizeof( MCGIDI::DataBuffer * ) * numIsotopes );
    cudaMemcpy( deviceBuffers_d, &deviceBuffers_h[0], sizeof( MCGIDI::DataBuffer * ) * numIsotopes, cudaMemcpyHostToDevice );

    setUp<<< numIsotopes, 32 >>>( numIsotopes, deviceBuffers_d );

    gpuErrchk( cudaPeekAtLastError( ) );
    gpuErrchk( cudaDeviceSynchronize( ) );

    if( doPrint ) {
        printData<<<1, 1>>>( reinterpret_cast<MCGIDI::ProtareSingle *>( deviceProtares[numIsotopes-1] ) );
        gpuErrchk( cudaPeekAtLastError( ) );
        gpuErrchk( cudaDeviceSynchronize( ) );
    }

    if( doCompare > 0 ) {
        int isoIndex = numIsotopes-1;
        size_t cpuSize = protares[isoIndex]->memorySize( );
        char *gidiBytes = (char *) malloc( cpuSize );
        cudaMemcpy( gidiBytes, deviceProtares[isoIndex], cpuSize, cudaMemcpyDeviceToHost );
        if( doCompare == 1 ) {
            FILE *outFile = fopen( "gidi_data.bin", "wb" );
            fwrite( gidiBytes, sizeof( char ), cpuSize, outFile );
            fclose( outFile ); }
        else {
            char *fileBytes = (char *) malloc( cpuSize );
            FILE *inFile = fopen( "gidi_data.bin", "rb" );
            fread( fileBytes, sizeof( char ), cpuSize, inFile );
            fclose( inFile );
            int errorCount = 0;
            int firstError = -1;
            for( int indx = 0; indx < cpuSize; indx++ ) {
                if( fileBytes[indx] == gidiBytes[indx] ) continue;
                if( firstError == -1 ) firstError = indx;
                errorCount++;
            }
            printf( "Out of %zu bytes, there were %d errors with the first one at %d\n", cpuSize, errorCount, firstError );
        }
    }

    int CPU_tally[numberOfTallies];
    if( numCollisions > 0 ) {
        timeval tv1, tv2;
        gettimeofday( &tv1, nullptr );
        MCGIDI::Protare &gidi_data = *( protares[numIsotopes-1] );
        double temperature = 2.58522e-8;
        TallyProductHandler products;
        products.setTally( CPU_tally );
        for( int i = 0; i < numberOfTallies; i++ ) CPU_tally[i] = 0;

        for( int collisionIndex = 0; collisionIndex < numCollisions; ++collisionIndex ) {
            uint64_t seed = collisionIndex + 1;
            double energy = pow( 10.0, myRNG( &seed ) * 1.3 );
            MCGIDI::Sampling::Input input( true, MCGIDI::Sampling::Upscatter::Model::B );
            int hashIndex = domainHash.index( energy );
            MCGIDI::URR_protareInfos urr;
            double crossSection = gidi_data.crossSection( urr, hashIndex, temperature, energy );
            int reactionIndex = MCProtare->sampleReaction( urr, hashIndex, temperature, energy, crossSection, (double (*)(void *)) myRNG, &seed );

            MCGIDI::Reaction const *reaction = gidi_data.reaction( reactionIndex );
            reaction->sampleProducts( &gidi_data, energy, input, (double (*)(void *)) myRNG, &seed, products );
        }
        gettimeofday( &tv2, nullptr );
        printf( "Host tally:       " );
        for( int i = 0; i < numberOfTallies; i++ ) printf( "%8d ", CPU_tally[i] );
        printf( "\nHost   Total time = %f seconds\n", (double) ( tv2.tv_usec - tv1.tv_usec ) / 1000000 + (double) ( tv2.tv_sec - tv1.tv_sec ) );
    }

    if( doPrint ) {
        int isoIndex = numIsotopes-1;
        MCGIDI::DataBuffer buf2_h;
        protares[isoIndex]->serialize( buf2_h, MCGIDI::DataBuffer::Mode::Count );
        buf2_h.allocateBuffers( );
        buf2_h.zeroIndexes( );
        size_t cpuSize = protares[isoIndex]->memorySize( );
        protares[isoIndex]->serialize( buf2_h, MCGIDI::DataBuffer::Mode::Pack );
        MCGIDI::ProtareSingle *MCProtare_h = nullptr;
        char *bigBuffer = (char *) malloc( cpuSize );
        MCProtare_h = new(bigBuffer) MCGIDI::ProtareSingle( );
        buf2_h.zeroIndexes( );
        buf2_h.m_placementStart = bigBuffer;
        buf2_h.m_placement = buf2_h.m_placementStart + sizeof( MCGIDI::ProtareSingle );
        buf2_h.m_maxPlacementSize = cpuSize;
        MCProtare_h->serialize( buf2_h, MCGIDI::DataBuffer::Mode::Unpack );
        buf2_h.m_placement = buf2_h.m_placementStart + sizeof( MCGIDI::ProtareSingle );
        if( !buf2_h.validate( ) ) printf( "Data went over memory pool size.\n" );

        MCGIDI::ProtareSingle &gidi_data = *MCProtare_h;
        for( int i1 = 0; i1 < numberOfReactions; ++i1 ) {
            MCGIDI::Reaction const *reaction = gidi_data.reaction( i1 );
            double threshold = gidi_data.threshold( i1 );

            printf( "HN: reaction(%d) = %s threshold = %g\n", i1, reaction->label( ).c_str( ), threshold );
        }

        buf2_h.nullOutPointers( );
    }

    int tallyDiff = 0;
    if( numCollisions > 0 ) {
        timeval tv1, tv2;
        gettimeofday( &tv1, nullptr  );
        int *tally_d, GPU_tally[numberOfTallies];
        cudaMalloc( (void **) &tally_d, numberOfTallies * sizeof( tally_d[0] ) ); 
        cudaMemset( tally_d, 0, numberOfTallies * sizeof( tally_d[0] ) ); 

        MCGIDI::ProtareSingle *MCProtare_d = reinterpret_cast<MCGIDI::ProtareSingle *>( deviceProtares[numIsotopes-1] );
        sample<<<(numCollisions+255)/256, 256>>>( MCProtare_d, numCollisions, tally_d );

        cudaDeviceSynchronize( );
        gettimeofday( &tv2, nullptr );

        cudaMemcpy( GPU_tally, tally_d, numberOfTallies * sizeof( tally_d[0] ), cudaMemcpyDeviceToHost );
        printf( "Device tally:     " );
        for( int i = 0; i < numberOfTallies; i++ ) printf( "%8d ", GPU_tally[i] );
        printf( "\nDevice Total time = %f seconds\n", (double) ( tv2.tv_usec - tv1.tv_usec ) / 1000000 + (double) ( tv2.tv_sec - tv1.tv_sec ) );
        cudaFree( tally_d );

        printf( "Tally difference: " );
        for( int i = 0; i < numberOfTallies; i++ ) {
            printf( "%8d ", CPU_tally[i] - GPU_tally[i] );
            tallyDiff += abs( CPU_tally[i] - GPU_tally[i] );
        }
        printf( "\n" );
        if( tallyDiff != 0 ) printf( "ERROR: tallyDiff = %d != 0\n", tallyDiff );
    }

    return( ( tallyDiff == 0 ) ? EXIT_SUCCESS : EXIT_FAILURE );
}

/*
=========================================================
*/
MCGIDI_HOST_DEVICE double myRNG( uint64_t *seed ) {

   *seed = 2862933555777941757ULL * ( *seed ) + 3037000493ULL;      // Update state from the previous value.
   
   return 5.42101086242752157e-20 * ( *seed );                      // Map state from [0,2**64) to double [0,1).
}

#else

/*
=========================================================
*/
int main( int argc, char **argv ) {

    std::cerr << "    gpuTest - no opt" << std::endl;

    return( EXIT_SUCCESS );
}

#endif

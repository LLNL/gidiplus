/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include <stdio.h>
#include <cuda.h>  
#include <math.h>
#include "MCGIDI.hpp"
#include <sys/time.h>

HOST_DEVICE double myRNG( uint64_t *state);

class TallyProductHandler : public MCGIDI::Sampling::ProductHandler {

    public:
        int *m_tally;

        HOST_DEVICE TallyProductHandler( ) : m_tally( nullptr ) { }
        HOST_DEVICE ~TallyProductHandler( ) {}

        HOST_DEVICE std::size_t size( ) { return 0; }
        HOST_DEVICE void clear( ) {  }

        HOST_DEVICE void setTally(int *a_tally) { m_tally = a_tally; }

        HOST_DEVICE void push_back( MCGIDI::Sampling::Product &a_product ) {
            int index = static_cast<int>(log10(a_product.m_kineticEnergy)) + 8;
            if (index < 0) index = 0;
            if (index > 9) index = 9;
            #ifdef __CUDA_ARCH__
            atomicAdd(&m_tally[index], 1);
            #else
            m_tally[index]++;
            #endif
        }
};


__global__ void sample(MCGIDI::ProtareSingle *MCProtare_d, int numCollisions, int *tally)
{
    MCGIDI::ProtareSingle &gidi_data = *MCProtare_d;
    MCGIDI::DomainHash domainHash( 4000, 1e-8, 10 );

    double temperature = 2.58522e-8;

    TallyProductHandler products;

    products.setTally(tally);

    int collisionIndex = blockIdx.x*blockDim.x + threadIdx.x;
    if (collisionIndex >= numCollisions) return;

    uint64_t seed = collisionIndex+1;
    double energy = pow(10.0, myRNG(&seed) * 1.3);

    MCGIDI::Sampling::Input input( true, MCGIDI::Sampling::Upscatter::Model::B );

    int hashIndex = domainHash.index( energy );

    MCGIDI::URR_protareInfos urr;
    double crossSection = gidi_data.crossSection( urr, hashIndex, temperature, energy );
    int reactionIndex = MCProtare_d->sampleReaction( urr, hashIndex, temperature, energy, crossSection, (double (*)(void *))myRNG, &seed );

    MCGIDI::Reaction const *reaction = gidi_data.reaction(reactionIndex);
    reaction->sampleProducts( &gidi_data, energy, input, myRNG, &seed, products );
}


// Call this each isotope per block and one warp only (i.e. <<< number_isotopes, 32>>>)
__global__ void setUp(int numIsotopes, MCGIDI::DataBuffer **p_buf)
{
    int isotopeIndex = blockIdx.x;

    MCGIDI::DataBuffer *buf = p_buf[isotopeIndex];
    MCGIDI::ProtareSingle *MCProtare_d = new(buf->m_placementStart) MCGIDI::ProtareSingle();

    buf->zeroIndexes();
    buf->m_placement = buf->m_placementStart + sizeof(MCGIDI::ProtareSingle);
    buf->m_maxPlacementSize = sizeof(*p_buf[isotopeIndex]) + sizeof(MCGIDI::ProtareSingle);

    MCProtare_d->serialize(*buf, MCGIDI::DataBuffer::Unpack);
    buf->m_placement = buf->m_placementStart + sizeof(MCGIDI::ProtareSingle);
}

// Called on blocks only.
__global__ void printData(MCGIDI::ProtareSingle *MCProtare_d)
{
    MCGIDI::ProtareSingle &gidi_data = *MCProtare_d;

    uint64_t seed = 1;
    int numberOfReactions = gidi_data.reactions().size();

    MCGIDI::Sampling::Input input( true, MCGIDI::Sampling::Upscatter::Model::B );
    MCGIDI::Sampling::MCGIDIVectorProductHandler products;

    for( int i1 = 0; i1 < numberOfReactions; ++i1 ) 
    {
        MCGIDI::Reaction const *reaction = gidi_data.reaction(i1);
        double threshold = gidi_data.threshold( i1 );
        printf( "D: reaction(%d) = %s threshold = %g\n", i1, reaction->label( ).c_str(), threshold);
    }
}

 
// main routine that executes on the host
int main(int argc, char *argv[])
{
    size_t my_size;
    cudaDeviceSetLimit(cudaLimitStackSize, 80*1024);
    cudaDeviceGetLimit( &my_size, cudaLimitStackSize ) ;
    printf("cudaLimitStackSize =  %dk\n", my_size/1024);
    cudaDeviceSetLimit (cudaLimitMallocHeapSize, 100*1024*1024 );
    cudaDeviceGetLimit( &my_size, cudaLimitMallocHeapSize ) ;
    printf("cudaLimitMallocHeapSize =  %dM\n", my_size / (1024*1024));
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 40 *1024 *1024);
    cudaDeviceGetLimit( &my_size, cudaLimitPrintfFifoSize ) ;
    printf("cudaLimitPrintfFifoSize =  %dM\n", my_size / (1024*1024));

    // doPrint == 0 means do not print out results from unpacked data
    int doPrint = 1;
    // Number of sample reactions
    int numCollisions = 0;
    // Number of isotopes
    int numIsotopes = 1;
    // doCompare to compare the bytes of gidi data. 0 - no comparson, 1 - No compare, write out data,
    //        2 - Read in data and compare
    int doCompare = 0;
    if (argc > 1) doPrint = atoi(argv[1]);
    if (argc > 2) numCollisions = atof(argv[2]);
    if (argc > 3) numIsotopes = atoi(argv[3]);
    if (numIsotopes > 100) numIsotopes = 100;
    if (argc > 4) doCompare = atoi(argv[4]);

    printf("doPrint = %d, numCollisions = %g, numIsotopes = %d, doCompare = %d\n", doPrint, static_cast<double>(numCollisions), numIsotopes, doCompare);

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
            "Mo95",  "Zr90",  "Kr80",  "Sr84",  "Ni62",  "Co58",  "V51",   "Ca44",  "Ti45", "Bk248"};

    std::vector<MCGIDI::ProtareSingle *>protares(numIsotopes);
    std::string mapFilename( "/usr/gapps/Mercury/data/nuclear/endl/2009.3_gp3.17/gnd/all.map" );
    PoPI::Database pops( "/usr/gapps/Mercury/data/nuclear/endl/2009.3/gnd/pops.xml" );

    std::ifstream meta_stream("/usr/gapps/data/nuclear/development/GIDI3/Versions/V10/metastables_alias.xml");
    std::string metastable_string((std::istreambuf_iterator<char>(meta_stream)), std::istreambuf_iterator<char>());
    pops.addDatabase(metastable_string, false);

    GIDI::Map::Map map( mapFilename, pops );
    int neutronIndex = pops[PoPI::IDs::neutron];
    MCGIDI::DomainHash domainHash( 4000, 1e-8, 10 );
    for (int isoIndex = 0; isoIndex < numIsotopes; isoIndex++)
    {
        std::string protareFilename( map.protareFilename( PoPI::IDs::neutron, isotopeNames[isoIndex] ) );
        GIDI::ProtareSingle *protare;
        std::vector<std::string> EmptyStringVec;

        try {
            GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, GIDI::Construction::PhotoMode::nuclearAndAtomic );
            protare = new GIDI::ProtareSingle( construction, protareFilename, GIDI::XML, pops, EmptyStringVec); }
        catch (char const *str) {
            std::cout << str << std::endl;
            exit( EXIT_FAILURE );
        }

        GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
        std::string label( temperatures[0].griddedCrossSection( ) );
        MCGIDI::Transporting::MC MC( pops, PoPI::IDs::neutron, &protare->styles( ), label, GIDI::Transporting::DelayedNeutrons::on, 20.0 );
        GIDI::Transporting::Particles particleList;
        GIDI::Transporting::MultiGroup continuous_energy_multigroup;
        GIDI::Transporting::Particle projectile("n", continuous_energy_multigroup);
        particleList.add( projectile );
        std::set<int> exclusionSet;

        try {
            protares[isoIndex] = new MCGIDI::ProtareSingle( *protare, pops, MC, particleList, domainHash, temperatures, exclusionSet ); }

        catch (char const *str) {
            std::cout << str << std::endl;
            exit( EXIT_FAILURE );
        }
    }


    MCGIDI::ProtareSingle *MCProtare = protares[numIsotopes-1];
    int numberOfReactions = MCProtare->reactions( ).size( );

    uint64_t seed = 1;
    MCGIDI::Sampling::Input input( true, MCGIDI::Sampling::Upscatter::Model::B );
    MCGIDI::Sampling::MCGIDIVectorProductHandler products;
    printf("CPU OUTPUT\n");
    if (doPrint) 
    {
        for( int i1 = 0; i1 < numberOfReactions; ++i1 ) 
        {
            MCGIDI::Reaction const *reaction = MCProtare->reaction(i1);
            double threshold = MCProtare->threshold( i1 );
    
            printf("HO: reaction(%d) = %s threshold = %g\n", i1, reaction->label( ).c_str(), threshold);
        }
    }

    std::vector<MCGIDI::DataBuffer *>deviceBuffers_h(numIsotopes);
    std::vector<char *>deviceProtares(numIsotopes);
    for (int isoIndex = 0; isoIndex < numIsotopes; isoIndex++)
    {
        MCGIDI::DataBuffer buf_h;
        protares[isoIndex]->serialize(buf_h, MCGIDI::DataBuffer::Count);
        buf_h.allocateBuffers();
        buf_h.zeroIndexes();

        protares[isoIndex]->serialize(buf_h, MCGIDI::DataBuffer::Pack);

        std::size_t cpuSize = sizeof(*protares[isoIndex]) + protares[isoIndex]->internalSize();
        deviceBuffers_h[isoIndex] = buf_h.copyToDevice(cpuSize, deviceProtares[isoIndex]);
    }

    MCGIDI::DataBuffer **deviceBuffers_d = nullptr;
    cudaMalloc((void **) &deviceBuffers_d, sizeof(MCGIDI::DataBuffer *) * numIsotopes);
    cudaMemcpy(deviceBuffers_d, &deviceBuffers_h[0], sizeof(MCGIDI::DataBuffer *) * numIsotopes, cudaMemcpyHostToDevice);

    setUp <<< numIsotopes, 32 >>> (numIsotopes, deviceBuffers_d);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    if (doPrint)
    {
        printData<<<1, 1>>> (reinterpret_cast<MCGIDI::ProtareSingle *>(deviceProtares[numIsotopes-1]));
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    if (doCompare > 0) {
        int isoIndex = numIsotopes-1;
        size_t cpuSize = sizeof(*protares[isoIndex]) + protares[isoIndex]->internalSize();
        char *gidiBytes = (char *)malloc(cpuSize);
        cudaMemcpy(gidiBytes, deviceProtares[isoIndex], cpuSize, cudaMemcpyDeviceToHost);
        if (doCompare == 1) {
            FILE *outFile = fopen("gidi_data.bin", "wb");
            fwrite(gidiBytes, sizeof(char), cpuSize, outFile);
            fclose(outFile);
        } else 
        {
            char *fileBytes = (char *)malloc(cpuSize);
            FILE *inFile = fopen("gidi_data.bin", "rb");
            fread(fileBytes, sizeof(char), cpuSize, inFile);
            fclose(inFile);
            int errorCount = 0;
            int firstError = -1;
            for (int indx = 0; indx < cpuSize; indx++) {
                if (fileBytes[indx] == gidiBytes[indx]) continue;
                if (firstError == -1) firstError = indx;
                errorCount++;
            }
            printf("Out of %d bytes, there were %d errors with the first one at %d\n", cpuSize, errorCount, firstError);
        }
    }

    int tally[10];
    if (numCollisions > 0)
    {
        timeval tv1, tv2;
        gettimeofday(&tv1, nullptr);
        MCGIDI::ProtareSingle &gidi_data = *(protares[numIsotopes-1]);
        double temperature = 2.58522e-8;
        TallyProductHandler products;
        products.setTally(tally);
        for (int i = 0; i < 10; i++) tally[i] = 0;

        for (int collisionIndex = 0; collisionIndex < numCollisions; ++collisionIndex) {
            uint64_t seed = collisionIndex+1;
            double energy = pow(10.0, myRNG(&seed) * 1.3);
            MCGIDI::Sampling::Input input( true, MCGIDI::Sampling::Upscatter::Model::B );
            int hashIndex = domainHash.index( energy );
            MCGIDI::URR_protareInfos urr;
            double crossSection = gidi_data.crossSection( urr, hashIndex, temperature, energy );
            int reactionIndex = MCProtare->sampleReaction( urr, hashIndex, temperature, energy, crossSection, (double (*)(void *))myRNG, &seed );

            MCGIDI::Reaction const *reaction = gidi_data.reaction(reactionIndex);
            reaction->sampleProducts( &gidi_data, energy, input, myRNG, &seed, products );
        }
        gettimeofday(&tv2, nullptr);
        printf("Host tally:   ");
        for (int i = 0; i < 10; i++) printf("%d ", tally[i]);
        printf("\nHost   Total time = %f seconds\n", (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec));
    }

    if (doPrint) 
    {
        int isoIndex = numIsotopes-1;
        MCGIDI::DataBuffer buf2_h;
        protares[isoIndex]->serialize(buf2_h, MCGIDI::DataBuffer::Count);
        buf2_h.allocateBuffers();
        buf2_h.zeroIndexes();
        size_t cpuSize = sizeof(*protares[isoIndex]) + protares[isoIndex]->internalSize();
        protares[isoIndex]->serialize(buf2_h, MCGIDI::DataBuffer::Pack);
        MCGIDI::ProtareSingle *MCProtare_h = nullptr;
        char *bigBuffer = (char *)malloc(cpuSize);
        MCProtare_h = new(bigBuffer) MCGIDI::ProtareSingle();
        buf2_h.zeroIndexes();
        buf2_h.m_placementStart = bigBuffer;
        buf2_h.m_placement = buf2_h.m_placementStart + sizeof(MCGIDI::ProtareSingle);
        buf2_h.m_maxPlacementSize = cpuSize;
        MCProtare_h->serialize(buf2_h, MCGIDI::DataBuffer::Unpack);
        buf2_h.m_placement = buf2_h.m_placementStart + sizeof(MCGIDI::ProtareSingle);
        if (!buf2_h.validate()) printf("Data went over memory pool size.\n");

        MCGIDI::ProtareSingle &gidi_data = *MCProtare_h;
        for( int i1 = 0; i1 < numberOfReactions; ++i1 ) {
            MCGIDI::Reaction const *reaction = gidi_data.reaction(i1);
            double threshold = gidi_data.threshold( i1 );

            printf("HN: reaction(%d) = %s threshold = %g\n", i1, reaction->label( ).c_str(), threshold);
        }

        buf2_h.nullOutPointers();
    }

    if (numCollisions> 0)
    {
        timeval tv1, tv2;
        gettimeofday(&tv1, nullptr);
        int *tally_d;
        cudaMalloc((void **) &tally_d, 10 * sizeof(int)); 
        cudaMemset(tally_d, 0, 10 * sizeof(int)); 

        MCGIDI::ProtareSingle *MCProtare_d = reinterpret_cast<MCGIDI::ProtareSingle *>(deviceProtares[numIsotopes-1]);
        sample<<<(numCollisions+255)/256, 256>>> (MCProtare_d, numCollisions, tally_d);

        cudaDeviceSynchronize();
        gettimeofday(&tv2, nullptr);

        cudaMemcpy(tally, tally_d, 10 * sizeof(int), cudaMemcpyDeviceToHost);
        printf("Device tally: ");
        for (int i = 0; i < 10; i++) printf("%d ", tally[i]);
        printf("\nDevice Total time = %f seconds\n", (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec));
        cudaFree( tally_d );
    }
}

HOST_DEVICE double myRNG( uint64_t *seed ) {
   // Reset the state from the previous value.
   *seed = 2862933555777941757ULL*(*seed) + 3037000493ULL;
   
   // Map the int state in (0,2**64) to double (0,1)
   // by multiplying by
   // 1/(2**64 - 1) = 1/18446744073709551615.
   return 5.4210108624275222e-20*(*seed);
}


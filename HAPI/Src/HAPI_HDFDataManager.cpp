/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "HAPI.hpp"
#include <vector>
//#include <time.h>

#ifdef HAPI_USE_HDF5
namespace HAPI {

    // constructor
    HDFDataManager::HDFDataManager(std::string const &a_filename) :
        m_filename( a_filename ) {

        m_file_id = H5Fopen( a_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
        H5Eset_auto1( nullptr, nullptr );

        m_dataset_ints = H5Dopen2( m_file_id, "iData", H5P_DEFAULT );
        m_iDataPresent = m_dataset_ints != H5I_INVALID_HID;
        if( m_iDataPresent ) m_dataspace_ints = H5Dget_space( m_dataset_ints );

        m_dataset_doubles = H5Dopen2( m_file_id, "dData", H5P_DEFAULT );
        m_dDataPresent = m_dataset_doubles != H5I_INVALID_HID;
        if( m_dDataPresent ) m_dataspace_doubles = H5Dget_space( m_dataset_doubles );

        m_stride[0] = 1;
        m_block[0] = 1;

        /*
        m_num_double_reads = 0;
        m_num_double_elem = 0;
        m_num_int_reads = 0;
        m_num_int_elem = 0;
        clock_gettime(CLOCK_MONOTONIC, &m_tstart);
        */
    }

    HDFDataManager::~HDFDataManager()
    {
      /*  timing info for debugging
      TimeType tstop;
      clock_gettime(CLOCK_MONOTONIC, &tstop);

      double t = tstop.tv_sec - m_tstart.tv_sec;
      t += (tstop.tv_nsec - m_tstart.tv_nsec)/1000000000.0;

      printf("\n");
      printf("HDFDataManager: num double reads: %ld\n", (long)m_num_double_reads);
      printf("HDFDataManager: num double reads: %ld\n", (long)m_num_double_elem);
      printf("HDFDataManager: num int reads:    %ld\n", (long)m_num_int_reads);
      printf("HDFDataManager: num int reads:    %ld\n", (long)m_num_int_elem);

      printf("HDFDataManager: Elapsed time:     %lf\n", t);

      size_t num_bytes_read = sizeof(double)*m_num_double_elem + sizeof(int)*m_num_int_elem;
      double MB = num_bytes_read/1024.0/1024.4;
      printf("HDFDataManager: Megabytes read:   %lf\n", MB);
      double MB_sec = MB / t;
      printf("HDFDataManager: Megabytes/second: %lf\n", MB_sec);
      */

        if( m_iDataPresent ) {
            H5Dclose(m_dataset_ints);
            H5Sclose(m_dataspace_ints);
        }   
        if( m_iDataPresent ) {
            H5Dclose(m_dataset_doubles);
            H5Sclose(m_dataspace_doubles);
        }
      H5Fclose(m_file_id);
    }

    void HDFDataManager::getDoubles(nf_Buffer<double> &result, size_t startIndex, size_t endIndex)
    {
        if( !m_dDataPresent ) throw LUPI::Exception( "HDFDataManager::getDoubles: HDF5 file " + m_filename + " has no 'dData' dataset." );

        hid_t memspace;
        herr_t status;

        hsize_t size = endIndex - startIndex;

        hsize_t dims[] {size};
        hsize_t offset[] {startIndex};
        hsize_t count[] {size};

        result.resize(size);
        m_num_double_reads ++;
        m_num_double_elem += size;
        
        // now can we access the allocated array and read into that?

        memspace = H5Screate_simple(1, dims, nullptr);
        status = H5Sselect_hyperslab(m_dataspace_doubles, H5S_SELECT_SET, offset, m_stride, count, m_block);
        if( status != 0 ) throw "H5Sselect_hyperslab error in HDFDataManager::getDoubles.";

        status = H5Dread(m_dataset_doubles, H5T_NATIVE_DOUBLE, memspace, m_dataspace_doubles, H5P_DEFAULT, result.data());
        if( status != 0 ) throw "H5Dread error in HDFDataManager::getDoubles.";

        H5Sclose(memspace);

    }

    void HDFDataManager::getInts(nf_Buffer<int> &result, size_t startIndex, size_t endIndex)
    {
        if( !m_iDataPresent ) throw LUPI::Exception( "HDFDataManager::getInts: HDF5 file " + m_filename + " has no 'iData' dataset." );

        hid_t memspace;
        herr_t status;
        hsize_t size = endIndex - startIndex;

        hsize_t dims[] {size};
        hsize_t offset[] {startIndex};
        hsize_t count[] {size};

        result.resize(size);

        m_num_int_reads ++;
        m_num_int_elem += size;

        memspace = H5Screate_simple(1, dims, nullptr);
        status = H5Sselect_hyperslab(m_dataspace_ints, H5S_SELECT_SET, offset, m_stride, count, m_block);
        if( status != 0 ) throw "H5Sselect_hyperslab error in HDFDataManager::getDoubles.";

        status = H5Dread(m_dataset_ints, H5T_NATIVE_INT, memspace, m_dataspace_ints, H5P_DEFAULT, result.data());
        if( status != 0 ) throw "H5Dread error in HDFDataManager::getDoubles.";

        H5Sclose(memspace);

    }

}
#endif

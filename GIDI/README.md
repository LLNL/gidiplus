# GIDI
GIDI (General Interaction Data Interface) is a C++ library for reading and writing nuclear reaction data stored in the GNDS (Generalized Nuclear Data Structure) format

# Working with Git LFS

## Installation
To clone and access GIDI the following commands are recommended:

    `git clone ssh://git@czgitlab.llnl.gov:7999/nuclear/gidiplus/gidi.git [local_folder_name]`

The GIDI data (Test/Data) is currently stored as a TAR compressed archive (Test/GIDI_Test_Data.tar.gz) in Git LFS. The above clone command automatically downloads Test/GIDI_Test_Data.tar.gz but an additional step may be required to initialize Git LFS (i.e. it may be necessary to install Git LFS and run the command `git lfs install`)

The GIDI data files need to be extracted from the TAR compressed archive with the following command:

    `cd Test; tar -xzvf GIDI_Test_Data.tar.gz`

Note that the resulting files in Test/Data are not tracked in Git and a seperate step is required to update any of the files in Git LFS

## Updating Git LFS data
Any updates to the files in the GIDI Test data require the following update to the corresponding Git LFS repo.

    `cd Test; rm GIDI_Test_Data.tar.gz; tar -czvf GIDI_Test_Data.tar.gz Data`

Note that the compressed archived file is recreated after deletion ... it is possible to simply update the exisiting archive but this sometimes results in a corrupted archive file.

This update will prompt Git to indicate that GIDI_Test_Data.tar.gz has been modified and a simple commit of this file ensure that Git LFS is updated, eg. the command

    `git commit GIDI_Test_Data.tar.gz -m "Updating the GIDI Test Data"`

will update Git LFS.
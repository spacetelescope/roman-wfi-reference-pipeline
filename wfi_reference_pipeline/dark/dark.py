import roman_datamodels.stnode as rds
from ..utilities.reference_file import ReferenceFile
import asdf
import numpy as np
# from astropy.stats import sigma_clipped_stats
import logging
import pandas as pd
import os




# Squash logging messages from stpipe.
logging.getLogger('stpipe').setLevel(logging.WARNING)


class Dark(ReferenceFile):
    """
    Class Dark() inherits the ReferenceFile() base class methods
    where static meta data for all reference file types are written. The
    method make_dark() implements specified MA table properties (number of
    reads per resultant). The dark asdf file has contains the averaged dark
    frame resultants.
    """

    def __init__(self, read_cube, meta_data, bit_mask=None, outfile=None, clobber=False):
        # If no output file name given, set default file name.
        self.outfile = outfile if outfile else 'roman_dark.asdf'

        # Access methods of base class ReferenceFile
        super(Dark, self).__init__(read_cube, meta_data, bit_mask=bit_mask, clobber=clobber)

        # Update metadata with dark file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Roman WFI dark reference file.'
        else:
            pass
        if 'reftype' not in self.meta.keys():
            self.meta['reftype'] = 'DARK'
        else:
            pass

    # def make_ramp_image
    # get all of the files and reads from a pwd

    def make_dark(self, n_result, n_read_per_result, det_str):

        """
        The method make_dark() generates a dark asdf file such that it contains
        a number of resultants that constructed from the mean of a number of reads
        as specified by the MA table string. The number of reads per resultant,
        the number of resultants, and the MA table string are inputs to creating
        the asdf dark file.

        NOTE: Future work will have the MA table name as input and internally
        reference the RTB Database to retrieve MA table properties (i.e. the
        number of reads per resultant and number of resultants, and possible
        sequence of reads to achieve unevenly spaced resultants.

        Parameters
        ----------
        n_result: integer; the number of resultants
            The number of final resultants in the dark asdf file.
        n_read_per_result: integer; the number of reads per resultant
            The number of reads to be averaged in creating a resultant.
        table_str: string; the MA table name
            The MA table name string is used to retrieve table parameters
            and inserted into the filename string.
        NOTE: Assuming equally spaced resultants.

        Outputs
        -------
        af: asdf file tree: {meta, data, dq, err}
            meta:
            data: averaged resultants per MA table specs
            dq: mask - data quality array
                masked hot pixels in rate image flagged 2**11
            err: zeros
        """

        # Check if the output file exists, and take appropriate action.
        self.check_output_file(self.outfile)

        # initialize dark cube with number of resultants from inputs
        dark_cube = np.zeros((n_result, 4096, 4096), dtype=np.float32)
        dark_cube_err = np.zeros((n_result, 4096, 4096), dtype=np.float32)

        # average over number of reads per resultant for each pixel
        # to make darks according to ma_table specs
        for i_res in range(0,n_result):
            i1 = i_res * n_read_per_result
            i2 = i1 + n_read_per_result
            dark_cube[i_res, :, :] = np.mean(self.data[i1:i2, :, :], axis=0)
            res_std = np.std(dark_cube[i_res, :, :])
            dark_cube_err[i_res, :, :] = np.round(np.random.uniform(.9*res_std, 1.1*res_std,
                                                                    size=(4096, 4096)).astype(np.float32),2)

        # update object for asdf file
        self.data = dark_cube
        # log info of darks made
        logging.info('Darks were made with %s resultants made with %s reads per resultant',
                     str(n_result), str(n_read_per_result))

        # hot pixel flagging will require
        fresult_avg = np.mean(self.data[-1])
        hot_pixels = np.where(self.data[-1] > 1.8*fresult_avg)
        self.mask[hot_pixels] = 2 ** 11
        num_hotpixels = np.count_nonzero(self.mask == 2 **11)

        #logging.info('Recombining dark reads into MA table specifications')
        # Calculate the dark image for each resultant. Each resultant is
        # treated as the average of the reads in it, so calculate the dark
        # image for each read and then combine them. If using all the reads,
        # don't average them.

        #dark_image = np.zeros((n_resultants, 4096, 4096), dtype=np.float32)
        #for i in range(n_resultants):
            #reads = np.zeros((n_reads, 4096, 4096), dtype=np.float32)
            #for j in range(n_reads):
                #read_time = (j + 1 + (n_reads * i)) * frame_time
                #reads[j, :, :] = self.data * read_time
            #if n_reads > 1:
                #dark_image[i, :, :] = np.mean(reads, axis=0)
            #else:
                #np.squeeze(reads)

        #logging.info('Constructing dark datamodel')
        # Construct the dark object from the data model.
        #dark_asdf = DarkModel(data=dark_image,
                              #err=np.zeros((n_resultants, 4096, 4096),
                                           #dtype=np.float32),
                              #dq=self.mask)

        #dark_asdf.history = self.meta['history']

        #logging.info(f'Saving dark reference file to {outfile}')

        # Construct the dark object from the data model.
        darkfile = rds.DarkRef()
        darkfile['meta'] = self.meta
        # Add in the MA table name to the meta data in the ASDF file.


        darkfile['data'] = self.data
        darkfile['dq'] = self.mask
        #darkfile['err'] = np.zeros(self.data.shape, dtype=np.float32)
        #darkfile['err'] = np.random.randint(3,8, size=(4096, 4096)).astype(np.float32)/10000.
        darkfile['err'] = dark_cube_err
        # Add in the meta data and history to the ASDF tree.
        af = asdf.AsdfFile()
        af.tree = {'roman': darkfile}
        af.write_to(self.outfile)


        #db_info = {
        #    'detector' : det_str,
        #    'type' : self.meta['reftype'],
        #    'useafter' : self.meta['useafter'],
        #    'mean': np.mean(self.data[-1]),
        #    'median': np.median(self.data[-1]),
        #    'stdev': np.std(self.data[-1]),
        #    '#hotpix' : num_hotpixels,
        #}

        #df = pd.DataFrame(db_info, index=[0])
        #print(df)

        #dbRFP_pwd = '/grp/roman/RFP/DEV/RTB_DataBase_RFPtmp/'
        #tmp_pwdfl = dbRFP_pwd + det_str + 'dark_db_metrics.csv'
        #df.to_csv(tmp_pwdfl, index=False)
        #os.chmod(tmp_pwdfl, 0o666)

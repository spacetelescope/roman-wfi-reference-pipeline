from ..utilities.reference_file import ReferenceFile
from astropy.stats import sigma_clipped_stats
from romancal.datamodels.reference_files.dark import DarkModel
import logging
import numpy as np

# Squash logging messages from stpipe.
logging.getLogger('stpipe').setLevel(logging.WARNING)


class Dark(ReferenceFile):
    """
    Base class for dark file creation. We can modify this for DCL
    data after we have the basic algorithm defined.
    """

    def __init__(self, ramp_image_list, meta_data, bit_mask=None,
                 outroot=None, clobber=False):

        self.outroot = outroot

        super(Dark, self).__init__(ramp_image_list, meta_data,
                                   bit_mask=bit_mask, clobber=clobber)

        # Update metadata with constants.
        self.meta['meta']['description'] = 'Dark file.'

        # Future arrays
        self.rate_image = None

    def compute_dark_rate(self, warm_thresh=3, hot_thresh=5,
                          unreliable_thresh=1, warm_bit=12, hot_bit=11,
                          unreliable_bit=23):

        logging.info('Computing sigma-clipped mean dark rate image')
        # Compute the mean and standard deviation of all of the darks in
        # the input list.
        self.rate_image, _, dark_noise = sigma_clipped_stats(self.data,
                                                             axis=0)

        logging.info('Computing sigma-clipped median and standard deviation '
                     'of mean dark rate image')
        # Compute the median and standard deviation of the average rate image.
        _, dark_rate_med, dark_rate_std = sigma_clipped_stats(self.rate_image)

        warm_rate = dark_rate_med + (warm_thresh * dark_rate_std)
        hot_rate = dark_rate_med + (hot_thresh * dark_rate_std)

        logging.info(f'\tmedian = {dark_rate_med:0.5f} ADU / sec')
        logging.info(f'\tstd. dev. = {dark_rate_std:0.5f} ADU / sec')

        logging.info('Flagging warm, hot, and unreliable pixels in DQ array')
        logging.info(f'\tWarm pixel threshold = {warm_rate:0.5f} ADU / sec')
        logging.info(f'\tHot pixel threshold = {hot_rate:0.5f} ADU / sec')
        logging.info(f'\tUnreliable pixel threshold = '
                     f'{unreliable_thresh:0.5f} ADU / sec')
        logging.info('DQ bit values:')
        logging.info(f'\tWarm = {warm_bit}')
        logging.info(f'\tHot = {hot_bit}')
        logging.info(f'\tUnreliable = {unreliable_bit}')

        # Mark warm/hot/unreliable pixels.
        warm_pixel = np.where((self.rate_image >= warm_rate) &
                              (self.rate_image < hot_rate))

        hot_pixel = np.where(self.rate_image >= hot_rate)

        unreliable_pixel = np.where(dark_noise > unreliable_thresh)

        self.mask[warm_pixel] += 2**warm_bit
        self.mask[hot_pixel] += 2**hot_bit
        self.mask[unreliable_pixel] += 2**unreliable_bit

    def make_dark(self, n_reads=1, n_resultants=1, frame_time=2.806,
                  ma_table_name='ALL'):

        # Set up the output file name, and check if it exists.
        outfile = f'{self.outroot}_{ma_table_name.lower()}_dark.asdf'
        self.check_output_file(outfile)

        logging.info('Recombining dark reads into MA table specifications')
        # Calculate the dark image for each resultant. Each resultant is
        # treated as the average of the reads in it, so calculate the dark
        # image for each read and then combine them. If using all the reads,
        # don't average them.
        dark_image = np.zeros((n_resultants, 4096, 4096), dtype=np.float32)
        for i in range(n_resultants):
            reads = np.zeros((n_reads, 4096, 4096), dtype=np.float32)
            for j in range(n_reads):
                read_time = (j + 1 + (n_reads * i)) * frame_time
                reads[j, :, :] = self.rate_image * read_time
            if n_reads > 1:
                dark_image[i, :, :] = np.mean(reads, axis=0)
            else:
                np.squeeze(reads)

        logging.info('Constructing dark datamodel')
        # Construct the dark object from the data model.
        dark_asdf = DarkModel(data=dark_image,
                              err=np.zeros((n_resultants, 4096, 4096),
                                           dtype=np.float32),
                              dq=self.mask)

        logging.info('Adding meta data to dark ASDF tree')
        # Add in the meta data and history to the ASDF tree.
        for key, value in self.meta['meta'].items():
            dark_asdf.meta[key] = value
        dark_asdf.history = self.meta['history']

        # Add in the MA table name to the meta data in the ASDF file.
        dark_asdf.meta['observation'] = {'ma_table_name': ma_table_name}

        logging.info(f'Saving dark reference file to {outfile}')
        # Write out the dark ASDF file.
        dark_asdf.save(outfile)

import datetime, os, sys, yaml
import numpy as np
if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources
from astropy.time import Time
from romancal.lib import dqflags
from wfi_reference_pipeline.version import version as PIPELINE_VERSION


class ReferenceFile:
    """
    Base class ReferenceFile() writes static metadata for all reference file types
    are written.

    Returns
    -------
    self.input_data: attribute;
        Class dependent variable assigned as attribute. Intended to be list of files or numpy array.
        If not used, returned as none.
    self.ancillary: attribute;
        Other data for WFI such as filter names, frame times, WFI mode.
    self.dqflag_defs:
    """

    def __init__(self, data, meta_data, bit_mask=None, clobber=False,
                 make_mask=False, mask_size=(4096, 4096)):

        self.data = data
        # TODO VERIFY THAT meta_data IS TYPE OF ONE OF THE REFERENCE FILE OBJECTS
        self.meta_data = meta_data

        # Load DQ flag definitions from romancal
        self.dqflag_defs = dqflags.pixel

        # TODO is this needed here or will this be reference type specific?, perhaps this hsould become an @abstractMethod ?
        if np.shape(bit_mask):
            print("Mask provided. Skipping internal mask generation.")
            self.mask = bit_mask.astype(np.uint32)
        else:
            if make_mask:
                self.mask = np.zeros(mask_size, dtype=np.uint32)
            else:
                self.mask = None


        # TODO - Is this needed here?
        # # Ancillary data for reference file modules
        # with importlib_resources.path('wfi_reference_pipeline.resources.data',
        #                               'ancillary.yaml') as afile:
        #     with open(afile) as af:
        #         self.ancillary = yaml.safe_load(af)

        # Other stuff.
        self.clobber = clobber

        # TODO remove the below line, it is here as an example for utilizing the abstractmethods
        self.meta_data.initialize_reference_data(self)

    def check_output_file(self, outfile):
        # Check if the output file exists, and take appropriate action.
        if os.path.exists(outfile):
            if self.clobber:
                os.remove(outfile)
            else:
                raise FileExistsError(f'''{outfile} already exists,
                                          and clobber={self.clobber}!''')

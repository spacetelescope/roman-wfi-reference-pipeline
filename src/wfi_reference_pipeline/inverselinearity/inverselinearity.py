import roman_datamodels.stnode as rds
import numpy as np
from ..utilities.reference_file import ReferenceFile
from astropy import units as u
from astropy.io import fits
import asdf


class InverseLinearity(ReferenceFile):
    """
    Class InverseLinearity() inherits the ReferenceFile() base class methods
    where static meta data for all reference file types are written. The
    method get_coeffs_from_dcl retrieves inverse linearity coefficients
    determined from DCL data by Bellini et al. (~2021), which are used to
    make these reference files with intended use by romanisim only. There
    is not currently any plan to produce these reference files for use
    in romancal standard processing.
    """

    def __init__(self, inv_linearity_image, meta_data, outfile='roman_inv_linearity.asdf', inv_coeffs=None,
                 bit_mask=None, clobber=False, ):
        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceFile()
        file base class.

        Parameters
        ----------

        inv_linearity_image: numpy.ndarray; Input image
         to perform the inverse linearity fit. It populates self.input_data.
        meta_data: dictionary; default = None
            Dictionary of information for reference file as required by romandatamodels.
        outfile: string; default = roman_inv_linearity.asdf
            Filename with path for saved inverse linearity reference file.
        inv_coeffs: numpy.ndarray; User input inverse linearity coefficients.
        bit_mask: 2D integer numpy array, default = None
            A 2D data quality integer array for supplying a mask for the creation of the dark reference file.
        clobber: Boolean; default = False
            True to overwrite the file name outfile if file already exists. False will not overwrite and exception
            will be raised if duplicate file is found.
        """

        # Access methods of base class ReferenceFile
        super().__init__(inv_linearity_image, meta_data, bit_mask=bit_mask, clobber=clobber, make_mask=True)

        # Update metadata with file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Roman WFI inverse linearity reference file.'
        if 'reftype' not in self.meta.keys():
            self.meta['reftype'] = 'INVERSELINEARITY'  # RTB coordinated with DMS and CRDS on reftype name of all caps
            # June 2023 - R. Cosentino, R. Klein, W. Jamieson

        # Initialize attributes
        self.outfile = outfile
        self.inv_coeffs = inv_coeffs

    def get_coeffs_from_dcl(self, wfi_det='WFI01'):
        """
        The method get_coeffs_from_dcl() will take as input a WFI detector id  and retrieve the inverse linearity
        coefficients from group roman on central storage from files generate by A. Bellini ~2021.

        Parameters
        ----------
        wfi_det: string; default = "WFI01"
            Variable to identify which detector maps to what sca file id.
        """

        # Update meta data based on input to getting DCL inverse linearity coefficients
        self.meta['instrument'].update({'detector': wfi_det})

        wfi_arr = ["WFI01", "WFI02", "WFI03", "WFI04", "WFI05", "WFI06", "WFI07", "WFI08", "WFI09", "WFI10", "WFI11",
                   "WFI12", "WFI13", "WFI14", "WFI15", "WFI16", "WFI17", "WFI18"]

        sca_id_arr = [22066, 21815, 21946, 22073, 21816, 20663, 22069, 21641, 21813, 22078, 21947, 22077, 22067, 21814,
                      21645, 21643, 21319, 20833]

        # Create a dictionary to map all wfi detectors to sca id numbers
        wfi_to_sca = dict(zip(wfi_arr, sca_id_arr))
        det_number = int(wfi_det[3:])
        sca_id = wfi_to_sca[wfi_det]

        # Make inverse linearity file string with absolute path to central storage from the detector input and the
        # mapping to WFI tags
        inv_file_dir = '/grp/roman/bellini/WFIsim/CNL/new/'
        inv_file = inv_file_dir + 'LNC_SCA' + str(det_number).zfill(2) + '.fits'

        # Open the SCA file to get inverse linearity coefficients as np.float32
        with fits.open(inv_file) as hdul:
            self.inv_coeffs = hdul[0].data.astype(np.float32)

        # Update meta data
        self.meta.update({'pedigree': 'GROUND'})
        self.meta['instrument'].update({'SCA': sca_id})

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """

        # Construct the dark object from the data model.
        inverselinearity_datamodel_tree = rds.InverseLinearityRef()
        inverselinearity_datamodel_tree['meta'] = self.meta
        inverselinearity_datamodel_tree['coeffs'] = self.inv_coeffs
        inverselinearity_datamodel_tree['dq'] = self.mask

        return inverselinearity_datamodel_tree

    def save_inverselinearity(self, datamodel_tree=None):
        """
        The method save_inverselinearity writes the reference file object to the specified asdf outfile.
        """

        # Use data model tree if supplied. Else write tree from module.
        af = asdf.AsdfFile()
        if datamodel_tree:
            af.tree = {'roman': datamodel_tree}
        else:
            af.tree = {'roman': self.populate_datamodel_tree()}
        af.write_to(self.outfile)

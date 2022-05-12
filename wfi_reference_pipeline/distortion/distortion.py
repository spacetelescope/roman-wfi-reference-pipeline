import roman_datamodels.stnode as rds
from ..utilities.reference_file import ReferenceFile
import asdf


class Distortion(ReferenceFile):
    """
    Class Distortion() inherits the ReferenceFile() base class methods
    where static meta data for all reference file types are written. The
    method make_gain() creates the asdf gain file.
    """

    def __init__(self, cdt_model, meta_data, bit_mask=None, outfile=None, clobber=False):
        # If no output file name given, set default file name.
        self.outfile = outfile if outfile else 'roman_distortion.asdf'

        # Access methods of base class ReferenceFile
        super(Distortion, self).__init__(cdt_model, meta_data, bit_mask=bit_mask, clobber=clobber)

        # Update metadata with distortion file type info if not included.
        if 'description' not in self.meta.keys():
            self.meta['description'] = 'Roman WFI distortion reference file.'
        else:
            pass
        if 'reftype' not in self.meta.keys():
            self.meta['reftype'] = 'DISTORTION'
        else:
            pass

    def make_distortion(self):
        """
        The method make_distortion() generates a distortion asdf file with the input data.

        Parameters
        ----------

        Outputs
        -------
        af: asdf file tree: {meta, model}
            meta:
        """
        # Check if the output file exists, and take appropriate action.
        self.check_output_file(self.outfile)

        filter = ["F062", "F087", "F106", "F129", "W146", "F158", "F184", "F213", "GRISM", "PRISM", "DARK"]

        # from https://github.com/spacetelescope/nircam_calib/blob/master/nircam_calib/reffile_creation/pipeline/distortion/nircam_distortion_reffiles_from_pysiaf.py
        # and https: // jira.stsci.edu / browse / RTB - 270
        # Find the distance between (0,0) and the reference location
        xshift, yshift = get_refpix(inst_siaf, full_aperture)

        # *****************************************************
        # If the user provides files containing distortion coefficients
        # (as output by the jwst_fpa package), use those rather than
        # retrieving coefficients from siaf.
        if dist_coeffs_file is not None:
            coeff_tab = read_distortion_coeffs_file(dist_coeffs_file)
            xcoeffs = convert_distortion_coeffs_table(coeff_tab, 'Sci2IdlX')
            ycoeffs = convert_distortion_coeffs_table(coeff_tab, 'Sci2IdlY')
            inv_xcoeffs = convert_distortion_coeffs_table(coeff_tab, 'Idl2SciX')
            inv_ycoeffs = convert_distortion_coeffs_table(coeff_tab, 'Idl2SciY')
        elif dist_coeffs_file is None:
            xcoeffs, ycoeffs = get_distortion_coeffs('Sci2Idl', siaf)
            inv_xcoeffs, inv_ycoeffs = get_distortion_coeffs('Idl2Sci', siaf)

        # V3IdlYAngle and V2Ref, V3Ref should always be taken from the latest version
        # of SIAF, rather than the output of jwst_fpa. Separate FGS/NIRISS analyses must
        # be done in order to modify these values.
        v3_ideal_y_angle = siaf.V3IdlYAngle * np.pi / 180.

        # *****************************************************
        # "Forward' transformations. science --> ideal --> V2V3
        # label = 'Sci2Idl'
        ##from_units = 'distorted pixels'
        ##to_units = 'arcsec'

        # xcoeffs, ycoeffs = get_distortion_coeffs(label, siaf)

        sci2idlx = Polynomial2D(degree, **xcoeffs)
        sci2idly = Polynomial2D(degree, **ycoeffs)

        # Get info for ideal -> v2v3 or v2v3 -> ideal model
        parity = siaf.VIdlParity
        # v3_ideal_y_angle = siaf.V3IdlYAngle * np.pi / 180.
        idl2v2v3x, idl2v2v3y = v2v3_model('ideal', 'v2v3', parity, v3_ideal_y_angle)

        # Finally, we need to shift by the v2,v3 value of the reference
        # location in order to get to absolute v2,v3 coordinates
        v2shift, v3shift = get_v2v3ref(siaf)

        #self.data = cdt_model done within the super in the base class merging

        # Construct the gain object from the data model.
        distortionfile = rds.DistortionRef()
        distortionfile['coordinate_distortion_transform'] = self.data
        distortionfile['meta'] = self.meta
        # Gain files do not have data quality or error arrays.

        # Add in the meta data and history to the ASDF tree.
        af = asdf.AsdfFile()
        af.tree = {'roman': distortionfile}
        af.write_to(self.outfile)


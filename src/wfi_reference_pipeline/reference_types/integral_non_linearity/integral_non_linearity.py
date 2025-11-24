

import numpy as np
import roman_datamodels.stnode as rds

from wfi_reference_pipeline.resources.wfi_meta_integral_non_linearity import (
    WFIMetaIntegralNonLinearity,
)

from ..reference_type import ReferenceType


class IntegralNonLinearity(ReferenceType):
    """
    Class IntegralNonLinearity() inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written. The
    method creates the asdf reference file.

    Example creation
    from wfi_reference_pipeline.reference_types.integral_non_linearity.integral_non_linearity import simulate_inl_correction_array
    from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
    from wfi_reference_pipeline.reference_types.integral_non_linearity.integral_non_linearity import IntegralNonLinearity

    for det in range(1, 19):
        arr = simulate_inl_correction_array()
        # Create meta data object for INL ref file
        tmp = MakeDevMeta(ref_type='INTEGRALNONLINEARITY')
        # Update meta per detector to get the right values from the form and update description
        tmp.meta_integral_non_linearity.instrument_detector = f"WFI{det:02d}"
        tmp.meta_integral_non_linearity.description = (
            "To support new integral non-linearity correction development for B21. "
            "The integral non-linearity correction is applied to each channel, "
            "numbered left to right from 1 to 32, with each channel containing 128 "
            "pixels in length."
            )
        # Update the file name to match the detector
        fl_name = 'new_roman_inl_' + tmp.meta_integral_non_linearity.instrument_detector
        # Instantiate an object and write the file out
        rfp_inl = IntegralNonLinearity(meta_data=tmp.meta_integral_non_linearity,
                                    ref_type_data=arr,
                                    clobber=True,
                                    outfile=fl_name+'.asdf')
        rfp_inl.generate_outfile()
    """

    def __init__(
            self,
            meta_data,
            file_list=None,
            ref_type_data=None,
            bit_mask=None,
            outfile="roman_inl.asdf",
            clobber=False,
    ):

        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceType()
        file base class.

        Parameters
        ----------
        meta_data: Object; default = None
            Object of meta information converted to dictionary when writing reference file.
        file_list: List of strings; default = None
            List of file names with absolute paths. Intended for primary use during automated operations.
        ref_type_data: numpy array; default = None
            Input which can be image array or data cube. Intended for development support file creation or as input
            for reference file types not generated from a file list.
        bit_mask: 2D integer numpy array, default = None
            A 2D data quality integer mask array to be applied to reference file.
        outfile: string; default = roman_flat.asdf
            File path and name for saved reference file.
        clobber: Boolean; default = False
            True to overwrite outfile if outfile already exists. False will not overwrite and exception
            will be raised if duplicate file found.
        ---------

        See reference_type.py base class for additional attributes and methods.
        """

        # Access methods of base class ReferenceType
        super().__init__(
            meta_data=meta_data,
            file_list=file_list,
            ref_type_data=ref_type_data,
            bit_mask=bit_mask,
            outfile=outfile,
            clobber=clobber
        )

        # Default meta creation for module specific ref type.
        if not isinstance(meta_data, WFIMetaIntegralNonLinearity):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaIntegralNonLinearity"
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI integral non linearity reference file."

        self.inl_correction = ref_type_data
        _, num_values = np.shape(ref_type_data)
        self.value_array = np.linspace(0, 65535, num_values, dtype=np.uint16)
        # TODO look at references for channel id number - https://roman-docs.stsci.edu/data-handbook-home/wfi-data-format/coordinate-systems

        self.outfile = outfile

    def calculate_error(self):
        """
        Abstract method not applicable.
        """
        pass

    def update_data_quality_array(self):
        """
        Abstract method not utilized.
        """
        pass

    def _make_inl_table(self):
        """
        Construct the integral non linearity correction table to populate the data model.
        """
        inl_table = {}

        # Assuming self.inl_correction is shaped (32, N)
        # Each row corresponds to a science chunk (0–31)
        for sci_chunk in range(32):
            # Reverse channel mapping: 0→32, 1→31, ..., 31→1 for a detector that is
            # needing to be flipped left right from detector to science coordinates
            # NOTE: other detectors
            channel = 32 - sci_chunk

            inl_table[str(sci_chunk)] = {
                'channel': channel,
                'correction': self.inl_correction[sci_chunk]
            }

        return inl_table


    def populate_datamodel_tree(self):
        """
        Build the Roman datamodel tree for the integral non-linearity reference.
        """
        try:
            # Placeholder until official datamodel exists
            inl_ref = rds.IntegralNonLinearity()
        except AttributeError:
            inl_ref = {"meta": {}, 
                       "inl_table": {},
                       "value": {}
                       }

        inl_ref["meta"] = self.meta_data.export_asdf_meta()
        inl_ref["inl_table"] = self._make_inl_table()
        inl_ref["value"] = self.value_array

        return inl_ref


def simulate_inl_correction_array():
    """
    Helper function to simulate array that will be used to create
    fully populated example reference file.

    Using a combination of a linear slope, saw tooth and sine curve with different periods
    and reflect about the mid point about the line y=x. Adding some noise and random
    phase shifts to look as much like synthetic INL data as possible from T. Brandt 2025.
    """

    n = 65536
    x = np.linspace(0, 65535, n)
    mid = (n - 1) // 2

    num_chan = 32
    inl_arrays = []

    for _ in range(num_chan):
        # Linear slope for first half
        linear_component = np.linspace(0, 3, mid+1)

        # Low-frequency sawtooth with random phase offset (0–180 degrees)
        phase_offset_deg = np.random.uniform(0, 180)
        phase_offset_rad = np.deg2rad(phase_offset_deg)
        num_humps_saw = 1.7
        phase_saw = ((num_humps_saw * x[:mid+1] / mid) * 2 * np.pi + phase_offset_rad) % (2 * np.pi)
        saw_component = 5 * (phase_saw / np.pi - 1)

        # High-frequency sine
        num_humps_sine = 4.3
        sine_cpononent = np.sin(3 * np.pi * num_humps_sine * x[:mid+1] / n)

        # Combine
        first_half_inl = linear_component + saw_component + sine_cpononent

        # diagonal symmetry
        second_half_inl = -first_half_inl[::-1]
        y_inl = np.concatenate([first_half_inl, second_half_inl])

        # Add Gaussian noise σ = 0.2
        noise = np.random.normal(0, 0.2, size=n)
        y_noisy = y_inl + noise

        inl_arrays.append(y_noisy)

    return inl_arrays

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
    from wfi_reference_pipeline.resources.make_dev_meta import MakeDevMeta
    from wfi_reference_pipeline.reference_types.integral_non_linearity.integral_non_linearity import IntegralNonLinearity

    arr - assumed here to be a numpy array that is derived in the instrument coordinate reference frame and properly
    transformed into the science coordinate reference frame for ingestion into the class.

    tmp = MakeDevMeta(ref_type='INTEGRALNONLINEARITY')
    rfp_inl = IntegralNonLinearity(meta_data=tmp.meta_integral_non_linearity,
                                ref_type_data=arr)
    rfp_inl.generate_outfile()

    Documentation and important notes. This effect comes from the Analog to Digital converters that read out the detector
    array. The dimensions of the detector are 4096x4096 with 32 A/D converters and amplifiers reading out every 128 pixels
    together at the same time. The correction implemented adjust the measured value to the actual value which varies across
    all possible values in UNIT16 from 0 to 65535. This module expects that the transformation from instrument to science 
    coordinate reference system has already taken place when the input data is passed into the class as ref_type_data.
    
    See https://roman-docs.stsci.edu/data-handbook/wfi-data-levels-and-products/coordinate-systems
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

        if ref_type_data is None and file_list is None:
            msg = "RFP is simulating the INL correction"
            print(msg)

            ref_type_data = np.asarray(simulate_inl_correction_array())

        # If INL correction arrays are provded as input into class then check everything needed
        # for the array to be valid.
        elif ref_type_data:
            # Convert to numpy array and enforce dtype in one go
            ref_type_data = np.asarray(ref_type_data, dtype=np.float64)

            # Print message only for detectors that need LR flip
            flip_lr_detectors = {
                "WFI03", "WFI06", "WFI09",
                "WFI12", "WFI15", "WFI18"
            }
            if self.meta_data.instrument_detector in flip_lr_detectors:
                msg = (
                    "RFP using input ref_type_data assumed to be transformed into the "
                    "proper science coordinate reference frame."
                )
                print(msg)

            # Validate array shape
            if ref_type_data.ndim != 2:
                raise ValueError(
                    "IntegralNonLinearity expects ref_type_data to be a 2D array "
                    "with shape (32, 65536)."
                )

            n_chan, n_val = ref_type_data.shape
            if n_chan != 32 or n_val != 65536:
                raise ValueError(
                    "Invalid INL correction array shape. "
                    f"Expected (32, 65536), got ({n_chan}, {n_val})."
                )

            # Set attributes
            self.inl_correction = ref_type_data
            self.value_array = np.linspace(0, 65535, n_val, dtype=np.uint16)

        elif file_list:
            raise ValueError(
                    "Module currently not capable to support file list input."
                    )

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
        Populate the INL table following the Roman INL reference schema.

        This method is explicit to map instrument channel number and index in the
        instrument coordinate reference frame to the science channel number and index
        in the science coorcinate reference frame. 

        Science channels are always numbered from bottom left to right for each detector
        from 1-32. The science coordinate reference frame is how the pixels on the detector
        look at the sky - after the hardware has been placed and some rotated to account
        for the position of detector electronics. 

        Instrument channels are indexed from 0-31 with the origin including detector electronics
        always in the lower left. The transformation from instrument to science coordinates is
        illustrated in https://roman-docs.stsci.edu/data-handbook/wfi-data-levels-and-products/coordinate-systems

        """
        table = {}
        for science_chan in range(32):
            key = f"science_channel_{science_chan+1:02d}"
            table[key] = {
                "instrument_channel": science_chan,
                "correction": self.inl_correction[science_chan],
            }
        return table

    def populate_datamodel_tree(self):
        """
        Build the Roman datamodel tree for the integral non-linearity reference.
        """
        inl_datamodel = rds.IntegralnonlinearityRef()
        inl_datamodel["meta"] = self.meta_data.export_asdf_meta()
        inl_datamodel["inl_table"] = self._make_inl_table()
        inl_datamodel["value"] = self.value_array

        return inl_datamodel


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

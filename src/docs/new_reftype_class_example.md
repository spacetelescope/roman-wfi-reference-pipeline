# General Module Design with Pseudo Code

In the pseudo code below, instances of `"ref type"` are intended to be filled out with the reference file type. There are two different occurrences which need modification by a user:

- **Camel case:** `RefType = Mask, Dark, Flat`
- **Lower case:** `ref_type = mask, dark, flat`

---

## Notes for Contributors

- Provide, at the end of the module docstring, some generic examples of how to execute the code, including instantiation and expected input variables.
- The `ReferenceType` base class handles many tasks for you, such as updating metadata and attributes.
- The check performed underneath the `super()` call ensures consistency:
  - You cannot instantiate a `Dark` with `ref_type = Flat` metadata.
  - This validation happens before writing the reference file.
  - RAD and RDM will prevent incorrect file writes, but this check helps catch issues earlier.

---

## Example Pseudo Code

```python
import logging
import numpy as np
import roman_datamodels.stnode as rds
from wfi_reference_pipeline.resources.wfi_{ref_type} import WFIMeta{RefType}

from ..reference_type import ReferenceType


class {RefType}(ReferenceType):
    """
    Class RefType inherits the ReferenceType() base class methods
    where static meta data for all reference file types are written.

    rfp_{ref_type} = RefType(meta_data, ref_type_data=)
    rfp_{ref_type}.make_{ref_type}_image()
    rfp_{ref_type}.generate_outfile()
    """

    def __init__(
        self,
        meta_data,
        file_list=None,
        ref_type_data=None,
        bit_mask=None,
        outfile="roman_{ref_type}.asdf",
        clobber=False,
    ):
        """
        Initialize the class with required inputs for the ReferenceType base class.

        Parameters
        ----------
        meta_data : Object
            Meta information object (converted to dictionary when writing).
        file_list : list of str, optional
            List of file paths (used in automated operations).
        ref_type_data : numpy.ndarray, optional
            Input data cube for development or non-file-based generation.
        bit_mask : numpy.ndarray, optional
            2D integer data quality mask.
        outfile : str
            Output file path (default: roman_{ref_type}.asdf).
        clobber : bool
            Overwrite existing file if True.

        See reference_type.py for additional attributes and methods.
        """

        super().__init__(
            meta_data=meta_data,
            file_list=file_list,
            ref_type_data=ref_type_data,
            bit_mask=bit_mask,
            outfile=outfile,
            clobber=clobber
        )

        # Validate metadata type
        if not isinstance(meta_data, WFIMeta{RefType}):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMeta{RefType}"
            )

        # Default description
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI {ref_type} reference file."

        logging.debug(f"Default {ref_type} reference file object: {outfile}")

        # Initialize attributes
        self.{ref_type}_image = None


# NOTE:
# This is where module-specific logic should handle input flows:
# - file_list inputs
# - 2D or 3D ref_type_data
# - other variations
# This may be refactored into a shared function in the future.


    def make_{ref_type}_image(self):
        """
        Primary method to generate the reference image array(s).
        Acts as the internal pipeline for the module.
        """
        pass


    def populate_datamodel_tree(self):
        """
        Populate the Roman data model tree with processed data.
        """
        pass

```
import os
import asdf
import crds
import numpy as np
import roman_datamodels.stnode as rds
from astropy.units import Quantity


class UpdateReferences:
    """
    A class to update Roman Space Telescope reference files to the latest data model.
    This is meant to only update files from the latest context from the CRDS server
    set in a users environment varilable.

    CRDS_SERVER_URL="https://roman-crds-test.stsci.edu" or CRDS_SERVER_URL="https://roman-crds-tvac.stsci.edu"
    """

    #TODO add more reference file types to this list here.
    ALLOWED_REF_TYPES = {"DARK", "GAIN", "READNOISE", "SATURATION", "PHOTOM"}

    def __init__(self, ref_type, input_dir, output_dir, file_suffix="_updated"):
        """
        Initialize the UpdateReferences class.

        ref_type: str;
            Type of reference file to update (e.g., DARK, GAIN, etc.). All caps.
        input_dir: str;
            Directory where the old reference files are located.
        output_dir: str; 
            Directory where updated files will be saved.
        file_suffix: str, default = "_updated";
            Suffix to append to updated file names.
        """
        if ref_type not in self.ALLOWED_REF_TYPES:
            raise ValueError(f"Invalid ref_type '{ref_type}'. Must be one of {self.ALLOWED_REF_TYPES}.")

        self.ref_type = ref_type
        self.input_dir = input_dir.rstrip("/") + "/"
        self.output_dir = output_dir.rstrip("/") + "/"
        self.file_suffix = file_suffix

        # Get the default context from the CRDS server and the associated pmap
        self.context = crds.get_default_context()
        self.pmap = crds.rmap.asmapping(self.context)

    def get_old_files(self):
        """Retrieve the list of old reference files from CRDS."""
        ref_map = self.pmap.get_imap('wfi').get_rmap(self.ref_type.lower())
        return ref_map.reference_names()

    def process_files(self, appended_description=None):
        """
        Process and update all reference files of the specified type.

        appended_description: str, default=None; 
            Custom string to append to the description metadata.
        """
        old_files = self.get_old_files()
        for old_file in old_files:
            self.generate_updated_file(old_file, appended_description)

    def generate_updated_file(self, old_file, appended_description=" Updated to latest roman data model."):
        """
        Update an individual reference file to the latest data model.

        old_file: str;
            Name of the old reference file.
        appended_description: str, default = " Updated to latest roman data model.";
            String to append to the description metadata.
        """
        old_path = os.path.join(self.input_dir, old_file)

        with asdf.open(old_path) as old_af:
            old_meta = old_af.tree['roman']['meta']
            
            # Ensure appended_description is a string.
            if not isinstance(appended_description, str):
                appended_description = " Updated to latest roman data model."

            # Create a copy of metadata.
            updated_meta = old_meta.copy()
            updated_meta["description"] += appended_description

            # Create new data model.
            datamodel_tree = self.create_datamodel(old_af, updated_meta)

            # Generate new filename with specified suffix.
            new_filename = old_file.replace(".asdf", f"{self.file_suffix}.asdf")
            new_path = os.path.join(self.output_dir, new_filename)

            # Write updated file.
            new_af = asdf.AsdfFile()
            new_af.tree = {'roman': datamodel_tree}
            new_af.write_to(new_path)
            new_af.close()

            # Set file permissions.
            os.chmod(new_path, 0o666)
            print(f"Created: {new_path}")

    def create_datamodel(self, old_af, updated_meta):
        """
        Create the appropriate data model for the specified reference type.

        old_af: str;
            ASDF file object of the old reference file.
        updated_meta: dict;
            Updated metadata for the new file.
        
        return: New reference data model tree.

        NOTE: This method extracts `.value` to remove units from arrays for Build 17 compatibility.
        Additional note: This might be where one would edit the code to do something different when
        updating the reference file, such as populating a new array, renaming an existing tree
        element.
        """
        #TODO add additional reference file types here. All have meta data, but the use of data, 
        # coefficients, etc. are reference file type specific and set in the schema from Roman
        # attribute dictionary. 
        if self.ref_type == "DARK":
            datamodel = rds.DarkRef()
            datamodel["data"] = old_af.tree['roman']['data'].value
            datamodel["dq"] = old_af.tree['roman']['dq']
            datamodel["dark_slope"] = old_af.tree['roman']['dark_slope'].value
            datamodel["dark_slope_error"] = old_af.tree['roman']['dark_slope_error'].value
        elif self.ref_type == "GAIN":
            datamodel = rds.GainRef()
            datamodel["data"] = old_af.tree['roman']['data'].value
        elif self.ref_type == "READNOISE":
            datamodel = rds.ReadnoiseRef()
            datamodel["data"] = old_af.tree['roman']['data'].value
        elif self.ref_type == "SATURATION":
            datamodel = rds.SaturationRef()
            datamodel["data"] = old_af.tree['roman']['data'].value
            datamodel["dq"] = old_af.tree['roman']['dq']
        elif self.ref_type == "PHOTOM":
            datamodel = rds.WfiImgPhotomRef()
            phot_table = old_af.tree['roman']['phot_table']
            new_dict = {}
            for filt, phot_values in phot_table.items():
                # Create an inner dictionary for each filter
                new_dict[filt] = {}
                for key, val in phot_values.items():
                    # Check if the value is a Quantity object
                    if isinstance(val, Quantity):
                        new_dict[filt][key] = float(val.value)  # Extract the numeric value from the Quantity
                    elif isinstance(val, np.float64):
                        new_dict[filt][key] = float(val)  # Convert np.float64 to a plain float
                    else:
                        new_dict[filt][key] = val  # Keep other values as they are
            datamodel['phot_table'] = new_dict
        else:
            raise ValueError(f"Unhandled reference type: {self.ref_type}")

        datamodel["meta"] = updated_meta
        return datamodel
    
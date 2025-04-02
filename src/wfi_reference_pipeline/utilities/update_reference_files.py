import os
import asdf
import crds
import roman_datamodels.stnode as rds

class UpdateReferences:
    """
    A class to update Roman Space Telescope reference files to the latest data model.
    """

    ALLOWED_REF_TYPES = {"DARK", "GAIN", "READNOISE", "SATURATION"}

    def __init__(self, ref_type, input_dir, output_dir, file_suffix="_updated"):
        """
        Initialize the UpdateReferences class.

        :param ref_type: Type of reference file to update (e.g., DARK, GAIN, etc.).
        :param input_dir: Directory where the old reference files are located.
        :param output_dir: Directory where updated files will be saved.
        :param file_suffix: Suffix to append to updated file names (default: "_updated").
        """
        if ref_type not in self.ALLOWED_REF_TYPES:
            raise ValueError(f"Invalid ref_type '{ref_type}'. Must be one of {self.ALLOWED_REF_TYPES}.")

        self.ref_type = ref_type
        self.input_dir = input_dir.rstrip("/") + "/"
        self.output_dir = output_dir.rstrip("/") + "/"
        self.file_suffix = file_suffix

        self.context = crds.get_default_context()
        self.pmap = crds.rmap.asmapping(self.context)

    def get_old_files(self):
        """
        Retrieve the list of old reference files from CRDS.

        :return: List of old reference file names.
        """
        ref_map = self.pmap.get_imap('wfi').get_rmap(self.ref_type.lower())
        return ref_map.reference_names()

    def process_files(self, appended_description=None):
        """
        Process and update all reference files of the specified type.
        """
        old_files = self.get_old_files()
        for old_file in old_files:
            self.generate_updated_file(old_file, appended_description)

    def generate_updated_file(self, old_file, appended_description=" Updated to latest roman data model."):
        """
        Update an individual reference file to the latest data model.

        :param old_file: Name of the old reference file.
        :param appended_description: String to append to the description metadata (default: " Updated to latest data model").
        """
        old_path = os.path.join(self.input_dir, old_file)

        with asdf.open(old_path, copy_arrays=True) as old_af:
            old_meta = old_af.tree['roman']['meta']
            old_meta["description"] += appended_description

            # Create new data model
            datamodel_tree = self.create_datamodel(old_af, old_meta)

            # Generate new filename with specified suffix
            new_filename = old_file.replace(".asdf", f"{self.file_suffix}.asdf")
            new_path = os.path.join(self.output_dir, new_filename)

            # Write updated file
            new_af = asdf.AsdfFile()
            new_af.tree = {'roman': datamodel_tree}
            new_af.write_to(new_path)
            new_af.close()

            # Set file permissions
            os.chmod(new_path, 0o666)
            print(f"Created: {new_path}")

    def create_datamodel(self, old_af, old_meta):
        """
        Create the appropriate data model for the specified reference type.

        :param old_af: ASDF file object of the old reference file.
        :param old_meta: Updated metadata for the new file.
        :return: New reference data model tree.

        NOTE - The code below only access the data values in the quantity array object and 
        ignores the unit to support the removal of units in roman data models in B17 for
        these reference file types. 
        """
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
        else:
            raise ValueError(f"Unhandled reference type: {self.ref_type}")

        datamodel["meta"] = old_meta
        return datamodel
    
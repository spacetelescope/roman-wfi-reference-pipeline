import os

import yaml

from wfi_reference_pipeline.utilities.multi_accumulation_tables.ma_table_xml_reader import (
    MATableReader,
)


class RefPipeMATableConfig:
    """
    A class to configure and generate reference pipeline MA tables configuration files.

    Example usage:
    from wfi_reference_pipeline.utilities.multi_accumulation_tables.rfp_ma_tables import RefPipeMATableConfig
    rfp_config = RefPipeMATableConfig()
    rfp_config._get_rfp_ma_tables_info()
    #TODO not completely figured out to make ma_table_config.yml
    rfp_config.write_rfp_ma_tables_to_txt() - not ideal for config yml file upload
    rfp_config.make_rfp_ma_table_config() - not working to maintain nested list of lists - NEEDED FOR DARK RESAMPLING
    """
    def __init__(self,
                 output_dir="/grp/roman/rcosenti/RFP_git_clone/"
                            "wfi_reference_pipeline/src/wfi_reference_pipeline/config",
                 output_file="rfp_ma_table_config.yml"):
        """
        Initializes the RefPipeMATableConfig with the specified output directory and file name.

        Parameters
        ----------
        output_dir (str): The directory where the configuration files will be saved. Default is the specified path.
        output_file (str): The name of the YAML configuration file. Default is 'rfp_ma_table_config.yml'.
        """
        self.output_dir = output_dir
        self.output_file = output_file
        self.rfp_ma_tables = None

    def _get_rfp_ma_tables_info(self):
        """
        Retrieves all RFP MA table information using the MATableReader.

        This method sets the rfp_ma_tables attribute with the retrieved information.
        """
        ma_tables = MATableReader()
        self.rfp_ma_tables = ma_tables.get_all_rfp_ma_table_info()

    def make_rfp_ma_table_config(self):
        """
        Generates a YAML configuration file containing the RFP MA table information.

        This method retrieves the MA table information, creates the output directory if it doesn't exist,
        and writes the information to a YAML file.
        """
        self._get_rfp_ma_tables_info()
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Write rfp_tables to the YAML file
        output_path = os.path.join(self.output_dir, self.output_file)
        with open(output_path, "w") as yaml_file:
            yaml.dump(self.rfp_ma_tables, yaml_file, default_flow_style=False, sort_keys=False)

    def write_rfp_ma_tables_to_txt(self):
        """
        Generates a text file containing the RFP MA table information.

        This method retrieves the MA table information, creates the output directory if it doesn't exist,
        and writes the information to a text file.
        """
        self._get_rfp_ma_tables_info()
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Write rfp_tables to a text file
        txt_output_file = "rfp_ma_tables.txt"
        txt_output_path = os.path.join(self.output_dir, txt_output_file)
        with open(txt_output_path, "w") as txt_file:
            txt_file.write(str(self.rfp_ma_tables))


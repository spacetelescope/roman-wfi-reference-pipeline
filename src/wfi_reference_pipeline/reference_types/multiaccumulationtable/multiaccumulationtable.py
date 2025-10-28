#import logging #uncomment if ever used

import asdf
import roman_datamodels.stnode as rds
import yaml

from wfi_reference_pipeline.resources.wfi_meta_multiaccumulationtable import (
    WFIMetaMultiAccumulationTable,
)

from ..reference_type import ReferenceType


class MultiAccumulationTable(ReferenceType):
    """
    Class MultiAccumulationTable() inherits the ReferenceType() base class methods where static meta data for all reference file types are written.
    """

    def __init__(self, meta_data, file_list, outfile='roman_matable.asdf', clobber=False):
        """
        The __init__ method initializes the class with proper input variables needed by the ReferenceType()
        base class.

        Parameters
        ----------
        meta_data: Object; default = None
            Object of meta information converted to dictionary when writing reference file.
        outfile: string; default = roman_matable.asdf
            Filename with path for saved inverse linearity reference file.
        clobber: Boolean; default = False
            True to overwrite the file name outfile if file already exists. False will not overwrite and exception
            will be raised if duplicate file is found.
        """

        # Access methods of base class ReferenceType
        super().__init__(meta_data, file_list=file_list, outfile=outfile, clobber=clobber)

        # Default meta creation for module specific ref type.
        if not isinstance(meta_data, WFIMetaMultiAccumulationTable):
            raise TypeError(
                f"Meta Data has reftype {type(meta_data)}, expecting WFIMetaMultiAccumulationTable"
            )
        if len(self.meta_data.description) == 0:
            self.meta_data.description = "Roman WFI multi-accumulation table reference file."

        # Ensure only one file name is passed to the open statement
        if len(self.file_list) > 1:
            raise Warning("Multiple files were provided in file list to MultiAccumulationTable(). Using first file.")
        self.ma_table_file = self.file_list[0]

    def generate_multi_accumulation_table_dict(self):
        """
        Create the dictionary of all required quantities needed for the MATABLE reference file. 
        """
        # open the yaml file
        with open(self.ma_table_file, 'r') as f:
            ma_tables_dict = yaml.safe_load(f)

        assert len(ma_tables_dict.keys()) == 2
        top_keys = list(ma_tables_dict.keys())
        return top_keys, [ma_tables_dict[key] for key in top_keys]

    def populate_asdf_tree(self):
        """
        Create the asdf element tree before the datamodel is created. 
        
        *** After the datamodel is created, we can delete this function. ***

        """
        matable_asdf_tree = dict()
        matable_asdf_tree['meta'] = self.meta_data.export_asdf_meta()
        keys, sub_dicts = self.generate_multi_accumulation_table_dict()
        for key, sub_dict in zip(keys, sub_dicts):
            matable_asdf_tree[key] = sub_dict

        return matable_asdf_tree

    def populate_datamodel_tree(self):
        """
        Create data model from DMS and populate tree.
        """
        # Construct the dark object from the data model.
        matable_datamodel_tree = rds.MatableRef()
        matable_datamodel_tree['meta'] = self.meta_data.export_asdf_meta()
        key, sub_dicts = self.generate_multi_accumulation_table_dict()
        for key, sub_dict in zip(key, sub_dicts):
            matable_datamodel_tree[key] = sub_dict

        return matable_datamodel_tree

    def save_multi_accumulation_table(self, datamodel_tree=None):
        """
        The method save_aperture_correction writes the reference file object to the specified asdf outfile.
        """
        # Use data model tree if supplied. Else write tree from module.
        af = asdf.AsdfFile()
        if datamodel_tree:
            af.tree = {'roman': datamodel_tree}
        else:
            af.tree = {'roman': self.populate_datamodel_tree()}

        # check to see if file currently exists
        self.check_outfile()
        af.write_to(self.outfile)

    # not needed for MA table files
    def calculate_error(self):
        return super().calculate_error()
    def update_data_quality_array(self):
        return super().update_data_quality_array()

"""Find and read an MA table from an XML file, exporting it into a class.

This module contains the class structure to extract a given MA table from a given XML file.

From WFI_Tools repo package developed by W. Schultz and modified and expanded for the RFP by R. Cosentino.

"""
from xml.etree import ElementTree
from pathlib import Path


class MATableReader:
    """Extract specified MA table from the provided XML file."""

    def __init__(self,
                 xml_filepath='./files/multi_accum_tables.xml',
                 ma_table_type='science'):
        self.xml_filepath = xml_filepath
        self.ma_table_type = ma_table_type

        # Verify xml_filepath using pathlib
        xml_path = Path(self.xml_filepath)
        if not xml_path.exists():
            raise IOError("Please check that xml_filepath exists.")

        # Ensure ma_table_type is 'science' or 'guide window'
        if self.ma_table_type not in ['science', 'guide window']:
            raise ValueError("Please ensure ma_table_type is 'science' or 'guide window'.")

        self.ma_table_number = None
        self.ma_table_name = None
        self.__name__ = None
        self.resultants = []

    def get_table_from_xml(self, ma_table_number=None, ma_table_name=None):
        """Extract XML element tree for requested MA table from the XML file.

        Also ensures that the file contains the correct tables and identifies which type of MA table is being requested.
        """

        self.ma_table_number = ma_table_number
        self.ma_table_name = ma_table_name

        # Ensure ma_table_number or ma_table_name is provided
        if (self.ma_table_number is None) and (self.ma_table_name is None):
            raise ValueError('Please specify either an MA table number (ma_table_number) or an MA table name (ma_table_name).')

        # Open the XML and create element tree
        tree = ElementTree.parse(self.xml_filepath)
        root = tree.getroot()

        # Confirm file contains MA tables
        if root.tag != "multi_accum_tables":
            raise ValueError(f"Can only read official multi_accum_tables_#.xml files. Expected 'multi_accum_tables' but got '{root.tag}'.")

        if len(root) <= 0:
            raise ValueError(f"{self.xml_filepath} contains no tables. Please use correct file.")

        # Parse tree to find the element corresponding to the desired table
        if self.ma_table_type == 'science':
            tables = root[1]
            self.__name__ = 'MAScienceTable'
        else:
            tables = root[2]
            self.__name__ = 'MAGuideWindowTable'

        if self.ma_table_number is not None:
            table_numbers = [int(elem.text) for t in tables for elem in t if elem.tag == 'ma_table_number']
            if self.ma_table_number not in table_numbers:
                raise ValueError("Please ensure ma_table_number is in the specified XML file.")
            ma_table = tables[table_numbers.index(self.ma_table_number)]
        else:
            table_names = [elem.text for t in tables for elem in t if elem.tag == 'ma_table_name']
            if self.ma_table_name not in table_names:
                raise ValueError("Please ensure ma_table_name is spelled correctly and is in the specified XML file.")
            ma_table = tables[table_names.index(self.ma_table_name)]

            if isinstance(ma_table, list):
                table_numbers = [int(elem.text) for t in ma_table for elem in t if elem.tag == 'ma_table_number']
                ma_table = ma_table[table_numbers.index(max(table_numbers))]

        # Save the table instance
        self.xml_table = ma_table
        self.create_table_class()

    def convert_xml_str_to_dtype(self, string):
        """Convert XML string values to the correct datatype."""
        try:
            value = float(string)
            if value % 1 == 0:
                return int(value)
            else:
                return value
        except ValueError:
            return string

    def create_table_class(self):
        """Parse through the XML table and save attributes to this class."""
        self.resultants = []  # Reset the resultants
        for elem in self.xml_table:
            if elem.tag == 'resultant':
                new_resultant = ResultantClass()
                for res_elem in elem:
                    elem_value = self.convert_xml_str_to_dtype(res_elem.text)
                    setattr(new_resultant, res_elem.tag, elem_value)
                self.resultants.append(new_resultant)
            else:
                elem_value = self.convert_xml_str_to_dtype(elem.text)
                setattr(self, elem.tag, elem_value)

    def __repr__(self):
        """Produce pretty print statements without recursion."""
        d = self.__dict__.copy()
        rstr = f"{self.__name__}(" if self.__name__ else "MATableReader("
        for k, v in d.items():
            if k == 'resultants':
                continue
            if k[0] != '_':
                rstr += f"{k}={v}, "
        if 'resultants' in d.keys():
            rstr += "resultants=[\n"
            for res in d['resultants']:
                rstr += f"\t{res},\n"
            rstr += ']'
        if rstr[-1] == " ":
            rstr = rstr[:-2]
        return rstr + ")"

    def count_tables(self):
        """Counts the number of tables in the specified XML file based on the ma_table_type."""
        # Open the XML and create element tree
        tree = ElementTree.parse(self.xml_filepath)
        root = tree.getroot()

        # Confirm the file contains MA tables
        if root.tag != "multi_accum_tables":
            raise ValueError(f"Expected 'multi_accum_tables' but got '{root.tag}'.")

        # Determine which type of tables to count
        if self.ma_table_type == 'science':
            tables = root[1]  # Assuming the second element contains science tables
        else:
            tables = root[2]  # Assuming the third element contains guide window tables

        return len(tables)

    def get_rfp_ma_table_info(self, ma_table_number):
        """Retrieve detailed information for the specified MA table number."""
        self.get_table_from_xml(ma_table_number=ma_table_number)
        rfp_ma_table_info = {
            "ma_table_number": self.ma_table_number,
            "ma_table_name": self.ma_table_name,
            "observing_mode": getattr(self, "observing_mode", None),
            "reset_reads": getattr(self, "reset_reads", None),
            "num_resultants": len(self.resultants),
            "read_pattern": []  # Initialize the read_pattern list
        }
        # Now make the read pattern
        rp = []
        current_frame = 1  # Initialize the current frame counter
        for resultant in self.resultants:
            read_frames = resultant.read_frames
            pre_resultant_skips = resultant.pre_resultant_skips

            # Generate the sequence of integers for this resultant
            resultant_sequence = list(range(current_frame, current_frame + read_frames))
            rp.append(resultant_sequence)

            # Update the current frame counter for the next resultant
            current_frame += read_frames + pre_resultant_skips

        rfp_ma_table_info['read_pattern'] = rp

        return rfp_ma_table_info

    def get_all_rfp_ma_table_info(self):
        """Retrieve detailed information for all MA tables and compile into a single dictionary."""
        rfp_ma_tables = {}
        num_tables = self.count_tables()
        for i in range(1, num_tables + 1):
            table_info = self.get_rfp_ma_table_info(ma_table_number=i)
            rfp_ma_tables[i] = table_info
        return rfp_ma_tables


class ResultantClass:
    """Initialize empty class to be used to store Resultant information."""

    def __repr__(self):
        """Produce pretty print statements without recursion."""
        d = self.__dict__.copy()
        rstr = f"{self.__class__.__name__}("
        for k, v in d.items():
            if k[0] != '_':
                rstr += f"{k}={v}, "
        if rstr[-1] == " ":
            rstr = rstr[:-2]
        return rstr + ")"


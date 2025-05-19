"""
Utility classes and functions for the WFI_REFERENCE_PIPELINE project
"""

class FilenameParser:
    """
        Extract level one naming convention information
            Level 1 Roman ASDF files (*_uncal.asdf ) comprising different Roman full focal plane exposures. Each of the 18 detectors is stored in a separate ASDF file. File names follow the Roman file naming convention, i.e.:

            r{PPPPP}{XX}{ppp}{SSS}{OOO}{VVV}_{gg}{s}{aa}_{EEEE}_WFI{NN}_uncal.asdf
            where the values in brackets correspond to:

            PPPPP   Program ID number
            XX	    Execution plan
            ppp     Pass
            SSS	    Segment
            OOO	    Observation
            VVV	    Visit ID number
            gg	    Visit file group
            s	    Visit file sequence
            aa	    Visit file activity
            EEEE	Exposure number
            NN	    WFI SCA number

    """

    def __init__(self, filename):
        filename = str(filename)
        self.program_id = filename[1:6]
        self.execution_plan = filename[6:8]
        self.pass_num = filename[8:11]
        self.segment = filename[11:14]
        self.observation = filename[14:17]
        self.visit_id_number = filename[17:20]
        # '_' changes index
        self.visit_file_group = filename[21:23]
        self.visit_file_sequence = filename[23:24]
        self.visit_file_activity = filename[24:26]
        # '_' changes index
        self.exposure_number = filename[27:31]
        # '_WFI' changes index
        self.wfi_sci_number = filename[35:37]
import asdf
import os
from pathlib import Path
from datetime import datetime, date
from romancal import step as rcal_steps
import romancal.pipeline as rcal_pipes
from ..constants import WFI_DETECTORS, WFI_REF_OPTICAL_ELEMENTS, WFI_TYPES

class ParameterFile():
    """
    Class ParameterFile() contains methods to create a new / or read in an existing Romancal parameter file for a provided step.
    """

    # define some constant lists of parameters
    COMMON_PARAMETERS = ['input_dir', 'output_dir', 'output_ext', 'output_file', 'output_use_index', 'output_use_model', 'post_hooks', 'pre_hooks', 'save_results', 'search_output_file', 'skip', 'suffix']
    UPDATE_METADATA_KEYS = ['author', 'useafter', 'pedigree', 'description']

    # gather lists of romancal steps/pipelines and suffixes
    ALL_STEP_NAMES = rcal_steps.__all__ 
    ALL_STEP_SUFFIXES = [step_name.lower() for step_name in ALL_STEP_NAMES]

    ALL_PIPE_NAMES = ['ExposurePipeline', 'MosaicPipeline']
    ALL_PIPE_SUFFIXES = [pipe_name.lower() for pipe_name in ALL_PIPE_NAMES]

    def __init__(self, parameter_filename):
        """
        The __init__ method initialized the class using the input variables.

        Parameters
        ----------
        parameter_filename: string
            Name of the parameter file to be created or simply the name of the step to create the file for (e.g. 'pars-jumpstep', 'pars-rampfitstep', 'pars-exposurepipeline')
        """
        # check that the desired Romancal step is with a configuration file of the supported steps
        self.file_suffix = self.check_file_is_implemented(parameter_filename)
        if 'pipeline' in self.file_suffix:
            self.is_pipeline = True
        else:
            self.is_pipeline = False
        self.input_filepath = Path(parameter_filename)
        self.romancal_class = self.get_romancal_class()
        
        # create the initial default information
        default_output_list = self.load_defaults()
        if self.is_pipeline:
            self.class_string, self.name_string, self.default_parameters, self.unique_parameter_keys, self.default_steps, self.step_names = default_output_list
        else:
            self.class_string, self.name_string, self.default_parameters, self.unique_parameter_keys = default_output_list
        self.default_meta = self.create_initial_meta()
        self.meta = self.default_meta.copy()

        # if the input filepath exists and has parameters in it, update the parameter attributes
        self.history_entries = []
        self._last_input_history = None
        if self.input_filepath.exists():
            self.check_and_unpack_input_file()
        
        # once finished with the import, print the parameters
        self.print_parameters()

    def check_file_is_implemented(self, filename):
        """
        Extract the parameter file suffix from the filename and ensure that the suffix matches an expected parameter file name.

        Parameters
        ----------
        filename: string
            The input filepath (or string) that was input to the class.
        """
        if ('step' in filename):
            possible_suffixes = self.ALL_STEP_SUFFIXES
        elif ('pipeline' in filename):
            possible_suffixes =self.ALL_PIPE_SUFFIXES
        else:
            raise ValueError('Expected a parameter filename for either a step or pipeline. Please confirm that the filename is in the same format as what is expected in CRDS.')
        
        # identify in any of the suffixes match
        matched_suffix_list = [suffix for suffix in possible_suffixes if suffix in filename]

        # ensure only one suffix matches
        if len(matched_suffix_list) == 1:
            return matched_suffix_list[0]
        elif len(matched_suffix_list) > 1:
            raise NotImplementedError("More than one instance of the file has been implemented. Please fix.")
        else:
            raise NotImplementedError("The provided file has not yet been implemented.")
        
    def get_romancal_class(self):
        """
        Parse the identified file suffix for the step or pipeline and match it to the corresponding Romancal class.
        """
        if self.is_pipeline:
            pipe_class_name = self.ALL_PIPE_NAMES[self.ALL_PIPE_SUFFIXES.index(self.file_suffix)]
            pipe_class = getattr(rcal_pipes, pipe_class_name)
            return pipe_class()
        else:
            step_class_name = self.ALL_STEP_NAMES[self.ALL_STEP_SUFFIXES.index(self.file_suffix)]
            step_class = getattr(rcal_steps, step_class_name)
            return step_class()

    def load_defaults(self):
        """
        Import the default values for the desired Romancal class from their native config file generation. Save the parameters as class attributes.
        """
        # write to temporary asdf file
        tmp_filename = f'pars-{self.file_suffix}_tmp_config.asdf' #TODO: think of a better filenaming convention for the temp file in case several are written to the same location at the same time.
        self.romancal_class.export_config(tmp_filename)

        # open temporary asdf file and save default values to the class
        with asdf.open(tmp_filename, 'r') as config_file:
            class_string = config_file['class']
            name_string = config_file['name']
            default_parameters = config_file['parameters']
            if self.is_pipeline:
                default_steps = config_file['steps']

        # delete the temporary asdf file
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)

        # save parameters and default values to class attributes
        for key,value in default_parameters.items():
            setattr(self, key, value)

        unique_parameter_keys = list(set(default_parameters.keys()) - set(self.COMMON_PARAMETERS))

        if self.is_pipeline:
            step_names = []
            for step in default_steps:
                setattr(self, step['name'], step)
                step_names.append(step['name'])
            return class_string, name_string, default_parameters, unique_parameter_keys, default_steps, step_names
        else:
            return class_string, name_string, default_parameters, unique_parameter_keys

    def create_initial_meta(self):
        """
        Construct a common default metadata dictionary for each parameter file.
        """
        default_meta = {
            "author": "Staff",
            "date": datetime.today().isoformat(),
            "description": f"Parameter File for WFI Romancal {self.name_string}.",
            "instrument": {
                "name": "WFI",
                "detector": "WFI01",
                "p_detector": "|".join(WFI_DETECTORS),
                "optical_element": "F062",
                "p_optical_element": "|".join(WFI_REF_OPTICAL_ELEMENTS),
                },
            "pedigree": "DEFAULT",
            "origin" : "STSCI",
            "reftype": f"pars-{self.file_suffix}",
            "telescope": "ROMAN",
            "title": f"{self.name_string} Parameters",
            "exposure": {
                "p_exptype": "|".join(WFI_TYPES), 
                "type": "WFI_IMAGE",
                },
            "useafter": "2020-01-01T00:00:00"
        }
        return default_meta

    def check_and_unpack_input_file(self):
        """
        Use the input file to update the class parameters and metadata.

        The input file must be an asdf file and contain the relevant elements. This function is designed to be used with a previous copy of a parameter file made by this code or Romancal itself. Using a different file format may not work and may lead to unexpected errors.
        """
        # ensure the file is an asdf file
        if self.input_filepath.suffix == '.asdf':
            with asdf.open(self.input_filepath, 'r') as input_file:
                input_keys = input_file.keys()
                # check to make sure the file is for the correct step using both the class and name
                if 'class' in input_keys:
                    if input_file['class'] != self.class_string:
                        raise ValueError(f"Input file does not have the same class name as the desired class. Expected: {self.class_string}. Received: {input_file['class']}")
                else:
                    raise Warning('Provided file does not contain a class identifier.')
                
                if 'name' in input_keys:
                    if input_file['name'] != self.name_string:
                        raise ValueError(f"Input file does not have the same name as the desired class. Expected: {self.name_string}. Received: {input_file['name']}")
                else:
                    raise Warning('Provided file does not contain a name identifier.')
                
                # parse the input metadata, if possible, and update vital metadata
                if 'meta' in input_keys:
                    for key, val in input_file['meta'].items():
                        if key in ['instrument', 'exposure']: # these are the metadata keys that might not need to be updated 
                            if isinstance(val, type(self.meta[key])):
                                if isinstance(val, dict):
                                    for subkey, subval in val.items():
                                        if (subkey in self.meta[key].keys()) and (subval != self.meta[key][subkey]):
                                            self.meta[key][subkey] = subval
                                            print(f'Value of meta[{key}][{subkey}] was updated using the input file.')
                                else:
                                    self.meta[key] = val
                                    print(f'Value of meta[{key}] was updated using the input file.')
                            else:
                                raise Warning(f'Type mismatch found between self.meta[{key}] and value from input file. Not updating the value.')
                # parse the input parameters, if possible, and update the class attributes 
                if 'parameters' in input_keys:
                    for key, val in input_file['parameters'].items():
                        if key in self.default_parameters.keys():
                            setattr(self, key, val)
                        else:
                            raise Warning(f'{key} specified in input file and not in expected parameters.')
                        
                # parse the input steps, if possible, and update the class attributes and dictionaries
                if 'steps' in input_keys:
                    for step in input_file['steps']:
                        step_name = step['name']
                        if step_name in self.step_names:
                            step_attr = getattr(self, step_name)
                            if step['class'] == step_attr['class']:
                                for key, val in step['parameters'].items():
                                    if key in step_attr['parameters'].keys():
                                        step_attr['parameters'][key] = val
                                    else:
                                        raise Warning(f'{key} specified in {step_name} step in input file and not in expected step parameters.')
                                setattr(self, step_name, step_attr)
                            else:
                                raise Warning(f'input file class name {step['class']} does not match expected step class name {step_attr['class']} for provided pipeline file.')
                        else:
                            raise Warning(f'Got an unexpected step {step['name']} in the {self.name_string} input file.')
                        
                # save the history of the file, if possible, to add to the next saved file when writing
                if 'history' in input_keys:
                    if 'entries' in input_file['history'].keys():
                        self.history_entries = input_file['history']['entries']
                        self._last_input_history = self.history_entries[-1]
        else:
            print("Unable to open input file as it is not an ASDF file. Not using information stored therein.")

    def print_parameters(self, unique=True):
        """
        Prints the values of the currently stored parameters into the terminal

        Parameters
        ----------
        unique: Boolean; default = True
            If True, the function will only print the step-specific parameters. If False, all parameters are printed.
        """
        if unique:
            param_keys = self.unique_parameter_keys
            print('Listing all step/pipeline specific parameters:')
        else:
            param_keys = self.default_parameters.keys()
            print('Listing all parameters:')
        for param_name in sorted(param_keys):
            print(f"{self.__class__.__name__}.{param_name} = {getattr(self, param_name)}")
    
    def write(self, history_entry_message, output_filepath='', use_input_filepath=False, as_dev_file=True):
        """
        Save the parameter file with the correct schema after performing some final metadata checks.

        All step specific parameters will be saved under 'parameters', but all non-default parameter names will be saved in 'non-default_parameters'.
        
        Note: CRDS automatically renames all the files they receive upon publishing them so the output filepath for these files is only for tracking them internally.

        Parameters
        ----------
        history_entry_message: string
            Message used for versioning to keep track of how the file has evolved. Should be a short and descriptive explanation of the changes.
        output_filepath: string; default = ''
            If specified, the provided path will be used to save the file. 
        use_input_filepath: Boolean; default = False
            If True, reuse the input_filepath to the class as the output filepath.
        as_dev_file: Boolean; default = True
            If True, output_filepath is blank, and use_input_filepath is False, append '_dev' to the end of the file name.
        """
        # check for a correct history message
        if (not isinstance(history_entry_message, str)) or (len(history_entry_message) < 1):
            raise ValueError("Please specify a history message to add to the parameter file.")
        
        # initialize the new asdf file
        asdf_file = asdf.AsdfFile()
        asdf_file.tree = {
                        'class': self.class_string,
                        'name':  self.name_string,
                    }
        # check that metadata is updated
        self.meta['date'] = datetime.today().isoformat()
        for key in self.UPDATE_METADATA_KEYS:
            if self.meta[key] == self.default_meta[key]:
                raise ValueError(f'Value of ParameterFile.meta["{key}"] was not updated from the default value. Please ensure all of the following keys are updated: {self.UPDATE_METADATA_KEYS}.')

        asdf_file['meta'] = self.meta

        # update the history entries
        asdf_file['history'] = {'entries': self.history_entries}
        asdf_file.add_history_entry(history_entry_message)
        
        # check if any parameters have non-default values
        parameters = dict()
        for key, val in sorted(self.default_parameters.items()):
            param_attribute = getattr(self, key)
            # if the parameter changed, add it to parameters dictionary
            if param_attribute != val:
                parameters[key] = param_attribute

        asdf_file['parameters'] = parameters

        # handle the steps in pipeline parameter files
        if self.is_pipeline:
            steps = []
            for step_name in self.step_names:
                step = getattr(self, step_name)
                # only save the skip boolean if it is true
                if step['parameters']['skip']:
                    steps.append({
                            'class': step['class'],
                            'name': step['name'],
                            'parameters': {'skip': True},
                        })
                    
            asdf_file['steps'] = steps

        # determine what the output filepath should be
        if use_input_filepath:
            output_filepath = self.input_filepath
        elif output_filepath == '':
            if as_dev_file:
                dev_str = '_dev'
            else:
                dev_str = ''
            date_str = date.today().strftime('%Y%m%d')
            output_filepath = f'./roman_wfi_pars-{self.file_suffix}_{date_str}{dev_str}.asdf'
        else: # If output_filepath is specified by a user
            if not Path(output_filepath).exists():
                raise ValueError(f'Provided output_filepath does not exists. output_filepath = {output_filepath}.')
            
        # save the file
        asdf_file.write_to(output_filepath)

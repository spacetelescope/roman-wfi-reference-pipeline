from romancal.dq_init import DQInitStep




class DarkPipeline():
    """

    """

    def __init__(self, config_file=None, input_dir=None, output_dir=None):
        pass

        # If no output file name given, set default file name.
        self.config_file = config_file
        self.input_dir = input_dir
        self.output_dir = output_dir

    def dark_pipeline_steps(self):
        """

        """

        result = DqInitStep.call(tmp)




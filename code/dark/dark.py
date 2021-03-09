class Dark:
    """
    Base class for dark file creation. We can modify this for DCL
    data after we have the basic algorithm defined.
    """

    def __init__(self, data_file, config):

        self.input_file = data_file
        self.configuration = config

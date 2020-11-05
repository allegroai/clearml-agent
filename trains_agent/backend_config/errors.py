class ConfigurationError(Exception):

    def __init__(self, msg, file_path=None, *args):
        """
        Initialize the message.

        Args:
            self: (todo): write your description
            msg: (str): write your description
            file_path: (str): write your description
        """
        super(ConfigurationError, self).__init__(msg, *args)
        self.file_path = file_path

from trains_agent.commands.base import ServiceCommandSection


class Config(ServiceCommandSection):

    def __init__(self, *args, **kwargs):
        """
        Initialize the configuration.

        Args:
            self: (todo): write your description
        """
        super(Config, self).__init__(*args, only_load_config=True, **kwargs)

    def config(self, **_):
        """
        Return a configuration object.

        Args:
            self: (todo): write your description
            _: (int): write your description
        """
        return self._session.print_configuration()

    def init(self, **_):
        """
        Initialize a new application.

        Args:
            self: (todo): write your description
            _: (int): write your description
        """
        # alias config init
        from .config import main
        return main()

    @property
    def service(self):
        """ The name of the REST service used by this command """
        return 'config'

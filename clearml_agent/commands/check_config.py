from clearml_agent.commands.base import ServiceCommandSection


class Config(ServiceCommandSection):

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, only_load_config=True, **kwargs)

    def config(self, **_):
        return self._session.print_configuration()

    def init(self, **_):
        # alias config init
        from .config import main
        return main()

    @property
    def service(self):
        """ The name of the REST service used by this command """
        return 'config'

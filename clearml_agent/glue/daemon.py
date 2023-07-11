from threading import Thread
from clearml_agent.session import Session


class K8sDaemon(Thread):

    def __init__(self, agent):
        super(K8sDaemon, self).__init__(target=self.target)
        self.daemon = True
        self._agent = agent
        self.log = agent.log
        self._session: Session = agent._session

    def target(self):
        pass

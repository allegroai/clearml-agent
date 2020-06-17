from typing import Text

from .base import PackageManager
from .requirements import SimpleSubstitution


class HorovodRequirement(SimpleSubstitution):

    name = "horovod"

    def __init__(self, *args, **kwargs):
        super(HorovodRequirement, self).__init__(*args, **kwargs)
        self.post_install_req = None

    def match(self, req):
        # match both horovod
        return req.name and self.name == req.name.lower()

    def post_install(self, session):
        if self.post_install_req:
            PackageManager.out_of_scope_install_package(self.post_install_req.tostr(markers=False))
            self.post_install_req = None

    def replace(self, req):
        """
        Replace a requirement
        :raises: ValueError if version is pre-release
        """
        # Store in post req install, and return nothing
        self.post_install_req = req
        # mark skip package, we will install it in post install hook
        return Text('')

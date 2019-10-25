from typing import Text

from .base import PackageManager
from .requirements import SimpleSubstitution


class CythonRequirement(SimpleSubstitution):

    name = "cython"

    def __init__(self, *args, **kwargs):
        super(CythonRequirement, self).__init__(*args, **kwargs)

    def match(self, req):
        # match both Cython & cython
        return self.name == req.name.lower()

    def replace(self, req):
        """
        Replace a requirement
        :raises: ValueError if version is pre-release
        """
        # install Cython before
        PackageManager.out_of_scope_install_package(str(req))
        return Text(req)

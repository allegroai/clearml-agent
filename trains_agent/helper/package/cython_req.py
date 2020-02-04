from typing import Text

from .base import PackageManager
from .requirements import SimpleSubstitution


class CythonRequirement(SimpleSubstitution):

    name = ("cython", "numpy", )

    def __init__(self, *args, **kwargs):
        super(CythonRequirement, self).__init__(*args, **kwargs)

    def match(self, req):
        # match both Cython & cython
        return req.name and req.name.lower() in self.name

    def replace(self, req):
        """
        Replace a requirement
        :raises: ValueError if version is pre-release
        """
        # install Cython before
        PackageManager.out_of_scope_install_package(str(req))
        return Text(req)

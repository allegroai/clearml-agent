from typing import Text

from .base import PackageManager
from .requirements import SimpleSubstitution


class PriorityPackageRequirement(SimpleSubstitution):

    name = ("cython", "numpy", "setuptools", )
    optional_package_names = tuple()

    def __init__(self, *args, **kwargs):
        super(PriorityPackageRequirement, self).__init__(*args, **kwargs)
        # check if we need to replace the packages:
        priority_packages = self.config.get('agent.package_manager.priority_packages', None)
        if priority_packages:
            self.__class__.name = priority_packages
        priority_optional_packages = self.config.get('agent.package_manager.priority_optional_packages', None)
        if priority_optional_packages:
            self.__class__.optional_package_names = priority_optional_packages

    def match(self, req):
        # match both Cython & cython
        return req.name and (req.name.lower() in self.name or req.name.lower() in self.optional_package_names)

    def replace(self, req):
        """
        Replace a requirement
        :raises: ValueError if version is pre-release
        """
        if req.name in self.optional_package_names:
            # noinspection PyBroadException
            try:
                if PackageManager.out_of_scope_install_package(str(req)):
                    return Text(req)
            except Exception:
                pass
            return Text('')
        PackageManager.out_of_scope_install_package(str(req))
        return Text(req)

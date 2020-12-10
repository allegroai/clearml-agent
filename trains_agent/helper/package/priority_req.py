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


class PackageCollectorRequirement(SimpleSubstitution):
    """
    This RequirementSubstitution class will allow you to have multiple instances of the same
    package, it will output the last one (by order) to be actually used.
    """
    name = tuple()

    def __init__(self, session, collect_package):
        super(PackageCollectorRequirement, self).__init__(session)
        self._collect_packages = collect_package or tuple()
        self._last_req = None

    def match(self, req):
        # match package names
        return req.name and req.name.lower() in self._collect_packages

    def replace(self, req):
        """
        Replace a requirement
        :raises: ValueError if version is pre-release
        """
        self._last_req = req.clone()
        return ''

    def post_scan_add_req(self):
        """
        Allows the RequirementSubstitution to add an extra line/requirements after
        the initial requirements scan is completed.
        Called only once per requirements.txt object
        """
        last_req = self._last_req
        self._last_req = None
        return last_req

from typing import Text

from .base import PackageManager
from .requirements import SimpleSubstitution


class PostRequirement(SimpleSubstitution):

    name = ("horovod", )
    optional_package_names = tuple()

    def __init__(self, *args, **kwargs):
        super(PostRequirement, self).__init__(*args, **kwargs)
        self.post_install_req = []
        # check if we need to replace the packages:
        post_packages = self.config.get('agent.package_manager.post_packages', None)
        if post_packages:
            self.__class__.name = post_packages
        post_optional_packages = self.config.get('agent.package_manager.post_optional_packages', None)
        if post_optional_packages:
            self.__class__.optional_package_names = post_optional_packages

    def match(self, req):
        # match both horovod
        return req.name and (req.name.lower() in self.name or req.name.lower() in self.optional_package_names)

    def post_install(self, session):
        for req in self.post_install_req:
            if req.name in self.optional_package_names:
                # noinspection PyBroadException
                try:
                    PackageManager.out_of_scope_install_package(req.tostr(markers=False))
                except Exception:
                    pass
            else:
                PackageManager.out_of_scope_install_package(req.tostr(markers=False))

        self.post_install_req = []

    def replace(self, req):
        """
        Replace a requirement
        :raises: ValueError if version is pre-release
        """
        # Store in post req install, and return nothing
        self.post_install_req.append(req)
        # mark skip package, we will install it in post install hook
        return Text('')

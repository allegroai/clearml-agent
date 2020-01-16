from collections import OrderedDict
from typing import Text

from .base import PackageManager
from .requirements import SimpleSubstitution


class ExternalRequirements(SimpleSubstitution):

    name = "external_link"

    def __init__(self, *args, **kwargs):
        super(ExternalRequirements, self).__init__(*args, **kwargs)
        self.post_install_req = []
        self.post_install_req_lookup = OrderedDict()

    def match(self, req):
        # match both editable or code or unparsed
        if not (not req.name or req.req and (req.req.editable or req.req.vcs)):
            return False
        if not req.req or not req.req.line or not req.req.line.strip() or req.req.line.strip().startswith('#'):
            return False
        return True

    def post_install(self):
        post_install_req = self.post_install_req
        self.post_install_req = []
        for req in post_install_req:
            try:
                freeze_base = PackageManager.out_of_scope_freeze() or ''
            except:
                freeze_base = ''
            PackageManager.out_of_scope_install_package(req.tostr(markers=False), "--no-deps")
            try:
                freeze_post = PackageManager.out_of_scope_freeze() or ''
                package_name = list(set(freeze_post['pip']) - set(freeze_base['pip']))
                if package_name and package_name[0] not in self.post_install_req_lookup:
                    self.post_install_req_lookup[package_name[0]] = req.req.line
            except:
                pass
            PackageManager.out_of_scope_install_package(req.tostr(markers=False), "--ignore-installed")

    def replace(self, req):
        """
        Replace a requirement
        :raises: ValueError if version is pre-release
        """
        # Store in post req install, and return nothing
        self.post_install_req.append(req)
        # mark skip package, we will install it in post install hook
        return Text('')

    def replace_back(self, list_of_requirements):
        if 'pip' in list_of_requirements:
            original_requirements = list_of_requirements['pip']
            list_of_requirements['pip'] = [r for r in original_requirements
                                           if r not in self.post_install_req_lookup]
            list_of_requirements['pip'] += [self.post_install_req_lookup.get(r, '')
                                            for r in self.post_install_req_lookup.keys() if r in original_requirements]
        return list_of_requirements

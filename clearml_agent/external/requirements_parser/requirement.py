from __future__ import unicode_literals
import re
from pkg_resources import Requirement as Req

from .fragment import get_hash_info, parse_fragment, parse_extras_require
from .vcs import VCS, VCS_SCHEMES


URI_REGEX = re.compile(
    r'^(?P<scheme>https?|file|ftps?)://(?P<path>[^#]+)'
    r'(#(?P<fragment>\S+))?'
)

VCS_REGEX = re.compile(
    r'^(?P<scheme>{0})://'.format(r'|'.join(
        [scheme.replace('+', r'\+') for scheme in VCS_SCHEMES])) +
    r'((?P<login>[^/@]+)@)?'
    r'(?P<path>[^#@]+)'
    r'(@(?P<revision>[^#]+))?'
    r'(#(?P<fragment>\S+))?'
)

VCS_EXT_REGEX = re.compile(
    r'^(?P<scheme>{0})(@)'.format(r'|'.join(
        [scheme.replace('+', r'\+') for scheme in ['git+git']])) +
    r'((?P<login>[^/@]+)@)?'
    r'(?P<path>[^#@]+)'
    r'(@(?P<revision>[^#]+))?'
    r'(#(?P<fragment>\S+))?'
)

# This matches just about everyting
LOCAL_REGEX = re.compile(
    r'^((?P<scheme>file)://)?'
    r'(?P<path>[^#]+)' +
    r'(#(?P<fragment>\S+))?'
)


class Requirement(object):
    """
    Represents a single requirement from clearml_agent.external.requirements_parser.requirement import Requirement

    Typically instances of this class are created with ``Requirement.parse``.
    For local file requirements, there's no verification that the file
    exists. This class attempts to be *dict-like*.

    See: http://www.pip-installer.org/en/latest/logic.html

    **Members**:

    * ``line`` - the actual requirement line being parsed
    * ``editable`` - a boolean whether this requirement is "editable"
    * ``local_file`` - a boolean whether this requirement is a local file/path
    * ``specifier`` - a boolean whether this requirement used a requirement
      specifier (eg. "django>=1.5" or "requirements")
    * ``vcs`` - a string specifying the version control system
    * ``revision`` - a version control system specifier
    * ``name`` - the name of the requirement
    * ``uri`` - the URI if this requirement was specified by URI
    * ``subdirectory`` - the subdirectory fragment of the URI
    * ``path`` - the local path to the requirement
    * ``hash_name`` - the type of hashing algorithm indicated in the line
    * ``hash`` - the hash value indicated by the requirement line
    * ``extras`` - a list of extras for this requirement
      (eg. "mymodule[extra1, extra2]")
    * ``specs`` - a list of specs for this requirement
      (eg. "mymodule>1.5,<1.6" => [('>', '1.5'), ('<', '1.6')])
    """

    def __init__(self, line):
        # Do not call this private method
        self.line = line
        self.editable = False
        self.local_file = False
        self.specifier = False
        self.vcs = None
        self.name = None
        self.subdirectory = None
        self.uri = None
        self.path = None
        self.revision = None
        self.hash_name = None
        self.hash = None
        self.extras = []
        self.specs = []

    def __repr__(self):
        return '<Requirement: "{0}">'.format(self.line)

    def __getitem__(self, key):
        return getattr(self, key)

    def keys(self):
        return self.__dict__.keys()

    @classmethod
    def parse_editable(cls, line):
        """
        Parses a Requirement from an "editable" requirement which is either
        a local project path or a VCS project URI.

        See: pip/req.py:from_editable()

        :param line: an "editable" requirement
        :returns: a Requirement instance for the given line
        :raises: ValueError on an invalid requirement
        """

        req = cls('-e {0}'.format(line))
        req.editable = True
        vcs_match = VCS_REGEX.match(line) or VCS_EXT_REGEX.match(line)
        local_match = LOCAL_REGEX.match(line)

        if vcs_match is not None:
            groups = vcs_match.groupdict()
            if groups.get('login'):
                req.uri = '{scheme}://{login}@{path}'.format(**groups)
            else:
                req.uri = '{scheme}://{path}'.format(**groups)
            req.revision = groups['revision']
            if groups['fragment']:
                fragment = parse_fragment(groups['fragment'])
                egg = fragment.get('egg')
                req.name, req.extras = parse_extras_require(egg)
                req.hash_name, req.hash = get_hash_info(fragment)
                req.subdirectory = fragment.get('subdirectory')
            for vcs in VCS:
                if req.uri.startswith(vcs):
                    req.vcs = vcs
        else:
            assert local_match is not None, 'This should match everything'
            groups = local_match.groupdict()
            req.local_file = True
            if groups['fragment']:
                fragment = parse_fragment(groups['fragment'])
                egg = fragment.get('egg')
                req.name, req.extras = parse_extras_require(egg)
                req.hash_name, req.hash = get_hash_info(fragment)
                req.subdirectory = fragment.get('subdirectory')
            req.path = groups['path']

        return req

    @classmethod
    def parse_line(cls, line):
        """
        Parses a Requirement from a non-editable requirement.

        See: pip/req.py:from_line()

        :param line: a "non-editable" requirement
        :returns: a Requirement instance for the given line
        :raises: ValueError on an invalid requirement
        """

        req = cls(line)

        vcs_match = VCS_REGEX.match(line) or VCS_EXT_REGEX.match(line)
        uri_match = URI_REGEX.match(line)
        local_match = LOCAL_REGEX.match(line)

        if vcs_match is not None:
            groups = vcs_match.groupdict()
            if groups.get('login'):
                req.uri = '{scheme}://{login}@{path}'.format(**groups)
            else:
                req.uri = '{scheme}://{path}'.format(**groups)
            req.revision = groups['revision']
            if groups['fragment']:
                fragment = parse_fragment(groups['fragment'])
                egg = fragment.get('egg')
                req.name, req.extras = parse_extras_require(egg)
                req.hash_name, req.hash = get_hash_info(fragment)
                req.subdirectory = fragment.get('subdirectory')
            for vcs in VCS:
                if req.uri.startswith(vcs):
                    req.vcs = vcs
        elif uri_match is not None:
            groups = uri_match.groupdict()
            req.uri = '{scheme}://{path}'.format(**groups)
            if groups['fragment']:
                fragment = parse_fragment(groups['fragment'])
                egg = fragment.get('egg')
                req.name, req.extras = parse_extras_require(egg)
                req.hash_name, req.hash = get_hash_info(fragment)
                req.subdirectory = fragment.get('subdirectory')
            if groups['scheme'] == 'file':
                req.local_file = True
        elif '#egg=' in line:
            # Assume a local file match
            assert local_match is not None, 'This should match everything'
            groups = local_match.groupdict()
            req.local_file = True
            if groups['fragment']:
                fragment = parse_fragment(groups['fragment'])
                egg = fragment.get('egg')
                name, extras = parse_extras_require(egg)
                req.name = fragment.get('egg')
                req.hash_name, req.hash = get_hash_info(fragment)
                req.subdirectory = fragment.get('subdirectory')
            req.path = groups['path']
        else:
            # This is a requirement specifier.
            # Delegate to pkg_resources and hope for the best
            req.specifier = True
            pkg_req = Req.parse(line)
            req.name = pkg_req.unsafe_name
            req.extras = list(pkg_req.extras)
            req.specs = pkg_req.specs
        return req

    @classmethod
    def parse(cls, line):
        """
        Parses a Requirement from a line of a requirement file.
        This is the main entry point for parsing a single requirements line (not parse_line!)

        :param line: a line of a requirement file
        :returns: a Requirement instance for the given line
        :raises: ValueError on an invalid requirement
        """
        line = line.lstrip()
        if line.startswith('-e') or line.startswith('--editable'):
            # Editable installs are either a local project path
            # or a VCS project URI
            return cls.parse_editable(
                re.sub(r'^(-e|--editable=?)\s*', '', line))
        elif '@' in line and ('#' not in line or line.index('#') > line.index('@')):
            # ClearML bug fix: support 'name @ git+' entries
            name, uri = line.split('@', 1)
            name = name.strip()
            uri = uri.strip()
            # noinspection PyBroadException
            try:
                # check if the name is valid & parsed
                Req.parse(name)
                # if we are here, name is a valid package name, check if the vcs part is valid
                if VCS_REGEX.match(uri) or VCS_EXT_REGEX.match(uri):
                    req = cls.parse_line(uri)
                    req.name = name
                    return req
                elif URI_REGEX.match(uri):
                    req = cls.parse_line(uri)
                    req.name = name
                    req.line = line
                    return req
            except Exception:
                pass

        return cls.parse_line(line)

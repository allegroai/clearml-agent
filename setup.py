"""
TRAINS - Artificial Intelligence Version Control
TRAINS-AGENT DevOps for machine/deep learning
https://github.com/allegroai/trains-agent
"""

import os.path
# Always prefer setuptools over distutils
from setuptools import setup, find_packages

def read_text(filepath):
    """
    Read text from file

    Args:
        filepath: (str): write your description
    """
    with open(filepath, "r") as f:
        return f.read()

here = os.path.dirname(__file__)
# Get the long description from the README file
long_description = read_text(os.path.join(here, 'README.md'))


def read_version_string(version_file):
    """
    Read the version string from version string.

    Args:
        version_file: (str): write your description
    """
    for line in read_text(version_file).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


version = read_version_string("trains_agent/version.py")

requirements = read_text(os.path.join(here, 'requirements.txt')).splitlines()

setup(
    name='trains_agent',
    version=version,
    description='TRAINS Agent - Auto-Magical DevOps for Deep Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    # The project's main homepage.
    url='https://github.com/allegroai/trains-agent',
    author='Allegroai',
    author_email='trains@allegro.ai',
    license='Apache License 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Intended Audience :: System Administrators',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: System :: Logging',
        'Topic :: System :: Monitoring',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: Apache Software License',
    ],

    keywords='trains devops machine deep learning agent automation hpc cluster',

    packages=find_packages(exclude=['contrib', 'docs', 'data', 'examples', 'tests*']),

    install_requires=requirements,
    extras_require={
    },
    package_data={
         'trains_agent': ['backend_api/config/default/*.conf']
    },
    include_package_data=True,
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': ['trains-agent=trains_agent.__main__:main'],
    },
)

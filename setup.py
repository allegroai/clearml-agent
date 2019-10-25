"""
TRAINS - Artificial Intelligence Version Control
TRAINS-AGENT DevOps for machine/deep learning
https://github.com/allegroai/trains-agent
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from six import exec_
from pathlib2 import Path


here = Path(__file__).resolve().parent

# Get the long description from the README file
long_description = (here / 'README.md').read_text()


def read_version_string():
    result = {}
    exec_((here / 'trains_agent/version.py').read_text(), result)
    return result['__version__']


version = read_version_string()

requirements = (here / 'requirements.txt').read_text().splitlines()


setup(
    name='trains_agent',
    version=version,
    description='Trains-Agent DevOps for deep learning (DevOps for TRAINS)',
    long_description=long_description,
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

#!/usr/bin/env python
from sys import argv, exit

if "upl" in argv[1:]:
    import os
    os.system("python setup.py register -r pypi")
    os.system("python setup.py sdist upload -r pypi")
    exit()


from distutils.core import setup
setup(
    name = 'exospoc',
    packages = ['exospoc'],
    version = '0.1',
    description = 'Provides astronomy ephemeris to plan telescope observations',
    author = 'Guillaume Schworer',
    author_email = 'guillaume.schworer@obspm.fr',
    url = 'https://github.com/ceyzeriat/exospoc',
    download_url = 'https://github.com/ceyzeriat/exospoc/tree/master/dist',
    keywords = ['astronomy','ephemeris','pyephem','iobserve','observatory','telescope','observer','target','airmass'], # arbitrary keywords
    classifiers = ['Development Status :: 4 - Beta','Environment :: Console','Intended Audience :: Education','Intended Audience :: Science/Research','License :: OSI Approved :: MIT License','Operating System :: Unix','Programming Language :: Python :: 2.7','Topic :: Documentation :: Sphinx', 'Topic :: Scientific/Engineering :: Physics'],
    install_requires=['pyephem','pytz']
    #package_data={'': ['obsData.txt']},
    #include_package_data=True
)

# http://peterdowns.com/posts/first-time-with-pypi.html

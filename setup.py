#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if __name__ == "__main__":
    setup(
        name='py3do',
        version='0.0.1',
        description ='py3do - Construct models for 3d printing in Python',
        author='Szymon Jaroszewicz',
        author_email='jszymon@gmail.com',
        license='GNU General Public License V.3 or later',
        url='https://github.com/jszymon/py3do',
        long_description=open('README.md').read(),
        python_requires=">=3.7",
        install_requires=['numpy>=1.18',
                          'matplotlib>=3.0.0',
                          'scipy>=1.4.0',
                          'shapely>=1.7.0',],
        
        packages=['py3do'],
)

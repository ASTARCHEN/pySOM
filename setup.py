#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages
from .pysom import __version__
setup(
    name="pySOM",
    version=__version__,
    description=(
        'Self-organizing Maps,SOM for python'
    ),
    long_description=open('README.rst').read(),
    author='A.Star',
    author_email='chenxiaolong12315@163.com',
    maintainer='A.Star',
    maintainer_email='chenxiaolong12315@163.com',
    license='MIT License',
    packages=find_packages(),
    platforms=["all"],
    url='https://gitee.com/hoops/pySOM',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries'
    ],
	install_requires=['matplotlib>=1.5.1',
                'numpy>=1.12.0',
                'scipy>=0.17.1']
)
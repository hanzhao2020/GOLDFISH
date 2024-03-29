import os
import codecs
from setuptools import setup

base_dir = os.path.abspath(os.path.dirname(__file__))
def read(fname):
    return codecs.open(os.path.join(base_dir, fname), encoding="utf-8").read()
    
setup(
    name='GOLDFISH',
    version='0.1',
    packages=['GOLDFISH'],
    url='https://github.com/hanzhao2020/GOLDFISH',
    license='GNU LGPLv3',
    author='Han Zhao',
    author_email='',
    description="Gradient-based Optimization, Large-scale Design Framework for Isogeometric SHells",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
)
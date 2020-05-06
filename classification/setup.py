from setuptools import setup
import sys
import os
import aws_bert_classification

install_requires = [
    'torch',
    'sagemaker',
    'pytest-runner',
    'numpy',
    'transformer',
    'pytorch-nlp',
    'pandas'
]

tests_requires = [
    'pytest',
    
]

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name=aws_bert_classification.__title__,
    version=aws_bert_classification.__version__,
    description=aws_bert_classification.__summary__,
    long_description=read('README.md'),
    license=aws_bert_classification.__license__,
    url=aws_bert_classification.__uri__,
    author=aws_bert_classification.__author__,
    author_email=aws_bert_classification.__email__,
    packages=['aws_adfs_auth'],
    install_requires=install_requires,
    tests_require=tests_requires,
    extras_requires={},
    data_files=[("", ["LICENSE"])],
    keywords="",
    test_suite='tests.aws_adfs_auth_test_suite',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Library',
        'Intended Audience :: Data Scientists',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX',
        'Topic :: Utilities',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
)

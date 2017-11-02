# from distutils.core import setup
from setuptools import setup, find_packages


setup(name='n_monomers',
    version='1.0',
    description='Explicit solutions of polymer equations',
    author='Oscar Ortega',
    author_email='oscar.ortega@vanderbilt.edu',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['pysb', 'numpy', 'sympy', 'pandas'],
    keywords=['systems', 'biology', 'model', 'polymer'],
    )


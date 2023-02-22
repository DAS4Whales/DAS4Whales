from setuptools import setup

from das4whales import __version__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='das4whales',
    version=__version__,
    description='Distributed acoustic sensing analysis tools for Bioacoustics',
    long_description=long_description,
    url='https://github.com/leabouffaut/DAS4Whales',
    author='Léa Bouffaut',
    author_email='lea.bouffaut@cornell.edu',
    license='Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International',
    py_modules=['das4whales'],
    install_requires=['requirements'],
)

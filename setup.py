from setuptools import setup

from das4whales import __version__

setup(
    name='das4whales',
    version=__version__,
    description='Machine Learning for Bioacoustics',
    url='https://github.com/leabouffaut/DAS4Whales',
    author='Léa Bouffaut',
    author_email='lea.bouffaut@cornell.edu',
    license='Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International',

    py_modules=['das4whales'],
)
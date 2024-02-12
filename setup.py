from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='das4whales',
    version='0.1.0',
    description='Distributed acoustic sensing analysis tools for Bioacoustics',
    long_description=long_description,
    url='https://github.com/leabouffaut/DAS4Whales',
    author='Léa Bouffaut',
    author_email='lea.bouffaut@cornell.edu',
    license='Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International',
    py_modules=['das4whales.data_handle', 'das4whales.dsp', 'das4whales.plot'],
    packages=['das4whales'],
    install_requires=[
        'h5py',
        'numpy',
        'DateTime',
        'scipy',
        'librosa',
        'matplotlib',
        'wget'],
)

# Import the setup() and find_packages() functions from setuptools,
# which are used to define and find your Python package for installation.
from setuptools import setup, find_packages

# Call the setup() function to tell Python how to install your package.
setup(
    name='piano_ai',  # The name of your package (what people will install with pip)
    version='0.1.0',  # The version of your package (update this when you make changes)
    author='Hadrien, Victor, Oumou, Anton, Jonade',  # The authors of the package
    description="""A deep learning toolkit for automatic piano transcription
from audio, including feature extraction, model training,
and MIDI output.""",  # A short description of what your package does
    packages=find_packages(),  # Automatically find all folders with __init__.py to include as packages
)

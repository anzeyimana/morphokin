from setuptools import setup, find_packages

VERSION = '1.0.0'
DESCRIPTION = 'MorphoKIN'
LONG_DESCRIPTION = 'Kinyarwanda Morphology Toolkit: Morphological Analysis, Synthesis, Tokenization and Text Normalization'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="morphokin",
    version=VERSION,
    author="Antoine Nzeyimana",
    author_email="<nzeyi@kinlp.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        "minineedle",
        "numpy"
    ],
    keywords=['python', 'morphokin', 'kinyarwanda', 'nlp', 'morphology'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Research and Development",
        "Programming Language :: Python :: 3",
        "Operating System :: Linux :: Linux OS",
    ]
)

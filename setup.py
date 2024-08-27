from setuptools import setup, find_packages

VERSION = "1.0.0"
DESCRIPTION = "A python module for generating thread art sequences."

with open("README.md", 'r') as f:
    LONG_DESCRIPTION = f.read()

with open("requirements.txt", 'r', encoding="utf-8") as f:
    requirements = [i.strip() for i in f.readlines()]

LICENSE = "MIT"

setup(name='threadArt',
      version=VERSION,
      description=DESCRIPTION,
      long_description_content_type="text/markdown",
      long_description=LONG_DESCRIPTION,
      license=LICENSE,
      author="Quintin Dunn",
      author_email="dunnquintin07@gmail.com",
      url="https://github.com/quintindunn/threadArt",
      packages=find_packages(),
      keywords=['art', 'string art', 'thread art'],
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Operating System :: Microsoft :: Windows :: Windows 10',
          'Programming Language :: Python :: 3',
      ],
      install_requires=requirements
      )

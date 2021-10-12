from setuptools import setup, find_packages


# with open('README.rst') as f:
#     readme = f.read()
#
# with open('LICENSE') as f:
#     license = f.read()

setup(
    name='synthobs',
    version='0.1.0',
    description='for creating synthetic observations from SPH simulations',
    # long_description=readme,
    author='Stephen Wilkins',
    author_email='s.wilkins@sussex.ac.uk',
    url='https://github.com/stephenmwilkins/SynthObs',
    # license=license,
    packages=find_packages(exclude=('examples'))
)

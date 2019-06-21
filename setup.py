from setuptools import find_packages, setup


NAME = 'oni'
VERSION = '0.1'
DESCRIPTION = """Calculator to balance machine resources in Oxygen Not Included"""
AUTHOR = 'Chris Beaumont'
EMAIL = 'chrisnbeaumont@gmail.com'

REQUIRED = [
    "pandas",
    "scipy",
    "PyYaml"
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    packages=find_packages(exclude=['tests', '__pycache__']),
    install_requires=REQUIRED,
    include_package_data=True,
    license="MIT",
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)

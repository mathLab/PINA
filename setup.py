from setuptools import setup, find_packages

meta = {}
with open("pina/meta.py") as fp:
    exec(fp.read(), meta)

# Package meta-data.
IMPORTNAME = meta['__title__']
PIPNAME = meta['__packagename__']
DESCRIPTION = 'Physic Informed Neural networks for Advance modeling.'
URL = 'https://github.com/mathLab/PINA'
MAIL = meta['__mail__']
AUTHOR = meta['__author__']
VERSION = meta['__version__']
KEYWORDS = 'physics-informed neural-network'

REQUIRED = [
    'numpy', 'matplotlib', 'torch', 'lightning'
]

EXTRAS = {
    'docs': ['sphinx', 'sphinx_rtd_theme'],
    'test': ['pytest', 'pytest-cov'],
}

LDESCRIPTION = (
    ""
)

setup(
    name=PIPNAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LDESCRIPTION,
    author=AUTHOR,
    author_email=MAIL,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    keywords=KEYWORDS,
    url=URL,
    license='MIT',
    packages=find_packages(),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    zip_safe=False,
)

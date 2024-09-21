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
    'numpy', 'matplotlib', 'torch', 'lightning', 'pytorch_lightning', 'torch_geometric', 'torch-cluster'
]

EXTRAS = {
    'docs': ['sphinx', 'sphinx_rtd_theme', 'sphinx_copybutton'],
    'test': ['pytest', 'pytest-cov'],
}

LDESCRIPTION = (
    "PINA is a Python package providing an easy interface to deal with "
    "physics-informed neural networks (PINN) for the approximation of "
    "(differential, nonlinear, ...) functions. Based on Pytorch, PINA "
    "offers a simple and intuitive way to formalize a specific problem "
    "and solve it using PINN. The approximated solution of a differential "
    "equation can be implemented using PINA in a few lines of code thanks "
    "to the intuitive and user-friendly interface."
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

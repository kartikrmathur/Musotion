<<<<<<< HEAD
from setuptools import setup, find_packages

setup(
    name='kaggle',
    version='1.5.6',
    description='Kaggle API',
    long_description=
    ('Official API for https://www.kaggle.com, accessible using a command line '
     'tool implemented in Python. Beta release - Kaggle reserves the right to '
     'modify the API functionality currently offered.'),
    author='Kaggle',
    author_email='support@kaggle.com',
    url='https://github.com/Kaggle/kaggle-api',
    keywords=['Kaggle', 'API'],
    entry_points={'console_scripts': ['kaggle = kaggle.cli:main']},
    install_requires=[
        # Restriction that urllib3's version is less than 1.25 needed to avoid
        # requests dependency problem.
        'urllib3 >= 1.21.1, < 1.25',
        'six >= 1.10',
        'certifi',
        'python-dateutil',
        'requests',
        'tqdm',
        'python-slugify'
    ],
    packages=find_packages(),
    license='Apache 2.0')
=======
from setuptools import setup, find_packages

setup(
    name='kaggle',
    version='1.5.6',
    description='Kaggle API',
    long_description=
    ('Official API for https://www.kaggle.com, accessible using a command line '
     'tool implemented in Python. Beta release - Kaggle reserves the right to '
     'modify the API functionality currently offered.'),
    author='Kaggle',
    author_email='support@kaggle.com',
    url='https://github.com/Kaggle/kaggle-api',
    keywords=['Kaggle', 'API'],
    entry_points={'console_scripts': ['kaggle = kaggle.cli:main']},
    install_requires=[
        # Restriction that urllib3's version is less than 1.25 needed to avoid
        # requests dependency problem.
        'urllib3 >= 1.21.1, < 1.25',
        'six >= 1.10',
        'certifi',
        'python-dateutil',
        'requests',
        'tqdm',
        'python-slugify'
    ],
    packages=find_packages(),
    license='Apache 2.0')
>>>>>>> e845925b6297e856d3840686af965c6579551143

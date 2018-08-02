from setuptools import setup, find_packages

setup(
    name='5Finder',
    version='1.0.0',
    packages=find_packages('5Finder'),
    package_dir={'': '5Finder'},
    include_package_data=True,
    install_requires=['pandas==0.23.0', 'unidecode==1.0.22', 'scipy==1.1.0'],

    python_requires='3.6.1',
    package_data={'': ['*.txt', '*.csv', '*md', '*rst']},


    author="Sebastián Mauricio Palacio",
    author_email="sebastian.,palacio@gmail.com",
    description="Detecting Outliers on underwriting",
    license="PSF",
    keywords="outlier deep learning underwriting",
    url="sebastian.mpalacio@gmail.com",


)
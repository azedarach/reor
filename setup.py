from setuptools import setup, find_packages


setup(
    name='reor',
    version='0.0.1',
    author='Dylan Harries',
    author_email='Dylan.Harries@csiro.au',
    description='Experiments with dimension reduction',
    long_description='',
    install_requires=[
        'cvxopt',
        'numpy',
        'scikit-learn',
        'scipy',
        'tensorflow>=2.0.0',
        'tensorflow-probability'],
    packages=find_packages('src'),
    package_dir={'':'src'},
    test_suite='tests',
    zip_safe=False
)

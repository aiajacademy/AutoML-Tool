from setuptools import setup, find_packages

setup(
    name='automl_tool',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'scikit-learn',
        'tensorflow',
        'torch',
        'hyperopt',
        'streamlit',
    ],
    entry_points={
        'console_scripts': [
            'automl=src.ui:main',
        ],
    },
)

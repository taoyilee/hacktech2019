from setuptools import setup, find_packages

REQUIRED_PACKAGES = ['cvxopt==1.2.2',
'lark-parser==0.6.5',
'lxml==4.2.5',
'matplotlib==2.2.4',
'numpy==1.15.4',
'pandas==0.23.4',
'pytest==4.0.2',
'scikit-learn==0.20.2',
'scipy==1.2.0',
'seaborn==0.9.0',
'sklearn==0.0',
'wfdb==2.2.1',
'tensorflow-gpu==1.12.0',
'Keras==2.2.4',
'xlrd==1.2.0',
'pandas==0.23.4',
'openpyxl==2.6.0',
'biosppy==0.5.1',
'configparser']

# Setup parameters for Google Cloud ML Engine
setup(name='hacktech2019',
      version='0.1',
      packages=find_packages(),
      description='Test running training on gcloud',
      author='Kenneth Stewart',
      license='Free',
      install_requires=REQUIRED_PACKAGES,
      zip_safe=False)

from setuptools import setup, find_packages

# Setup parameters for Google Cloud ML Engine
setup(name='hacktech2019',
      python_requires='>=3.5',
      version='0.1',
      packages=find_packages(),
      description='Test running training on gcloud',
      author='Kenneth Stewart',
      license='Free',
      install_requires=[
            'lxml==4.2.5',
            'matplotlib==3.0.2',
            'numpy==1.15.4',
            'pandas==0.23.4',
            'pytest==4.0.2',
            'scikit-learn==0.20.2',
            'scipy==1.2.0',
            'sklearn==0.0',
            'wfdb==2.2.1',
            'tensorflow-gpu==1.12.0',
            'Keras==2.2.4',
            'xlrd==1.2.0',
            'pandas==0.23.4',
            'openpyxl==2.6.0',
      ],
      zip_safe=False)

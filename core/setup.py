from setuptools import setup, find_packages

# Setup parameters for Google Cloud ML Engine
setup(name='core',
      version='0.1',
      packages=find_packages(),
      description='Test running training on gcloud',
      author='Kenneth Stewart',
      license='Free',
      install_requires=[
          'keras',
          'h5py',
      ],
      zip_safe=False)

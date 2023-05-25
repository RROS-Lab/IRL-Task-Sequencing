from setuptools import setup, find_packages

setup(
    name="IRL-Task-Sequencing", 
    version="0.0.1", 
    packages=find_packages(exclude=[]),
    include_package_data = True,
    license='MIT',
    description = 'Generates task sequencing policies for robotic execution of processing parts in manufacturing applications',
    author = 'Omey Manyar',
    author_email = 'manyar@usc.edu',
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/rros/IRL-Task-Sequencing',
    keywords = [
        'robotics',
        'machine learning',
        'imitation learning',
        'reward functions',
        'task sequencing'
    ],
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'seaborn'
    ],
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Researchers',
        'Topic :: Scientific/Engineering :: Robotics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)
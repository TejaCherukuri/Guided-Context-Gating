from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(filepath:str) -> List[str]:
    with open(filepath) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='gcg',
    version='0.1.0',
    description='Diabetic Retinopathy Classification based on Guided Context Gating Attention',
    author='Teja Cherukuri',
    author_email='tejakrishnacherukuri@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
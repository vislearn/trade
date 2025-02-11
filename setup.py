from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')
install_requires = [x.strip() for x in all_reqs if x.strip() != ""]

setup(
    name='trade',
    description='Code for the paper "TRADE: Transfer of Distributions between External Conditions with Normalizing Flows"',
    version='0.1dev',
    packages=find_packages(exclude=['tests']),
    license='MIT',
    author='Armand Rousselot',
    author_email='armand.rousselot@iwr.uni-heidelberg.de',
    install_requires=install_requires,
    url='https://github.com/vislearn/trade'
)
from setuptools import setup

setup(
    name='mlserver',
    version='1.0',
    packages=['src'],
    url='',
    license='',
    author='nosp',
    author_email='',
    install_requires= [
        'tensorflow', 'flask', 'numpy', 'pandas'
    ],
    description='Server with neural bot'
)

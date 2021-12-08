from setuptools import setup

setup(
    name='marmot',
    version='0.1',
    py_modules=['marmot'],
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        marmot=marmot:cli
    ''',
)
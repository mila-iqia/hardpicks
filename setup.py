from setuptools import setup, find_packages

setup(
    name='hardpicks',
    version='0.1.0',
    packages=find_packages(include=['hardpicks',
                                    'hardpicks.*']),
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'main=hardpicks.main:main'
        ],
    }
)

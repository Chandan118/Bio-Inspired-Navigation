from setuptools import setup
from pathlib import Path

package_name = 'bio_nav_data_ros'

here = Path(__file__).parent
launch_files = [str(path) for path in (here / 'launch').glob('*.py')]

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', [str(here / 'resource' / package_name)]),
        ('share/' + package_name, [str(here / 'package.xml')]),
        ('share/' + package_name + '/launch', launch_files),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Chandan Sheikder',
    author_email='chandan@example.com',
    maintainer='Chandan Sheikder',
    maintainer_email='chandan@example.com',
    description='ROS2 interface for the Bio-Inspired Navigation data generation toolkit.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'data_generation_node = bio_nav_data_ros.data_generation_node:main',
        ],
    },
)

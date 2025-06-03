import os
from glob import glob
from setuptools import setup
from setuptools import find_packages, setup

package_name = 'cdlc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share',package_name),glob('launch/launch.[pxy][yam]')),
        (os.path.join('share',package_name),glob('launch/.[pxy][yam]')),
        (os.path.join('share',package_name, 'config'),glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ivndx',
    maintainer_email='ivanuriel34@hotmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        "test_lidar = cdlc.test_lidar:main",
        "mybug = cdlc.mybug:main",
        "odometry_covariance = cdlc.odometry_covariance:main",
        "bug_2 = cdlc.bug_2:main",
        "odom_cov_2 = cdlc.odom_cov_2:main",
        "odom_cov_3 = cdlc.odom_cov_3:main",
        "bug_two = cdlc.bug_two:main",
        "metrics = cdlc.metrics:main",
        "state_machine_fl = cdlc.state_machine_fl:main",
        "sm_back_2 = cdlc.sm_back_2:main"
        ],
    },
)

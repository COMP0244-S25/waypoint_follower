import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'waypoint_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='jiaojh1994@gmail.com',
    description='Waypoint follower node with PD control and dynamic waypoint subscription',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'waypoint_follower = waypoint_follower.waypoint_follower:main',
            'path_follower = waypoint_follower.path_follower:main',
            'goalpose_follower = waypoint_follower.goalpose_follower:main',
            'odometry_conversion = waypoint_follower.odometry_conversion:main',
            'publish_eight_shape = waypoint_follower.publish_eight_shape:main',
            'publish_ellipse_shape = waypoint_follower.publish_ellipse_shape:main',
            'publish_cosince_shape = waypoint_follower.publish_cosince_shape:main'
        ],
    },
)

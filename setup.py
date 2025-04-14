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
            'publish_cosince_shape = waypoint_follower.publish_cosince_shape:main',
            'publish_shape1 = waypoint_follower.publish_shape1:main',
            'publish_shape2 = waypoint_follower.publish_shape2:main',
            'publish_shape3 = waypoint_follower.publish_shape3:main',
            'path_marking = waypoint_follower.path_marking:main',
            'path_follower_team1 = waypoint_follower.path_follower_team1:main',
            'path_follower_team10 = waypoint_follower.path_follower_team10:main',
            'path_follower_team11 = waypoint_follower.path_follower_team11:main',
            'path_follower_team12 = waypoint_follower.path_follower_team12:main',
            'path_follower_team2 = waypoint_follower.path_follower_team2:main',
            'path_follower_team3 = waypoint_follower.path_follower_team3:main',
            'path_follower_team4 = waypoint_follower.path_follower_team4:main',
            'path_follower_team5 = waypoint_follower.path_follower_team5:main',
            'path_follower_team6_0 = waypoint_follower.path_follower_team6_0:main',
            'path_follower_team6_1 = waypoint_follower.path_follower_team6_1:main',
            'path_follower_team7_0 = waypoint_follower.path_follower_team7_0:main',
            'path_follower_team7_1 = waypoint_follower.path_follower_team7_1:main',
            'path_follower_team7_2 = waypoint_follower.path_follower_team7_2:main',
            'path_follower_team8H = waypoint_follower.path_follower_team8H:main',
            'path_follower_team8M = waypoint_follower.path_follower_team8M:main',
            'path_follower_team9 = waypoint_follower.path_follower_team9:main',
        ],
    },
)

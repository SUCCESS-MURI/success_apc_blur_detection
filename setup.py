## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    scripts=['blur_detection_ros_main.py'],
    packages=['success_apc_blur_detection'],
    package_dir={'': 'src'}
)

setup(**setup_args)

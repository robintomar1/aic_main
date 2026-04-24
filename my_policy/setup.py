from setuptools import find_packages, setup

package_name = "my_policy"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="rs1",
    maintainer_email="rs1@clutterbot.com",
    description="Participant policy for the AI for Industry Challenge",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
)

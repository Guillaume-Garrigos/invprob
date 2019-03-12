from setuptools import setup

setup(
        name="inverse_problems",
        version="0.0.3",
        author="Guillaume Garrigos",
        author_email="guillaume.garrigos@gmail.com",
        packages=["inverseproblemspkg"],
        package_dir={"inverseproblemspkg":"inverseproblemspkg"},
        url="https://github.com/Guillaume-Garrigos/inverse_problems/",
        license="MIT",
        install_requires=["numpy >= 1.9"]
)
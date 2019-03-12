from setuptools import setup

setup(
        name="inverse-problems",
        version="0.0.1",
        author="guillaume garrigos",
        author_email="guillaume.garrigos@lpsm.paris",
        packages=["inverse-problems"],
        package_dir={"inverse-problems":"inverse-problems"},
        url="https://github.com/Guillaume-Garrigos/inverse-problems",
        license="MIT",
        install_requires=["numpy >= 1.8"]
)
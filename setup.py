from setuptools import setup

setup(
        name="inverse_problems",
        version="0.0.1",
        author="guillaume garrigos",
        author_email="guillaume.garrigos@lpsm.paris",
        packages=["inverse_problems"],
        package_dir={"inverse_problems": "inverse_problems"},
        url="https://github.com/Guillaume-Garrigos/inverse_problems/",
        license="MIT",
        install_requires=["numpy >= 1.8"]
)
from setuptools import setup

setup(
        name="invprob",
        version="0.0.1",
        author="Guillaume Garrigos",
        author_email="guillaume.garrigos@lpsm.paris",
        packages=["invprob"],
        package_dir={"invprob":"invprob"},
        url="https://github.com/Guillaume-Garrigos/invprob/",
        license="MIT",
        install_requires=["numpy >= 1.9"]
)
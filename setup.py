from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="invprob",
    version="0.0.13",
    author="Guillaume Garrigos",
    author_email="guillaume.garrigos@lpsm.paris",
    license="MIT",
    description="A package containing useful functions for solving inverse problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Guillaume-Garrigos/invprob/",
    packages=["invprob", ],
    classifiers=[
        "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
            "numpy >= 1.9",
            "pylab",
            "matplotlib",
            "scipy",
            "scikit-image"
    ],
    include_package_data=True,
)
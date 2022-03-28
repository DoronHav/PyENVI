import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyENVI",
    version="0.0.1",
    author="Doron Haviv",
    author_email="doron.haviv@gmail.com",
    description="Integrating scRNAseq with spatial sequencing data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
#     package_dir={"": ""},
#     packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
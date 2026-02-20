import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stc_cc",
    author_email="susyustc@gmail.com",
    description="Repository for the paper: stochastic tensor contraction for quantum chemistry",
    long_description=long_description,
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=["pyscf", "numba", "torch>=2.9", "sympy", "threadpoolctl"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

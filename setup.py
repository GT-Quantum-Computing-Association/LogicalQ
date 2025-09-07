from setuptools import setup, find_packages

setup(
    name="LogicalQ",
    version="7.1.3",
    author="GT-QCA",
    author_email="aliceandbob@gatechquantum.com",
    url="https://github.com/GT-Quantum-Computing-Association/gatech-qec-project/",
    license="LICENSE",
    description="A toolkit for quantum circuit development with built-in, generalized quantum error correction.",
    long_description=open("README.md").read(),
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "qiskit>=2.1.0",
        "qiskit-aer",
        "qiskit_addon_utils",
        "qiskit_ibm_runtime",
        "pytket",
        "pytket-qiskit",
        "pytket-quantinuum",
        "qbraid",
    ],
)


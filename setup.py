from setuptools import setup, find_packages

setup(
    name="LogicalQ",
    version="0.1.0",
    author="GT-QCA",
    author_email="aliceandbob@gatechquantum.com",
    url="https://github.com/GT-Quantum-Computing-Association/LogicalQ/",
    license="LICENSE",
    description="A toolkit for quantum circuit development with built-in, generalized quantum error mitigation, detection, and correction.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "qiskit>=2.1.0",
        "qiskit-aer",
        "qiskit-addon-utils",
        "qiskit-experiments",
        "qiskit-ibm-runtime==0.41.1",
        "pytket",
        "pytket-qiskit",
        "pytket-quantinuum",
        "qbraid",
    ],
    extras_require={
        "docs": [
            "sphinx",
            "sphinx-autoapi",
            "sphinx-rtd-theme",
            "python-markdown-math",
        ],
    },
)


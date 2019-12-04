import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="superdifferentiator-cs207", # Replace with your own username
    version="0.0.1",
    author="Qiang Fei, Jordan Turley, Shucheng Yan",
    author_email="jordan_turley@g.harvard.edu",
    description="Autodifferentiation package for CS207 at Harvard.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/super-differentiator/cs207-FinalProject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
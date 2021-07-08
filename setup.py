from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="msynth",
    version="1.0.0",
    author="Tim Blazytko",
    author_email="tim@blazytko.to",
    description="Code deobfuscation framework to simplify Mixed Boolean-Arithmetic (MBA) expressions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mrphrazer/msynth",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: POSIX :: Linux",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)

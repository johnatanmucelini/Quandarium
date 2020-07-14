import setuptools

with open("README.md", "r") as fh: 
    long_description = fh.read()
    
setuptools.setup( 
    name="quamdariu-johnatanmucelini", # Replace with your own username 
    version="0.0.1", 
    author="Johnatan Mucelini", 
    author_email="johnatan.mucelini@gmail.com", 
    description="A quantum chemistry tools package focus in data mining of atomic properties", 
    long_description=long_description, 
    long_description_content_type="text/markdown", 
    url="https://github.com/johnatanmucelini/quandarium", 
    packages=setuptools.find_packages(), 
    classifiers=[ 
        "Programming Language :: Python :: 3", 
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent", 
    ], 
    python_requires='>=3.6',
)

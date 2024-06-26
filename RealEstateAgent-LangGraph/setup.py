from setuptools import setup, find_packages

setup(
    name="real_estate",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "chromadb==0.5.3",
        "pillow==10.3.0",
        "open-clip-torch==2.24.0",
        "tqdm==4.66.4",
        "matplotlib==3.9.0",
        "pandas==2.2.2",
        "langchain==0.2.5",
        "langchain_openai==0.1.9",
        "langchain-google-genai==1.0.5",
        "langchain-core==0.2.9",
        "langchain_experimental==0.0.61",
        "langgraph==0.1.1",
        "langsmith==0.1.81"
    ],
    python_requires='==3.10.14'
)

# Llama-Langchain-Module
1 -  to activate venv

Set-ExecutionPolicy Unrestricted -Scope Process
.\.venv\Scripts\activate


2 - to install dependencies:
pip install text-generation
pip install langchain
pip install langchain-community
pip install --upgrade --quiet  llama-cpp-python
pip install langchain_experimental

clone repo:

git clone --recursive -j8 https://github.com/abetlen/llama-cpp-python.git


set parameters:

set FORCE_CMAKE=1
set CMAKE_ARGS=-DLLAMA_CUBLAS=OFF
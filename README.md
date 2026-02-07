# MCP Server for Ax

## To start with local LLM with Ollama:
```
python bridge.py
```
## To start with local LLM and python llama.cpp (to run GGUF file models):
```
python directserver.py
```
## Requirements

To run with ollama, the requirements.txt will do. 

To add GPU support for llama.cpp:

At command line: 
```
export CMAKE_ARGS="-DGGML_CUDA=on"
export FORCE_CMAKE=1
pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
```
# Whisper optimization using ORT toolchain
This folder contains a sample use case of Olive to optimize a [Whisper](https://huggingface.co/openai/whisper-tiny) model using ONNXRuntime tools.

Performs optimization pipeline:
- *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Quantized Onnx Model -> Insert Beam Search Op -> Insert Pre/Post Processing Ops -> Tune performance*

Outputs the best metrics, model, and corresponding Olive config.

## Prerequisites
### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

## To optimize Whisper model run the sample config
First, install required packages according to passes.
```
python -m olive.workflows.run --config whisper_cpu_config.json --setup
```
Then, optimize the model
```
python -m olive.workflows.run --config whisper_cpu_config.json
```
or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("whisper_cpu_config.json")
```

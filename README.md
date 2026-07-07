# slm-in-a-box

An experiment in running small language models with minimal dependencies.

A 50M parameter language model packed into a single HTML file.
Open it in Chrome. No install, no server, no internet required.

## Download

→ [supra50m_chat.html](link) (~124 MB) — save and open locally

Works on any machine (desktop, phone, etc.) with a modern web browser.

## Features

- Fully offline after download — works on a USB stick
- Adjustable temperature, top-p, top-k, max tokens, repetition penalty
- System prompt, conversation history, stop/regenerate/copy controls
- ~45 tok/s in Chrome on a modern laptop

> **Note:** The WebGPU backend option will likely freeze your browser tab.
> Stick with CPU (WASM), which is the default.

## Build It Yourself

Requires the model files from
[onnx-community/Supra-1.5-50M-Instruct-exp-ONNX](https://huggingface.co/onnx-community/Supra-1.5-50M-Instruct-exp-ONNX)
on HuggingFace (just `model_int8.onnx`, `tokenizer.json`, `config.json`,
`generation_config.json`).

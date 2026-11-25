# Chapter 05 Tracker

| File | Description | Notes |
|---------|------------------|----------------|
| 05-01.py | Convert text into a numpy enbedding vector, using an embedding model | openai, numpy |
| 05-02.py | Embeds three texts and compares them using cosine similarity | openai, numpy |
| 05-03.py | Splits text into 'chunks' | n/a |
| 05-04.py | Vector search: Using embedding, sorts 5 texts for similarity to another one | openai, numpy |
| 05-05.py |  | openai, numpy |

---
### Install the default Ollama Qwen 3 Embedding 4B (Q4 quantization = 4-bit integers)

```bash
ollama run qwen3-embedding:4b
ollama show qwen3-embedding:4b
  Model
    architecture        qwen3
    parameters          4.0B 
    context length      40960
    embedding length    2560 
    quantization        Q4_K_M
```

In Python code: MODEL_NAME = "qwen3-embedding:4b"

---
### Install the real Qwen 3 Embedding 4B (No quantization, F16 = 16-bit floating-point) in Ollama

```bash
ollama pull hf.co/Qwen/Qwen3-Embedding-4B-GGUF:F16

# Create a Modelfile that sets the full 32K context window by default
cat << 'EOF' > Modelfile
FROM hf.co/Qwen/Qwen3-Embedding-4B-GGUF:F16
PARAMETER num_ctx 32768
EOF

# Build alias
ollama create qwen3-embedding:4b-f16-32k -f Modelfile

# Tidy up $ ollama list (the weights are not removed as long as the alias is there)
ollama rm hf.co/Qwen/Qwen3-Embedding-4B-GGUF:F16

ollama show qwen3-embedding:4b-f16-32k
  Model
    architecture        qwen3    
    parameters          4.0B     
    context length      40960    
    embedding length    2560     
    quantization        F16      

  Parameters
    num_ctx    32768  
```
8 GB weigths + 4 GB Context buffer (32768 KV Cache) + 1.5 GB CUDA = 14 GB VRAM (85% of the RTX 5060 Ti 16 GB RAM )   \
In Python code: MODEL_NAME = "qwen3-embedding:4b-f16-32k"    \
   \
Why the "real" Qwen 3 Embedding 4B (no quantization, F16 weights) runs faster than the default Ollama Qwen 3 Embedding 4B (Q4 quantization)   \ 
- With F16, weights are loaded directly from GDDR7 VRAM straight into the FP16 Tensor Cores of the RTX 5060 Ti (Blackwell).
- With Q4, GPU Tensor Cores cannot natively multiply 4-bit directly. The CUDA cores must first dequantize (Q4 -> FP16) the weights and then feed them to the Tensor Cores.
- Quantization saves lots of VRAM but the dequantization math slows down execution compared to native F16 weights that go straight through.
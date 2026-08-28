**Instalación del entorno** \

```bash
conda create -n ai_agent_scratch python=3.13 -y \
conda activate ai_agent_scratch \
pip install openai
```

Para comprobar versiones ejecutar `/appendix_a/version.py`

conda create -n llm_scratch_cuda python=3.11 -y \
conda activate llm_scratch_cuda \
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia \


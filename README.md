**Instalación del entorno** \
conda create -n llm_scratch_cuda python=3.11 -y \
conda activate llm_scratch_cuda \
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia \


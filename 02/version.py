import openai
import sys, platform
import importlib.metadata
# import torch

os_info = platform.freedesktop_os_release()
ubuntu_version = os_info.get("PRETTY_NAME", "Linux")

print(f"\nOS: {ubuntu_version}")
print(f"Python: {sys.version_info.major}.{sys.version_info.minor}")
print(f"OpenAI API: {openai.__version__}")
print(f"Tavily API: {importlib.metadata.version('tavily-python')}\n")  # There is no attribute tavily.__version__ in Tavily :-(







# print("PyTorch version:", torch.__version__)
# print("Apple Silicon acceleration:", torch.backends.mps.is_available())
# print("CUDA enabled:", torch.cuda.is_available())
# if torch.cuda.is_available():
#    print("CUDA Compute Platform.", torch.version.cuda)
#    print(f"GPU: {torch.cuda.get_device_name(0)}")
# else:
#     print(f"CUDA not available, using CPU.") 


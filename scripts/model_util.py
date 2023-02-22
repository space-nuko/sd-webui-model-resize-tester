import os.path

def is_safetensors(path):
  return os.path.splitext(path)[1].lower() == '.safetensors'

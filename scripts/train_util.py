import hashlib
import safetensors.torch
from io import BytesIO

def model_hash(filename):
  """Old model hash used by stable-diffusion-webui"""
  try:
    with open(filename, "rb") as file:
      m = hashlib.sha256()

      file.seek(0x100000)
      m.update(file.read(0x10000))
      return m.hexdigest()[0:8]
  except FileNotFoundError:
    return 'NOFILE'


def calculate_sha256(filename):
  """New model hash used by stable-diffusion-webui"""
  hash_sha256 = hashlib.sha256()
  blksize = 1024 * 1024

  with open(filename, "rb") as f:
    for chunk in iter(lambda: f.read(blksize), b""):
      hash_sha256.update(chunk)

  return hash_sha256.hexdigest()


def precalculate_safetensors_hashes(tensors, metadata):
  """Precalculate the model hashes needed by sd-webui-additional-networks to
  save time on indexing the model later."""

  # Because writing user metadata to the file can change the result of
  # sd_models.model_hash(), only retain the training metadata for purposes of
  # calculating the hash, as they are meant to be immutable
  metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

  bytes = safetensors.torch.save(tensors, metadata)
  b = BytesIO(bytes)

  model_hash = addnet_hash_safetensors(b)
  legacy_hash = addnet_hash_legacy(b)
  return model_hash, legacy_hash


def addnet_hash_legacy(b):
  """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
  m = hashlib.sha256()

  b.seek(0x100000)
  m.update(b.read(0x10000))
  return m.hexdigest()[0:8]


def addnet_hash_safetensors(b):
  """New model hash used by sd-webui-additional-networks for .safetensors format files"""
  hash_sha256 = hashlib.sha256()
  blksize = 1024 * 1024

  b.seek(0)
  header = b.read(8)
  n = int.from_bytes(header, "little")

  offset = n + 8
  b.seek(offset)
  for chunk in iter(lambda: b.read(blksize), b""):
    hash_sha256.update(chunk)

  return hash_sha256.hexdigest()

import hashlib


def file_md5(file_path: str) -> str:
    """Compute the md5 hex digest of a binary file with reasonable memory usage."""
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            md5_hash.update(data)
    return md5_hash.hexdigest()


def model_id(model_name, model_pkl_file) -> str:
    """Return a model ID based on a readable name and a file hash.
    Can be used e.g. for prefixing."""
    return "{model_lower}{pkl_hash}".format(
        model_lower=model_name.lower(), pkl_hash=file_md5(model_pkl_file)[:8])

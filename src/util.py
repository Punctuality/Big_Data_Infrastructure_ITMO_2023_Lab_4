import hashlib
import os


def md5_hash(file_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()

def file_size(file_path):
    return os.stat(file_path).st_size

def head_file(file_path, n=5):
    with open(file_path, "r") as f:
        for _ in range(n):
            print(f.readline().strip())
import os
from pathlib import Path
import urllib.request
from src.graphconversion.MiniDataset import MiniDataset
import gzip
import shutil
from torch_geometric.datasets import Entities


def extract_nt_file(dataset_name):
    """Extracts .nt.gz file from PyG Entities dataset structure and places unzipped .nt in the expected raw folder."""
    dataset_name = dataset_name.lower()
    gzip_folder = Path(f"data/{dataset_name}/raw")
    output_path = Path(f"data/{dataset_name}/raw/{dataset_name}_stripped.nt")

    for file in gzip_folder.glob("*.nt.gz"):
        print(f"[Found] Compressed RDF file: {file.name}")
        os.makedirs(output_path.parent, exist_ok=True)
        with gzip.open(file, 'rb') as f_in, open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"[Extracted] {file.name} → {output_path}")
        return

    print(f"[ERROR] No .nt.gz file found in {gzip_folder}")



def download_file(url, dest_path):
    """Download file from URL to destination if not exists."""
    if not os.path.exists(dest_path):
        print(f"[Downloading] {url}")
        urllib.request.urlretrieve(url, dest_path)
        print(f"[Downloaded] → {dest_path}")
    else:
        print(f"[Exists] Skipping download → {dest_path}")

def prepare_aifb():
    print("[Preparing] AIFB dataset via PyG...")
    Entities(root="data", name="AIFB")
    extract_nt_file("aifb")

def prepare_mutag():
    print("[Preparing] MUTAG dataset via PyG...")
    Entities(root="data", name="MUTAG")
    extract_nt_file("mutag")

def prepare_mini():
    print("[Preparing] MINI synthetic dataset...")
    _ = MiniDataset()

def prepare_dataset(dataset_name):
    dataset = dataset_name.lower()
    if dataset == "aifb":
        prepare_aifb()
    elif dataset == "mutag":
        prepare_mutag()
    elif dataset == "mini":
        prepare_mini()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

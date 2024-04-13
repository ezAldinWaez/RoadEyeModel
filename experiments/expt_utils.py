import os

def make_out_dir(exptNum: str) -> str:
    OUT_DIR = f"out/{exptNum}"
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR

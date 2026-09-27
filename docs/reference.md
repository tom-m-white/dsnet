## Setup

Python 3.10, PyTorch 2.4.1 (CUDA 12.1), developed on Windows 11 with an RTX 4060.

```bash
conda create -y -n dsnet python=3.10
conda activate dsnet
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric h5py pyyaml tqdm pytest "scipy<1.14" numpy==1.23.5 ortools==9.7.2996 opencv-python==4.8.1.78 "protobuf<5" psutil threadpoolctl
```

The pinned versions matter ^^^: DSNet's original code uses `np.bool` and the pre-9.8 OR-Tools knapsack
API, both removed in newer releases. `numpy==1.23.5` and `ortools==9.7.2996` keep that code running
unmodified. `evaluate.py` itself needs only numpy, scipy, h5py and pyyaml.
Not following these pinned verisions caused errors.

### Data

Preprocessed features (h5) from the DSNet release, 1024-dim GoogLeNet pool5 features for every
15th frame, plus human annotations, KTS shot boundaries and the 5 train/test splits:

```bash
mkdir -p DSNet/datasets && cd DSNet/datasets
curl -L -o dsnet_datasets.zip https://www.dropbox.com/s/tdknvkpz1jp6iuz/dsnet_datasets.zip?dl=1
unzip dsnet_datasets.zip
```

For TVSum τ/ρ you also need the original TVSum release, which contains the per-annotator 1–5
ratings (`ydata-tvsum50-anno.tsv`). The h5 file does not: its per-annotator field is binary.

```bash
mkdir -p tvsum_original && cd tvsum_original
curl -L -O https://people.csail.mit.edu/yalesong/tvsum/tvsum50_ver_1_1.tgz   # 644 MB
tar -xzf tvsum50_ver_1_1.tgz && cd ydata-tvsum50-v1_1 && unzip ydata-tvsum50-data.zip
```

`python -m scripts.inspect_datasets` prints what is inside the h5 files and verifies the splits.
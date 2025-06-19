import os
import zipfile
import numpy as np

def parse_bpseq_file(filepath):
    """Parses a single .bpseq into (sequence, dot-bracket)"""
    bases = []
    pair_map = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            idx = int(parts[0]) - 1        # 0-based index
            base = parts[1].upper()
            pair = int(parts[2]) - 1       # 0-based pairing index
            bases.append(base)
            if pair >= 0:
                pair_map[idx] = pair

    # build dot-bracket
    dot = ['.' for _ in bases]
    used = set()
    for i, j in pair_map.items():
        if i not in used and j not in used and 0 <= i < len(bases) and 0 <= j < len(bases):
            dot[i], dot[j] = '(', ')'
            used.update((i, j))

    return ''.join(bases), ''.join(dot)


def parse_three_line_txt(filepath):
    """Parses a 3-line .txt record into (sequence, dot-bracket)"""
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    # Expecting groups of 3: header, seq, structure
    out = []
    for i in range(0, len(lines), 3):
        if i+2 < len(lines) and lines[i].startswith('>'):
            seq = lines[i+1].upper()
            struct = lines[i+2]
            if len(seq) == len(struct):
                out.append((seq, struct))
    return out


def load_from_inner_folder(main_path, name_filter=None, allowed_exts=('.bpseq', '.txt')):
    all_data = []
    for root, dirs, files in os.walk(main_path):
        for fname in files:
            if not fname.lower().endswith(allowed_exts):
                continue
            if name_filter and name_filter.lower() not in fname.lower():
                continue

            fpath = os.path.join(root, fname)

            try:
                if fname.endswith('.bpseq'):
                    seq, struct = parse_bpseq_file(fpath)
                    if set(seq).issubset({'A', 'U', 'G', 'C'}) and len(seq) == len(struct):
                        all_data.append((seq, struct))
                elif fname.endswith('.txt'):
                    for seq, struct in parse_three_line_txt(fpath):
                        if set(seq).issubset({'A', 'U', 'G', 'C'}) and len(seq) == len(struct):
                            all_data.append((seq, struct))
            except Exception as e:
                print(f"Skipping {fpath}: {e}")

    return all_data
# One-hot encoding function for RNA sequences
def one_hot_encode(sequence):
    mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
    one_hot = np.zeros((len(sequence), 4))
    for i, base in enumerate(sequence):
        if base in mapping:
            one_hot[i, mapping[base]] = 1
    return one_hot

# Convert dot-bracket structure to labels
def structure_to_labels(dotbracket):
    mapping = {'.': 0, '(': 1, ')': 2}
    return np.array([mapping[char] for char in dotbracket])

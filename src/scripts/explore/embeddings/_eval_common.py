from dataclasses import dataclass
from pathlib import Path
from csv import DictReader
import numpy as np
from sklearn.preprocessing import normalize
import json
from typing import Any
from cclang.core.storage import StorageManager

@dataclass
class EmbeddingData:
    raw: np.ndarray
    normed: np.ndarray
    centered: np.ndarray
    word2idx: dict[str, int]
    idx2word: list[str]
    tf: np.ndarray
    df: np.ndarray

    @property
    def vocab_size(self) -> int:
        return len(self.idx2word)

    @property
    def dim(self) -> int:
        return self.raw.shape[1]



def load_vocab(storage: StorageManager, vocab_path: Path)->tuple[list[str], dict[str, int], np.ndarray, np.ndarray]:
    idx2word: list[str] = []
    word2idx: dict[str, int] = {}
    tf: np.ndarray
    df: np.ndarray

    with storage.open(vocab_path, mode='r', encoding='utf-8') as f:
        reader = DictReader(f, delimiter='\t')
        tf_list = []
        df_list = []
        for row in reader:
            idx = int(row['idx'])
            idx2word.append(row['lemma'])
            word2idx[row['lemma']] = idx
            tf_list.append(int(row['tf']))
            df_list.append(int(row['df']))

        tf = np.array(tf_list)
        df = np.array(df_list)

    return idx2word, word2idx, tf, df

def load_embeddings(storage: StorageManager, vocab_path: Path, embedding_path: Path)->EmbeddingData:
    idx2word, word2idx, tf, df = load_vocab(storage, vocab_path)
    with storage.open(embedding_path, mode='rb') as f:
        embeddings = np.load(f)

    if len(idx2word) != embeddings.shape[0]:
        raise RuntimeError(f'Size mismatch between {len(idx2word)} and {embeddings.shape[0]}')

    normed = normalize(embeddings, axis=1, norm="l2")
    centered = embeddings - embeddings.mean(axis=0)
    embedding_data = EmbeddingData(
        raw = embeddings,
        normed = normed,
        centered = centered,
        word2idx = word2idx,
        idx2word = idx2word,
        tf = tf,
        df = df,
    )
    return embedding_data

def add_common_args(parser: "argparse.ArgumentParser") -> None:
    """Add --embeddings-path, --vocab-path, --output arguments."""
    import argparse  # noqa: local to avoid top-level cost

    parser.add_argument(
        "--embeddings-path",
        "-e",
        type=Path,
        required=True,
        help="Path to .npy file with V·Σ^α matrix (vocab_size × k)",
    )
    parser.add_argument(
        "--vocab-path",
        "-v",
        type=Path,
        required=True,
        help="Path to vocabulary .tsv (idx, lemma, tf, df)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output JSON path (default: stdout)",
    )

    parser.add_argument('--base-path',
                        type=Path,
                        default=Path('data'))

def write_report(report: dict[str, Any], output_path: Path | None) -> None:
    """Write JSON report to *output_path* or stdout."""
    text = json.dumps(report, indent=2, ensure_ascii=False, default=_json_default)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    else:
        print(text)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

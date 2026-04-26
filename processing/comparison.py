"""
Modul comparison — Perbandingan chain sequence antar subjek.

Pipeline:
  Chain DataFrame (dari chunking) → Filter (scenario, task, channel,
  subband, feature) → Pair-wise comparison → Consecutive match segments
  → Summary table + Detail table.

Mendukung perbandingan semua pasangan unik dari 170 subjek EEGET-ALS.
Output: tabel summary per pasangan + tabel detail segmen matching.
"""

import logging
from itertools import combinations

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Core: Find consecutive matching segments                           #
# ------------------------------------------------------------------ #

def find_consecutive_matches(seq1, seq2, min_length=2):
    """Temukan segmen bit berturut-turut yang identik antara 2 sequence.

    Kedua sequence di-truncate ke panjang terpendek agar perbandingan
    adil (fair comparison).

    Parameters
    ----------
    seq1 : str
        Bit string pertama, mis. ``"110100110"``.
    seq2 : str
        Bit string kedua.
    min_length : int
        Panjang minimum segmen berturut-turut yang dicatat (default: 2).

    Returns
    -------
    dict
        Keys:
        - ``segments`` : list[dict] — tiap dict berisi:
            - ``start_bit``    : posisi awal (1-indexed)
            - ``end_bit``      : posisi akhir (1-indexed, inclusive)
            - ``length``       : panjang segmen
            - ``matching_bits``: string bit yang cocok
        - ``compared_length``     : panjang setelah truncate
        - ``total_matching_bits`` : jumlah total bit yang ada di segmen ≥ min_length
        - ``total_identical_bits``: jumlah total bit yang identik (semua, termasuk < min_length)
        - ``match_percentage``    : persentase bit matching (segmen ≥ min_length)
        - ``identity_percentage`` : persentase bit identik total
    """
    if not seq1 or not seq2:
        return {
            "segments": [],
            "compared_length": 0,
            "total_matching_bits": 0,
            "total_identical_bits": 0,
            "match_percentage": 0.0,
            "identity_percentage": 0.0,
        }

    # Truncate ke panjang terpendek
    n = min(len(seq1), len(seq2))
    s1 = seq1[:n]
    s2 = seq2[:n]

    # Scan bit-by-bit
    segments = []
    current_start = None
    total_identical = 0

    for i in range(n):
        if s1[i] == s2[i]:
            total_identical += 1
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                seg_len = i - current_start
                if seg_len >= min_length:
                    segments.append({
                        "start_bit": current_start + 1,  # 1-indexed
                        "end_bit": i,                     # 1-indexed, inclusive
                        "length": seg_len,
                        "matching_bits": s1[current_start:i],
                    })
                current_start = None

    # Flush segmen terakhir
    if current_start is not None:
        seg_len = n - current_start
        if seg_len >= min_length:
            segments.append({
                "start_bit": current_start + 1,
                "end_bit": n,
                "length": seg_len,
                "matching_bits": s1[current_start:n],
            })

    total_matching = sum(seg["length"] for seg in segments)
    match_pct = round(total_matching / n * 100, 2) if n > 0 else 0.0
    identity_pct = round(total_identical / n * 100, 2) if n > 0 else 0.0

    return {
        "segments": segments,
        "compared_length": n,
        "total_matching_bits": total_matching,
        "total_identical_bits": total_identical,
        "match_percentage": match_pct,
        "identity_percentage": identity_pct,
    }


# ------------------------------------------------------------------ #
#  Build detail table for one pair                                    #
# ------------------------------------------------------------------ #

def build_detail_table(match_result, subj_a, subj_b):
    """Buat DataFrame detail segmen matching untuk satu pasangan subjek.

    Parameters
    ----------
    match_result : dict
        Output dari ``find_consecutive_matches``.
    subj_a, subj_b : str
        ID subjek.

    Returns
    -------
    pd.DataFrame
        Kolom: segment_no, bit_range, length, matching_bits.
        Kosong jika tidak ada segmen.
    """
    segments = match_result.get("segments", [])
    if not segments:
        return pd.DataFrame(columns=[
            "subject_a", "subject_b", "segment_no",
            "bit_range", "length", "matching_bits",
        ])

    rows = []
    for i, seg in enumerate(segments, start=1):
        rows.append({
            "subject_a": subj_a,
            "subject_b": subj_b,
            "segment_no": i,
            "bit_range": f"Bit {seg['start_bit']} – {seg['end_bit']}",
            "length": seg["length"],
            "matching_bits": seg["matching_bits"],
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
#  Compare all pairs for a given filter combination                   #
# ------------------------------------------------------------------ #

def compare_all_pairs(chain_df, scenario_id=None, task=None,
                      channel=None, subband=None, feature=None,
                      min_length=2, progress_callback=None):
    """Bandingkan chain_sequence antar semua pasangan unik subjek.

    Parameters
    ----------
    chain_df : pd.DataFrame
        Output chain dari chunking pipeline. Harus punya kolom:
        ``subject_id``, ``scenario_id``, ``task``, ``channel``,
        ``subband``, ``feature``, ``chain_sequence``.
    scenario_id : int | None
        Filter scenario. None = semua.
    task : str | None
        Filter task. None = semua.
    channel : str | None
        Filter channel. None = semua.
    subband : str | None
        Filter subband. None = semua.
    feature : str | None
        Filter feature. None = semua.
    min_length : int
        Minimum panjang segmen berturut-turut (default: 2).
    progress_callback : callable | None
        ``(current, total)`` untuk progress bar.

    Returns
    -------
    tuple (summary_df, all_details_df)
        - summary_df : DataFrame ringkasan per pasangan.
            Kolom: subject_a, subject_b, scenario_id, task, channel,
            subband, feature, seq_a, seq_b, compared_length,
            n_matching_segments, matching_segments_desc,
            total_matching_bits, match_percentage,
            total_identical_bits, identity_percentage.
        - all_details_df : DataFrame detail semua segmen matching.
            Kolom: subject_a, subject_b, scenario_id, task, channel,
            subband, feature, segment_no, bit_range, length,
            matching_bits.
    """
    if chain_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Apply filters
    df = chain_df.copy()
    if scenario_id is not None:
        df = df[df["scenario_id"] == scenario_id]
    if task is not None:
        df = df[df["task"] == task]
    if channel is not None:
        df = df[df["channel"] == channel]
    if subband is not None:
        df = df[df["subband"] == subband]
    if feature is not None:
        df = df[df["feature"] == feature]

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Tentukan group keys (kombinasi filter yang berlaku)
    group_cols = []
    for col in ["scenario_id", "task", "channel", "subband", "feature"]:
        if col in df.columns and df[col].nunique() > 0:
            group_cols.append(col)

    summary_rows = []
    detail_rows = []

    # Group by each unique combination of (scenario, task, channel, subband, feature)
    if not group_cols:
        # Jika semua filter tunggal, langsung compare
        _compare_group(df, {}, min_length, summary_rows, detail_rows)
    else:
        groups = list(df.groupby(group_cols))
        total_groups = len(groups)

        for g_idx, (group_key, grp) in enumerate(groups):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_meta = dict(zip(group_cols, group_key))

            _compare_group(grp, group_meta, min_length,
                           summary_rows, detail_rows)

            if progress_callback:
                progress_callback(g_idx + 1, total_groups)

    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    all_details_df = pd.DataFrame(detail_rows) if detail_rows else pd.DataFrame()

    return summary_df, all_details_df


def _compare_group(grp_df, group_meta, min_length,
                   summary_rows, detail_rows):
    """Bandingkan semua pasangan subjek dalam satu group."""
    # Ambil unique subjects dengan sequences
    subj_seqs = {}
    for _, row in grp_df.iterrows():
        subj = row.get("subject_id", "")
        seq = row.get("chain_sequence", "")
        if subj and seq:
            subj_seqs[subj] = seq

    subjects = sorted(subj_seqs.keys(), key=_subject_sort_key)
    if len(subjects) < 2:
        return

    for subj_a, subj_b in combinations(subjects, 2):
        seq_a = subj_seqs[subj_a]
        seq_b = subj_seqs[subj_b]

        result = find_consecutive_matches(seq_a, seq_b, min_length)

        # Buat deskripsi segmen
        seg_descs = []
        for seg in result["segments"]:
            seg_descs.append(
                f"Bit {seg['start_bit']}–{seg['end_bit']} "
                f"({seg['length']} bit)"
            )
        matching_desc = "; ".join(seg_descs) if seg_descs else "Tidak ada"

        summary_row = dict(group_meta)
        summary_row.update({
            "subject_a": subj_a,
            "subject_b": subj_b,
            "seq_a": seq_a,
            "seq_b": seq_b,
            "compared_length": result["compared_length"],
            "n_matching_segments": len(result["segments"]),
            "matching_segments_desc": matching_desc,
            "total_matching_bits": result["total_matching_bits"],
            "match_percentage": result["match_percentage"],
            "total_identical_bits": result["total_identical_bits"],
            "identity_percentage": result["identity_percentage"],
        })
        summary_rows.append(summary_row)

        # Detail rows
        for i, seg in enumerate(result["segments"], start=1):
            detail_row = dict(group_meta)
            detail_row.update({
                "subject_a": subj_a,
                "subject_b": subj_b,
                "segment_no": i,
                "bit_range": f"Bit {seg['start_bit']} – {seg['end_bit']}",
                "length": seg["length"],
                "matching_bits": seg["matching_bits"],
            })
            detail_rows.append(detail_row)


def _subject_sort_key(subj_id):
    """Sort key numerik untuk subject_id seperti 'id1', 'id12', etc."""
    num = subj_id.replace("id", "")
    try:
        return int(num)
    except ValueError:
        return 999999

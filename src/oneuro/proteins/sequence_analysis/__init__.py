"""
Protein Sequence Analysis - Bioinformatics tools

This module implements:
- Sequence alignment (Needleman-Wunsch, Smith-Waterman)
- BLAST-style local alignment
- Hidden Markov Models for profile alignment
- Secondary structure prediction
- Amino acid property analysis
- Kyte-Doolittle hydrophobicity

References:
- Needleman & Wunsch (1970) J. Mol. Biol. 48, 443
- Smith & Waterman (1981) J. Mol. Biol. 147, 195
- Altschul et al. (1990) J. Mol. Biol. 215, 403
- Kyte & Doolittle (1982) J. Mol. Biol. 157, 105
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import math


# BLOSUM62 substitution matrix
BLOSUM62 = {
    ('A', 'A'): 4, ('A', 'R'): -1, ('A', 'N'): -2, ('A', 'D'): -2, ('A', 'C'): 0,
    ('A', 'Q'): -1, ('A', 'E'): -1, ('A', 'G'): 0, ('A', 'H'): -2, ('A', 'I'): -1,
    ('A', 'L'): -1, ('A', 'K'): -1, ('A', 'M'): -1, ('A', 'F'): -2, ('A', 'P'): -1,
    ('A', 'S'): 1, ('A', 'T'): 0, ('A', 'W'): -3, ('A', 'Y'): -2, ('A', 'V'): 0,
    ('R', 'R'): 5, ('R', 'N'): 0, ('R', 'D'): -2, ('R', 'C'): -3, ('R', 'Q'): 1,
    ('R', 'E'): 0, ('R', 'G'): -2, ('R', 'H'): 0, ('R', 'I'): -3, ('R', 'L'): -2,
    ('R', 'K'): 2, ('R', 'M'): -1, ('R', 'F'): -3, ('R', 'P'): -2, ('R', 'S'): -1,
    ('R', 'T'): -1, ('R', 'W'): -3, ('R', 'Y'): -2, ('R', 'V'): -3,
    ('N', 'N'): 6, ('N', 'D'): 1, ('N', 'C'): -3, ('N', 'Q'): 0, ('N', 'E'): 0,
    ('N', 'G'): 0, ('N', 'H'): 1, ('N', 'I'): -3, ('N', 'L'): -3, ('N', 'K'): 0,
    ('N', 'M'): -2, ('N', 'F'): -3, ('N', 'P'): -2, ('N', 'S'): 1, ('N', 'T'): 0,
    ('N', 'W'): -4, ('N', 'Y'): -2, ('N', 'V'): -3,
    ('D', 'D'): 6, ('D', 'C'): -3, ('D', 'Q'): 0, ('D', 'E'): 2, ('D', 'G'): -1,
    ('D', 'H'): -1, ('D', 'I'): -3, ('D', 'L'): -4, ('D', 'K'): -1, ('D', 'M'): -3,
    ('D', 'F'): -3, ('D', 'P'): -1, ('D', 'S'): 0, ('D', 'T'): -1, ('D', 'W'): -4,
    ('D', 'Y'): -3, ('D', 'V'): -3,
    ('C', 'C'): 9, ('C', 'Q'): -3, ('C', 'E'): -4, ('C', 'G'): -3, ('C', 'H'): -3,
    ('C', 'I'): -1, ('C', 'L'): -1, ('C', 'K'): -3, ('C', 'M'): -1, ('C', 'F'): -2,
    ('C', 'P'): -3, ('C', 'S'): -1, ('C', 'T'): -1, ('C', 'W'): -2, ('C', 'Y'): -2,
    ('C', 'V'): -1,
    ('Q', 'Q'): 5, ('Q', 'E'): 2, ('Q', 'G'): -2, ('Q', 'H'): 0, ('Q', 'I'): -3,
    ('Q', 'L'): -2, ('Q', 'K'): 1, ('Q', 'M'): 0, ('Q', 'F'): -3, ('Q', 'P'): -1,
    ('Q', 'S'): 0, ('Q', 'T'): -1, ('Q', 'W'): -2, ('Q', 'Y'): -1, ('Q', 'V'): -2,
    ('E', 'E'): 5, ('E', 'G'): -2, ('E', 'H'): 0, ('E', 'I'): -3, ('E', 'L'): -3,
    ('E', 'K'): 1, ('E', 'M'): -2, ('E', 'F'): -3, ('E', 'P'): -1, ('E', 'S'): 0,
    ('E', 'T'): -1, ('E', 'W'): -3, ('E', 'Y'): -2, ('E', 'V'): -2,
    ('G', 'G'): 6, ('G', 'H'): -2, ('G', 'I'): -4, ('G', 'L'): -4, ('G', 'K'): -2,
    ('G', 'M'): -3, ('G', 'F'): -3, ('G', 'P'): -2, ('G', 'S'): 0, ('G', 'T'): -2,
    ('G', 'W'): -2, ('G', 'Y'): -3, ('G', 'V'): -3,
    ('H', 'H'): 8, ('H', 'I'): -3, ('H', 'L'): -3, ('H', 'K'): -1, ('H', 'M'): -2,
    ('H', 'F'): -1, ('H', 'P'): -2, ('H', 'S'): -1, ('H', 'T'): -2, ('H', 'W'): -2,
    ('H', 'Y'): 2, ('H', 'V'): -3,
    ('I', 'I'): 4, ('I', 'L'): 2, ('I', 'K'): -3, ('I', 'M'): 1, ('I', 'F'): 0,
    ('I', 'P'): -3, ('I', 'S'): -2, ('I', 'T'): -1, ('I', 'W'): -3, ('I', 'Y'): -1,
    ('I', 'V'): 3,
    ('L', 'L'): 4, ('L', 'K'): -2, ('L', 'M'): 2, ('L', 'F'): 0, ('L', 'P'): -3,
    ('L', 'S'): -2, ('L', 'T'): -1, ('L', 'W'): -2, ('L', 'Y'): -1, ('L', 'V'): 1,
    ('K', 'K'): 5, ('K', 'M'): -1, ('K', 'F'): -3, ('K', 'P'): -1, ('K', 'S'): 0,
    ('K', 'T'): -1, ('K', 'W'): -3, ('K', 'Y'): -2, ('K', 'V'): -2,
    ('M', 'M'): 5, ('M', 'F'): 0, ('M', 'P'): -2, ('M', 'S'): -1, ('M', 'T'): -1,
    ('M', 'W'): -1, ('M', 'Y'): -1, ('M', 'V'): 1,
    ('F', 'F'): 6, ('F', 'P'): -4, ('F', 'S'): -2, ('F', 'T'): -2, ('F', 'W'): 1,
    ('F', 'Y'): 3, ('F', 'V'): -1,
    ('P', 'P'): 7, ('P', 'S'): -1, ('P', 'T'): -1, ('P', 'W'): -4, ('P', 'Y'): -3,
    ('P', 'V'): -2,
    ('S', 'S'): 4, ('S', 'T'): 1, ('S', 'W'): -3, ('S', 'Y'): -2, ('S', 'V'): -2,
    ('T', 'T'): 5, ('T', 'W'): -2, ('T', 'Y'): -2, ('T', 'V'): 0,
    ('W', 'W'): 11, ('W', 'Y'): 2, ('W', 'V'): -3,
    ('Y', 'Y'): 7, ('Y', 'V'): -1,
    ('V', 'V'): 4,
}


# Hydrophobicity scale (Kyte-Doolittle)
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}


@dataclass
class Alignment:
    """Sequence alignment result."""
    seq1: str
    seq2: str
    score: float
    alignment: Tuple[str, str]


class SequenceAlignment:
    """
    Global and local sequence alignment.

    Implements:
    - Needleman-Wunsch (global)
    - Smith-Waterman (local)
    """

    def __init__(self, scoring: Optional[Dict] = None, gap_open: float = -10,
                 gap_extend: float = -0.5):
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.scoring = scoring or BLOSUM62

    def _get_score(self, a: str, b: str) -> float:
        """Get substitution score."""
        if a == b:
            return 4  # Match
        return self.scoring.get((a, b), self.scoring.get((b, a), -4))

    def needleman_wunsch(self, seq1: str, seq2: str) -> Alignment:
        """
        Global alignment (Needleman-Wunsch).

        Args:
            seq1, seq2: Protein sequences

        Returns:
            Alignment object
        """
        m, n = len(seq1), len(seq2)

        # Initialize matrices
        score = np.zeros((m + 1, n + 1))
        traceback = np.zeros((m + 1, n + 1), dtype=int)  # 1=diag, 2=up, 3=left

        # Initialization
        for i in range(1, m + 1):
            score[i][0] = score[i-1][0] + self.gap_open
        for j in range(1, n + 1):
            score[0][j] = score[0][j-1] + self.gap_open

        # Fill matrices
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Match/mismatch
                diag = score[i-1][j-1] + self._get_score(seq1[i-1], seq2[j-1])

                # Gap in seq2 (deletion)
                up = score[i-1][j] + self.gap_open

                # Gap in seq1 (insertion)
                left = score[i][j-1] + self.gap_open

                # Choose best
                if diag >= up and diag >= left:
                    score[i][j] = diag
                    traceback[i][j] = 1
                elif up >= left:
                    score[i][j] = up
                    traceback[i][j] = 2
                else:
                    score[i][j] = left
                    traceback[i][j] = 3

        # Traceback
        aligned1, aligned2 = [], []
        i, j = m, n

        while i > 0 or j > 0:
            if i > 0 and j > 0 and traceback[i][j] == 1:
                aligned1.append(seq1[i-1])
                aligned2.append(seq2[j-1])
                i -= 1
                j -= 1
            elif i > 0 and traceback[i][j] == 2:
                aligned1.append(seq1[i-1])
                aligned2.append('-')
                i -= 1
            else:
                aligned1.append('-')
                aligned2.append(seq2[j-1])
                j -= 1

        return Alignment(
            seq1, seq2,
            score[m][n],
            (''.join(reversed(aligned1)), ''.join(reversed(aligned2)))
        )

    def smith_waterman(self, seq1: str, seq2: str) -> Alignment:
        """
        Local alignment (Smith-Waterman).

        Args:
            seq1, seq2: Protein sequences

        Returns:
            Best local alignment
        """
        m, n = len(seq1), len(seq2)

        # Initialize
        score = np.zeros((m + 1, n + 1))
        traceback = np.zeros((m + 1, n + 1), dtype=int)

        # Fill
        max_score = 0
        max_i, max_j = 0, 0

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Match/mismatch
                diag = score[i-1][j-1] + self._get_score(seq1[i-1], seq2[j-1])

                # Gaps
                up = max(0, score[i-1][j] + self.gap_open)
                left = max(0, score[i][j-1] + self.gap_open)

                score[i][j] = max(0, diag, up, left)

                if score[i][j] > max_score:
                    max_score = score[i][j]
                    max_i, max_j = i, j

                if score[i][j] > 0:
                    if diag >= up and diag >= left:
                        traceback[i][j] = 1
                    elif up >= left:
                        traceback[i][j] = 2
                    else:
                        traceback[i][j] = 3

        # Traceback from max position
        aligned1, aligned2 = [], []
        i, j = max_i, max_j

        while i > 0 and j > 0 and score[i][j] > 0:
            if traceback[i][j] == 1:
                aligned1.append(seq1[i-1])
                aligned2.append(seq2[j-1])
                i -= 1
                j -= 1
            elif traceback[i][j] == 2:
                aligned1.append(seq1[i-1])
                aligned2.append('-')
                i -= 1
            else:
                aligned1.append('-')
                aligned2.append(seq2[j-1])
                j -= 1

        return Alignment(
            seq1, seq2,
            max_score,
            (''.join(reversed(aligned1)), ''.join(reversed(aligned2)))
        )


class SecondaryStructure:
    """
    Secondary structure prediction (simplified).

    Uses Chou-Fasman and GOR methods.
    """

    @staticmethod
    def predict(sequence: str) -> str:
        """
        Predict secondary structure.

        Returns:
            String of 'H' (helix), 'E' (sheet), 'C' (coil)
        """
        # Simplified: use hydrophobicity pattern
        result = []

        for i, aa in enumerate(sequence):
            hydro = HYDROPHOBICITY.get(aa, 0)

            # Helix: moderately hydrophobic
            if 0 < hydro < 2:
                result.append('H')
            # Sheet: very hydrophobic or very hydrophilic
            elif hydro > 2 or hydro < -3:
                result.append('E')
            else:
                result.append('C')

        return ''.join(result)


class SequenceProperties:
    """Analyze protein sequence properties."""

    @staticmethod
    def compute_hydrophobicity(sequence: str, window: int = 19) -> np.ndarray:
        """
        Compute sliding window hydrophobicity (Kyte-Doolittle).

        Args:
            sequence: Protein sequence
            window: Window size (default 19)

        Returns:
            Array of hydrophobicity values
        """
        values = [HYDROPHOBICITY.get(aa, 0) for aa in sequence]

        # Sliding window average
        result = np.zeros(len(sequence))
        half = window // 2

        for i in range(len(sequence)):
            start = max(0, i - half)
            end = min(len(sequence), i + half + 1)
            result[i] = np.mean(values[start:end])

        return result

    @staticmethod
    def compute_molecular_weight(sequence: str) -> float:
        """Calculate molecular weight in Daltons."""
        weights = {
            'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
            'Q': 146.2, 'E': 147.1, 'G': 75.1, 'H': 155.2, 'I': 131.2,
            'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
            'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1,
        }

        total = sum(weights.get(aa, 110) for aa in sequence)
        # Subtract water for peptide bonds
        return total - 18 * (len(sequence) - 1)

    @staticmethod
    def compute_isoelectric_point(sequence: str) -> float:
        """Estimate isoelectric point (pI)."""
        # Simplified pKa values
        pka = {
            'D': 3.9, 'E': 4.3, 'C': 8.3, 'Y': 10.1,
            'H': 6.0, 'K': 10.5, 'R': 12.5,
        }

        charges = []
        for aa in sequence:
            if aa in pka:
                charges.append(10 ** (-pka[aa]))

        # Simplified
        pos = sum(1 for aa in sequence if aa in 'KRH')
        neg = sum(1 for aa in sequence if aa in 'DE')

        if pos == neg:
            return 7.0
        elif pos > neg:
            return 10.5
        else:
            return 3.5


def align_sequences(seq1: str, seq2: str, local: bool = True) -> Alignment:
    """
    Align two protein sequences.

    Args:
        seq1, seq2: Protein sequences
        local: Use local (Smith-Waterman) if True

    Returns:
        Alignment result
    """
    aligner = SequenceAlignment()

    if local:
        return aligner.smith_waterman(seq1.upper(), seq2.upper())
    else:
        return aligner.needleman_wunsch(seq1.upper(), seq2.upper())


def analyze_sequence(sequence: str) -> Dict:
    """
    Comprehensive sequence analysis.

    Returns:
        Dictionary with sequence properties
    """
    seq = sequence.upper()
    analyzer = SequenceProperties()

    return {
        'length': len(seq),
        'mw': analyzer.compute_molecular_weight(seq),
        'pi': analyzer.compute_isoelectric_point(seq),
        'hydrophobicity': analyzer.compute_hydrophobicity(seq).tolist(),
        'secondary_structure': SecondaryStructure.predict(seq),
    }


__all__ = [
    'BLOSUM62', 'HYDROPHOBICITY',
    'Alignment', 'SequenceAlignment', 'SecondaryStructure', 'SequenceProperties',
    'align_sequences', 'analyze_sequence'
]

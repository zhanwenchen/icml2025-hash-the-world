"""
hashing_bucketers.py
====================

This module implements a simple, training‑free exploration mechanism for
language‑mediated reinforcement learning. It discretizes both the state and
action spaces using hashing: states are mapped to compact indices using the
`feature_hash_string` function, while actions are grouped into similarity
preserving buckets using the SimHash algorithm. The `(state_idx, action_bucket)`
tuple serves as the key for a tabular Q‑table and a visit counter. An
intrinsic bonus proportional to the inverse square root of visit counts
encourages exploration.

The approach is described in the paper *Hash the World: Language‑Native
Hashing for Paraphrase‑Robust Exploration in LM‑RL* (ICML 2025). See the
repository README for an overview and toy usage example.
"""

from __future__ import annotations

import re
import mmh3
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple, Iterable

def feature_hash_string(s: str, B: int = 8192, seed: int = 0) -> np.ndarray:
    """Hash a string into a length‑``B`` signed one‑hot vector.

    The hash is performed with two independent MurmurHash3 calls: one to
    determine the index and one to determine the sign. The resulting vector
    contains exactly one non‑zero entry of ±1. This simple feature hashing
    scheme is fast, seed‑stable and works on arbitrary strings.

    Parameters
    ----------
    s: str
        The input string to hash.
    B: int, default 8192
        The number of bins (dimension of the output vector).
    seed: int, default 0
        A seed to randomize the hash functions.

    Returns
    -------
    ndarray
        A 1‑D numpy array of length ``B`` with one non‑zero entry at the
        hashed index.
    """
    v = np.zeros(B, dtype=np.float32)
    # index hash
    i = mmh3.hash(s, seed) % B
    # sign hash (use a different seed)
    sgn = 1 if (mmh3.hash(s, seed + 1) & 1) == 0 else -1
    v[i] = sgn
    return v

def char_ngrams(text: str, n: int = 4) -> Iterable[str]:
    """Generate character n‑grams from the input text.

    The text is lowercased and normalized by collapsing whitespace. For
    very short strings, the entire text is yielded once.
    """
    t = re.sub(r"\s+", " ", text.lower()).strip()
    if len(t) < n:
        yield t
        return
    for i in range(len(t) - n + 1):
        yield t[i : i + n]

def simhash(text: str, n: int = 4, bits: int = 64, seed: int = 0) -> int:
    """Compute a 64‑bit SimHash over character n‑grams of a string.

    The SimHash algorithm projects the input set of n‑grams into a binary
    signature that preserves cosine similarity: strings with overlapping
    n‑grams tend to produce similar signatures. This implementation uses
    MurmurHash3 to hash each n‑gram and majority voting to form the final
    signature.

    Parameters
    ----------
    text: str
        The input string.
    n: int, default 4
        The length of character n‑grams.
    bits: int, default 64
        The number of bits in the signature.
    seed: int, default 0
        A seed for hashing.

    Returns
    -------
    int
        The 64‑bit SimHash signature as an integer.
    """
    mask = (1 << bits) - 1
    votes = [0] * bits
    for g in char_ngrams(text, n=n):
        h = mmh3.hash64(g, seed=seed)[0] & mask
        for b in range(bits):
            votes[b] += 1 if (h >> b) & 1 else -1
    sig = 0
    for b in range(bits):
        if votes[b] > 0:
            sig |= (1 << b)
    return sig

def hamming_distance(a: int, b: int) -> int:
    """Compute the Hamming distance between two integers.

    This counts the number of differing bits when the integers are treated
    as unsigned bit strings of equal length.
    """
    return (a ^ b).bit_count()

@dataclass
class ActionBucketInfo:
    """Internal record for an action bucket.

    Attributes
    ----------
    sig: int
        The representative SimHash signature of the bucket.
    members: Set[str]
        A set of action strings that currently belong to the bucket.
    count: int
        The number of times actions in this bucket have been assigned to a
        key (i.e., the visit count).
    """
    sig: int
    members: Set[str] = field(default_factory=set)
    count: int = 0

class ActionBucketer:
    """Similarity‑preserving bucketer for actions using SimHash.

    New action strings are assigned to existing buckets if they are
    sufficiently similar (under a Hamming distance threshold) and the
    existing bucket has reached a minimum count. Otherwise a new bucket is
    created. Bucket signatures are occasionally recomputed to recenter
    clusters based on their current members.
    """

    def __init__(self,
                 n_gram: int = 4,
                 bits: int = 64,
                 seed: int = 0,
                 dist_thr: int = 18,
                 min_count_to_absorb: int = 5,
                 recenter_every: int = 25):
        self.n_gram = n_gram
        self.bits = bits
        self.seed = seed
        self.dist_thr = dist_thr
        self.min_count_to_absorb = min_count_to_absorb
        self.recenter_every = recenter_every
        self._registry: Dict[int, ActionBucketInfo] = {}
        self._inv: Dict[str, int] = {}
        self._next_id = 0
        self._inserts_since_center: Dict[int, int] = defaultdict(int)

    def _new_bucket(self, sig: int, text: str) -> int:
        bid = self._next_id
        self._next_id += 1
        info = ActionBucketInfo(sig=sig, members={text}, count=1)
        self._registry[bid] = info
        self._inv[text] = bid
        self._inserts_since_center[bid] = 0
        return bid

    def _nearest_bucket(self, sig: int) -> Tuple[Optional[int], Optional[int]]:
        if not self._registry:
            return None, None
        best_bid, best_d = None, float('inf')
        for bid, info in self._registry.items():
            d = hamming_distance(sig, info.sig)
            if d < best_d:
                best_bid, best_d = bid, d
        return best_bid, int(best_d)

    def assign(self, action_text: str) -> int:
        """Assign an action string to a bucket and return its bucket id.

        If the action has been seen before, it is mapped to its existing
        bucket and the count is incremented. Otherwise a SimHash signature
        is computed. The action is absorbed into the nearest dense bucket if
        its Hamming distance is below `dist_thr` and the target bucket has
        reached `min_count_to_absorb` visits. Otherwise a new bucket is
        created. Bucket signatures are occasionally recentered to prevent
        drift.
        """
        # Check exact match first
        if action_text in self._inv:
            bid = self._inv[action_text]
            self._registry[bid].count += 1
            return bid
        # Compute signature for the new action
        sig = simhash(action_text, n=self.n_gram, bits=self.bits, seed=self.seed)
        # Find closest existing bucket
        bid, d = self._nearest_bucket(sig)
        if bid is None:
            return self._new_bucket(sig, action_text)
        dense = self._registry[bid].count >= self.min_count_to_absorb
        if d <= self.dist_thr and dense:
            # absorb into existing bucket
            self._registry[bid].members.add(action_text)
            self._registry[bid].count += 1
            self._inv[action_text] = bid
            # track insertions for recentering
            self._inserts_since_center[bid] += 1
            if self._inserts_since_center[bid] >= self.recenter_every:
                self.recenter(bid)
                self._inserts_since_center[bid] = 0
            return bid
        else:
            return self._new_bucket(sig, action_text)

    def recenter(self, bid: int) -> None:
        """Recompute the representative signature of a bucket.

        The bucket's signature is updated by majority voting over the
        SimHash signatures of all its members. This helps to keep the
        centroid close to the evolving cluster of paraphrases.
        """
        info = self._registry[bid]
        bits = self.bits
        votes = [0] * bits
        for a in info.members:
            h = simhash(a, n=self.n_gram, bits=bits, seed=self.seed)
            for b in range(bits):
                votes[b] += 1 if (h >> b) & 1 else -1
        sig = 0
        for b in range(bits):
            if votes[b] > 0:
                sig |= (1 << b)
        info.sig = sig

class StateBucketer:
    """Compactly hash state strings to a single active index.

    The class wraps `feature_hash_string` and returns only the index of the
    non‑zero element, which can be used directly as part of a key.
    """
    def __init__(self, B: int = 8192, seed: int = 0):
        self.B = B
        self.seed = seed

    def index(self, state_text: str) -> int:
        vec = feature_hash_string(state_text, B=self.B, seed=self.seed)
        # return the index of the single non‑zero entry
        return int(np.argmax(np.abs(vec)))

class QTable:
    """Tabular value table with count‑based novelty bonus.

    Q‑values and visit counts are stored in dictionaries keyed by `(state_idx,
    action_bucket)`. When a key is updated, an intrinsic bonus is added to
    the external reward based on the inverse square root of the visit count.
    """
    def __init__(self, c_bonus: float = 0.5, gamma: float = 0.99, alpha: float = 0.1):
        self.Q = defaultdict(float)
        self.N = defaultdict(int)
        self.c_bonus = c_bonus
        self.gamma = gamma
        self.alpha = alpha

    def key(self, s_idx: int, a_bid: int) -> Tuple[int, int]:
        return (s_idx, a_bid)

    def bonus(self, key: Tuple[int, int]) -> float:
        # Return an intrinsic reward bonus based on visit count
        return self.c_bonus / np.sqrt(self.N[key] + 1)

    def update(self,
               s_idx: int,
               a_bid: int,
               ext_r: float,
               next_s_idx: Optional[int] = None,
               next_a_bid: Optional[int] = None) -> None:
        k = self.key(s_idx, a_bid)
        r = ext_r + self.bonus(k)
        # Bootstrap value for SARSA; pass None to avoid bootstrap at terminal states
        if (next_s_idx is not None) and (next_a_bid is not None):
            k2 = self.key(next_s_idx, next_a_bid)
            td_target = r + self.gamma * self.Q[k2]
        else:
            td_target = r
        self.Q[k] += self.alpha * (td_target - self.Q[k])
        self.N[k] += 1

class HashingAgentIndex:
    """High‑level wrapper that exposes `(state_text, action_text)` → key mapping.

    This class combines the state and action bucketers with a Q‑table. The
    `update` method performs a TD update for a single transition, and the
    `value` method returns the current Q‑value for a `(state, action)` pair.
    """
    def __init__(self,
                 B_state: int = 8192,
                 act_dist_thr: int = 18,
                 act_min_absorb: int = 5,
                 c_bonus: float = 0.5,
                 gamma: float = 0.99,
                 alpha: float = 0.1):
        self.state_bucketer = StateBucketer(B=B_state)
        self.action_bucketer = ActionBucketer(dist_thr=act_dist_thr,
                                              min_count_to_absorb=act_min_absorb)
        self.qtable = QTable(c_bonus=c_bonus, gamma=gamma, alpha=alpha)

    def key_sa(self, state_text: str, action_text: str) -> Tuple[int, int]:
        s_idx = self.state_bucketer.index(state_text)
        a_bid = self.action_bucketer.assign(action_text)
        return (s_idx, a_bid)

    def update(self,
               state_text: str,
               action_text: str,
               ext_r: float,
               next_state_text: Optional[str] = None,
               next_action_text: Optional[str] = None) -> None:
        # Assign keys for the current state and action (creates buckets if unseen)
        s_idx, a_bid = self.key_sa(state_text, action_text)
        if (next_state_text is not None) and (next_action_text is not None):
            ns_idx = self.state_bucketer.index(next_state_text)
            # peek bucket id for next action: do not increment counts yet
            na_bid = self.action_bucketer._inv.get(next_action_text)
        else:
            ns_idx, na_bid = None, None
        self.qtable.update(s_idx, a_bid, ext_r, ns_idx, na_bid)

    def value(self, state_text: str, action_text: str) -> float:
        # Non‑mutating Q lookup; returns 0.0 for unseen pairs
        s_idx = self.state_bucketer.index(state_text)
        bid = self.action_bucketer._inv.get(action_text)
        if bid is None:
            return 0.0
        return self.qtable.Q.get((s_idx, bid), 0.0)

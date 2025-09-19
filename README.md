# Hash the World: Language‑Native Hashing for Paraphrase‑Robust Exploration in LM‑RL

This repository accompanies our ICML 2025 submission *Hash the World: Language‑Native Hashing for Paraphrase‑Robust Exploration in LM‑RL*. It provides the reference implementation and pseudocode for our exploration algorithm as well as a demonstration of how to integrate it with a simple text environment.

## Abstract

Exploration in language‑mediated reinforcement learning (LM‑RL) is challenging because observations and actions arrive as unbounded natural language. Popular intrinsic rewards either (i) measure density in a learned embedding space (ε‑ball or k‑NN novelty), which adds encoder cost and can drift during training, or (ii) rely on prediction error (e.g. random network distillation), which introduces extra networks and non‑stationarity. We propose a training‑free, language‑native alternative: **hash the world**. States are reduced to compact, seed‑stable indices via signed **feature hashing**; actions are grouped online with **similarity‑preserving SimHash** into dynamic paraphrase buckets. A single discrete key, `(state_idx, action_bucket)`, becomes the unit of exploration with a monotone count‑based bonus. A simple distance‑and‑density gate (Hamming threshold plus minimum cluster size) stabilizes bucket growth and curbs premature over‑merging; lightweight maintenance (recentering or split–merge) keeps clusters healthy. The method runs in constant time per step, is drift‑resistant, explicitly credits action‑side paraphrases, and drops into existing LM‑RL stacks regardless of how actions are proposed (templates, knowledge graphs, or LLMs). We also describe probabilistic scaffolds—Bloom filters for episodic/global novelty memory and Count‑Min Sketch for fixed‑budget counting—that preserve constant‑time updates at scale.

## Contributions

1. **Language‑native discretization for exploration.** We unify feature‑hashed states with SimHash‑bucketed actions to form a seed‑stable key `(state_idx, action_bucket)` that enables paraphrase‑robust credit assignment without encoders, k‑NN search, or auxiliary predictors.
2. **Stable, monotone novelty with cheap online clustering.** We pair a count‑based bonus on the discrete key with a distance‑and‑density gate for action buckets, plus optional recenter/split‑merge. This yields low‑variance, drift‑resistant intrinsic rewards; Bloom filters and Count‑Min Sketch provide scalable, fixed‑memory implementations.
3. **Plug‑and‑play wrapper for LM‑RL.** The hashing layer decouples feasibility from exploration—it wraps any action proposer (templates, knowledge graphs, or LLMs) and any control algorithm (SARSA, Q‑learning, or actor‑critic), delivering constant overhead while explicitly rewarding under‑explored paraphrases.

## Repository contents

- `hashing_bucketers.py` – The main implementation of our hashing‑based exploration algorithm, including action and state bucketers, a tabular Q‑table with intrinsic rewards, and a convenience wrapper for integration.
- `toy_env_example.py` – A minimal example of how to use the hashing agent in a simple text environment. This file is provided for demonstration purposes and mirrors the pseudocode in our paper.

## Usage

Install the `mmh3` dependency (for MurmurHash3) with pip:

```sh
pip install mmh3
```

To run the toy example:

```sh
python toy_env_example.py
```

The script will create a small text environment, instantiate the `HashingAgentIndex`, and run a few episodes to demonstrate learning and clustering behaviour. Modify the example to connect the hashing agent to your own LM‑RL environment.

For more details, please refer to the paper.

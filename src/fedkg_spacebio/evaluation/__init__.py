"""Evaluation (phase 3): the split and leakage guard.

Leave-one-site-out and fixed, shared out-of-fold splits. A held-out site's samples must never
reach training, and every fitted transform (normalization, filtering, feature selection, batch
correction) is fit inside training folds only. Tests must cover both.
"""

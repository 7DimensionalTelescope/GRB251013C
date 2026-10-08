#!/usr/bin/env python3
"""Minimal reproducer: does `a - b` (or `a + b`, `a * b`) inside a function overwrite a large local array?

Seen 2026-10-08 with Python 3.14.1 + NumPy 2.2.4 (conda env "yp"): NumPy's temporary elision reuses the buffer of an
operand whose reference count looks like that of a temporary, and Python 3.14 pushes borrowed references to local
variables, so a large (>= ~256 KiB) local array operand is overwritten with the result. Views, dictionary items and
module globals are not affected, nor are small arrays. Exit status 1 if the bug is present, 0 otherwise.

    python modeling/test/check_numpy_elision.py
"""
import sys

import numpy as np


def subtract_local(n):
    a = np.arange(n, dtype=float)
    b = np.ones(n)
    keep = a.copy()
    _ = a - b                       # (the result is irrelevant; a must stay unchanged)
    return np.array_equal(a, keep)


def add_right_local(n):
    b = np.arange(n, dtype=float)
    ones = np.ones(n)
    keep = b.copy()
    _ = ones[:] + b                 # left is a view (never reused): the right-hand local gets overwritten
    return np.array_equal(b, keep)


def multiply_local(n):
    a = np.arange(n, dtype=float)
    keep = a.copy()
    _ = a * np.full(n, 2.0)
    return np.array_equal(a, keep)


def main():
    print(f"Python {sys.version.split()[0]}, NumPy {np.__version__}")
    bad = False
    for name, test in (("a - b", subtract_local), ("view + b (right operand)", add_right_local), ("a * x", multiply_local)):
        for n in (1_000, 100_000, 1_000_000):
            safe = test(n)
            bad |= not safe
            print(f"  {name:<22} n = {n:>9,d} ({8 * n / 1024:>7.0f} KiB): operand unchanged = {safe}")
    print("ELISION BUG PRESENT" if bad else "no operand mutation: OK")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

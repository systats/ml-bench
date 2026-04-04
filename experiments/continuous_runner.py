#!/usr/bin/env python3
"""Continuous Landscape Experiment Runner.

Runs all leakage landscape experiments continuously, expanding coverage.
Crash-safe: SQLite checkpoint tracks completed cells.
Append-only: JSONL output flushed after each row.
Resource-friendly: --workers, --nice, --max-memory.

Usage:
    python3 continuous_runner.py --data-dir ../data --workers 2
    python3 continuous_runner.py --data-dir ../data --only an,ap
    python3 continuous_runner.py --data-dir ../data --status  # print progress

Architecture:
    ExperimentRegistry → lists all experiments with configs
    Checkpoint (SQLite) → tracks (experiment, dataset_key) done tuples
    Each experiment script is imported and called per-dataset
    Results appended to canonical JSONL files in data-dir
"""

import argparse
import json
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENTS_DIR = Path(__file__).parent
DATA_DIR_DEFAULT = EXPERIMENTS_DIR.parent / "data"
CHECKPOINT_DB = "continuous_checkpoint.db"

# Canonical output files — these names match what compile_claims.py reads
EXPERIMENT_REGISTRY = {
    # V1: full 21-experiment suite (A-W)
    "v1": {
        "script": "run_leakage_landscape.py",
        "output": "leakage_landscape_v1_final.jsonl",
        "description": "21 leakage experiments (A-W) across all datasets",
        "min_rows": 100,
        "priority": 1,
    },
    # V2: seed inflation, screen/algorithm selection
    "v2": {
        "script": "run_leakage_landscape.py",  # same script, v2 experiments
        "output": "leakage_landscape_v2.jsonl",
        "description": "Seed inflation + algorithm selection bias",
        "min_rows": 100,
        "priority": 2,
    },
    # V3 AN: N-scaling (peeking, seed, normalize, oversample × sample size)
    "an": {
        "script": "run_v3_experiments.py",
        "output": "v3_an.jsonl",
        "description": "N-scaling: leakage × sample size",
        "min_rows": 200,
        "priority": 3,
    },
    # V3 AP: Seed dose-response (K=5,10,25,50,100)
    "ap": {
        "script": "run_v3_experiments.py",
        "output": "v3_ap.jsonl",
        "description": "Seed dose-response at varying K",
        "min_rows": 100,
        "priority": 4,
    },
    # V3 AO: CV coverage gap (actual vs nominal 95% CI)
    "ao": {
        "script": "run_v3_experiments.py",
        "output": "v3_ao_merged.jsonl",
        "description": "CV coverage: nominal vs actual 95% CI",
        "min_rows": 100,
        "priority": 5,
    },
    # V3 AN Peeking V2: model selection bias × N-scaling (19 configs)
    "an_peeking": {
        "script": "run_v3_an_peeking_v2.py",
        "output": "v3_an_peeking_v2.jsonl",
        "description": "Model selection peeking × sample size (19 configs)",
        "min_rows": 2000,
        "priority": 6,
    },
    # V3 AN Peeking Ext: 5K, 10K sample sizes
    "an_peeking_ext": {
        "script": "run_v3_an_peeking_v2_ext.py",
        "output": "v3_an_peeking_v2_ext.jsonl",
        "description": "Peeking extension: n=5K,10K",
        "min_rows": 10000,
        "priority": 7,
    },
    # V3 AN Peeking Ext2: 50K, 100K sample sizes
    "an_peeking_ext2": {
        "script": "run_v3_an_peeking_v2_ext2.py",
        "output": "v3_an_peeking_v2_ext2.jsonl",
        "description": "Peeking extension: n=50K,100K",
        "min_rows": 100000,
        "priority": 8,
    },
    # AC2: Compound class II (screen→tune→seed)
    "ac2": {
        "script": "exp_ac2_compound.py",
        "output": "v3_ac2.jsonl",
        "description": "Compound class II: screen→tune→seed pipeline",
        "min_rows": 200,
        "priority": 9,
    },
    # AN 100K: N-scaling at extreme sizes
    "an_100k": {
        "script": "exp_an_100k.py",
        "output": "v3_an_100k.jsonl",
        "description": "N-scaling at n=1K-100K (large datasets only)",
        "min_rows": 100000,
        "priority": 10,
    },
    # AT: Temporal HP tuning leakage
    "at_temporal": {
        "script": "exp_at_temporal.py",
        "output": "v3_att.jsonl",
        "description": "Temporal HP tuning: walk-forward vs IID selection",
        "min_rows": 500,
        "priority": 11,
    },
    # AT V2: Practitioner-realistic temporal optimism
    "at_temporal_v2": {
        "script": "exp_at_temporal_v2.py",
        "output": "v3_att2.jsonl",
        "description": "Temporal optimism: practitioner-realistic protocol",
        "min_rows": 500,
        "priority": 12,
    },
    # AT Tune NScaling: HP inflation × sample size × algorithm
    "at_tune": {
        "script": "exp_at_tune_nscaling.py",
        "output": "v3_at.jsonl",
        "description": "HP tuning inflation × n × algorithm × K",
        "min_rows": 200,
        "priority": 13,
    },
    # Boundary: temporal + group leakage (hand-picked datasets)
    "boundary": {
        "script": "exp_boundary.py",
        "output": "v3_boundary.jsonl",
        "description": "Temporal + group boundary leakage (13 datasets)",
        "min_rows": 500,
        "priority": 14,
    },
    # Boundary Full: automated discovery of temporal/group datasets
    "boundary_full": {
        "script": "exp_boundary_full.py",
        "output": "v3_boundary_full.jsonl",
        "description": "Auto-discover temporal/group datasets",
        "min_rows": 500,
        "priority": 15,
    },
}

# ---------------------------------------------------------------------------
# Checkpoint (SQLite)
# ---------------------------------------------------------------------------

class Checkpoint:
    """SQLite-backed checkpoint for crash-safe progress tracking."""

    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS completed (
                experiment TEXT NOT NULL,
                dataset_key TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                elapsed_s REAL,
                status TEXT DEFAULT 'ok',
                PRIMARY KEY (experiment, dataset_key)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS errors (
                experiment TEXT NOT NULL,
                dataset_key TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                error TEXT,
                PRIMARY KEY (experiment, dataset_key)
            )
        """)
        self.conn.commit()

    def is_done(self, experiment, dataset_key):
        cur = self.conn.execute(
            "SELECT 1 FROM completed WHERE experiment=? AND dataset_key=?",
            (experiment, dataset_key))
        return cur.fetchone() is not None

    def mark_done(self, experiment, dataset_key, elapsed_s=0.0, status="ok"):
        self.conn.execute(
            "INSERT OR REPLACE INTO completed VALUES (?, ?, ?, ?, ?)",
            (experiment, dataset_key, datetime.now(timezone.utc).isoformat(), elapsed_s, status))
        self.conn.commit()

    def mark_error(self, experiment, dataset_key, error_msg):
        self.conn.execute(
            "INSERT OR REPLACE INTO errors VALUES (?, ?, ?, ?)",
            (experiment, dataset_key, datetime.now(timezone.utc).isoformat(), str(error_msg)[:500]))
        self.conn.commit()

    def count_done(self, experiment=None):
        if experiment:
            cur = self.conn.execute(
                "SELECT COUNT(*) FROM completed WHERE experiment=?", (experiment,))
        else:
            cur = self.conn.execute("SELECT COUNT(*) FROM completed")
        return cur.fetchone()[0]

    def count_errors(self, experiment=None):
        if experiment:
            cur = self.conn.execute(
                "SELECT COUNT(*) FROM errors WHERE experiment=?", (experiment,))
        else:
            cur = self.conn.execute("SELECT COUNT(*) FROM errors")
        return cur.fetchone()[0]

    def summary(self):
        """Return dict of experiment → {done, errors}."""
        cur = self.conn.execute(
            "SELECT experiment, COUNT(*) FROM completed GROUP BY experiment")
        done = dict(cur.fetchall())
        cur = self.conn.execute(
            "SELECT experiment, COUNT(*) FROM errors GROUP BY experiment")
        errs = dict(cur.fetchall())
        return {exp: {"done": done.get(exp, 0), "errors": errs.get(exp, 0)}
                for exp in set(list(done.keys()) + list(errs.keys()))}

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# Dataset inventory
# ---------------------------------------------------------------------------

def load_dataset_inventory(data_dir):
    """Load dataset inventory from V1 JSONL. Returns list of (name, source, n_rows)."""
    v1_path = data_dir / "leakage_landscape_v1_final.jsonl"
    if not v1_path.exists():
        # Try extended
        v1_path = data_dir / "leakage_landscape_v1_extended.jsonl"
    if not v1_path.exists():
        print(f"ERROR: No V1 inventory found in {data_dir}")
        sys.exit(1)

    inventory = []
    seen = set()
    with open(v1_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = f"{r['name']}|{r['source']}"
            if key in seen:
                continue
            seen.add(key)
            inventory.append({
                "name": r["name"],
                "source": r["source"],
                "n_rows": r.get("n_rows", 0),
            })
    return inventory


def get_existing_keys(jsonl_path):
    """Read existing JSONL and return set of completed dataset keys."""
    keys = set()
    if not os.path.exists(jsonl_path):
        return keys
    with open(jsonl_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                key = f"{r['name']}|{r.get('source', 'openml')}"
                keys.add(key)
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


# ---------------------------------------------------------------------------
# Result writer (append-only, flush per row)
# ---------------------------------------------------------------------------

class ResultWriter:
    """Append-only JSONL writer with per-row flush."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Record initial row count — we must never have fewer rows
        self._initial_rows = 0
        if os.path.exists(path):
            with open(path) as f:
                self._initial_rows = sum(1 for _ in f)
        self._fh = open(path, "a")  # append-only, never truncate

    def write(self, result_dict):
        line = json.dumps(result_dict, default=str)
        self._fh.write(line + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self):
        self._fh.close()


# ---------------------------------------------------------------------------
# Experiment dispatcher
# ---------------------------------------------------------------------------

def run_experiment_cell(experiment_name, dataset_info, data_dir, experiments_dir):
    """Run a single (experiment, dataset) cell. Returns result dict or None.

    Each experiment script has its own run logic. This dispatcher imports
    and calls the right function. If an experiment script doesn't have a
    `run_single(name, source, data_dir)` function, we skip it.
    """
    name = dataset_info["name"]
    source = dataset_info["source"]
    n_rows = dataset_info["n_rows"]

    # Import experiment-specific runner
    # For now, we delegate to the existing scripts' internal functions
    # by adding experiments_dir to sys.path
    if str(experiments_dir) not in sys.path:
        sys.path.insert(0, str(experiments_dir))

    if experiment_name in ("an", "ap", "ao"):
        return _run_v3_cell(experiment_name, name, source, n_rows, data_dir)
    elif experiment_name == "an_peeking":
        return _run_an_peeking_cell(name, source, n_rows, data_dir)
    elif experiment_name == "ac2":
        return _run_ac2_cell(name, source, n_rows, data_dir)
    elif experiment_name == "at_tune":
        return _run_at_tune_cell(name, source, n_rows, data_dir)
    elif experiment_name in ("v1", "v2"):
        return _run_v1v2_cell(experiment_name, name, source, n_rows, data_dir)
    else:
        # Experiments that need special handling (boundary, temporal)
        # These have hand-picked datasets or discovery logic
        return None


def _run_v3_cell(exp_type, name, source, n_rows, data_dir):
    """Run a single V3 experiment cell (AN, AP, or AO)."""
    import run_v3_experiments as v3

    X, y = v3.load_dataset(name, source)
    if X is None:
        return None

    X, y = v3.prepare_binary(X, y)
    if X is None or len(X) < 50:
        return None

    result = {
        "name": name,
        "source": source,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "v3_status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if exp_type == "an":
        if n_rows < 200:
            return None
        result.update(_run_an(X, y, v3))
    elif exp_type == "ap":
        result.update(_run_ap(X, y, v3))
    elif exp_type == "ao":
        result.update(_run_ao(X, y, v3))

    return result


def _run_an(X, y, v3):
    """N-scaling: peeking, seed, normalize, oversample at varying n."""
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.preprocessing import StandardScaler

    n_full = min(len(X), 2000)
    ns = [n for n in v3.SUBSAMPLE_NS_MAIN if n <= n_full]
    if not ns:
        return {"v3_status": "skip"}

    rng = np.random.RandomState(v3.SEED)
    results = {
        "an_n_levels": ns,
        "an_n_full": n_full,
        "an_peeking_means": [],
        "an_seed_means": [],
        "an_normalize_means": [],
        "an_oversample_means": [],
    }

    for n in ns:
        # Subsample
        if len(X) > n:
            idx = rng.choice(len(X), n, replace=False)
            Xn, yn = X[idx], y[idx]
        else:
            Xn, yn = X, y

        if len(np.unique(yn)) < 2:
            for k in ("peeking", "seed", "normalize", "oversample"):
                results[f"an_{k}_means"].append(None)
            continue

        # Peeking: best-of-5 configs vs random pick
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                Xn, yn, test_size=0.3, random_state=v3.SEED, stratify=yn)
            configs = [
                v3.make_lr(v3.SEED),
                v3.make_rf(v3.SEED, 10),
                v3.make_rf(v3.SEED, 50),
                v3.make_rf(v3.SEED, 100),
                v3.make_dt(v3.SEED),
            ]
            aucs = [v3.fit_score(c, X_tr, y_tr, X_te, y_te) for c in configs]
            honest = aucs[rng.randint(len(aucs))]
            peeking_delta = max(aucs) - honest
        except Exception:
            peeking_delta = None
        results["an_peeking_means"].append(peeking_delta)

        # Seed inflation: best-of-10 seeds vs mean
        try:
            seed_aucs = []
            for s in range(10):
                clf = v3.make_rf(v3.SEED + s, 50)
                seed_aucs.append(v3.fit_score(clf, X_tr, y_tr, X_te, y_te))
            seed_delta = max(seed_aucs) - float(np.mean(seed_aucs))
        except Exception:
            seed_delta = None
        results["an_seed_means"].append(seed_delta)

        # Normalization: global vs per-fold
        try:
            skf = StratifiedKFold(n_splits=min(5, len(yn) // 2), shuffle=True, random_state=v3.SEED)
            global_aucs, perfold_aucs = [], []
            scaler_global = StandardScaler().fit(Xn)
            Xn_global = scaler_global.transform(Xn)

            for tr_idx, te_idx in skf.split(Xn, yn):
                clf_g = v3.make_lr(v3.SEED)
                global_aucs.append(v3.fit_score(clf_g, Xn_global[tr_idx], yn[tr_idx],
                                                 Xn_global[te_idx], yn[te_idx]))
                scaler_fold = StandardScaler().fit(Xn[tr_idx])
                Xtr_f = scaler_fold.transform(Xn[tr_idx])
                Xte_f = scaler_fold.transform(Xn[te_idx])
                clf_f = v3.make_lr(v3.SEED)
                perfold_aucs.append(v3.fit_score(clf_f, Xtr_f, yn[tr_idx], Xte_f, yn[te_idx]))

            norm_delta = float(np.mean(global_aucs) - np.mean(perfold_aucs))
        except Exception:
            norm_delta = None
        results["an_normalize_means"].append(norm_delta)

        # Oversampling: before vs after split
        try:
            minority = yn == np.argmin(np.bincount(yn.astype(int)))
            n_minority = minority.sum()
            if n_minority < 5 or n_minority > len(yn) - 5:
                oversample_delta = None
            else:
                # Before split (leaky)
                idx_dup = rng.choice(np.where(minority)[0], n_minority, replace=True)
                Xo = np.vstack([Xn, Xn[idx_dup]])
                yo = np.concatenate([yn, yn[idx_dup]])
                X_tr_o, X_te_o, y_tr_o, y_te_o = train_test_split(
                    Xo, yo, test_size=0.3, random_state=v3.SEED, stratify=yo)
                clf_leaky = v3.make_lr(v3.SEED)
                auc_leaky = v3.fit_score(clf_leaky, X_tr_o, y_tr_o, X_te_o, y_te_o)

                # After split (honest)
                X_tr_h, X_te_h, y_tr_h, y_te_h = train_test_split(
                    Xn, yn, test_size=0.3, random_state=v3.SEED, stratify=yn)
                min_tr = y_tr_h == np.argmin(np.bincount(y_tr_h.astype(int)))
                idx_dup_h = rng.choice(np.where(min_tr)[0], min_tr.sum(), replace=True)
                X_tr_aug = np.vstack([X_tr_h, X_tr_h[idx_dup_h]])
                y_tr_aug = np.concatenate([y_tr_h, y_tr_h[idx_dup_h]])
                clf_honest = v3.make_lr(v3.SEED)
                auc_honest = v3.fit_score(clf_honest, X_tr_aug, y_tr_aug, X_te_h, y_te_h)

                oversample_delta = auc_leaky - auc_honest
        except Exception:
            oversample_delta = None
        results["an_oversample_means"].append(oversample_delta)

    return results


def _run_ap(X, y, v3):
    """Seed dose-response at K=5,10,25,50,100."""
    from sklearn.model_selection import train_test_split

    result = {}
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=v3.SEED, stratify=y)
    except Exception:
        return {"v3_status": "skip"}

    for algo_name, make_fn in [("lr", v3.make_lr), ("rf", lambda s: v3.make_rf(s, 50))]:
        # Pool 100 seed AUCs
        pool = []
        for s in range(100):
            clf = make_fn(v3.SEED + s)
            pool.append(v3.fit_score(clf, X_tr, y_tr, X_te, y_te))
        pool = np.array(pool)
        mean_auc = float(np.mean(pool))

        rng = np.random.RandomState(v3.SEED)
        for K in v3.SEED_KS:
            inflations = []
            for _ in range(10):  # 10 bootstrap samples
                picks = rng.choice(pool, K, replace=False)
                inflations.append(float(np.max(picks) - mean_auc))
            result[f"ap_{algo_name}_inflation_k{K}"] = float(np.mean(inflations))

    return result


def _run_ao(X, y, v3):
    """CV coverage: actual vs nominal 95% CI."""
    from sklearn.model_selection import StratifiedKFold

    result = {}
    for algo_name, make_fn in [("lr", v3.make_lr), ("rf", lambda s: v3.make_rf(s, 50))]:
        rep_means = []
        for rep in range(v3.N_REPS_AO):
            skf = StratifiedKFold(n_splits=v3.CV_FOLDS, shuffle=True,
                                   random_state=v3.SEED + rep)
            fold_aucs = []
            for tr_idx, te_idx in skf.split(X, y):
                clf = make_fn(v3.SEED + rep)
                fold_aucs.append(v3.fit_score(clf, X[tr_idx], y[tr_idx],
                                               X[te_idx], y[te_idx]))
            rep_means.append(float(np.mean(fold_aucs)))

        arr = np.array(rep_means)
        mu = np.mean(arr)
        se = np.std(arr, ddof=1) / np.sqrt(len(arr))

        # z-based 95% CI
        z_lo, z_hi = mu - 1.96 * se, mu + 1.96 * se
        # t-based 95% CI
        from scipy.stats import t
        t_crit = t.ppf(0.975, len(arr) - 1)
        t_lo, t_hi = mu - t_crit * se, mu + t_crit * se

        result[f"ao_{algo_name}_mean"] = float(mu)
        result[f"ao_{algo_name}_se"] = float(se)
        result[f"ao_{algo_name}_ci_z"] = [float(z_lo), float(z_hi)]
        result[f"ao_{algo_name}_ci_t"] = [float(t_lo), float(t_hi)]
        result[f"ao_{algo_name}_n_reps"] = len(rep_means)

    return result


def _run_an_peeking_cell(name, source, n_rows, data_dir):
    """Run AN peeking V2 for a single dataset."""
    import run_v3_an_peeking_v2 as pv2
    import run_v3_experiments as v3

    X, y = v3.load_dataset(name, source)
    if X is None:
        return None
    X, y = v3.prepare_binary(X, y)
    if X is None or len(X) < 2000:
        return None

    result = {
        "name": name,
        "source": source,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "v3_status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        # Use the script's internal experiment logic
        configs = pv2.CONFIGS if hasattr(pv2, "CONFIGS") else pv2.build_configs()
        n_configs = len(configs)
        SUBSAMPLE_NS = [50, 100, 200, 500, 1000, 2000]
        PEEK_KS = [1, 2, 5, 10, 15, n_configs]
        N_REPS = 5

        from sklearn.model_selection import train_test_split
        rng = np.random.RandomState(42)

        for n_sub in SUBSAMPLE_NS:
            if n_sub > len(X):
                continue
            for K in PEEK_KS:
                if K > n_configs:
                    continue
                deltas = []
                for rep in range(N_REPS):
                    seed = 42 + rep
                    if len(X) > n_sub:
                        idx = np.random.RandomState(seed).choice(len(X), n_sub, replace=False)
                        Xs, ys = X[idx], y[idx]
                    else:
                        Xs, ys = X, y
                    if len(np.unique(ys)) < 2:
                        continue
                    try:
                        X_tr, X_te, y_tr, y_te = train_test_split(
                            Xs, ys, test_size=0.3, random_state=seed, stratify=ys)
                    except Exception:
                        continue

                    aucs = []
                    for cfg in configs[:n_configs]:
                        clf = pv2.build_model(cfg, seed) if hasattr(pv2, "build_model") else cfg
                        aucs.append(v3.fit_score(clf, X_tr, y_tr, X_te, y_te))
                    if not aucs:
                        continue
                    best_k = sorted(range(len(aucs)), key=lambda i: aucs[i], reverse=True)[:K]
                    honest = aucs[rng.randint(len(aucs))]
                    deltas.append(np.mean([aucs[i] for i in best_k]) - honest)

                if deltas:
                    result[f"an2_peeking_k{K}_n{n_sub}"] = float(np.mean(deltas))

    except Exception as e:
        result["v3_status"] = "error"
        result["error"] = str(e)[:200]

    return result


def _run_ac2_cell(name, source, n_rows, data_dir):
    """Run compound class II for a single dataset."""
    import run_v3_experiments as v3

    X, y = v3.load_dataset(name, source)
    if X is None:
        return None
    X, y = v3.prepare_binary(X, y)
    if X is None or len(X) < 200:
        return None

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    result = {
        "name": name,
        "source": source,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "v3_status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)

        # Screen: 6 algorithms
        screen_configs = [
            ("LR_C1", Pipeline([("s", StandardScaler()),
                                ("m", LogisticRegression(C=1, max_iter=1000, random_state=42))])),
            ("LR_C10", Pipeline([("s", StandardScaler()),
                                 ("m", LogisticRegression(C=10, max_iter=1000, random_state=42))])),
            ("RF_50", RandomForestClassifier(n_estimators=50, random_state=42)),
            ("RF_100", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("DT", DecisionTreeClassifier(random_state=42)),
        ]

        # Try to add XGB
        try:
            from xgboost import XGBClassifier
            screen_configs.append(
                ("XGB", XGBClassifier(n_estimators=100, random_state=42,
                                       eval_metric="logloss", verbosity=0)))
        except ImportError:
            pass

        # Leaky screen: pick by test AUC
        screen_aucs_test = []
        screen_aucs_cv = []
        for label, clf in screen_configs:
            auc_test = v3.fit_score(clf, X_tr, y_tr, X_te, y_te)
            screen_aucs_test.append(auc_test)

            # Honest: 3-fold CV on train
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_aucs = []
            for tri, tei in skf.split(X_tr, y_tr):
                from sklearn.base import clone
                c = clone(clf)
                cv_aucs.append(v3.fit_score(c, X_tr[tri], y_tr[tri], X_tr[tei], y_tr[tei]))
            screen_aucs_cv.append(float(np.mean(cv_aucs)))

        leaky_best = int(np.argmax(screen_aucs_test))
        honest_best = int(np.argmax(screen_aucs_cv))

        result["leaky_screen_best"] = screen_configs[leaky_best][0]
        result["leaky_screen_auc"] = screen_aucs_test[leaky_best]
        result["honest_screen_best"] = screen_configs[honest_best][0]
        result["honest_screen_auc"] = screen_aucs_cv[honest_best]

        # Seed inflation on screen winner
        seed_aucs_leaky = []
        seed_aucs_honest = []
        for s in range(10):
            from sklearn.base import clone
            clf_l = clone(screen_configs[leaky_best][1])
            if hasattr(clf_l, "random_state"):
                clf_l.random_state = 42 + s
            seed_aucs_leaky.append(v3.fit_score(clf_l, X_tr, y_tr, X_te, y_te))

            clf_h = clone(screen_configs[honest_best][1])
            if hasattr(clf_h, "random_state"):
                clf_h.random_state = 42 + s
            seed_aucs_honest.append(v3.fit_score(clf_h, X_tr, y_tr, X_te, y_te))

        result["leaky_test_auc"] = float(max(seed_aucs_leaky))
        result["honest_test_auc"] = float(np.mean(seed_aucs_honest))
        result["ac2_compound_delta"] = result["leaky_test_auc"] - result["honest_test_auc"]

    except Exception as e:
        result["v3_status"] = "error"
        result["error"] = str(e)[:200]

    return result


def _run_at_tune_cell(name, source, n_rows, data_dir):
    """Run HP tuning inflation × sample size for a single dataset."""
    import run_v3_experiments as v3

    X, y = v3.load_dataset(name, source)
    if X is None:
        return None
    X, y = v3.prepare_binary(X, y)
    if X is None or len(X) < 200:
        return None

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    result = {
        "name": name,
        "source": source,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "v3_status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": {},
    }

    N_LEVELS = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    K_BUDGETS = [10, 50, 100]
    N_REPS = 5

    def make_hp_configs(algo, K, rng):
        configs = []
        for _ in range(K):
            if algo == "lr":
                C = float(10 ** rng.uniform(-3, 3))
                configs.append({"algo": "lr", "C": C})
            elif algo == "knn":
                k = int(rng.choice([3, 5, 7, 11, 15, 21, 31]))
                configs.append({"algo": "knn", "n_neighbors": k})
            elif algo == "rf":
                n_est = int(rng.choice([10, 50, 100, 200]))
                max_d = rng.choice([3, 5, 10, None])
                configs.append({"algo": "rf", "n_estimators": n_est, "max_depth": max_d})
        return configs

    def build_from_config(cfg, seed):
        if cfg["algo"] == "lr":
            return Pipeline([("s", StandardScaler()),
                             ("m", LogisticRegression(C=cfg["C"], max_iter=1000, random_state=seed))])
        elif cfg["algo"] == "knn":
            return Pipeline([("s", StandardScaler()),
                             ("m", KNeighborsClassifier(n_neighbors=cfg["n_neighbors"]))])
        elif cfg["algo"] == "rf":
            return RandomForestClassifier(
                n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"], random_state=seed)

    try:
        for algo in ["lr", "knn", "rf"]:
            result["results"][algo] = {}
            for n_sub in N_LEVELS:
                if n_sub > len(X):
                    continue
                result["results"][algo][str(n_sub)] = {}
                for K in K_BUDGETS:
                    deltas = []
                    for rep in range(N_REPS):
                        seed = 42 + rep
                        rng = np.random.RandomState(seed)

                        if len(X) > n_sub:
                            idx = rng.choice(len(X), n_sub, replace=False)
                            Xs, ys = X[idx], y[idx]
                        else:
                            Xs, ys = X, y

                        if len(np.unique(ys)) < 2:
                            continue

                        # 60/20/20 split
                        try:
                            X_tr, X_rest, y_tr, y_rest = train_test_split(
                                Xs, ys, test_size=0.4, random_state=seed, stratify=ys)
                            X_val, X_te, y_val, y_te = train_test_split(
                                X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest)
                        except Exception:
                            continue

                        hp_configs = make_hp_configs(algo, K, np.random.RandomState(seed + 1000))

                        test_aucs = []
                        val_aucs = []
                        for cfg in hp_configs:
                            clf = build_from_config(cfg, seed)
                            test_aucs.append(v3.fit_score(clf, X_tr, y_tr, X_te, y_te))
                            clf2 = build_from_config(cfg, seed)
                            val_aucs.append(v3.fit_score(clf2, X_tr, y_tr, X_val, y_val))

                        # Leaky: best by test, report test
                        leaky_auc = max(test_aucs)
                        # Honest: best by val, report test
                        honest_idx = int(np.argmax(val_aucs))
                        honest_auc = test_aucs[honest_idx]
                        deltas.append(leaky_auc - honest_auc)

                    if deltas:
                        result["results"][algo][str(n_sub)][f"k{K}"] = float(np.mean(deltas))

    except Exception as e:
        result["v3_status"] = "error"
        result["error"] = str(e)[:200]

    return result


def _run_v1v2_cell(version, name, source, n_rows, data_dir):
    """V1/V2 are already complete — skip re-running."""
    # V1 and V2 are complete datasets. The runner only extends V3+ experiments.
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_continuous(data_dir, experiments_dir, checkpoint, experiments=None,
                   limit=None, nice=0):
    """Main loop: pick next unfinished cell, run it, write result."""

    if nice > 0:
        os.nice(nice)

    inventory = load_dataset_inventory(data_dir)
    print(f"Loaded {len(inventory)} datasets from inventory")

    # Filter to requested experiments
    registry = EXPERIMENT_REGISTRY.copy()
    if experiments:
        registry = {k: v for k, v in registry.items() if k in experiments}

    # Skip V1/V2 (already complete)
    registry.pop("v1", None)
    registry.pop("v2", None)

    # Sort by priority
    sorted_exps = sorted(registry.items(), key=lambda x: x[1]["priority"])

    # Writers
    writers = {}
    for exp_name, exp_cfg in sorted_exps:
        out_path = str(data_dir / exp_cfg["output"])
        existing = get_existing_keys(out_path)
        writers[exp_name] = {
            "writer": ResultWriter(out_path),
            "existing": existing,
        }

    total_run = 0
    total_skip = 0
    total_error = 0
    start_time = time.time()

    # Graceful shutdown
    shutdown = False
    def _handle_signal(sig, frame):
        nonlocal shutdown
        print(f"\n[SIGNAL {sig}] Finishing current cell, then stopping...")
        shutdown = True
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    for exp_name, exp_cfg in sorted_exps:
        if shutdown:
            break

        min_rows = exp_cfg["min_rows"]
        eligible = [d for d in inventory if d["n_rows"] >= min_rows]
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name} — {exp_cfg['description']}")
        print(f"Eligible datasets: {len(eligible)}, min_rows={min_rows}")
        print(f"Already done (JSONL): {len(writers[exp_name]['existing'])}")
        print(f"Already done (checkpoint): {checkpoint.count_done(exp_name)}")
        print(f"{'='*60}")

        for i, ds in enumerate(eligible):
            if shutdown:
                break
            if limit and total_run >= limit:
                print(f"Limit reached ({limit})")
                return

            key = f"{ds['name']}|{ds['source']}"

            # Skip if already in JSONL or checkpoint
            if key in writers[exp_name]["existing"] or checkpoint.is_done(exp_name, key):
                total_skip += 1
                continue

            # Run the cell
            t0 = time.time()
            try:
                result = run_experiment_cell(exp_name, ds, data_dir, experiments_dir)
                elapsed = time.time() - t0

                if result is None:
                    checkpoint.mark_done(exp_name, key, elapsed, "skip")
                    total_skip += 1
                    continue

                writers[exp_name]["writer"].write(result)
                writers[exp_name]["existing"].add(key)
                checkpoint.mark_done(exp_name, key, elapsed, "ok")
                total_run += 1

                # Progress
                rate = total_run / (time.time() - start_time) * 3600
                remaining = len(eligible) - i - 1
                eta_h = remaining / max(rate, 1) if rate > 0 else 0
                status = result.get("v3_status", "ok")
                print(f"  [{total_run}] {exp_name}/{ds['name'][:40]:40s} "
                      f"{elapsed:.1f}s status={status} "
                      f"({rate:.0f}/hr ETA={eta_h:.1f}h)")

            except Exception as e:
                elapsed = time.time() - t0
                checkpoint.mark_error(exp_name, key, str(e))
                total_error += 1
                print(f"  [ERR] {exp_name}/{ds['name'][:40]:40s} {elapsed:.1f}s: {e}")

    # Cleanup
    for w in writers.values():
        w["writer"].close()

    elapsed_total = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DONE: {total_run} run, {total_skip} skipped, {total_error} errors "
          f"in {elapsed_total/60:.1f}min")


def print_status(data_dir, checkpoint):
    """Print progress summary."""
    inventory = load_dataset_inventory(data_dir)
    summary = checkpoint.summary()

    print(f"{'Experiment':<20} {'Done':>6} {'Errors':>6} {'JSONL':>6} {'Eligible':>8} {'%':>6}")
    print("-" * 60)

    for exp_name, exp_cfg in sorted(EXPERIMENT_REGISTRY.items(), key=lambda x: x[1]["priority"]):
        if exp_name in ("v1", "v2"):
            continue
        min_rows = exp_cfg["min_rows"]
        eligible = len([d for d in inventory if d["n_rows"] >= min_rows])
        out_path = data_dir / exp_cfg["output"]
        jsonl_count = len(get_existing_keys(str(out_path)))
        s = summary.get(exp_name, {"done": 0, "errors": 0})
        pct = (s["done"] / eligible * 100) if eligible > 0 else 0
        print(f"{exp_name:<20} {s['done']:>6} {s['errors']:>6} {jsonl_count:>6} {eligible:>8} {pct:>5.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Continuous Leakage Landscape Runner")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT,
                        help="Directory with JSONL data files")
    parser.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS_DIR,
                        help="Directory with experiment scripts")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint SQLite DB")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated list of experiments to run")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max cells to run before stopping")
    parser.add_argument("--nice", type=int, default=10,
                        help="Nice priority (0=normal, 19=lowest)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (future)")
    parser.add_argument("--status", action="store_true",
                        help="Print progress and exit")

    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = str(args.data_dir / CHECKPOINT_DB)

    checkpoint = Checkpoint(args.checkpoint)

    if args.status:
        print_status(args.data_dir, checkpoint)
        checkpoint.close()
        return

    experiments = None
    if args.only:
        experiments = set(args.only.split(","))

    print("Continuous Landscape Runner")
    print(f"  data-dir: {args.data_dir}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  experiments: {experiments or 'all'}")
    print(f"  nice: {args.nice}")
    print(f"  limit: {args.limit or 'unlimited'}")
    print(f"  PID: {os.getpid()}")
    print(f"  started: {datetime.now(timezone.utc).isoformat()}")
    print()

    try:
        run_continuous(
            data_dir=args.data_dir,
            experiments_dir=args.experiments_dir,
            checkpoint=checkpoint,
            experiments=experiments,
            limit=args.limit,
            nice=args.nice,
        )
    finally:
        checkpoint.close()


if __name__ == "__main__":
    main()

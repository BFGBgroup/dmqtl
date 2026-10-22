"""The batch null fit must hand optim_maxit and rng_seed to the R script (positions 9 and 10)."""
import csv
import subprocess

import numpy as np

from dmqtl import nullmodel


def _fake_run(record):
    def run(cmd, capture_output, text, timeout):
        record.append(cmd)
        out_prefix = cmd[4]
        with open(cmd[2]) as f:
            rows = list(csv.reader(f))
        genes = []
        for r in rows[1:]:
            if r[0] not in genes:
                genes.append(r[0])
        n = len(rows[0]) - 2
        with open(f"{out_prefix}_phi.csv", "w") as f:
            f.write('"gene_id","phi","converged","boundary_hit","outlier_flag"\n')
            for g in genes:
                f.write(f'"{g}",50,TRUE,FALSE,FALSE\n')
        with open(f"{out_prefix}_eta.csv", "w") as f:
            f.write('"gene_id","feature_id",' + ",".join(f'"S{i + 1}"' for i in range(n)) + "\n")
            for g in genes:
                f.write(f'"{g}","{g}__PAS0",' + ",".join("0.1" for _ in range(n)) + "\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return run


def test_maxit_and_seed_reach_rscript(monkeypatch):
    rec = []
    monkeypatch.setattr(nullmodel.subprocess, "run", _fake_run(rec))
    gy = {"A": np.ones((5, 2)), "B": np.ones((5, 2))}
    X = np.ones((5, 1))
    fits = nullmodel.fit_null_drimseq_batch(gy, X, ["Intercept"], optim_maxit=1000, rng_seed=7)
    assert rec[0][-3:] == ["1000", "7", "1"] and len(rec[0]) == 13
    assert fits["A"].converged
    nullmodel.fit_null_drimseq_batch(gy, X, ["Intercept"], cr_logdet=False)
    assert rec[1][-3:] == ["", "", "0"]

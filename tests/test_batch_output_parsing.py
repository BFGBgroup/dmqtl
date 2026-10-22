"""Reading the batch R script's CSV output must not fail the whole chunk when one gene's fit is NA.

DRIMSeq leaves genewise precision NA for genes it cannot fit; their dmFit
coefficients, and therefore their eta rows, are NA as well. Those genes are
already marked converged=FALSE and skipped downstream, but parsing "NA" with
float() raised ValueError and discarded the results of every other gene in
the chunk.
"""
import numpy as np

from dmqtl.nullmodel import _read_batch_outputs


def test_na_gene_is_marked_failed_and_others_survive(tmp_path):
    n = 4
    sample_cols = [f"S{i + 1}" for i in range(n)]
    gene_ys = {"A": np.ones((n, 3)), "B": np.ones((n, 2))}
    X = np.ones((n, 1))
    prefix = tmp_path / "batch_out"
    (tmp_path / "batch_out_phi.csv").write_text(
        '"gene_id","phi","converged","boundary_hit","outlier_flag"\n'
        '"A",55.5,TRUE,FALSE,FALSE\n'
        '"B",NA,FALSE,FALSE,FALSE\n')
    (tmp_path / "batch_out_eta.csv").write_text(
        '"gene_id","feature_id",' + ",".join(f'"{c}"' for c in sample_cols) + "\n"
        '"A","A__PAS0",0.1,0.2,0.3,0.4\n'
        '"A","A__PAS1",-0.1,-0.2,-0.3,-0.4\n'
        '"B","B__PAS0",NA,NA,NA,NA\n')
    fits = _read_batch_outputs(prefix, ["A", "B"], gene_ys, X, sample_cols)
    assert fits["A"].converged and fits["A"].phi == 55.5
    np.testing.assert_array_equal(fits["A"].eta, [[0.1, -0.1], [0.2, -0.2], [0.3, -0.3], [0.4, -0.4]])
    assert not fits["B"].converged and np.isnan(fits["B"].phi)
    assert fits["B"].eta.shape == (n, 1) and np.isnan(fits["B"].eta).all()

import numpy as np
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from dmqtl.dmglm import softmax_ref

rng = np.random.default_rng(42)
n_ind = 80
sample_ids = [f"IND{i:03d}" for i in range(n_ind)]

n_genes = 12
genes = [f"GENE{g}" for g in range(n_genes)]
gene_chrom = {g: "chr1" for g in genes}
gene_start = {g: 1_000_000 + i * 2_000_000 for i, g in enumerate(genes)}

pas_per_gene = {g: rng.integers(2, 5) for g in genes}  # K in [2,4]

# covariates: sex (0/1) + 2 PCs
cov_names = ["sex", "PC1", "PC2"]
cov_vals = np.vstack([
    rng.integers(0, 2, size=n_ind).astype(float),
    rng.normal(size=n_ind),
    rng.normal(size=n_ind),
])
X_full = np.column_stack([np.ones(n_ind), cov_vals.T])  # (n, 1+3)

pas_rows = []   # (pas_id, gene_id, counts[n_ind])
snp_rows = []   # (variant_id, chrom, pos, genotype[n_ind])
causal_snp_for_gene = {}

for gi, gene in enumerate(genes):
    K = pas_per_gene[gene]
    p = X_full.shape[1]
    beta = rng.normal(scale=0.3, size=(p, K - 1))
    phi = rng.uniform(40, 120)

    # candidate SNPs around the gene: 1 causal (for first half of genes) + several null
    n_null_snp = 15
    is_causal_gene = gi < n_genes // 2
    snp_ids_this_gene = []

    causal_delta = None
    if is_causal_gene:
        causal_delta = rng.normal(scale=1.2, size=K - 1)

    all_snps_this_gene = (["causal"] if is_causal_gene else []) + [f"null{j}" for j in range(n_null_snp)]
    gvecs = {}
    for tag in all_snps_this_gene:
        maf = rng.uniform(0.1, 0.45)
        g = rng.binomial(2, maf, size=n_ind).astype(float)
        gvecs[tag] = g

    eta = X_full @ beta
    if is_causal_gene:
        eta = eta + np.outer(gvecs["causal"] - gvecs["causal"].mean(), causal_delta)

    pi = softmax_ref(eta)
    N = rng.poisson(150, size=n_ind).astype(float)
    N[N < 5] = 5
    y = np.zeros((n_ind, K))
    for i in range(n_ind):
        alpha_i = phi * pi[i]
        p_i = rng.dirichlet(alpha_i)
        y[i] = rng.multinomial(int(N[i]), p_i)

    for k in range(K):
        pas_id = f"{gene}_PAS{k}"
        pas_rows.append((pas_id, gene, y[:, k]))

    pos0 = gene_start[gene]
    for j, tag in enumerate(all_snps_this_gene):
        vid = f"{gene}_snp_{tag}"
        pos = pos0 + (j - len(all_snps_this_gene)//2) * 5000
        snp_rows.append((vid, gene_chrom[gene], pos, gvecs[tag]))
        if tag == "causal":
            causal_snp_for_gene[gene] = vid

    # also throw in a handful of far-away (trans / out-of-window) SNPs for realism
    for j in range(3):
        vid = f"{gene}_far_snp{j}"
        pos = pos0 + 5_000_000 + j * 10000  # outside 1Mb cis window
        maf = rng.uniform(0.1, 0.4)
        g = rng.binomial(2, maf, size=n_ind).astype(float)
        snp_rows.append((vid, gene_chrom[gene], pos, g))

# write genotype.txt
with open("genotype.txt", "w") as f:
    f.write("id\t" + "\t".join(sample_ids) + "\n")
    for vid, chrom, pos, g in snp_rows:
        f.write(vid + "\t" + "\t".join(f"{v:.0f}" for v in g) + "\n")

# write snp_pos.txt
with open("snp_pos.txt", "w") as f:
    f.write("variant_id\tchr\tpos\n")
    for vid, chrom, pos, g in snp_rows:
        f.write(f"{vid}\t{chrom}\t{pos}\n")

# write pas_counts.txt
with open("pas_counts.txt", "w") as f:
    f.write("id\t" + "\t".join(sample_ids) + "\n")
    for pas_id, gene, y in pas_rows:
        f.write(pas_id + "\t" + "\t".join(f"{v:.0f}" for v in y) + "\n")

# write pas_to_gene.txt
with open("pas_to_gene.txt", "w") as f:
    f.write("pas_id\tgene_id\n")
    for pas_id, gene, y in pas_rows:
        f.write(f"{pas_id}\t{gene}\n")

# write covariates.txt
with open("covariates.txt", "w") as f:
    f.write("id\t" + "\t".join(sample_ids) + "\n")
    for name, row in zip(cov_names, cov_vals):
        f.write(name + "\t" + "\t".join(f"{v:.4f}" for v in row) + "\n")

# write gene_pos.txt
with open("gene_pos.txt", "w") as f:
    f.write("gene_id\tchr\tstart\tend\n")
    for gene in genes:
        f.write(f"{gene}\t{gene_chrom[gene]}\t{gene_start[gene]}\t{gene_start[gene]+1000}\n")

print("causal genes:", {g: causal_snp_for_gene[g] for g in genes if g in causal_snp_for_gene})
print("wrote synthetic input files")

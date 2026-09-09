# Installing R and DRIMSeq

`dmqtl.nullmodel.fit_null_drimseq` requires a working `Rscript` executable with both `DRIMSeq` (a Bioconductor package) and `jsonlite` (a CRAN package used by `r_null_fit.R` to write JSON output).

Choose one of the installation methods below according to your environment. After installation, use the verification commands at the end of this document to confirm that everything is working correctly.

## Option A: Conda / Mamba (Recommended)

This is the recommended approach for most users. Bioconda provides precompiled packages for DRIMSeq, so you do not need to install R system-wide or compile Bioconductor packages from source.

This approach also keeps the R environment isolated from other projects and from the system R installation.

### Using Conda

```bash
conda create -n dmqtl-r -c conda-forge -c bioconda \
    r-base r-jsonlite bioconductor-drimseq

conda activate dmqtl-r

Rscript -e 'library(DRIMSeq); library(jsonlite); cat("OK\n")'
```

### Using Mamba

If you use `mamba`, dependency resolution is usually much faster, especially for environments with many Bioconductor dependencies:

```bash
mamba create -n dmqtl-r -c conda-forge -c bioconda \
    r-base r-jsonlite bioconductor-drimseq
```

Then activate the environment:

```bash
mamba activate dmqtl-r
```

When running `dmqtl`, set `rscript_bin` to the `Rscript` executable in this environment. You can also provide its absolute path without activating the environment:

```python
run_dmqtl(
    ...,
    use_drimseq="auto",
    rscript_bin="/path/to/miniconda3/envs/dmqtl-r/bin/Rscript"
)
```

## Option B: System R + BiocManager

You can also install R directly on your operating system and then install DRIMSeq through Bioconductor.

### Ubuntu / Debian

```bash
sudo apt-get update && sudo apt-get install -y r-base
```

If the R version provided by your system package repository is too old, DRIMSeq may require a newer R version. In that case, consider using the official CRAN APT repository:

https://cran.r-project.org/bin/linux/ubuntu/README.html

Alternatively, use the Conda/Mamba approach described in Option A to avoid system R version conflicts.

### macOS

Install R using Homebrew:

```bash
brew install r
```

Alternatively, download and install the official R package from:

https://cran.r-project.org/bin/macosx/

### Install DRIMSeq and jsonlite

After installing R, start R by running:

```bash
R
```

or use `Rscript -e` directly.

Inside R:

```r
install.packages("BiocManager")
BiocManager::install("DRIMSeq")
install.packages("jsonlite")
```

## Option C: Docker

Docker is suitable for HPC or cluster environments where a clean and reproducible software environment is preferred.

The official Bioconductor Docker image provides R and the basic Bioconductor infrastructure. DRIMSeq and `jsonlite` can then be installed inside the container.

```bash
docker run -it --rm -v $(pwd):/work bioconductor/bioconductor_docker:RELEASE_3_19 \
    Rscript -e 'BiocManager::install("DRIMSeq"); install.packages("jsonlite")'
```

When running `dmqtl`, use the container as the R execution environment. For example, the Python workflow can be run inside the same container, or a wrapper can be used to execute `Rscript` through Docker.

## Verify the Installation

After installation, verify that both required R packages are available:

```bash
Rscript -e 'if (!requireNamespace("DRIMSeq", quietly=TRUE)) stop("DRIMSeq is not installed correctly") else cat("DRIMSeq OK\n")'

Rscript -e 'if (!requireNamespace("jsonlite", quietly=TRUE)) stop("jsonlite is not installed correctly") else cat("jsonlite OK\n")'
```

You can also check availability directly from Python:

```python
from dmqtl.nullmodel import drimseq_available

print(drimseq_available())                       # True/False
print(drimseq_available("/path/to/Rscript"))     # Check a specific Rscript
```

Once `drimseq_available()` returns `True`, `run_dmqtl(...)` with the default setting

```python
use_drimseq="auto"
```

will automatically use the DRIMSeq backend. No changes to the main function call are required.

## Verify the DRIMSeq Reference-Category Convention

A successful installation is not the final step. The DRIMSeq reference-category convention should also be manually verified against the Python implementation.

See:

```text
examples/verify_drimseq_reference_category.py
```

and the documentation and comments in:

```text
tests/test_r_bridge_smoke.py
```

These checks are intended to confirm that the reference-category convention used by the real R/DRIMSeq implementation is consistent with the Python implementation.

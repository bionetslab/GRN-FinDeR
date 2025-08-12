# %%
import scanpy as scanpy
import numpy as np
import scanpy as sc
import seaborn as sns
from scipy.stats import median_abs_deviation
import pandas as pd
import SEACells
import argparse


def main(cell_type: str):
    
    cell_map = {
        'myeloid leukocyte': 'monocyte',
        't cell': 't_cell',
        'lymphocyte of b lineage': 'b_cell'
    }

    if cell_type not in cell_map:
        raise ValueError(f"Invalid cell type '{cell_type}'. Choose from: {list(cell_map.keys())}")

    output_file_name = cell_map[cell_type]

    adata = sc.read_h5ad('/data_nfs/og86asub/datasets/cellxgene/blood_sc/raw/988defcd-7e39-4d07-91b9-a9853af1e769.h5ad')

    biomart = pd.read_csv('/data_nfs/og86asub/scFL-Green/raw/biomart.txt', sep = '\t')
    biomart = biomart.loc[:,["Gene stable ID", 'Gene name']]
    biomart = biomart.drop_duplicates()


    obs_var = 'broad_cell_class'



    # %%
    adata = adata[adata.obs[obs_var] == cell_type, :]

    # %%
    # Replace gene names with ensemble ids.
    adata.var = adata.var.reset_index()
    adata.var = adata.var.merge(biomart, left_on='ensg', right_on='Gene stable ID', how='left')


    # %%

    adata.var['keep'] = ~pd.isna(adata.var['Gene name'])
    adata = adata[:, adata.var.keep]
    # mitochondrial genes (already annotated)
    #adata.var["mt2"] = adata.var['Gene name'].str.startswith(("MT-"))
    # ribosomal genes
    #Duplication fixes weird pandas errror.
    adata.var["ribo"] = adata.var['Gene name'].str.startswith("RPS|RPL")
    adata.var["ribo"] = adata.var['Gene name'].str.startswith("RPS|RPL")
    # hemoglobin genes.
    adata.var["hb"] = adata.var['Gene name'].str.contains("^HB[^(P)]")


    # %%

    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20], log1p=True
    )

    adata.obs["mt_outlier"] = (adata.obs["pct_counts_mt"] > 10)

    adata = adata[ (~adata.obs.mt_outlier)].copy()
    sc.pp.filter_genes(adata, min_cells=20)
    sc.pp.filter_cells(adata, min_genes=200)

    scales_counts = sc.pp.normalize_total(adata, target_sum=10000, inplace=False)
    adata.layers['counts'] = scales_counts["X"]
    adata.layers["log1p"] = sc.pp.log1p(adata.layers['counts'], copy=True)
    sc.pp.highly_variable_genes(adata, layer="log1p", n_top_genes=4000)

    sc.pp.scale(adata, layer = 'log1p')
    sc.pp.pca(adata, svd_solver="arpack", use_highly_variable=True, layer='log1p')
    sc.pp.neighbors(adata)
    #sc.tl.umap(adata)

    # %%
    n_SEACells = int(np.floor(adata.shape[0]/75))
    build_kernel_on = 'X_pca' # key in ad.obsm to use for computing metacells
                            # This would be replaced by 'X_svd' for ATAC data

    ## Additional parameters
    n_waypoint_eigs = 10


    model = SEACells.core.SEACells(adata, 
                    build_kernel_on=build_kernel_on, 
                    n_SEACells=n_SEACells, 
                    n_waypoint_eigs=n_waypoint_eigs,
                    convergence_epsilon = 1e-5)

    model.construct_kernel_matrix()
    M = model.kernel_matrix

    model.initialize_archetypes()
    model.fit(min_iter=10, max_iter=50)

    # %%
    model.plot_convergence()

    # %%
    SEACell_ad = SEACells.core.summarize_by_SEACell(adata, SEACells_label='SEACell', summarize_layer='X')

    # %%
    SEACells.plot.plot_2D(adata, key='X_umap', colour_metacells=False)

    # %%
    SEACells.plot.plot_SEACell_sizes(adata, bins=10)

    # %%
    sc.pp.normalize_per_cell(SEACell_ad)
    sc.pp.scale(SEACell_ad)
    SEACell_ad.write(f"/data_nfs/og86asub/datasets/cellxgene/blood_sc/metacells/tabula_sapiens/{output_file_name}.h5ad")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SEACell metacells for a selected cell type.")
    parser.add_argument(
        "--cell_type",
        type=str,
        required=True,
        help="Cell type to process. Options: 'myeloid leukocyte', 't cell', 'b cell'"
    )
    args = parser.parse_args()
    main(args.cell_type)
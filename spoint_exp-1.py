from SPACEL import Spoint
import scanpy as sc
import numpy as np
import matplotlib
import wandb

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.serif'] = ['Arial']
sc.settings.set_figure_params(dpi=50,dpi_save=300,facecolor='white',fontsize=10,vector_friendly=True,figsize=(3,3))
sc.settings.verbosity = 3


sc_ad = sc.read_h5ad('/scratch/harsha.vasamsetti/data/visium_human_DLPFC/human_MTG_snrna_norm_by_exon.h5ad')
st_ad = sc.read_h5ad('/scratch/harsha.vasamsetti/data/visium_human_DLPFC/human_DLPFC_spatial_151676.h5ad')

sc.pp.filter_genes(st_ad,min_cells=1)  
sc.pp.filter_genes(sc_ad,min_cells=1) 

sc.pp.filter_cells(st_ad,min_genes=1) 
sc.pp.filter_cells(sc_ad,min_genes=1)  

spoint_model = Spoint.init_model(sc_ad,st_ad,celltype_key='cluster_label',deg_method='t-test',sm_size=100000,
                                 use_gpu=True,
                                use_wandb=True,           # Enable wandb
                                wandb_project="spacel",   # Project name
                                wandb_name="spoint-run1"  # Run name
                                 )


spoint_model.train(max_steps=5000, batch_size=512)

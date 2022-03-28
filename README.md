PyENVI for Python3
======================


ENVI is a deep learnining based variational inference method to integrate scRNAseq with spatial sequencing data. 
It creates a combined latent space for both data modalities, from which missing gene can be imputed from spatial data, cell types labels can be transfered
and the spatial niche can be reconstructed for the dissociated scRNAseq data

This implementation is written in Python3 and relies on tensorflow>2.4, tensorflow_probability, sklearn, scipy and scanpy.  


To install PhenoGraph, simply run the setup script:

    python3 setup.py install

Or use:

    pip3 install git+https://github.com/doronhaviv/PyENVI.git


Expected use is within a script or interactive kernel running Python `3.x`. Data are expected to be passed as anndata

To run ENVI:

    import PyENVI #import ENVI
    model = PyENVI.ENVI(spatial_data = Spatial_anndata, sc_data = scRNAseq_anndata) #Initialize Model
    model.Train() #Train VAE
    model.impute() #impute missing genes for spatial data, see model.spatial_data.obsm['imputation']
    model.pred_type(pred_on = 'sc')/model.pred_type(pred_on = 'spatial')# transfer cell_type labels, see model.spatial_data/sc_data.obs['cell_type']
    model.reconstruct_niche(pred_key = 'cell_type') #reconstruct spatial niche for sRNAseq, see model.sc_data.obsm['niche_by_type']
    
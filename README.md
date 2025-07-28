# ST-GCP
A Graph Convolutional Network Model with Contrastive Learning and permutation for Spatial Transcriptomics
![Figure 1](https://github.com/user-attachments/assets/9b9dc299-4dba-4900-ab95-f60f3dc8e4af)


## Requirement

- Reference

https://docs.anaconda.com/anaconda/install/index.html

https://pytorch.org/

- torch==1.13.0
- scipy>=1.10.1
- numpy>=1.24.4
- scikit-learn>=1.3.2
- pandas>=2.0.3
- tqdm>=4.66.5
- scanpy>=1.9.8
- anndata==0.9.2
- matplotlib==3.7.5
- sklearn
- Seurat v3
- seaborn 0.13.2

## Usage

#### Clone this repo.

```
git clone https://github.com/wyk-wuhan/ST-GCP.git
cd ST-GCP
```

#### Code description

- Cal_Spatial_Net : Construct the spatial neighbor graph using the Radius method
- get_feature : Randomly mask features in the gene expression matrix
- dropout_edge : Probabilistically prune the edges of the spatial neighbor graph
- STGCM : Obtain low-dimensional latent features through a graph autoencoder framework
- torch.cosine_similarity : Compute the contrastive loss between the original graph and the masked graph
- mclust_R : Perform clustering analysis based on mclust algorithm

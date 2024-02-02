from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, ColorBar, LinearColorMapper
from bokeh.transform import linear_cmap
from bokeh import palettes
from bokeh.layouts import column
from rdkit import Chem
# from useful_rdkit_utils import mol2numpy_fp
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from bokeh.io import output_notebook
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import rdMolDraw2D
from bokeh.plotting import ColumnDataSource, figure, output_file, show
import numba
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# import hdbscan
from itertools import product
import concurrent.futures
from rdkit.Chem import AllChem
from umap import UMAP
import numpy as np

def smi2svg(smi):
    mol = Chem.MolFromSmiles(smi)
    d2d = rdMolDraw2D.MolDraw2DSVG(200, 100)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()
    

def umap(df, nbits=2048):
    """
    Compute UMAP projections for molecular data.

    Parameters:
    - df: Dataframe containing a 'ROMol' column with molecular data.

    Returns:
    - res: UMAP reduced dimensionality output.
    """
    # Compute Morgan Fingerprints
    df['fp'] = df['ROMol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=nbits))
    df['svg'] = df['Smiles'].apply(lambda x: smi2svg(x))
    # Tanimoto Distance function
    def tanimoto_dist(a, b):
        dotprod = np.dot(a, b)
        tc = dotprod / (np.sum(a) + np.sum(b) - dotprod)
        return 1.0 - tc

    # UMAP dimensionality reduction
    fps = df['fp'].apply(lambda fp: np.array(fp)).tolist()
    from umap import UMAP
    reducer = UMAP(metric=tanimoto_dist)
    res = reducer.fit_transform(fps)
    
    return res


def plot_umap(df, res, colour='cnnaffinity'):
    # Data Preparation

    data = dict(
        x=res[:, 0],
        y=res[:, 1],
        enamine_ids= df['enamine_id'],
        img=df['svg'],
        sf1_values=df[colour],
        cnn=df['cnnaffinity'],
        cluster=df["cluster"],
        # plip=df['plip'],
        # run=df['exp']
    )

    # Tooltip for hover functionality
    TOOLTIPS = """
    <div>
        @img{safe}
        sf1 Value: @sf1_values<br>
        cycle: @cycle <br>
        cnnaff: @cnn <br>
        plip: @plip <br>
        enamine id: @enamine_ids <br>
        al exp: @run <br>
    </div>
    """

    # Bokeh visualization

    # make the circles smaller for the noise (==0) in the cluster
    data["sizes"] = [2 if c == 0 else 10 for c in picked_df.cluster]

    source = ColumnDataSource(data)
    p = figure(tooltips=TOOLTIPS, width=1000, height=500, title="UMAP Projection of Molecular Fingerprints")

    colors = df["cluster"].astype('float').values
    mapper = linear_cmap(field_name='cluster', palette=palettes.Turbo256, low=min(colors), high=max(colors))

    p.circle('x', 'y', size="sizes", source=source, color=mapper, alpha=0.9)

    # Create a color bar based on sf1 values
    # color_mapper = LinearColorMapper(palette=Viridis256, low=min(colors), high=max(colors))
    # color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, location=(0,0), title='sf1 Value')

    # Add the color bar to the plot
    # p.add_layout(color_bar, 'right')
    output_file(filename="al.html")
    # Display the plot
    output_notebook()
    show(column(p))

'''e min_cluster_size and min_samples, and experiment with cluster_selection_epsilon and cluster_selection_method.'''

def cluster_data(df, res, min_samples, min_cluster_size):
    # Apply HDBSCAN clustering on the UMAP results
    clusterer = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size, cluster_selection_method='leaf')
    cluster_labels = clusterer.fit_predict(res)

    # Add the cluster labels to the original DataFrame
    df['cluster'] = cluster_labels
    return df


import seaborn as sns
import matplotlib.pyplot as plt
from rdkit.Chem import Descriptors


if __name__ == "__main__":
    mols = list(Chem.SDMolSupplier("available_enamines_cnnaffinity_sign.sdf"))
    smiles = [m.GetProp("filename") for m in mols]
    cnnaffinities = [m.GetProp("cnnaffinity") for m in mols]
    enamine_id = [m.GetProp("enamine_id") for m in mols]

    # change noise in the cluster to be 0
    clusters = np.loadtxt("labels.dat") + 1
    picked_df = pd.DataFrame({"ROMol": mols, 
        "Smiles": smiles, 
        "cnnaffinity": cnnaffinities, 
        "enamine_id": enamine_id,
        "cluster": clusters, 
        })

    res = umap(picked_df)

    #make sure everything is filled and +ve
    # picked_df[['cnnaffinity', 'plip']] = picked_df[['cnnaffinity', 'plip']].fillna(0).astype(float).abs()
    # picked_df['cnn_norm'] = picked_df['cnn_norm'].fillna(0).abs()
    plot_umap(picked_df, res, 'cnnaffinity')
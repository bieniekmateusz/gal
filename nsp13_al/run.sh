export FG_ANIMODEL_PATH=/home/dresio/newcastle/sarscov2_reconstruction_random_search/random_dask/animodel.pt
export FG_GNINA_PATH=/home/dresio/gnina
export FG_PROTEIN_PATH=/home/dresio/newcastle/sarscov2_reconstruction_random_search/random_dask/protein.pdb

export FG_MW_LIMIT=200
export FG_MAX_MOLS=100
export FG_SCALE=5

#export FG_TMP_DIR=/scratch
export FG_TMP_DIR=None

python rsearcher.py
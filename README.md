# BiofuelDiscovery
In BiofuelDiscovery project, we trained three classifier models using Gaussian processes, random forests, and semi-supervised learning algorithms. We then used the SMILES of biofuels classified by these models to train the GAN model. Finally, we added the unique molecules created by the GAN to the total database and repeated the training cycle.
# Dataset
In Dataset fold, we have four datasets, "Dataset.csv" is used for three Classifiers, “Smiles list.csv" is used for Molgan to generate SMILES, "SemiSupervised_Dataset.csv" is used for Semi-Supervised Learning model, but it have some problem. In addition, we also have a file called "Rd_SMILES".csv, which is used to expand the database.
# Classifier
In Classifier fold, we have three models, these classifiers are used to distinguish biofuels from fossil fuels in the "Dataset.csv".
# Gan
In Gan fold, we use molgan to generated unique vaild molecules.
The molgan code comes from the deepchem tutorial (https://github.com/deepchem/deepchem/blob/master/examples/tutorials/Generating_molecules_with_MolGAN.ipynb)
# RDkit
In this fold, we have two tools, "SMILES TO Random SMILES.py" can convert SMILES to Random SMILES, "Descriptor.py" can calculate all properties of a molecule based on the SMILES.
# PairPlot
We use PairPlot to visualize the relationships between important descriptors for data exploration.
# Others
If you want to run Molgan, please check whether your third-party library version is consistent with that in requirement.txt.

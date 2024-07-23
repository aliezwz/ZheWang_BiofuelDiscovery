# BiofuelDiscovery
## 1. Introduction
In BiofuelDiscovery project, we trained three classifier models using Gaussian processes, random forests, and semi-supervised learning algorithms. We then used the SMILES of biofuels classified by these models to train the GAN model. Finally, we added the unique molecules created by the GAN to the total database and repeated the training cycle.
## 2. Project Structure
### 2.1 Datasets Folder
#### 2.1.1 Seed_Dataset
**Seed_Dataset.csv** is the original dataset used to train the classifier, which contains the names of molecules, smiles, labels (Types), and descriptors.
#### 2.1.2 Lotus_Dataset
The Lotus database is a natural product database that contains a total of 276,518 natural products. This dataset is used to trian Molgan.  
You can access the [Lotus dataset](https://lotus.naturalproducts.net/) directly.
#### 2.1.3 Smile_Dataset
This database contains a large number of SMILES of biofuel molecules and is used to train MolGan to generate new molecules.
### 2.2 Classifiers Folder
In Classifiers Folder, we have three classifiers written by different algorithms, namely RF_Classifier.py(Random Forest), GP_Classifier.py(Gaussian Process) and semi-supervised learning classifier(Semi-supervised Learning). 
### 2.3 Molgans Folder
In Molgans Folder, we have a **Modified_MolGAN.py**, this python file is used to generated unique and novel SMILES. In addition, there is a subfolder to store the molecules generated by Molgan.
### 2.4 Archive Folder
This folder contains many test files, please ignore this file.

## 3. Tutorial
### 3.1 Prepare the database
To run BiofuelDiscovery project, firstly you need to prepare a dataset containing biofuel molecules and fossil fuel molecules. Here, we call it "Seed_Dataset". This dataset should contains columns for "Name", "Smiles" and "Types". The "Name" column contains the name of the molecule. The "Smiles" column contains the **SMILES**(simplified molecular-input line-entry system) of the molecule, **SMILES** is a specification in the form of a line notation for describing the structure of chemical species using short ASCII strings[[1]](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system), you can search for smiles for each molecule on PubChem or use rdkit to generate smiles.  

In addition to the "Seed_Dataset", you still need to prepare a large database containing many biomass molecules. Here we use "Lotus Dataset".
### 3.2 Installation
The python version we use is 3.8.  
In addition to some commonly used data science libraries, you still need to install some machine learning and chemical information software libraries.   

```
pip install scikit-learn  
pip install torch  
pip install tensorflow  
pip install deepchem  
```  
Additionally, if you want to know all the libraries needed to run Molgan, please check requirements.txt. Also more information you can check the deepchem tutorials.  
[Deepchem tutorials](https://github.com/deepchem/deepchem/blob/master/examples/tutorials/Generating_molecules_with_MolGAN.ipynb)
### 3.3 How to 
In order to run the project, user should follow the following steps:  
1-Select any model in the Classifier fold and train it using **Seed_Dataset**. You will get a trianed modle.  
2-Use the trained model to classify your dataset. You will get a database with labels "biofuel" and "non-biofuel".  
3-Save the dataset you used in step 1 and the smiles of the molecules with the label "biofuel" in the newly obtained dataset, and create a new dataset.  
4-Use SMILES dataset to train MolGAN. The MolGAN will generate more unique and vaild smiles.  
5-Merge smiles generated by molgan into "Dataset_Mixed.csv"  
6-Repeat steps 2-5 and you will get a large number of unique and meaningful biofuel molecules.

# 4.Result
This is your "Classification Dataset"
| Name | Smiles | Types |
|-------------------|------------------|---------------|
| 2-Methylpentane   | CCCC(C)C         |      |
| 3-Methylpentane   | CCC(C)CC         |      |
| 2,3-Dimethylbutane| CC(C)C(C)C       |      |
| 2-Methylhexane    | CCCCC(C)C        |      |
| 3-Methylhexane    | CCCC(C)CC        |      |
| 2,2-Dimethylpentane| CCCC(C)(C)C     |      |
| 3,3-Dimethylpentane| CCC(C)(C)CC     |      |

After step 2, you will get a dataset with label.
| Name | Smiles | Types |
|-------------------|------------------|---------------|
| 2-Methylpentane   | CCCC(C)C         | Biofuel       |
| 3-Methylpentane   | CCC(C)CC         | Not biofuel   |
| 2,3-Dimethylbutane| CC(C)C(C)C       | Not biofuel   |
| 2-Methylhexane    | CCCCC(C)C        | Biofuel       |
| 3-Methylhexane    | CCCC(C)CC        | Not biofuel   |
| 2,2-Dimethylpentane| CCCC(C)(C)C      | Biofuel       |
| 3,3-Dimethylpentane| CCC(C)(C)CC      | Biofuel       |

For Molgan, every time you run it, you will get a series of smiles.
| Generated SMILES                   |
|------------------------------------|
| CCCC                               |
| CCCO                               |
| CC12CC3(C4OC41)C1CC34CC124         |
| CCC                                |
| CCC1C2C3C4C3C4C(C)C12C             |
| OCCO                               |
| CC                                 |
| CCC1(C)CC1                         |
| CCOO                               |
| CC12C3CCC4C15C1C32C415             |
| C=CC                               |
| O=CO                               |
| CCC=CC1CC1                         |
| CCO                                |
| CC1=CC1                            |
| CC(C)(C)C1(C)CC1                   |
| CC=CCCC                            |
| CC(CO)C1(C)C(C)CC12CC2             |
| …………                               |
# 5. Classifier
In Classifier fold, we have three classifiers, They used Gaussian process, random forest and semi-supervised learning algorithms respectively.
## 5.1 Common functions
For Gaussian process classifier and random forest classifier, they both have some common function.  
### 5.1.1 read_data
```
def read_data(csv_file):
    print("Reading dataset...")
    df = pd.read_csv(csv_file)
    assert 'Smile' in df.columns, "Smile column is missing in the dataset"
    assert 'Type' in df.columns, "Type column is missing in the dataset"
    print("Dataset reading completed.")
    return df
```
**Meaning**: Reading the database.  
**Input**: Dataset.csv  
**Output**: Dataset.csv  
**Usage**:
```
csv_file = 'Dataset.csv'
df = read_data(csv_file)
print(df)
```  
### 5.1.2 encode_labels
```
def encode_labels(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le
```
**Meaning**: Convert category labels to numeric labels.  
**Input**: **y**  This is the column in your database that contains the categories.  
**Output**: The encoded numeric labels and the LabelEncoder object used for encoding.
```
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 0 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 0 1 0
 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
 0 1 1 1 1 0 0 1 0 1 1 0 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 1 1 0 0 1
 0 1 1 1 0 1 1 1 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 0 1 1 1 1 1 0 0 0 1 1
 0 1 0 1 0 1 1 0 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 0 0 1 1 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 1 1 1 0 1 0 0 0 1 0 0 1 1 1 1 0 0 1 1 0 0 0 1 0 1 1 0 1 0 1 0
 1 1 0 1 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 1 0 1 1
 1 1 1 1 1 0 0 1 1 0 0 0 0 0 0 0 1 0 0 0 1 0 0 1 1 1 1 1 0 0 0 1 0 0 0 1 0
 1 1 1 1 1 0 0 0 0 1 0 0 1 1 1 0 0 1 1 1 1 1 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 1 1 0 0 0 1 0 0 0 1 0 0 0 1 0 1 1 1 0 1 0 0 1 0 1 1 0 0 0
 1 0 1 1 0 0 0 0 0 0 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 0 0 1
 1 0 0 1 1 0 0 1 1 0 0 1 1 1 1 0 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 0 0 0 0 1 0 0 1 0 0 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 1 1 1 1]  
LabelEncoder()
```
**Usage**:  
```
y, label_encoder = encode_labels(df['Type'].values)
print(y)
print(label_encoder)
```
### 5.1.3 smiles_to_descriptors
```
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"RDKit could not parse Smile: {smiles}"
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in Descriptors._descList])
    descriptors = calc.CalcDescriptors(mol)
    return np.array(descriptors)
```
**Meaning**: Convert SMILES representations of chemical molecules into molecular descriptors.  
**Input**: SMILES
**Output**: Molecular descriptors  
**Usage**:  
```
df['Descriptors'] = df['Smile'].apply(smiles_to_descriptors)
print(df['Descriptors'])
```
### 5.1.4 apply_descriptors  
```
def apply_descriptors(df):
    print("Applying molecular descriptors...")
    df['Descriptors'] = df['Smile'].apply(smiles_to_descriptors)
    X = np.stack(df['Descriptors'])
    y, label_encoder = encode_labels(df['Type'].values)
    smiles = df['Smile'].values  # Store Smile strings
    print("Descriptor conversion completed.")
    return X, y, smiles, label_encoder
```
**Meaning**: Convert the SMILES string for each chemical molecule in the dataset into a molecular descriptor and encode the target label. This is equivalent to the combination of the two functions encode_labels and smiles_to_descriptors.  
**Input**: **Dataframe**  Here you can get the dataframe by running **read_data**.  
**Output**: Features array, label array, SMILES string array, and label encoder object.  
**Usage**:  
```
X, y, smiles, label_encoder = apply_descriptors(df)
print(X,y,smiles,label_encoder)
```
### 5.1.5 split_data
```
def split_data(X, y, smiles):
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(X, y, smiles, test_size=0.2, random_state=42)
    assert len(X_train) > 0 and len(X_test) > 0, "Training or testing set is empty"
    print("Data splitting completed.")
    return X_train, X_test, y_train, y_test, smiles_train, smiles_test
```
**Meaning**: Divide the dataset into training and test sets, and ensure that the training and test sets after the split are not empty.  
**Input**: The feature array X, the label array y, and the SMILES string array smiles.  
**Output**: Split training and test set features, labels, and SMILES strings.  
**Usage**:  
Here, You can achieve different ratios by changing the parameter **test_size**.
```
X_train, X_test, y_train, y_test, smiles_train, smiles_test = split_data(X, y, smiles)
print(X_train, X_test, y_train, y_test, smiles_train, smiles_test)
```
### 5.1.6 low_variance_filter
```
def low_variance_filter(X, threshold=0.01):
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    assert X_filtered.shape[1] > 0, "All features removed by low variance filter"
    return X_filtered
```
**Meaning**: Apply a low variance filter to remove features whose variance is below a specified threshold.  
**Input**: The feature matrix X and the variance threshold threshold.    
**Output**: Feature matrix after low variance filtering.    
**Usage**:  
```
X_train = low_variance_filter(X_train)
print(X_train)
```
### 5.1.7 correlation_filter
```
def correlation_filter(X_train, X_test, threshold=0.95):
    corr_matrix = np.corrcoef(X_train, rowvar=False)
    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    to_drop = [column for column in range(corr_matrix.shape[1]) if any(corr_matrix[column, row] > threshold for row in range(corr_matrix.shape[0]))]
    X_train_filtered = np.delete(X_train, to_drop, axis=1)
    X_test_filtered = np.delete(X_test, to_drop, axis=1)
    assert X_train_filtered.shape[1] > 0, "All features removed by correlation filter"
    return X_train_filtered, X_test_filtered
```
**Meaning**: By calculating the correlation between the features in the training set, highly correlated features are removed.  
**Input**: Training set feature matrix **X_train**, test set feature matrix **X_test**, correlation threshold **threshold**.  
**Output**: Feature matrices of the training and test sets after correlation filtering.  
**Usage**:  
```
X_train, X_test = correlation_filter(X_train, X_test)
print(X_train,X_test)
```
### 5.1.8 preprocess_data
```
def preprocess_data(X_train, X_test, smiles_train, smiles_test, apply_low_variance=True, apply_correlation_filter=True):
    print("Starting preprocessing...")
    if apply_low_variance:
        print("Applying low variance filter...")
        X_train = low_variance_filter(X_train)
        X_test = low_variance_filter(X_test)
    if apply_correlation_filter:
        print("Applying correlation filter...")
        X_train, X_test = correlation_filter(X_train, X_test)
    print("Preprocessing completed.")
    return X_train, X_test
```
**Meaning**: Preprocess the training and test feature data based on the user's choice, including applying low variance filters and correlation filters.  
**Input**: The training set feature matrix **X_train**, the test set feature matrix **X_test**, the SMILES strings **smiles_train** and **smiles_test** for the training and test sets, and two Boolean parameters **apply_low_variance** and **apply_correlation_filter** to control whether to apply the corresponding filters.  
**Output**: Feature matrices of the training and test sets after preprocessing.  
**Usage**:  
Here, apply_low_variance and apply_correlation_filter default to True, you can change them to False.
```
    apply_low_variance = False
    apply_correlation_filter = False

    X_train_preprocessed, X_test_preprocessed = preprocess_data(X_train, X_test, smiles_train, smiles_test, apply_low_variance, apply_correlation_filter)
```

### 5.1.9 baseline_model  
This funcation contains **compute_metrics** function. And here the Gaussian process classifier is used as the baseline model.  
```
def baseline_model(X_train, y_train, X_test, y_test, label_encoder):
    print("Training baseline model...")
    kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2))
    model = GaussianProcessClassifier(kernel=kernel, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    # Additional assertions
    assert len(np.unique(y_pred)) > 1, "Prediction resulted in only one class"
    assert y_proba.min() >= 0 and y_proba.max() <= 1, "Probabilities are not between 0 and 1"
    print(f"Debug: y_test: {y_test[:10]}, y_proba: {y_proba[:10]}")
    metrics = compute_metrics(y_test, y_pred, y_proba, label_encoder)
    print("Baseline model training completed.")
    return model, metrics
```
**Meaning**: Use the GaussianProcessClassifier as the baseline model and calculate various performance indicators of the model on the test set.  
**Input**:  
**X_train**: training set feature matrix.
**y_train**: training set label array.
**X_test**: test set feature matrix.
**y_test**: test set label array.
**label_encoder**: label encoder object, used to convert numerical labels back to original category labels.
**Output**: The trained model and the performance indicators of the model on the test set.  
**Usage**:  
```
baseline_model_result, baseline_metrics = baseline_model(X_train_preprocessed, y_train, X_test_preprocessed, y_test, label_encoder)
print(baseline_model_result,baseline_metrics)
```

### 5.1.10 compute_metrics
```
def compute_metrics(y_true, y_pred, y_proba, label_encoder):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'f1_score': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, output_dict=True, target_names=label_encoder.classes_)
    }
    return metrics
```
**Meaning**: Calculates and returns various performance metrics predicted by the model.  
**Input**:  
**y_true**: true label array.  
**y_pred**: predicted label array.  
**y_proba**: predicted probability array.  
**label_encoder**: label encoder object, used to convert numerical labels back to original category labels.  
**Output**: A dictionary containing various performance metrics.  
**Usage**:  
```
metrics = compute_metrics(y_test, y_pred, y_proba, label_encoder)
print(metrics)
```
### 5.1.11 plot_and_save_roc
```
def plot_and_save_roc(y_true, y_proba, label_encoder, filename='roc_curve.png'):
    # Find the positive label
    pos_label = label_encoder.transform(['biofuel'])[0]
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    print(f"Debug: FPR: {fpr}, TPR: {tpr}, Thresholds: {thresholds}")
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
```
**Meaning**: Plot the ROC curve and save it as an image file.  
**Input**:  
**y_true**: true label array.   
**y_proba**: predicted probability array.  
**label_encoder**: label encoder object, used to convert numerical labels back to original category labels.  
**filename**: The filename to save the image to (default is 'roc_curve.png').
**Output**: Generate and save ROC curve graphs.  
**Usage**:  
```
plot_and_save_roc(y_test, baseline_model_result.predict_proba(X_test_preprocessed)[:, 1], label_encoder, filename='baseline_roc_curve.png')
```
### 5.1.12 optimize_hyperparameters_tpot
```
def optimize_hyperparameters_tpot(X, y):
    print("Starting hyperparameter optimization with TPOT...")
    tpot = TPOTClassifier(generations=5, population_size=20, cv=5, random_state=42, verbosity=2, config_dict='TPOT sparse')
    tpot.fit(X, y)
    print("Hyperparameter optimization with TPOT completed.")
    return tpot.fitted_pipeline_, tpot.fitted_pipeline_.get_params()
```
**Meaning**: Automatically optimize hyperparameters using TPOT (Tree-based Pipeline Optimization Tool). **TPOT** is an open source Python library that uses genetic algorithms to automatically find the best machine learning pipeline.  
**Input**:  
**X**: feature matrix.  
**y**: label array.  
**Output**: The optimized pipeline and its hyperparameters.  
**Usage**:  
```
optimize_hyperparameters_tpot(X_train_preprocessed, y_train)
```
### 5.1.13 optimize_hyperparameters_grid_search
```
def optimize_hyperparameters_grid_search(X, y):
    print("Starting hyperparameter optimization with Grid Search...")
    param_grid = {
        'kernel__k1__constant_value': [0.1, 1, 10],
        'kernel__k2__length_scale': [0.1, 1, 10],
        'kernel__k1__constant_value_bounds': [(1e-2, 1e2)],
        'kernel__k2__length_scale_bounds': [(1e-2, 1e2)],
    }
    kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2))
    grid_search = GridSearchCV(estimator=GaussianProcessClassifier(kernel=kernel, random_state=42),
                               param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    print("Hyperparameter optimization with Grid Search completed.")
    return grid_search.best_estimator_, grid_search.best_params_
```
**Meaning**: Use Grid Search to optimize the hyperparameters.  
**Input**:  Same with optimize_hyperparameters_tpot.  
**Output**: The best estimator and the best parameter combination.  
**Usage**：  
```
optimize_hyperparameters_grid_search(X_train_preprocessed, y_train)
```
# 6.Gan
In Gan fold, we use MolGAN to generated unique vaild molecules.  
The MolGAN network was initially presented by Cao and Kipf in their work "MolGAN: An implicit generative model for small molecular graphs[[2]](https://arxiv.org/abs/1805.11973)  
In MolGAN, we have many parameters, here we will introduce some important parameters in MolGAN. You can change these parameters to explore MolGAN.  
```
num_atoms = 12
```
The **num_atoms** parameter determines the maximum size of molecules to be processed and generated. The size of the num_atoms parameter directly affects the model's efficiency, data processing capabilities, and the diversity and complexity of the generated results.
```
gan.fit_gan(iterbatches(25), generator_steps=0.2, checkpoint_interval=5000)
```
Here, iterbatches is a funcation。 In this function, you need to pass in the value of **epoch**(25). This means the model will be trained for 25 epochs, with each epoch going through the entire dataset.The size of the epochs value directly affects the model training time, performance, convergence, and risk of overfitting.  

The **generator_steps** parameter controls the frequency and number of steps that the generator updates relative to the discriminator. Here, generator_steps is set to 0.2, so the generator will be updated 0.2 times after each discriminator update, that is, the generator will be updated once after every 5 discriminator updates. The size of the generator_steps parameter affects the update frequency and balance of the generator and discriminator during training.  

```
generated_data = gan.predict_gan_generator(10000)
```
Here, we define the number of molecules we want to generate as 10000. A larger number means more molecules will be generated and more time will be required to generate them. But please make sure you have enough molecular samples for GAN to learn.

# 7.RDkit
Once you have the **SMILES** of a molecule, you also need enough descriptors for the classifier to learn the molecular features. These descriptors can be calculated using RDkit.
```
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"RDKit could not parse Smile: {smiles}"
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in Descriptors._descList])
    descriptors = calc.CalcDescriptors(mol)
    return np.array(descriptors)
```
The **smiles_to_descriptors** function converts a molecule's SMILES representation into a set of molecular descriptors. Specifically, it uses RDKit to parse the SMILES string into a molecule object, checks if the parsing is successful, calculates all the molecular descriptors using a molecular descriptor calculator, and finally returns the descriptors as a NumPy array. These descriptors can be used as features in cheminformatics and molecular modeling to predict the chemical and physical properties of the molecule.
# 8. Pairplot
This is the data exploration part. In this part, you can use the code in pairplot to see the relationship between the different descriptors.  
```
data = pd.read_csv("Orignal Dataset.csv")
selected_feature = ['MinPartialCharge', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'Chi2v','VSA_EState8', "Type"]
```
Here you just need to substitute your database and change the features (descriptors) you want to see.
![image](https://github.com/aliezwz/ZheWang_BiofuelDiscovery/assets/147146769/d38cfa2c-083f-4694-923c-af63de4896a9)

# 9.Reference
[1] [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system)  
[2] [Molgan](https://arxiv.org/abs/1805.11973)

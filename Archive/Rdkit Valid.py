from rdkit import Chem
from rdkit.Chem import Draw

def validate_and_draw_smiles(smiles):
    # Create a molecule object from the SMILES string
    molecule = Chem.MolFromSmiles(smiles)

    # Check if the molecule object was created successfully (valid SMILES)
    is_valid_smiles = molecule is not None

    if is_valid_smiles:
        print("The SMILES string is valid.")
        # Generate a 2D image of the molecule
        molecule_image = Draw.MolToImage(molecule)
        molecule_image.show()
    else:
        print("The SMILES string is invalid.")

# Example usage
smiles_string = "COc1ccc2[nH+]ccc2ncc1CC(=O)NCC1"
validate_and_draw_smiles(smiles_string)

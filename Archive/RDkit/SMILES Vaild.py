from rdkit import Chem
from rdkit.Chem import Draw


def validate_and_draw_smiles(smiles):
    # Try to create a molecule from the SMILES string
    molecule = Chem.MolFromSmiles(smiles)

    # Check if the molecule was successfully created
    if molecule is not None:
        print("The SMILES string is valid.")
        # Draw the 2D structure of the molecule
        return Draw.MolToImage(molecule)
    else:
        print("Invalid SMILES string.")
        return None


# Example SMILES string
smiles = "CC(C)(CO)C(O)O"
image = validate_and_draw_smiles(smiles)

# If image is not None, display the image
if image:
    image.show()

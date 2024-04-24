from pathlib import Path

from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Draw import rdMolDraw2D


def create_mol_image(smiles: str, filepath: str) -> None:
    mol = MolFromSmiles(smiles)
    d = rdMolDraw2D.MolDraw2DSVG(-1, -1)
    d.drawOptions().padding = 0.02
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    with open(filepath, "w") as file:
        file.write(d.GetDrawingText())


if __name__ == "__main__":
    plots_dir = "../plots/molecules"
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # Decalin
    decalin_smiles = "C1CCC2CCCCC2C1"
    create_mol_image(decalin_smiles, f"{plots_dir}/mol_decalin.svg")

    # Bicyclopentyl
    bicyclopentyl_smiles = "C1CCC(C1)C2CCCC2"
    create_mol_image(bicyclopentyl_smiles, f"{plots_dir}/mol_bicyclopentyl.svg")

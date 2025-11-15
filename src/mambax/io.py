"""File I/O operations for molecular data."""

import logging
import re
from typing import Any, List, Tuple

from rdkit.Chem import AllChem as Chem


def read_xyz(filename: str, skipH: bool = False) -> Tuple[List[str], List[List[float]], float, str]:
    """Read XYZ file and extract atom types, coordinates, charge, and title."""
    with open(filename, 'r') as fin:
        natoms = int(fin.readline())
        title = fin.readline()[:-1]
        q = 0
        qin = re.search(r"(?:CHARGE|CHG)=([-+]?\d*\.\d+|\d+|[-+]?\d)", title, re.IGNORECASE)
        if qin:
            q = float(qin.group(1))
        coords = []
        atomtypes = []
        for x in range(natoms):
            line = fin.readline().split()
            if (line[0].lower() == 'h') and skipH:
                continue
            atomtypes.append(line[0])
            coords.append([float(line[1]), float(line[2]), float(line[3])])

    return atomtypes, coords, q, title


def read_mol(mol: Any) -> Tuple[List[str], List[List[float]], int, str]:
    """Extract atom types, coordinates, charge, and title from RDKit molecule."""
    title = ""
    coords = []
    atomtypes = []
    for i, atom in enumerate(mol.GetAtoms()):
        an = atom.GetSymbol()
        if an == "H":
            logging.warning("PDB file should not contain hydrogens!")
        atomtypes.append(an)
        pos = mol.GetConformer().GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])

    q = Chem.GetFormalCharge(mol)
    return atomtypes, coords, q, title


def read_pdbfile(filename: str, skipH: bool = False) -> Tuple[List[str], List[List[float]], int, str]:
    """Read PDB file and extract atom types, coordinates, charge, and title."""
    mol = Chem.MolFromPDBFile(filename)
    atomtypes, coords, q, title = read_mol(mol)
    return atomtypes, coords, q, title


def create_sdfile(name: str, atomtypes: List[str], coords: List[List[float]], df) -> str:
    """Create SDF molblock string from atom types, coordinates, and bond dataframe."""
    df = df[df.bond > 0]

    ins = name + "\n"
    ins += "ML generated sdf\n"
    ins += "\n"
    ins += "%3d%3d  0  0  0  0  0  0  0  0  1 V2000\n" % (len(atomtypes), (df.bond > 0).sum())

    for at, xyz in zip(atomtypes, coords):
        ins += "%10.4f%10.4f%10.4f %-2s 0  0  0  0  0\n" % (xyz[0], xyz[1], xyz[2], at)

    for index, row in df.iterrows():
        ins += "%3d%3d%3d  0  0  0  0\n" % (row['id1'], row['id2'], row['bond'])

    ins += "M  END"
    return ins


def mol2xyz(mol: Any) -> str:
    """Convert RDKit molecule to XYZ format string."""
    atomtypes, coords, q, _ = read_mol(mol)

    xyz_str = "%d\n\n" % len(atomtypes)
    for at, xyz in zip(atomtypes, coords):
        xyz_str += "%3s%10.4f%10.4f%10.4f\n" % (at, xyz[0], xyz[1], xyz[2])

    return xyz_str


def convert_smiles2sdfile(
    smifile: str = "./smi_repository/bbp2.smi",
    method: str = 'UFF',
    outdat: str = 'train_dat.csv'  # Unused but kept for API compatibility
) -> str:
    """Generate 3D SDF file from SMILES using force-field optimization."""
    logging.info("Selected optimization method: %s"%(method))
    mols = Chem.SmilesMolSupplier(smifile,delimiter='\t ',titleLine=False, sanitize=True)
    mols = [x for x in mols if x is not None]

    sdfile = smifile.replace('.smi', '_' + method + '.sdf')
    w = Chem.SDWriter(sdfile)

    for i, m in enumerate(mols):
        tmp = Chem.GetMolFrags(m, asMols=True)
        if len(tmp) > 1:
            logging.info("Using only first fragment: %s" % Chem.MolToSmiles(m))
            m = tmp[0]
        m = Chem.AddHs(m)
        conv = 0
        try:
            if method == 'UFF':
                Chem.EmbedMolecule(m)
                conv = Chem.UFFOptimizeMolecule(m, maxIters=200)
            elif method == 'MMFF':
                Chem.EmbedMolecule(m)
                conv = Chem.MMFFOptimizeMolecule(m, maxIters=200)
            elif method == 'ETKDG':
                Chem.EmbedMolecule(m, Chem.ETKDG())
            else:
                Chem.EmbedMultipleConfs(m, 10, Chem.ETKDG())

            if conv == 0 or conv == 1:
                Chem.SanitizeMol(m)
                smiles = Chem.MolToSmiles(Chem.RemoveHs(m))
                m.SetProp("SMILES", smiles)
                w.write(m)
            if conv == 1:
                logging.info("Optimization not converged for molecule: %d with SMILES: %s" % (i, Chem.MolToSmiles(Chem.RemoveHs(m))))
            elif conv == -1:
                logging.info("Forcefield could not be setup for molecule: %d with SMILES: %s" % (i, Chem.MolToSmiles(Chem.RemoveHs(m))))

        except ValueError:
            logging.info("Optimization problem for molecule: %d with SMILES: %s"%(i,Chem.MolToSmiles(Chem.RemoveHs(m))))

    w.close()
    logging.info("Writing to SD file:" + sdfile)
    return sdfile


from ase.build import bulk
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from ase.units import GPa

calc = FAIRChemCalculator.from_model_checkpoint(
    "/mnt/shared-storage-user/lijielan/checkpoints/uma-s-1p1.pt", task_name="omat", device="cpu"
)

atoms = bulk("Fe")
atoms.calc = calc

print(f"Energy (eV)                 = {atoms.get_potential_energy()}")
print(f"Energy per atom (eV/atom)   = {atoms.get_potential_energy()/len(atoms)}")
print(f"Forces of first atom (eV/A) = {atoms.get_forces()[0]}")
print(f"Stress[0][0] (eV/A^3)       = {atoms.get_stress(voigt=False)[0][0]}")
print(f"Stress[0][0] (GPa)          = {atoms.get_stress(voigt=False)[0][0] / GPa}")

opt = FIRE(FrechetCellFilter(atoms))
opt.run(0.05, 100)
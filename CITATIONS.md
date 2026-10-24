# Citations

If you use this pipeline or the curated ChEMBL library in your research,
please cite the underlying tools and data resources.

## Pipeline components

- **DiffDock** (blind diffusion docking)
  Corso, G., Stärk, H., Jing, B., Barzilay, R., & Tommi, J. (2023).
  DiffDock: Diffusion steps, twists, and turns for molecular docking.
  *International Conference on Learning Representations (ICLR)*.
  https://github.com/gcorso/DiffDock

- **GNINA** (3D CNN docking & affinity rescoring)
  McNutt, A. et al. (2021). GNINA 1.0: molecular docking with deep learning.
  *PLOS Computational Biology*, 17(6): e1009070.
  https://github.com/ccob00/gnina

- **P2Rank** (binding pocket detection)
  Kran, M. et al. (2020). P2Rank: machine learning based binding site
  prediction and its application to drug design.
  *Nucleic Acids Research*, 48(W1): W33–W39.
  https://github.com/rdk/p2rank

- **ADMET-AI** (BBB, hERG, DILI triage)
  Cherkasov, A. et al. ADMET-AI: an accessible artificial intelligence-driven
  ADMET evaluation platform. (See https://admet-ai.github.io/)
  https://github.com/AdrianoBaffi/admet-ai

- **REINVENT4** (de novo ligand generation)
  MolecularAI. REINVENT4.
  https://github.com/MolecularAI/REINVENT4

- **LocalColabFold / AlphaFold2** (structure prediction)
  Jumper, J. et al. (2021). Highly accurate protein structure prediction with
  AlphaFold. *Nature*, 596, 583–589.
  Mirdita, M. et al. (2022). ColabFold—making protein folding accessible.
  *Nature Methods*, 19, 679–682.
  https://github.com/YoshitakaMo/localcolabfold

- **RDKit** (cheminformatics)
  Landrum, G. et al. RDKit: Open-source cheminformatics.
  https://www.rdkit.org

## Data resources

- **ChEMBL** (curated LRRK2/GBA1 bioactivities)
  Gaulton, A. et al. (2023). ChEMBL: a large-scale bioactivity database for
  drug discovery. *Nucleic Acids Research*, 51(D1): D1152–D1159.
  https://www.ebi.ac.uk/chembl

- **RCSB PDB** (LRRK2 7LHT, GBA1 2V3D structures)
  Berman, H. M. et al. (2000). The Protein Data Bank.
  *Nucleic Acids Research*, 28(1): 235–242.
  https://www.rcsb.org

## Software dependencies

- OpenMM, PDBFixer, PDB2PQR (target preparation)
- Open Babel (format conversion)
- Meeko (AutoDock Vina/Meeko PDBQT conversion)
- PubChemPy, chembl_webresource_client (data access)
- Paramiko (SSH orchestration)

## Disclaimer

This is a computational triage workflow. Docking scores and ADMET predictions
are noisy in-silico estimates that prioritize compounds for experiments; they
are not evidence of efficacy or safety, and nothing here is medical advice.
Target PDB IDs and the curated compound library are provided for research use;
confirm construct, mutation status and compound identity before drawing
biological conclusions.

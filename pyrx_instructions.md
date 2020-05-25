1. Download PyRx from https://sourceforge.net/projects/pyrx/

2. Watch this video as a tutorial https://www.youtube.com/watch?v=2t12UlI6vuw&t=1092s

3. Download SARS-CoV-2 main protease with unliganded active site from PDB website https://www.rcsb.org/. Just look for the PDB id _6M03_ in the search box or click here https://www.rcsb.org/structure/6M03

4. Open PyRX and edit some settings `Edit > Preferences`. Modify the number of CPUs and the energy minimization in Open Babel

5. Load the .pdb file of the macromolecule. `File > Load Molecule`. Right click on the molecule in the Navigator pane and click `AutoDock > Make Macromolecule`

6. Import the ligands. `File > Import > Chemical Table File - SDF`. Now in `Controls` window go to `Open Babel`, right clik and select minimize all. This  finds the minimun energy configuration for each of the molecules. Now right click again and select `Convert All to AutoDock Ligand (pdbqt)`

7. In the AutoDock tab select all ligands and the macromolecule. Now in `Controls > Vina Wizard > Select Molecules` will show the number of ligands and the macromolecule selected. Click `Forward`. Select the grid and click `Forward`.

8. The simulations starts. It will take about 20 seconds per ligand
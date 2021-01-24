Contributors: Emily Nguyen UC Berkeley, Roy Ben-Shalom UCSF
Last Edited: written Sept 28 2020, documented Jan 22 2021.
Hidden Markov Model of Na1.2 channel.

Developed to better fit a model to Recovery from Inactivation (RFI) experimental data
	compared to a Hodgkin–Huxley model.

See 'genSimData_Na12_mut_HMM_tutorial.ipynb' for example on how to run.

Required files to run 'genSimData_Na12_mut_HMM.py':
1) mod file
	- 'Na15.mod' 
		-- Currently configured to be run.
		-- Fits best to mutant experimental data 
	- 'na12.mod' 
	- 'na12_mut.mod'

2) 'state_variables.py'

3) 'vclmp_pl.mod'
	- same code as vclmp_pl.mod for HH model 

Notes: 
* To run different mod files, 
	some code in genSimData_Na12_mut_HMM.py may need to be changed.
* Code must be compiled before running.
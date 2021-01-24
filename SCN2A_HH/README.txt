Last Edited: written Fall 2019, latest edit Dec 2020.
Hodgkin–Huxley Model of Na1.2 channel.

See ipynb for example on how to run.

Required files to run 'generalized_genSim_shorten_time.py':
1) mod file
	- 'na12_mut.mod'
		-- Currently configured to be run.

2) 'vclmp_pl.mod'

Notes: 
* To run different mod files, 
	some code in genSim.py may need to be changed.
* Code must be compiled before running.

References :

genSim functions: 
    DOI: https://doi.org/10.1038/s41598-019-53662-9
        --5_sl_inact_rec_static.py for RFI and RFI tau functions
        --1_I_V_relationship_static.py for act and inact functions
        

RFI tau calculation: 
    DOI: https://doi.org/10.4049/jimmunol.0803003
        -- eqn (4)
        -- conventional two pulse protocol 
# -*- coding: utf-8 -*-

## This code calculates the influence matrix and response matrix for a
## Shack-Hartmann wavefront sensor + deformable mirror system

def shack_hartmann_calibrator(wf, shwfs, shwfse, det, dm, amp):
    
    num_modes = len(dm.influence_functions)
    
    act_levels = np.zeros(num_modes)
    
    Infmat = []
    
    for dm_mode in np.arange(num_modes):
        
        act_levels = np.zeros(num_modes)
        act_levels[dm_mode] = amp
        dm.actuators = act_levels
        
        wfdm = dm.forward(wf)
        wfmla = shwfs.forward(wfdm)
        det.integrate(wfmla, 1, 1)
        mla_img = det.read_out()
        
        Splus = shwfse.estimate([mla_img])
        
        act_levels = np.zeros(num_modes)
        act_levels[dm_mode] = -amp
        dm.actuators = act_levels
        
        wfdm = dm.forward(wf)
        wfmla = shwfs.forward(wfdm)
        det.integrate(wfmla, 1, 1)
        mla_img = det.read_out()
        
        Sminus = shwfse.estimate([mla_img])
        
        shift = (Splus + Sminus) / (2 * amp)
        Infmat.append(shift)
    
    return Infmat
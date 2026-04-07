#!/usr/bin/env python3
import argparse, json
import numpy as np
import meep as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulationTemplate import MEEP_2x2Splitter, MEEP_1x2Splitter
#from datetime import datetime

mp.verbosity(0) # suppress meep output

def simulate_designs(hole_flag, diameters, template):
    if template == "2x2":
        # parameters for the simulation are all contained within the template 
        device_params = MEEP_2x2Splitter(tapered=False)
                      
        # refactor the design and hole mapping
        hf = np.asarray(hole_flag).reshape(30,14) # since the design region is divided into 30x14 cells
        di = np.asarray(diameters).reshape(30,14)
        
        cylinders = []
        for i in range(30):
            cx = -3.6/2 + (i + 0.5) * (3.6/30)
            for j in range(14):
                cy = 1.68/2 - (j + 0.5) * (1.68/14)
                if hf[i,j] > 0:
                    d = float(di[i,j])
                    r = 0.5 * d
                    cylinders.append(mp.Cylinder(
                        center=mp.Vector3(cx,cy),
                        radius=r,
                        height=mp.inf, # 2D sims do not have height
                        material=device_params.SiO2
                    ))
        
        # adding the geometry hole to the geometry block
        for c in cylinders:
            device_params.geometry.append(c)
            
        # source
        device_params.defineSources()
            
        # define simulation object
        device_params.defineSim()
        
        # add flux monitors
        device_params.addFluxMonitors()
        
        # plot this shit so I can see if it's crap
        #time_stmp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        device_params.sim.plot2D()
        plt.savefig('current_sim_design.png')
        
        # run the simulation under the above defined parameters
        device_params.sim.run(
                until=mp.stop_when_fields_decayed(
                    50,
                    mp.Ey,
                    mp.Vector3(device_params.sx*0.5 - device_params.waveguide_length*0.3,
                                  device_params.arm_separation*0.5 + device_params.waveguide_width*0.5),
                    1e-1
                    )
                )
                
        # retrieve the fluxes
        incidentFluxBot = mp.get_fluxes(device_params.incidentMonitorBot)
        transFluxTop = mp.get_fluxes(device_params.transMonitorTop)
        transFluxBot = mp.get_fluxes(device_params.transMonitorBot)
        freqs = mp.get_flux_freqs(device_params.incidentMonitorTop)
        
        # calculate Tt and Tb
        W = []
        Tt = []
        Tb = []
        for i in range(len(freqs)):
            W.append(1.0/(freqs[i]))
            Tt.append(transFluxTop[i]/(incidentFluxBot[i]))
            Tb.append(transFluxBot[i]/(incidentFluxBot[i]))
            
        Trans = [Tt, Tb]
    
    else:
        device_params = MEEP_1x2Splitter()
        
        hf = np.asarray(hole_flag).reshape(16,16)
        di = np.asarray(diameters).reshape(16,16)
        
        cylinders = []
        for i in range(16):
            cx = -2/2 + (i + 0.5) * (2/16)
            for j in range(16):
                cy = 2/2 - (j + 0.5) * (2/16)
                if hf[i,j] > 0:
                    d = float(di[i,j])
                    r = 0.5 * d
                    cylinders.append(mp.Cylinder(
                        center=mp.Vector3(cx,cy),
                        radius=r,
                        height=mp.inf, # 2D sims do not have height
                        material=device_params.SiO2
                    ))
        
        # adding the geometry hole to the geometry block
        for c in cylinders:
            device_params.geometry.append(c)
        
        # define sources
        device_params.defineSources()
        
        # define simulation object
        device_params.defineSim()
        
        # add flux monitors
        device_params.addFluxMonitors()
        
        # plot this shit so I can see if it's crap
        #time_stmp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        device_params.sim.plot2D()
        plt.savefig('current_sim_design.png')
            
        device_params.sim.run(
                until=mp.stop_when_fields_decayed(
                    50,
                    mp.Ey,
                    mp.Vector3(device_params.MMI_x*0.5 + device_params.taper_length*0.65, 
                                0.25+device_params.waveguide_l_port*0.25),
                    1e-1
                    )
                )
        
        # retrieve frequencies
        flux_freqs = mp.get_flux_freqs(device_params.incidentFluxMonitor)
    
        # retieve fluxes
        incident_flux = mp.get_fluxes(device_params.incidentFluxMonitor)
        
        output_flux_one = mp.get_fluxes(device_params.outputFluxMonitorOne)
        output_flux_two = mp.get_fluxes(device_params.outputFluxMonitorTwo)
        
        # plot transmissions
        T1 = list()
        T2 = list()
        Ws = list()
        
        for i in range(len(flux_freqs)):
                Ws.append(1/flux_freqs[i])
                T1.append(output_flux_one[i]/incident_flux[i])
                T2.append(output_flux_two[i]/incident_flux[i])
                
        Trans = [T1, T2]
    
    return {"Transmission": Trans}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--param_file")
    p.add_argument("--out_file")
    p.add_argument("--template")
    args = p.parse_args()

    with open(args.param_file, "r") as f:
        param_dict = json.load(f)
    
    template = args.template if args.template in ["1x2","2x2"] else "2x2"
    
    # Convert lists back to numpy
    hole_flag = np.array(param_dict["hole_flag"])
    diameters = np.array(param_dict["diameters"])

    result = simulate_designs(hole_flag, diameters, template)

    with open(args.out_file, "w") as f:
        json.dump(result, f)


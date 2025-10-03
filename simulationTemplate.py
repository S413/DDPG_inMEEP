# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 23:39:16 2024

@author: sergi
"""

import meep as mp
import matplotlib.pyplot as plt

# we need a template for the meep simulation, such that we can test the GAN
# In actuality, it'd be best ot just have this for all applications, since it'd be more consistent and clean

class MEEP_2x2Splitter():
    def __init__(self):
        # I say we put here some things that will likely not change since they have been constant
        self.waveguide_width = 0.45
        self.design_region_width = 3.6
        self.design_region_height = 1.68
        self.arm_separation =  0.6 # min to avoid jumping source
        self.waveguide_length = 3.0
        self.pml_size = 1.0
        self.resolution = 100
        
        self.Si = mp.Medium(index=3.4)
        self.SiO2 = mp.Medium(index=1.4)
        
        self.sx = 2*self.pml_size + 2*self.waveguide_length + self.design_region_width
        self.sy = 2*self.pml_size + self.design_region_height + 1.0 # magick number, just to give some space between edges and pml prob
        self.cell_size = mp.Vector3(self.sx, self.sy)
        
        self.pml_layers = [mp.PML(self.pml_size),
              mp.Absorber(self.pml_size)]
        
        self.geometry = [
            mp.Block(
                center=mp.Vector3(-self.sx/2 + self.pml_size + 0.1, self.arm_separation/2 + self.waveguide_width/2, 0),
                material=self.Si,
                size=mp.Vector3(self.sx/2, self.waveguide_width)
                ), # top left waveguide
            mp.Block(
                center=mp.Vector3(-self.sx/2 + self.pml_size + 0.1, -self.arm_separation/2 - self.waveguide_width/2, 0),
                material=self.Si,
                size=mp.Vector3(self.sx/2, self.waveguide_width),
                ), # bottom left waveguide
            mp.Block(
                center=mp.Vector3(self.sx/2 - self.pml_size - 0.1, self.arm_separation/2 + self.waveguide_width/2, 0),
                material=self.Si,
                size=mp.Vector3(self.sx/2, self.waveguide_width),
                ), # top right waveguide
            mp.Block(
                center=mp.Vector3(self.sx/2 - self.pml_size - 0.1, -self.arm_separation/2 - self.waveguide_width/2, 0),
                material=self.Si,
                size=mp.Vector3(self.sx/2, self.waveguide_width),
                ), # bottom right waveguide
            mp.Block(
                center=mp.Vector3(),
                material=self.Si,
                size=mp.Vector3(self.design_region_width, self.design_region_height),
                ), # design region
            ]
        
        self.sources = None
        
        self.sim = None
        
    def defineSources(self):
        fcen = 1.0/1.55
        fwidth = fcen * 0.04
        
        src_center = mp.Vector3(-self.sx/2 + self.pml_size + self.waveguide_length/3, -self.arm_separation/2 - self.waveguide_width/2, 0)
        src_size = mp.Vector3(0, self.waveguide_width*1.2, 0) 
        
        eig_parity = mp.EVEN_Z + mp.ODD_Y
        kpoint = mp.Vector3(0, 0, 0)
        
        src = mp.GaussianSource(fcen, fwidth)
        
        self.sources = [
            mp.EigenModeSource(
                src,
                eig_band=1,
                # direction=mp.NO_DIRECTION,
                # eig_kpoint=kpoint,
                eig_parity=eig_parity,
                size=src_size,
                center=src_center,
                eig_match_freq=True,
                )
            ]
        
    
    def defineSim(self):
        self.sim = mp.Simulation(
            resolution = self.resolution,
            cell_size = self.cell_size,
            boundary_layers = self.pml_layers,
            default_material = self.SiO2,
            sources = self.sources,
            geometry = self.geometry
            )
    
    def plotGeometry(self, filename):
        if self.sim != None:
            self.sim.plot2D()
            plt.savefig(filename)
        else:
            self.sim = mp.Simulation(
                resolution = self.resolution,
                cell_size = self.cell_size,
                boundary_layers = self.pml_layers,
                default_material = self.SiO2,
                geometry = self.geometry
                )
            self.sim.plot2D()
            plt.savefig(filename)
            
            self.sim.reset_meep()
            self.sim = None
            
    def addCylinderGeometries(self, stat, holeRadius):
        cylinders = []
        for i in range(30):
            for j in range(14):
                if stat[i,j]==0:
                    c = mp.Cylinder(
                        center=mp.Vector3(-self.design_region_width/2 + 0.06 + (i)*0.12, self.design_region_height/2 - 0.06 - (j)*0.12),
                        radius=holeRadius,
                        height=mp.inf,
                        material=self.SiO2
                    )
                    cylinders.append(c)
                    
        for c in cylinders:
            self.geometry.append(c)
        
    def reset_sim(self):
        if self.sim != None:
            self.sim.reset_meep()
        else:
            print("There was no simulation object deifned anyhow!")
    
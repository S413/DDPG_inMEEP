# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 23:39:16 2024

@author: sergi
"""

import meep as mp
import matplotlib.pyplot as plt

#TODO: template should take params to decide: source type, etched holes, tapering

class MEEP_2x2Splitter():
    def __init__(self, tapered=True):
        self.waveguide_width = 0.5 
        self.waveguide_width_large = 0.8
        self.design_region_width = 3.6 
        self.design_region_height = 1.68 
        self.arm_separation =  0.4 # min to avoid jumping source?
        self.waveguide_length = 4  
        self.waveguide_length_tape = 4
        self.pml_size = 1.0
        self.resolution = 100
        
        self.Si = mp.Medium(index=3.45)
        self.SiO2 = mp.Medium(index=1.45)
        
        if tapered:
        
            self.sx = 2*self.pml_size + 2*self.waveguide_length + 2*self.waveguide_length_tape + self.design_region_width
            self.sy = 2*self.pml_size + self.design_region_height + 1.0 # magick number, just to give some space between edges and pml prob
            self.cell_size = mp.Vector3(self.sx, self.sy)
            
            self.pml_layers = [mp.PML(self.pml_size)]
            
            # vertices for the tapered waveguides
            in_top = [
                        mp.Vector3(-self.sx*0.5, self.arm_separation*0.5 + self.waveguide_width), #top left vertex
                        mp.Vector3(-self.design_region_width*0.5 - self.waveguide_length_tape, self.arm_separation*0.5 + self.waveguide_width), #top start of taper
                        mp.Vector3(-self.design_region_width*0.5, self.design_region_height*0.5), #top end of taper
                        mp.Vector3(-self.design_region_width*0.5, 0), #bot end of taper
                        mp.Vector3(-self.design_region_width*0.5 - self.waveguide_length_tape, self.arm_separation*0.5), #bot start of taper
                        mp.Vector3(-self.sx*0.5, self.arm_separation*0.5), #bot left vertex
                    ]
            
            
            in_bot = [
                        mp.Vector3(-self.sx*0.5, -self.arm_separation*0.5), #top left vertex
                        mp.Vector3(-self.sx + self.pml_size + self.waveguide_length, -self.arm_separation*0.5), #top start of taper
                        mp.Vector3(-self.design_region_width*0.5, 0), #top end of taper
                        mp.Vector3(-self.design_region_width*0.5, -self.design_region_height*0.5), #bot end of taper
                        mp.Vector3(-self.design_region_width*0.5 - self.waveguide_length_tape, -self.arm_separation*0.5 - self.waveguide_width), #bot start of taper
                        mp.Vector3(-self.sx*0.5, -self.arm_separation*0.5 - self.waveguide_width), #bot left vertex
                    ]
                    
            out_top = [
                        mp.Vector3(self.design_region_width*0.5, self.design_region_height*0.5), #top left vertex
                        mp.Vector3(self.design_region_width*0.5 + self.waveguide_length_tape, self.arm_separation*0.5 + self.waveguide_width), #top start of taper
                        mp.Vector3(self.sx*0.5, self.arm_separation*0.5 + self.waveguide_width), #top end of waveguide
                        mp.Vector3(self.sx*0.5, self.arm_separation*0.5), #bot end of waveguide
                        mp.Vector3(self.design_region_width*0.5 + self.waveguide_length_tape, self.arm_separation*0.5), #bot start of taper
                        mp.Vector3(self.design_region_width*0.5, 0), #bot left vertex
                    ]
                    
            out_bot = [
                        mp.Vector3(self.design_region_width*0.5, 0), #top left vertex
                        mp.Vector3(self.design_region_width*0.5 + self.waveguide_length_tape, -self.arm_separation*0.5), #top start of taper
                        mp.Vector3(self.sx*0.5, -self.arm_separation*0.5), #top end of waveguide
                        mp.Vector3(self.sx*0.5, -self.arm_separation*0.5 - self.waveguide_width), #bot end of waveguide
                        mp.Vector3(self.design_region_width*0.5 + self.waveguide_length_tape, -self.arm_separation*0.5 - self.waveguide_width), #bot start of taper
                        mp.Vector3(self.design_region_width*0.5, -self.design_region_height*0.5), #bot left vertex
                    ] 
                    
            self.geometry = [
                mp.Prism(in_top, height=mp.inf, material=self.Si),
                mp.Prism(in_bot, height=mp.inf, material=self.Si),
                mp.Block(
                    center=mp.Vector3(), 
                    size=mp.Vector3(self.design_region_width, self.design_region_height),
                    material=self.Si,
                    ),
                mp.Prism(out_top, height=mp.inf, material=self.Si),
                mp.Prism(out_bot, height=mp.inf, material=self.Si),
            ]
        
        else:
            self.waveguide_length = 3.0 
            self.sx = 2*self.pml_size + 2*self.waveguide_length + self.design_region_width
            self.sy = 2*self.pml_size + self.design_region_height + 1.0 # magick number, just to give some space between edges and pml prob
            self.cell_size = mp.Vector3(self.sx, self.sy)
            
            self.pml_layers = [mp.PML(self.pml_size)]
            
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
        self.fcen = 1.0/1.55
        f_min = 1.0/1.6
        f_max = 1.0/1.5
        
        self.fwidth = f_max - f_min
        
        src_center = mp.Vector3(-self.sx*0.5 + self.pml_size + self.waveguide_length*0.25, -self.arm_separation*0.5 - self.waveguide_width*0.5 + 0.1, 0)
        src_size = mp.Vector3(0, self.waveguide_width*1.2, 0) 
        
        src = mp.GaussianSource(self.fcen, self.fwidth)
        
        self.sources = [
            mp.Source(
                src,
                size=src_size,
                center=src_center,
                component=mp.Ey,
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
            
    def addFluxMonitors(self):
        nfreq = 100
    
        self.incidentMonitorTop = self.sim.add_flux(
        self.fcen,
        self.fwidth,
        nfreq,
        mp.FluxRegion(
            center=mp.Vector3(-self.sx*0.5 + self.pml_size + self.waveguide_length*0.5,
                              self.arm_separation*0.5 + self.waveguide_width*0.5 - 0.1),
            size=mp.Vector3(0, self.waveguide_width*1.2)
            )
        )
        self.incidentMonitorBot = self.sim.add_flux(
            self.fcen,
            self.fwidth,
            nfreq,
            mp.FluxRegion(
                center=mp.Vector3(-self.sx*0.5 + self.pml_size + self.waveguide_length*0.5,
                                  -self.arm_separation*0.5 - self.waveguide_width*0.5 + 0.1),
                size=mp.Vector3(0, self.waveguide_width*1.2)
                )
            )
        self.transMonitorTop = self.sim.add_flux(
            self.fcen,
            self.fwidth,
            nfreq,
            mp.FluxRegion(
                center=mp.Vector3(self.sx*0.5 - self.waveguide_length*0.3,
                                  self.arm_separation*0.5 + self.waveguide_width*0.5),
                size=mp.Vector3(0, self.waveguide_width*1.2)
                )
            )
        transMonitorBot = self.sim.add_flux(
            self.fcen,
            self.fwidth,
            nfreq,
            mp.FluxRegion(
                center=mp.Vector3(self.sx*0.5 - self.waveguide_length*0.3,
                                  -self.arm_separation*0.5 - self.waveguide_width*0.5),
                size=mp.Vector3(0, self.waveguide_width*1.2)
                )
            )

class MEEP_1x2Splitter():
    def __init__(self):
        # simulation parameters
        self.MMI_x = 2.0 # 2 microns
        self.MMI_y = 2.0 # 2 microns 
        self.waveguide_s_port = 0.5 # half a microns
        self.waveguide_l_port = 1.0 # 1 micron
        self.taper_length = 3.0 # 3 microns taper length: total 0.5 micron grain over 3 microns taper distance
        # now decide on the waveguide lengths to allow enough room for proper propagation and the monitors
        self.straight_len = 3.0 # let's give it 3 microns freo the edges of the simulation cell
        
        # dpml layer sizes
        self.dpml = 1.0
        
        # simulation cell dimensions
        self.cell_x = self.straight_len*2 + self.taper_length*2 + self.MMI_x + self.dpml*2 # have a little extra space for PML and whatnot
        self.cell_y = self.MMI_y + self.dpml*4 # also have a little extra, and wavbeguides should be contained within this width
        
        # etched holes
        self.separation_holes = 0.1 # 100 nanometers
        self.hole_radius = 0.04 # 40 nanometers, sounds familiar
        
        # materials
        self.Si = mp.Medium(index=3.5)
        self.SiO2 = mp.Medium(index=1.45)
        
        # MMI design region is transformed into a 20x20 grid
        
        self.resolution = 100 # 100 pixels per micron
        
        # creating the splitter from prisms and blocks
        # prism is the whole tapered waveguide si we need at least 6 points defined clockwise from left of the simulation cell
        in_vertices = [
                       mp.Vector3(-self.MMI_x*0.5 - self.taper_length - self.straight_len - self.dpml, self.waveguide_s_port*0.5), # left top corner of waveguide
                       mp.Vector3(-self.MMI_x*0.5 - self.taper_length, self.waveguide_s_port*0.5), # end of straight, beginning of tapering
                       mp.Vector3(-self.MMI_x*0.5, self.waveguide_l_port*0.5), # end of waveguide top right
                       mp.Vector3(-self.MMI_x*0.5, -self.waveguide_l_port*0.5), # end of waveguide bottom right
                       mp.Vector3(-self.MMI_x*0.5 - self.taper_length, -self.waveguide_s_port*0.5), # start of bottom taper
                       mp.Vector3(-self.MMI_x*0.5 - self.taper_length - self.straight_len - self.dpml, -self.waveguide_s_port*0.5) # left bottom corner of waveguide
                       ]
        
        # these I will try to define from the other end, so definition will start from the right end of the cell defined counter-clockwise
        out_vertices_one = [
                           mp.Vector3(self.MMI_x*0.5 + self.taper_length + self.straight_len + self.dpml, 0.25 + self.waveguide_s_port), # top right corver of waveguide, 0.25 accounts for the separation between top and botton waveguides
                           mp.Vector3(self.MMI_x*0.5 + self.taper_length, 0.25 + self.waveguide_s_port), # end of straight, beginning of taper
                           mp.Vector3(self.MMI_x*0.5, self.waveguide_l_port), # end of taper, top left corner of port
                           mp.Vector3(self.MMI_x*0.5, 0), # end of taper, bottom left corner of port
                           mp.Vector3(self.MMI_x*0.5 + self.taper_length, 0.25), # start of taper on bottom
                           mp.Vector3(self.MMI_x*0.5 + self.taper_length + self.straight_len + self.dpml, 0.25), # right bottom corner of waveguide
                           ]
                           
        out_vertices_two = [
                       mp.Vector3(self.MMI_x*0.5 + self.taper_length + self.straight_len + self.dpml, -0.25), # starting with top right corner of bottom exit waveguide port
                       mp.Vector3(self.MMI_x*0.5 + self.taper_length, -0.25),
                       mp.Vector3(self.MMI_x*0.5, 0),
                       mp.Vector3(self.MMI_x*0.5, -self.waveguide_l_port),
                       mp.Vector3(self.MMI_x*0.5 + self.taper_length, -0.25 - self.waveguide_s_port),
                       mp.Vector3(self.MMI_x*0.5 + self.taper_length + self.straight_len + self.dpml, -0.25 - self.waveguide_s_port),
                       ]
                       
        # now we create the prisms based on the vertices and add the block for the MMI region 
        self.geometry = [
                    mp.Prism(in_vertices, height=mp.inf, material=self.Si),
                    mp.Block(center=mp.Vector3(), material=self.Si, size=mp.Vector3(self.MMI_x, self.MMI_y)),
                    mp.Prism(out_vertices_one, height=mp.inf, material=self.Si),
                    mp.Prism(out_vertices_two, height=mp.inf, material=self.Si)
                    ]
                    
        self.sources = None
        self.sim = None
        
    def defineSources(self):
        lambda_min = 1.45
        lambda_max = 1.65
        fmin = 1.0/lambda_max
        fmax = 1.0/lambda_min
        self.fcen = 0.5*(fmin+fmax)
        self.df = fmax-fmin
        
        source_center = [-self.MMI_x*0.5 - self.taper_length - self.straight_len*0.75, 0, 0]
        src = mp.GaussianSource(self.fcen, self.df)
        source_size = source_size = mp.Vector3(0, self.waveguide_s_port, 0)
        
        self.sources = [
            mp.Source(
                src,
                size=source_size,
                center=source_center,
                component=mp.Ey
                )
            ]
            
    def defineSim(self):
        self.sim = mp.Simulation(
            resolution=self.resolution,
            cell_size=mp.Vector3(self.cell_x, self.cell_y),
            boundary_layers=[mp.PML(self.dpml)],
            default_material=self.SiO2,
            sources=self.sources,
            geometry=self.geometry
            )
                    
    def addFluxMonitors(self):
        nfreq = 100    
        # flux Regions
        # incident
        self.incidentFluxRegion = mp.FluxRegion(center=mp.Vector3(-self.MMI_x*0.5 - self.taper_length - self.straight_len*0.45, 0),
                                           size=mp.Vector3(0, self.waveguide_s_port*1.5))
        self.incidentFluxMonitor = self.sim.add_flux(self.fcen, self.df, nfreq, self.incidentFluxRegion) 
        # output arms
        self.outputFluxRegionOne = mp.FluxRegion(center=mp.Vector3(self.MMI_x*0.5 + self.taper_length*0.65, 0.25+self.waveguide_l_port*0.25),
                                           size=mp.Vector3(0, self.waveguide_l_port*0.75))
        self.outputFluxMonitorOne = self.sim.add_flux(self.fcen, self.df, nfreq, self.outputFluxRegionOne)
        
        self.outputFluxRegionTwo = mp.FluxRegion(center=mp.Vector3(self.MMI_x*0.5 + self.taper_length*0.65, -0.25-self.waveguide_l_port*0.25),
                                           size=mp.Vector3(0, self.waveguide_l_port*0.75))
        self.outputFluxMonitorTwo = self.sim.add_flux(self.fcen, self.df, nfreq, self.outputFluxRegionTwo)
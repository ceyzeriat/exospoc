# -*- coding: utf-8 -*-
#################################################################
#############             EXOSPOC v1.0              #############
#############          Guillaume Schworer           #############
#############     guillaume.schworer@obspm.fr       #############
#################################################################


#################################################################
#############              Libraries                #############
#################################################################


import numpy as np
from scipy import interpolate
import time
import pylab as plt
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

import misc.funcs as funcs
import astro
import angle
import flux
import pyexo


#################################################################
#############           Global Functions            #############
#################################################################


#returns 1 callable function able to interpolate 3D data from the input file
#input file columns : first shall be X, others shall be n columns: Y1,..Yi,..Yn
def export_function_3D(q_data, q_pt, file_in, row_skip, delimiter=',', kind='linear', kindq='linear'):
    "input: q_data is the (1,..i,..n) vector defining the Y1,..Yi,..Yn columns used for 3D interpolation, q_pt is the value at which Yq is obtained from Y1,..Yi,..Yn, file_in is path/file.ext, row_skip is how many rows on top of the file must be skiped when reading it, delimiter is column split letter, kind is the interpolation method for XY, kindq is the interpolation method for Y1,..Yi,..Yn; output: callable function"
    res_out=np.loadtxt(file_in, delimiter=delimiter,skiprows=row_skip, unpack=True)
    if res_out.shape[0]<3: return None # exists if data is actually 2D
    yq=[interpolate.interp1d(q_data,res_out[1:].T[i],kind=kindq)(q_pt) for i in range(res_out.shape[1])] # calculates the Yq values based on the Y1,..Yi,..Yn columns, interpolates Y along i values
    return interpolate.interp1d(res_out[0],yq,kind=kind) # interpolates Yq along X values


#returns 1 callable function able to interpolate data from the input file
def export_function(file_in, row_skip, delimiter=',',kind='linear'):
    "input: file_in is path/file.ext, row_skip is how many rows on top of the file must be skiped when reading it, delimiter is column split letter, kind is the interpolation method; output: callable function"
    res_out=np.loadtxt(file_in, delimiter=delimiter,skiprows=row_skip, unpack=True)
    return interpolate.interp1d(res_out[0],res_out[1],kind=kind) # first column is X, second is Y


# calculates the equivalent geometric albedo of a surface defined with its single-scattering albedo, assuming Rayleigh scattering
def rayleigh_ag_from_w(scat_albedo):
    "#takes single-scattering albedo, returns its related geometric albedo considering Rayleigh scattering"
    s = np.sqrt(1-np.asarray(scat_albedo))
    return 0.7977*((1-0.23*s)*(1-s))/((1+0.72*s)*(0.95+0.08*np.asarray(scat_albedo)))


# calculates the phase integral of a Lambertian sphere with alpha phase angle
def Lambertian_phase_function(alpha):
    "input: alpha is phase angle [deg]; output: phase integral [%]"
    alpharad = alpha*angle.DEG2RAD
    return (np.sin(alpharad)+(np.pi-alpharad)*np.cos(alpharad))/np.pi


def kill_discontinuity(x, y, z=None):
    "takes x and y vectors and returns them so that there is no 'plot looping over a discontinuity'"
    if np.sum(np.diff(x)<0)<2:
        index_sorted = np.argsort(x)
        x = x[index_sorted]
        y = y[index_sorted]
        if z is not None: z = z[index_sorted]
    else:
        x_min = np.argmin(x) #gets indexes for x sorted in increasing values
        x = np.roll(x,-x_min)
        y = np.roll(y,-x_min)
        if z is not None: z = np.roll(z,-x_min)
    if np.sum(np.diff(y)<0)<2:
        y_max = np.argmax(y)+1
        if z is None:
            return x[:y_max], x[y_max:], y[:y_max], y[y_max:]
        else:
            return x[:y_max], x[y_max:], y[:y_max], y[y_max:], z[:y_max], z[y_max:]
    if z is None:
        return x, None, y, None
    else:
        return x, None, y, None, z, None



#################################################################
#############            Classe Planet              #############
#################################################################


class c_planet:
    """
    ################################################################################


    ###  Exospoc, developped by Guillaume Schworer, guillaume.schworer@obspm.fr  ###


    *** DESCRIPTION
    Exospoc is a code that, given orbital and reflectance parameters, will output
    parameters among angular separation, contrast ratio, polarization degree,
    orbital fraction, Julian date of a Keplerian-orbit body around its
    star.


    *** PLANET OBJECT - INPUT PARAMETERS
    type:
    myplanet = c_planet(nu=.., fr=.., date=.., t=.., a=.., e=.., i=.., w=.., o=..,
        tperi=.., distance=.., radius=.., albedo_scat=.., albedo_geo=..)
    to create the planet object.

    As input parameters:
    Give i (inclination), w (argument at periapsis), o (argument of ascending node)
    in degrees
    Give tperi (time at periapsis) in JD (use the function astro.cal2JD([year,
        month, day, hour, minute, second]) to convert dates if needed)
    Give t (period) in days
    Give e (eccetricity), albedo_scat (single-scattering albedo for Rayleigh),
    albedo_geo (geometric albedo) between 0 and 1
    Give a (semi-major axis) in AU
    Give distance (d to star) in pc
    Give radius (r of planet) in Jupiter radii

    Orbital positions for processing:
    date (julian day), fr (orbital fraction) and nu (true anomaly) indicate the
    planet position around the star for which you want to calculate the output
    parameters.
    If you want to process the whole orbit, give e.g. fr = np.linspace(0,1,1000)
    If you want to process only several dates, give date = [D1, D2, D3, ...] in
    Julian date or date = np.linspace(D_start,D_end, n_points)
    Only one of these 3 parameters is required. In case several are given, priority
    is given to date, then fr, then nu.


    *** COMPUTE PLANET
    Once the planet object is created type 'myplanet.compute()' to run the
    calculations.

    Then, 'myplanet.available_output' gives the outputs processed from the input
    parameters


    *** PLOTING
    Use myplanet.SPOC_diag(...) to display the SPOC diagram of the planet
    Use myplanet.hist(...) to display the SPOC histogram of the planet
    Use myplanet.plot_diag(...) for any other custom ploting


    *** AVAILABLE METHODS AND ATTRIBUTES
    Type 'myplanet.' then press <tab> to see all methods and attributes available

    Type 'myplanet.attribute_or_method?' (question mark at the end) to see the help
     of the attribute or method

    All output parameters are vectors that have the same dimension as the given
    planet orbital position in input (either fr, nu or date).
    The i-th element of any output parameter corresponds to the i-th orbital
    position given in input.
    You can access the output parameters typing 'myplanet.output'. They are all
    numpy arrays.

    ################################################################################
    """

    _pyexo_mapping = {'eccentricity':'e','inclination':'i','radius':'radius','orbital_period':'t','semi_major_axis':'a','omega':'w','tperi':'tperi','albedo':'albedo_geo','star_distance':'distance'}
    _all_outputs = ['alpha', 'fr', 'nu', 'r', 'r_rel', 'sep', 'sep_rel', 'date', 'date_rel', 'cr', 'cr_rel', 'pol', 'pol_ang', 'pol_ang_rel', 'crpol', 'crpol_rel', 'north_angle', 'north_angle_rel']

    def __init__(self, nu=None, fr=None, date=None, t=None, a=None, e=None, i=None, w=None, o=None, tperi=None, distance=None, radius=None, albedo_scat=None, albedo_geo=None, exoplanet=None, star=None):
        self.e = 0. # default
        self.i = 45. # default
        self.w = 0. # default
        if exoplanet is not None:
            if not isinstance(exoplanet, pyexo.pyexo): exoplanet = pyexo.pyexo(exoplanet)
            if hasattr(exoplanet,'names'): raise Exception, "exospoc works with just one planet at a time, %s were found" % len(exoplanet.names)
            for key in self._pyexo_mapping.keys():
                if exoplanet[key] is not None: setattr(self, self._pyexo_mapping[key], float(exoplanet[key]))
            if star is None:
                if exoplanet['star_teff'] is None and exoplanet['star_sp_type'] is not None: exoplanet['star_teff'] = astro.sptype2star(exoplanet['star_sp_type'])[0]
                if exoplanet['star_teff'] is not None:
                    self.star = flux.bbody_temp(ref_mag=float(exoplanet['mag_v']), ref_band='V', teff=float(exoplanet['star_teff']))
                    self.star.fluxV = self.star.flux(np.linspace(506e-9,594e-9,500), unit='Ph').sum() * 52.8 * 3600 # photons per h on vlt
        if star is not None:
            if isinstance(star, flux.bbody_temp):
                self.star = star
                self.star.fluxV = self.star.flux(np.linspace(506e-9,594e-9,500), unit='Ph').sum() * 52.8 * 3600 # photons per h on vlt
            else:
                raise Exception, "Could not understand star parameter"
        # overwrite pyexo with user imput
        if e is not None: self.e = e #[deg] eccentricity, 1 double value
        if w is not None: self.w = w #[deg] argument of periapsis, 1 double value
        if i is not None: self.i = i #[deg] inclination, 1 double value
        if o is not None: self.o = o #[deg] longitude of ascending node, 1 double value
        if tperi is not None:
            self.tperi = tperi #[julian] time at periapsis, 0 if undefined, 1 double value
            if self.tperi!=0 and self.tperi<50000.: self.tperi=self.tperi+50000. #correction if tperi is not complete
            if self.tperi!=0 and self.tperi<2400000.: self.tperi=self.tperi+2400000. #correction if tperi is not complete
        if t is not None: self.t = t #[days] period, 1 double value
        if a is not None: self.a = a #[au] semi-major axis, 1 double value
        if nu is not None: self.nu = nu # [deg] or [None] if undefined, vector of double
        if fr is not None: self.fr = fr # [-] or [None] if undefined, vector of double
        if date is not None: self.date = date # [julian] or [None] if undefined, vector of double
        if albedo_scat is not None: self.albedo_scat = albedo_scat # [0-1] Single-scattering Albedo for Rayleigh scattering
        if albedo_geo is not None: self.albedo_geo = albedo_geo # [0-1] Geometric Albedo
        if distance is not None: self.distance = distance # [m], distance host-star to observer
        if radius is not None: self.radius = radius # [m]
        self._which_outputs()


    def _which_outputs(self):
        """
        Defines which outputs are calcul-able depending on the inputs, updates the output list
        """
        
        self.available_output = ['alpha', 'fr', 'nu'] # always known outputs
        
        if hasattr(self, 'a'):
            self.available_output.append('r') # if a given then alway know r too
            if hasattr(self, 'distance'): # if a and distance given then alway know r separation too
                self.available_output.append('sep')
                if hasattr(self, 't'): self.available_output.append('mv')
        else: # if a not given then only know relative r and separation
            self.available_output.append('r_rel') 
            self.available_output.append('sep_rel')
            if hasattr(self, 't'): self.available_output.append('mv_rel')
        
        if hasattr(self,'t'):
            if hasattr(self,'tperi'):
                self.available_output.append('date')
            else:
                self.available_output.append('date_rel')
        
        if hasattr(self,'radius') and hasattr(self,'albedo_geo'):
            if hasattr(self,'a'):
                self.available_output.append('cr')
            else:
                self.available_output.append('cr_rel')
            if hasattr(self,'albedo_scat'):
                self.available_output.append('pol')
                if hasattr(self,'o'):
                    self.available_output.append('pol_ang')
                else:
                    self.available_output.append('pol_ang_rel')
                if self._has_output('cr'):
                    self.available_output.append('crpol')
                else:
                    self.available_output.append('crpol_rel')
        
        if hasattr(self,'o'):
            self.available_output.append('north_angle')
        else:
            self.available_output.append('north_angle_rel')


    def _has_output(self, output=""):
        return self.available_output.count(output)!=0


    def compute(self, silent=False): # run simu
        self._which_outputs()
        if hasattr(self, 'tperi') and hasattr(self, 't'): self.tperi = astro.next_date(self.tperi, self.t)
        if hasattr(self, 'radius'): self.radius=self.radius*astro.JUPRADIUS
        # depending on what the user gave (fr or nu or date), this calculates the 2 other informations from the given one
        if not silent: print "* Starting calculation *"
        if hasattr(self,'date') and self._has_output('date'):
            self.fr = ((self.date-self.tperi) / self.t)%1 #[0-1]
            self.nu = astro.fr_to_nu(self.fr, self.e, 1.e-15, degrees=True) #[deg]
        elif hasattr(self,'nu'):
            fr_initial = astro.nu_to_fr(self.nu, self.e, degrees=True)
            self.nu = np.asarray(self.nu)*1. %360 #[deg]
            self.fr = astro.nu_to_fr(self.nu, self.e, degrees=True) #[0-1]
        elif hasattr(self,'fr'):
            fr_initial = np.asarray(self.fr)*1.
            self.fr = fr_initial%1 #[0-1]
            self.nu = astro.fr_to_nu(self.fr, self.e, 1.e-15, degrees=True) #[deg]
        else:
            if not silent: print 'No orbital position defined, assumed 100 positions linearly spaced in time'
            self.fr = np.linspace(0,1,101)[:-1] #[0-1]
            fr_initial = self.fr
            self.nu = astro.fr_to_nu(self.fr, self.e, 1.e-15, degrees=True) #[deg]
        if not hasattr(self,'date'):
            if self._has_output('date'):
                self.date = fr_initial * self.t + self.tperi # [days]
            elif self._has_output('date_rel'):
                self.date_rel = fr_initial * self.t # [days]
        r, r_proj, north_angle, self.alpha = astro.orbital_pos(self.nu, self.e, self.i, self.w, getattr(self, 'a', 1.)*astro.AU2M, getattr(self, 'o', 0.), degrees=True) # all outputs are vectors of double, same size as nu (or fr or date)
        if self._has_output('north_angle'):
            self.north_angle = north_angle # checks if north_angle is part of outputs
            if self._has_output('pol_ang'):
                self.pol_ang = north_angle*2. % 360
        else:
            self.north_angle_rel = north_angle
            if self._has_output('pol_ang_rel'):
                self.pol_ang_rel = north_angle*2. % 360
        if self._has_output('r'): self.r = r # checks if orbital radius is part of outputs
        if self._has_output('r_rel'): self.r_rel = r/r.max()
        if self._has_output('sep'): self.sep = angle.agl(r_proj, self.distance*astro.PC2M, unit='arcsec').v # angular separation in [arcsec]
        if self._has_output('sep_rel'):
            self.sep_rel = angle.agl(r_proj, 1., unit='arcsec').v # angular separation in [arcsec]
            self.sep_rel = self.sep_rel/self.sep_rel.max()
        #self.mean_sep = self.separation(percentile=50, nb_pts=100, unit='arcsec') # mean angular separation in [arcsec]
        if not self._has_output('pol'):
            if self._has_output('cr_rel'):
                self.cr_rel = (self.radius/self.r_rel)**2 * Lambertian_phase_function(self.alpha) * self.albedo_geo
                self.cr_rel = self.cr_rel/self.cr_rel.max()
            if self._has_output('cr'): self.cr = (self.radius/self.r)**2 * Lambertian_phase_function(self.alpha) * self.albedo_geo
        else:
            Rayleigh_phase_function = export_function_3D([0.,1.],self.albedo_scat,'rayleigh_phase_integral_from_orbital_phase.txt', row_skip=1, kind=3, kindq='linear') #Rayleigh
            self.albedo_geo_Rayleigh = rayleigh_ag_from_w(self.albedo_scat)
            if not silent and self.albedo_geo_Rayleigh>self.albedo_geo: print 'WARNING: geometric albedo from rayleigh scattering was higher than given geometric albedo. Geometric albedo value was updated'
            self.albedo_geo = np.max([self.albedo_geo_Rayleigh, self.albedo_geo]) # albedo_geo_Rayleigh cannot be lower than albedo_geo
            self._phased_albedo_R = Rayleigh_phase_function(self.alpha) * self.albedo_geo_Rayleigh
            self._phased_albedo_L = Lambertian_phase_function(self.alpha) * (self.albedo_geo - self.albedo_geo_Rayleigh)
            self._phased_albedo_RL = self._phased_albedo_R + self._phased_albedo_L
            peak_pol = export_function('rayleigh_polarization_peak_from_scat_albedo.txt', row_skip=1, kind=3)
            pol = export_function('rayleigh_polarization_from_orbital_phase.txt', row_skip=1, kind=3)
            pol_R = pol(self.alpha)/peak_pol(0.1)*peak_pol(self.albedo_scat)
            self.pol = np.true_divide(pol_R, 1.+np.true_divide(self._phased_albedo_L, self._phased_albedo_R)).repeat(1)
            if self._has_output('cr_rel'):
                self.cr_rel = (self.radius/self.r_rel)**2 * self._phased_albedo_RL
                self.cr_rel = self.cr_rel/self.cr_rel.max()
            if self._has_output('cr'): self.cr = (self.radius/self.r)**2 * self._phased_albedo_RL
        if self._has_output('crpol_rel'):
            self.crpol_rel = self.cr_rel*self.pol
            self.crpol_rel = self.crpol_rel/self.crpol_rel.max()
        if self._has_output('crpol'): self.crpol = self.cr*self.pol
        if hasattr(self, 'radius'): self.radius = self.radius/astro.JUPRADIUS
        if hasattr(self, 'r'): self.r = self.r*astro.M2AU

        # orbital motion
        if hasattr(self,'t') and self._has_output('sep'):
            dates = getattr(self, 'date'*self._has_output('date') + 'date_rel'*self._has_output('date_rel'))
            if (np.diff(dates)>0).all():
                angles = getattr(self, 'north_angle'*self._has_output('north_angle') + 'north_angle_rel'*self._has_output('north_angle_rel'))
                self.mv = 1000*np.sqrt(self.sep**2+np.roll(self.sep,-1)**2-2*self.sep*np.roll(self.sep,-1)*np.cos((angles-np.roll(angles,-1))*np.pi/180.%(2*np.pi)))/((np.roll(dates,-1)-dates)%self.t*24)
            else:
                if not silent: print "Dates are not monotically increasing, can't compute the relative movement per unit time of the planet"

        # star stuff
        if hasattr(self, 'star') and self._has_output('cr'):
            self.fluxV = self.cr*self.star.fluxV # photons per h on vlt already
        if not silent: print "Done"


    def SPOC_diag(self, label=None, legend=True, bw=True, markers='tnrp', log='xy', x_range=None, y_range=None, z_range=None):
        "Draws the SPOC diagram "
        if not ((hasattr(self,'cr') or hasattr(self,'cr_rel')) and (hasattr(self,'sep') or hasattr(self,'sep_rel'))):
            print "Contrast ratio or Separation is unknown, cannot process SPOC diagram."
            return
        self.plot_diag(x='sep'*self._has_output('sep')+'sep_rel'*self._has_output('sep_rel'), y='cr'*self._has_output('cr')+'cr_rel'*self._has_output('cr_rel'), z='pol'*self._has_output('pol'), markers=markers, log=log, minimap=True, label=label, overplot_fig=False, legend=legend, bw=bw, x_range=x_range, y_range=y_range, z_range=z_range)
        return


    def plot_diag(self, x, y, z=None, label=None, legend=True, bw=True, markers='t', log='', minimap=False, overplot_fig=False, x_range=None, y_range=None, z_range=None, color=None):#, leg_half=False):
        """x, y, (z) must be strings part of 'available_output' argument.
        * Log is a string that can be 'x', 'y' or 'xt'.
        * Markers is a string that can contain 't' for time markers, 'n' for the current position, 'r' for radius or 'p' for phase angle, or any combination of these (eg. 'trp')
        * Overplot_fig set to False creates a new Figure, set to True overplots on the current Figure, or can be set to a Figure number.
        * Ranges are 25 elements list that set the [min, max] bounds of the axis"""
        if hasattr(self,'pol'): self.pol = self.pol*100
        if hasattr(self,'cr'): self.cr = np.log10(self.cr)
        if hasattr(self,'crpol'): self.crpol = np.log10(self.crpol)
        if not (hasattr(self,x) and hasattr(self,y) and (self._has_output(z) or z is None)):
            print "Cannot find such outputs"
            return
        axes_labels={'alpha':r'Phase angle [$^\circ$]', 'fr':'Orbital fraction', 'nu':r'True anomaly [$^\circ$]', 'r':'Star-planet distance [AU]', 'sep':'Angular separation [arcsec]','sep_rel':'Relative angular separation', 'date':'Date (JD)', 'date_rel':'Time (days)', 'cr':'Contrast ratio [LOG10]', 'cr_rel':'Relative contrast ratio [log10]', 'pol':'Degree of polarization [%]', 'pol_ang':r'Angle of polarization [$^\circ$]','pol_ang_rel':r'Relative angle of polarization [$^\circ$]', 'north_angle':r'Angle to the North [$^\circ$]', 'north_angle_rel':r'Angle around the star (arbitrary) [$^\circ$]', 'crpol':'Polarized contrast ratio [log10]','crpol_rel':'Relative polarized contrast ratio [log10]', 'mv':'Orbital motion [mas/h]'}
        marker_size = 9
        nTimeMarkers = 10
        #checks for overploting options
        if overplot_fig is False or z is not None:
            if overplot_fig and z is not None: print "No overploting available with color axis"
            fig = plt.figure(figsize=(8,6))
            if z is not None:
                ax = fig.add_axes([0.11, 0.16, 0.75, 0.80])
            else:
                ax = fig.add_axes([0.11, 0.09, 0.75, 0.88])
        elif isinstance(overplot_fig,int):
            fig = plt.figure(overplot_fig)
            ax = plt.gca()
        else:
            fig = plt.gcf()
            ax = plt.gca()
        #plots for 2 or 3 axis
        if z is not None: #color axis case
            if z_range is None: z_range=[getattr(self,z).min(), getattr(self,z).max()]
            if bw:
                cdict = { 'red'  :  (   (0., 0., 0.9),
                                        (0.6, 0.45, 0.45),
                                        (1., 0., 0.)),
                          'green':  (   (0., 0., 0.9),
                                        (0.6, 0.6, 0.6),
                                        (1., 0., 0.)),
                          'blue' :  (   (0., 0., 0.9),
                                        (0.6, 0.8, 0.8),
                                        (1., 0.05, 0.))}
                thecmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
            else: thecmap=plt.get_cmap('jet')
            x_val, x_val2, y_val, y_val2, z_val, z_val2 = kill_discontinuity(getattr(self,x),getattr(self,y),getattr(self,z))
            points = np.asarray([x_val, y_val]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            sm = plt.cm.ScalarMappable(cmap=thecmap, norm=plt.Normalize(z_range[0], z_range[1]))
            lc = LineCollection(segments, cmap=thecmap, norm=plt.Normalize(z_range[0], z_range[1]))
            lc.set_array(z_val)
            lc.set_linewidth(3)
            ax.add_collection(lc)
            if z_val2 is not None:
                points = np.asarray([x_val2, y_val2]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                sm = plt.cm.ScalarMappable(cmap=thecmap, norm=plt.Normalize(z_range[0], z_range[1]))
                lc = LineCollection(segments, cmap=thecmap, norm=plt.Normalize(z_range[0], z_range[1]))
                lc.set_array(z_val2)
                lc.set_linewidth(3)
                ax.add_collection(lc)
            sm._A = []
            #cbaxes = fig.add_axes([0.91, 0.09, 0.03, 0.85])
            cbaxes = fig.add_axes([0.1, 0.05, 0.8, 0.03])
            cb = plt.colorbar(sm, cax = cbaxes, orientation='horizontal')
            #cb=plt.colorbar(sm)
            cb.ax.text(0.3, 0.05, axes_labels[z], rotation=0, fontsize=12)
        else: # 2 axis case
            x_val, x_val2, y_val, y_val2 = kill_discontinuity(getattr(self,x),getattr(self,y))
            legend_size=0
            if fig.axes[0].get_legend() is not None: legend_size=len(fig.axes[0].get_legend().texts)
            if label is None: label="Planet "+str(legend_size+1)
            if color is not None:
                plt.plot(x_val,y_val, label=label, color=color)
            else: plt.plot(x_val,y_val, label=label)
            if x_val2 is not None: plt.plot(x_val2,y_val2,fig.axes[0].lines[-1].get_color())
        ax.set_xlabel(axes_labels[x])
        ax.set_ylabel(axes_labels[y])
        #checks for markers
        if markers.upper().find('T')!=-1: #show linearly spaced time markers
            if self.fr.min()<=2./self.fr.size and 1-self.fr.max()<=2./self.fr.size: #whole fr loop
                fr=np.linspace(self.fr.min(),self.fr.max(), nTimeMarkers+1)[:-1]
            else:
                fr=np.linspace(self.fr.min(),self.fr.max(), nTimeMarkers)
            pTimeMarkers=c_planet(fr=fr, t=getattr(self,'t',None), a=getattr(self,'a',None), e=getattr(self,'e',None), i=getattr(self,'i',None), w=getattr(self,'w',None), tperi=getattr(self,'tperi',None), distance=getattr(self,'distance',None), radius=getattr(self,'radius',None), albedo_scat=getattr(self,'albedo_scat',None), albedo_geo=getattr(self,'albedo_geo',None), o=getattr(self,'o',None))
            pTimeMarkers.compute(silent=True)
            if hasattr(pTimeMarkers,'pol'): pTimeMarkers.pol *= 100
            if hasattr(pTimeMarkers,'cr'): pTimeMarkers.cr = np.log10(pTimeMarkers.cr)
            if hasattr(pTimeMarkers,'crpol'): pTimeMarkers.crpol = np.log10(pTimeMarkers.crpol)
            if fig.axes[0].get_legend() is None: #legend creation
                if legend=='half':
                    ax.plot(getattr(pTimeMarkers,x), getattr(pTimeMarkers,y), '+k', label=r'$\Delta t$', mec='k', mew=2.5, ms=marker_size)
                else:
                    ax.plot(getattr(pTimeMarkers,x), getattr(pTimeMarkers,y), '+k', label=r'$\Delta t$'+' = ' + str(round(pTimeMarkers.t/nTimeMarkers,2)) + 'd, n=' + str(nTimeMarkers), mec='k', mew=2.5, ms=marker_size)
            else:
                ax.plot(getattr(pTimeMarkers,x), getattr(pTimeMarkers,y), '+k', mec='k', mew=2.5, ms=marker_size)
        if markers.upper().find('N')!=-1 and self._has_output('date'): #show now marker
            fr=((astro.cal2JD()-self.tperi)/self.t)%1
            fr=[fr,fr+1/6.]
            pTimeMarkers=c_planet(fr=fr, t=getattr(self,'t',None), a=getattr(self,'a',None), e=getattr(self,'e',None), i=getattr(self,'i',None), w=getattr(self,'w',None), tperi=getattr(self,'tperi',None), distance=getattr(self,'distance',None), radius=getattr(self,'radius',None), albedo_scat=getattr(self,'albedo_scat',None), albedo_geo=getattr(self,'albedo_geo',None), o=getattr(self,'o',None))
            pTimeMarkers.compute(silent=True)
            if hasattr(pTimeMarkers,'pol'): pTimeMarkers.pol *= 100
            if hasattr(pTimeMarkers,'cr'): pTimeMarkers.cr = np.log10(pTimeMarkers.cr)
            if hasattr(pTimeMarkers,'crpol'): pTimeMarkers.crpol = np.log10(pTimeMarkers.crpol)
            if fig.axes[0].get_legend() is None: #legend creation
                if legend=='half':
                    ax.plot(getattr(pTimeMarkers,x)[0], getattr(pTimeMarkers,y)[0], 'xk', label=r'$t_{Obs}$', mec='k', mew=2.5, ms=marker_size)
                else:
                    ax.plot(getattr(pTimeMarkers,x)[0], getattr(pTimeMarkers,y)[0], 'xk', label='t='+'/'.join(map(str,time.gmtime()[:3]))+'-'+':'.join(map(str,time.gmtime()[3:6])), mec='k', mew=2.5, ms=marker_size)
                if legend=='half':
                    ax.plot(getattr(pTimeMarkers,x)[1],getattr(pTimeMarkers,y)[1],'dk', label=r'$t_{Obs}+T/6$', mec='k', mew=2.5, ms=marker_size)
                else:
                    ax.plot(getattr(pTimeMarkers,x)[1],getattr(pTimeMarkers,y)[1],'dk', label='t+T/6 ('+str(np.round(self.t/6,2))+'d)', mec='k', mew=2.5, ms=marker_size)
            else:
                ax.plot(getattr(pTimeMarkers,x)[0], getattr(pTimeMarkers,y)[0], 'xg', mec='k', mew=2.5, ms=marker_size)
                ax.plot(getattr(pTimeMarkers,x)[1], getattr(pTimeMarkers,y)[1], 'dg', mec='k', mew=2.5, ms=marker_size)
        if markers.upper().find('R')!=-1: #show radii
            r=getattr(self,'r',0)+getattr(self,'r_rel',0)
            rmm = [np.argmin(r), np.argmax(r)]
            if legend=='half':
                ax.plot(getattr(self,x)[rmm[0]],getattr(self,y)[rmm[0]],'>w',mec='k',ms=marker_size)
                ax.plot(getattr(self,x)[rmm[1]],getattr(self,y)[rmm[1]],'>k',mec='k',ms=marker_size)
            else:
                ax.plot(getattr(self,x)[rmm[0]],getattr(self,y)[rmm[0]],'>w',label=r'$r_{orbit,min}=$' + str(round(r[rmm[0]]*astro.M2AU,3)) + r'$AU$',mec='k',ms=marker_size)
                ax.plot(getattr(self,x)[rmm[1]],getattr(self,y)[rmm[1]],'>k',label=r'$r_{orbit,max}=$' + str(round(r[rmm[1]]*astro.M2AU,3)) + r'$AU$',mec='k',ms=marker_size)
        if markers.upper().find('P')!=-1: #show phases
            amm = [np.argmin(getattr(self,'alpha')), np.argmax(getattr(self,'alpha'))]
            if legend=='half':
                ax.plot(getattr(self,x)[amm[0]],getattr(self,y)[amm[0]],'ow',mec='k',ms=marker_size)
                ax.plot(getattr(self,x)[amm[1]],getattr(self,y)[amm[1]],'ok',mec='k',ms=marker_size)
            else:
                ax.plot(getattr(self,x)[amm[0]],getattr(self,y)[amm[0]],'ow',label=r'$\alpha_{min}=$' + str(round(getattr(self,'alpha')[amm[0]],1)) + r'$^{\circ}$',mec='k',ms=marker_size)
                ax.plot(getattr(self,x)[amm[1]],getattr(self,y)[amm[1]],'ok',label=r'$\alpha_{max}=$' + str(round(getattr(self,'alpha')[amm[1]],1)) + r'$^{\circ}$',mec='k',ms=marker_size)
        #creates legend
        if legend: ax.legend(loc=0, ncol=1, fancybox=True, shadow=True, numpoints=1)
        #checks for log axes
        if log.upper().find('X')!=-1 and log.upper().find('Y')!=-1:
            plt.loglog()
        elif log.upper().find('X')!=-1:
            plt.semilogx()
        elif log.upper().find('Y')!=-1:
            plt.semilogy()
        #apply ranges
        if overplot_fig is False:
            if x_range is None: ax.set_xlim([getattr(self,x).min()*1.05-getattr(self,x).max()*0.05,getattr(self,x).max()*1.05-getattr(self,x).min()*0.05])
            if y_range is None: ax.set_ylim([getattr(self,y).min()*1.05-getattr(self,y).max()*0.05,getattr(self,y).max()*1.05-getattr(self,y).min()*0.05])
        else:
            ax.autoscale()
        if x_range is not None: ax.set_xlim(x_range)
        if y_range is not None: ax.set_ylim(y_range)
        fig.canvas.draw()
        # abs flux
        if hasattr(self, 'star') and (y.upper()=='CR' or y.upper()=='CRPOL'):
            ax2 = ax.twinx()
            tick_loc = np.asarray([item.get_position()[1] for item in ax.yaxis.get_ticklabels() if item.get_text()!=''])
            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticks(tick_loc)
            vals = 10**tick_loc*self.star.fluxV
            order = int(np.log10(np.min(vals)))
            vals = (np.round(vals/10**order, 2)*10**order).astype(int)
            if vals.max()>10000:
                vals = np.asarray(['%1.1e'%F for F in vals])
            ax2.set_yticklabels(vals.astype(str))
            ax2.set_ylabel(r'Flux [photons]')

        if minimap:
            x_mini='sep'*self._has_output('sep')+'sep_rel'*self._has_output('sep_rel')
            ang='north_angle'*self._has_output('north_angle')+'north_angle_rel'*self._has_output('north_angle_rel')
            #create mini map
            axisbg='y'
            if bw: axisbg='w'
            ax2=plt.axes([.08,.18,.3,.3], axisbg=axisbg, polar=True)
            ax2.patch.set_alpha(0.3)
            ax2.xaxis.set_ticklabels('')
            ax2.yaxis.set_ticklabels('')
            ax2.plot(0,0,'*k',ms=10)
            ax2.set_ylim([0,getattr(self,x_mini).max()*1.25])
            #adds the markers
            if markers.upper().find('N')!=-1 and self._has_output('date'): #show now marker
                fr=((astro.cal2JD()-self.tperi)/self.t)%1
                fr=[fr,fr+1/6.]
                pTimeMarkers=c_planet(fr=fr, t=getattr(self,'t',None), a=getattr(self,'a',None), e=getattr(self,'e',None), i=getattr(self,'i',None), w=getattr(self,'w',None), tperi=getattr(self,'tperi',None), distance=getattr(self,'distance',None), radius=getattr(self,'radius',None), albedo_scat=getattr(self,'albedo_scat',None), albedo_geo=getattr(self,'albedo_geo',None), o=getattr(self,'o',None))
                pTimeMarkers.compute(silent=True)
                ax2.plot(getattr(pTimeMarkers,ang)[0]*angle.DEG2RAD,getattr(pTimeMarkers,x_mini)[0],'xk',mec='k',mew=2.5,ms=marker_size)
                ax2.plot(getattr(pTimeMarkers,ang)[1]*angle.DEG2RAD,getattr(pTimeMarkers,x_mini)[1],'dk',mec='k',mew=2.5,ms=marker_size)
            if markers.upper().find('R')!=-1: #show radii
                r=getattr(self,'r',0)+getattr(self,'r_rel',0)
                rmm = [np.argmin(r), np.argmax(r)]
                ax2.plot(getattr(self,ang)[rmm[0]]*angle.DEG2RAD,getattr(self,x_mini)[rmm[0]],'>w',mec='k',ms=marker_size)
                ax2.plot(getattr(self,ang)[rmm[1]]*angle.DEG2RAD,getattr(self,x_mini)[rmm[1]],'>k',mec='k',ms=marker_size)
            if markers.upper().find('P')!=-1: #show phases
                amm = [np.argmin(getattr(self,'alpha')), np.argmax(getattr(self,'alpha'))]
                ax2.plot(getattr(self,ang)[amm[0]]*angle.DEG2RAD,getattr(self,x_mini)[amm[0]],'ow',mec='k',ms=marker_size)
                ax2.plot(getattr(self,ang)[amm[1]]*angle.DEG2RAD,getattr(self,x_mini)[amm[1]],'ok',mec='k',ms=marker_size)
            #mini-map curve
            if self._has_output(z):
                if bw:
                    cdict = { 'red'  :  (   (0., 0., 0.9),
                                            (0.6, 0.45, 0.45),
                                            (1., 0., 0.)),
                              'green':  (   (0., 0., 0.9),
                                            (0.6, 0.6, 0.6),
                                            (1., 0., 0.)),
                              'blue' :  (   (0., 0., 0.9),
                                            (0.6, 0.8, 0.8),
                                            (1., 0.05, 0.))}
                    thecmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
                else: thecmap=plt.get_cmap('jet')
                points = np.asarray([getattr(self,ang)*angle.DEG2RAD, getattr(self,x_mini)]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                sm = plt.cm.ScalarMappable(cmap=thecmap, norm=plt.Normalize(getattr(self,z).min(), getattr(self,z).max()))
                lc = LineCollection(segments, cmap=thecmap, norm=plt.Normalize(getattr(self,z).min(), getattr(self,z).max()))
                lc.set_array(getattr(self,z))
                lc.set_linewidth(3)
                ax2.add_collection(lc)
            else:
                plt.plot(getattr(self, ang), getattr(self, x_mini))
            #mini-map horizontality marker
            ax2.add_patch(Polygon(np.asarray([getattr(self,ang)[self.alpha<=90]*angle.DEG2RAD,getattr(self,x_mini)[self.alpha<=90]]).T, closed=True, fill=True, facecolor='k', edgecolor='none', alpha=0.2))
        fig.show()
        if hasattr(self,'pol'): self.pol = self.pol/100.
        if hasattr(self,'cr'): self.cr = 10**self.cr
        if hasattr(self,'crpol'): self.crpol = 10**self.crpol
        return

    def SPOC_hist(self, deltacr=0.5, cat=[0,0.1,1/3.,0.5,2/3.,0.9,1], cat_percent=True, crpol=False, return_val=False):
        "Draws an histogram for planet visibility"
        if not ((hasattr(self,'cr') or hasattr(self,'cr_rel')) and (hasattr(self,'sep') or hasattr(self,'sep_rel'))):
            print "Contrast ratio or Separation is unknown, cannot process SPOC diagram."
            return
        if crpol and not hasattr(self,'pol'):
            print "No polarization output available, switching to intensity contrast ratio."
        color = ['w','#FFDD78','#7878FF','#D26E23','#AAFCED','k']
        hatch = ['','','','/','/']
        coloredge = ['k','k','k','k','k']
        pTimeMarkers=c_planet(fr=np.linspace(0,1,1001)[:-1], t=getattr(self,'t',None), a=getattr(self,'a',None), e=getattr(self,'e',None), i=getattr(self,'i',None), w=getattr(self,'w',None), tperi=getattr(self,'tperi',None), distance=getattr(self,'distance',None), radius=getattr(self,'radius',None), albedo_scat=getattr(self,'albedo_scat',None), albedo_geo=getattr(self,'albedo_geo',None), o=getattr(self,'o',None))
        pTimeMarkers.compute(silent=True)
        cat=np.sort(cat)[::-1]
        sep = getattr(self,'sep',0)+getattr(self,'sep_rel',0)
        if crpol:
            cr = np.log10(getattr(self,'crpol',0)+getattr(self,'crpol_rel',0))
            xlabel='Polarized contrast ratio [log10]'
        else:
            cr = np.log10(getattr(self,'cr',0)+getattr(self,'cr_rel',0))
            xlabel='Contrast ratio [log10]'
        sepmax = sep.max()*cat_percent+1*(not cat_percent)
        if cat[0]<sep.max(): cat = np.r_[sep.max(),cat] ###
        if cat[-1]>sep.min(): cat = np.r_[cat,0] ###
        bins = np.linspace(np.floor(cr.min()/deltacr)*deltacr, np.ceil(cr.max()/deltacr)*deltacr, np.ceil(cr.max()/deltacr)-np.floor(cr.min()/deltacr)+1)
        norma = 100./cr.size
        bottom = np.zeros(bins.size-1)
        fig = plt.figure()
        values=[]
        for i in range(cat.size):
            hh = cr[sep/sepmax>cat[i]]
            if return_val: values.append(plt.histogram(hh, bins=bins)[0]*norma)
            if hh.size!=0:
                a = plt.histogram(hh, bins=bins)[0]
                p = plt.bar((bins[:-1]+deltacr/2.)[a>0], a[a>0]*norma, deltacr/2., hatch=hatch[(i-1)%np.size(hatch)], color=color[(i-1)%np.size(color)], edgecolor=coloredge[(i-1)%np.size(coloredge)], bottom=bottom[a>0], label=str(round(cat[i]*sepmax,3)) + "-" + str(round(cat[i-1]*sepmax,3)))
                bottom = bottom+a*norma
                cr = cr[sep/sepmax<=cat[i]]
                sep = sep[sep/sepmax<=cat[i]]
        plt.xticks(bins+deltacr/4., np.round(bins,3))
        plt.legend(title="Separation in [arcsec]", loc=0, fancybox=True, shadow=True, prop={'size':12})
        plt.xlabel(xlabel)
        plt.ylabel('Duration [% of Period]')
        plt.show()
        if return_val: return bins[:-1]+deltacr/2., cat*sepmax, values


if False:
    #################################################################
    #############               Examples                #############
    #################################################################


    print """\n\n\n\n\n
    ################################################################################

    Fig 1: Solar-system angular separation versus contrast ratio diagram, 30 degrees inclination with the observer.

    ################################################################################
    """

    plt.figure(1)
    plt.clf()
    system_inclination = 45. #degrees
    Mercury = c_planet(fr=np.linspace(0,1,1001)[:-1], t=87.969, a=0.387, e=0.205, i=7.05+system_inclination, w=29.124, distance=10., radius=0.0348, albedo_geo=0.142)
    Venus = c_planet(fr=np.linspace(0,1,1001)[:-1], t=224.701, a=0.7233, e=0.0068, i=3.39+system_inclination, w=54.85, distance=10., radius=0.086, albedo_geo=0.65)
    Earth = c_planet(fr=np.linspace(0,1,1001)[:-1], t=365.26, a=1., e=0.0167, i=0+system_inclination, w=114.2, distance=10., radius=0.091, albedo_geo=0.367)
    Mars = c_planet(fr=np.linspace(0,1,1001)[:-1], t=686.9, a=1.52, e=0.093, i=1.85+system_inclination, w=286.46, distance=10, radius=0.0484, albedo_geo=0.25)
    Jupiter = c_planet(fr=np.linspace(0,1,1001)[:-1], t=4335.35, a=5.2034, e=0.04839, i=1.3+system_inclination, w=275.066, distance=10., radius=1, albedo_geo=0.52)
    Saturn = c_planet(fr=np.linspace(0,1,1001)[:-1], t=10757.7, a=9.537, e=0.0541, i=2.48+system_inclination, w=338.7168, distance=10., radius=0.82, albedo_geo=0.47)
    Uranus = c_planet(fr=np.linspace(0,1,1001)[:-1], t=30799.1, a=19.2294, e=0.0444, i=0.77+system_inclination, w=96.54, distance=10., radius=0.365, albedo_geo=0.51)
    Neptune = c_planet(fr=np.linspace(0,1,1001)[:-1], t=60224.9, a=30.104, e=0.0085, i=1.769+system_inclination, w=273.25, distance=10., radius=0.3517, albedo_geo=0.41)

    Mercury.compute(silent=True)
    Venus.compute(silent=True)
    Earth.compute(silent=True)
    Mars.compute(silent=True)
    Jupiter.compute(silent=True)
    Saturn.compute(silent=True)
    Uranus.compute(silent=True)
    Neptune.compute(silent=True)

    Mercury.plot_diag(x='sep', y='cr', label='Mercury', markers='', log='', minimap=False, overplot_fig=1)
    Venus.plot_diag(x='sep', y='cr', label='Venus', markers='', log='', minimap=False, overplot_fig=1)
    Earth.plot_diag(x='sep', y='cr', label='Earth', markers='', log='', minimap=False, overplot_fig=1)
    Mars.plot_diag(x='sep', y='cr', label='Mars', markers='', log='', minimap=False, overplot_fig=1)
    Jupiter.plot_diag(x='sep', y='cr', label='Jupiter', markers='', log='', minimap=False, overplot_fig=1)
    Saturn.plot_diag(x='sep', y='cr', label='Saturn (no-ring)', markers='', log='', minimap=False, overplot_fig=1)
    Uranus.plot_diag(x='sep', y='cr', label='Uranus', markers='', log='', minimap=False, overplot_fig=1)
    Neptune.plot_diag(x='sep', y='cr', label='Neptune', markers='', log='', minimap=False, overplot_fig=1)

    """
    plt.plot(Mercury.sep, Mercury.cr, label='Mercury', lw=3)
    plt.plot(Venus.sep, Venus.cr, label='Venus', lw=3)
    plt.plot(Earth.sep, Earth.cr, label='Earth', lw=3)
    plt.plot(Mars.sep, Mars.cr, label='Mars', lw=3)
    plt.plot(Jupiter.sep, Jupiter.cr, label='Jupiter', lw=3)
    plt.plot(Saturn.sep, Saturn.cr, label='Saturn (no ring)', lw=3)
    plt.plot(Uranus.sep, Uranus.cr, label='Uranus', lw=3)
    plt.plot(Neptune.sep, Neptune.cr, label='Neptune', lw=3)

    plt.plot(Mercury.sep, Mercury.cr, color='k', lw=3)
    plt.plot(Venus.sep, Venus.cr, color='k', lw=3)
    plt.plot(Earth.sep, Earth.cr, color='k', lw=3)
    plt.plot(Mars.sep, Mars.cr, color='k', lw=3)
    plt.plot(Jupiter.sep, Jupiter.cr, color='k', lw=3)
    plt.plot(Saturn.sep, Saturn.cr, color='k', lw=3)
    plt.plot(Uranus.sep, Uranus.cr, color='k', lw=3)
    plt.plot(Neptune.sep, Neptune.cr, color='k', lw=3)

    Mercury.plot_diag(x='sep', y='cr', markers='', log='', minimap=False, overplot_fig=1, color='k')
    Venus.plot_diag(x='sep', y='cr', markers='', log='', minimap=False, overplot_fig=1, color='k')
    Earth.plot_diag(x='sep', y='cr', markers='', log='', minimap=False, overplot_fig=1, color='k')
    Mars.plot_diag(x='sep', y='cr', markers='', log='', minimap=False, overplot_fig=1, color='k')
    Jupiter.plot_diag(x='sep', y='cr', markers='', log='', minimap=False, overplot_fig=1, color='k')
    Saturn.plot_diag(x='sep', y='cr', markers='', log='', minimap=False, overplot_fig=1, color='k')
    Uranus.plot_diag(x='sep', y='cr', markers='', log='', minimap=False, overplot_fig=1, color='k')
    Neptune.plot_diag(x='sep', y='cr', markers='', log='', minimap=False, overplot_fig=1, color='k')
    """
    dummy = raw_input("Press Enter to continue...")

    print """\n\n\n\n\n
    ################################################################################

    Fig 2: Contrast ratio elovution over time, 30 degrees inclination with the observer.

    ################################################################################
    """

    plt.figure(2)
    plt.clf()
    duration = 365.26*15 #days
    Venus = c_planet(fr=np.linspace(0,duration/224.701,15000), t=224.701, a=0.7233, e=0.0068, i=3.39+system_inclination, w=54.85, distance=10, radius=0.086, albedo_geo=0.65)
    Earth = c_planet(fr=np.linspace(0,duration/365.26,15000), t=365.26, a=1, e=0.0167, i=0+system_inclination, w=114.2, distance=10, radius=0.091, albedo_geo=0.367)
    Jupiter = c_planet(fr=np.linspace(0,duration/4335.35,15000), t=4335.35, a=5.2034, e=0.04839, i=1.3+system_inclination, w=275.066, distance=10, radius=1, albedo_geo=0.52)

    Venus.compute(silent=True)
    Earth.compute(silent=True)
    Jupiter.compute(silent=True)

    Venus.plot_diag(x='date_rel', y='cr', label='Venus', markers='', log='y', minimap=False, overplot_fig=2)
    Earth.plot_diag(x='date_rel', y='cr', label='Earth', markers='', log='y', minimap=False, overplot_fig=2)
    Jupiter.plot_diag(x='date_rel', y='cr', label='Jupiter', markers='', log='y', minimap=False, overplot_fig=2)

    dummy = raw_input("Press Enter to continue...")


    print """\n\n\n\n\n
    ################################################################################

    Fig 3 & 4: SPOC diagram and histogram for HD 80606 b

    ################################################################################
    """

    HD80606b = c_planet(fr=np.linspace(0,1,10001)[:-1], t=111.436, tperi=2454424.857, a=0.449, e=0.93366, i=89.285, w=300.651, distance=58.4, radius=0.921, albedo_geo=0.5, albedo_scat=0.43)

    HD80606b.compute(silent=True)

    HD80606b.SPOC_diag(bw=False)

    HD80606b.SPOC_hist()

    dummy = raw_input("Press Enter to continue...")


    print """\n\n\n\n\n
    ################################################################################

    Fig 5: Date versus contrast ratio for HD80606b, with observability zone (assuming 10-6 CR and 3mas separation is suitable...)

    ################################################################################
    """
    print "Duration when HD 80606 b is above 10-6 contrast ratio (days): ", (HD80606b.cr>=10**-6).sum()*1./HD80606b.cr.size*HD80606b.t
    print "Duration when HD 80606 b is above 10-6 polarized contrast ratio (days): ", (HD80606b.crpol>=10**-6).sum()*1./HD80606b.cr.size*HD80606b.t
    print "Duration when HD 80606 b is above 3 mas angular separation (days): ", (HD80606b.sep>=0.003).sum()*1./HD80606b.cr.size*HD80606b.t
    print "Duration when HD 80606 b is above 3 mas angular separation AND above 10-6 contrast ratio (days): ", ((HD80606b.sep>=0.003) & (HD80606b.cr>=10**-6)).sum()*1./HD80606b.cr.size*HD80606b.t
    date_begin = funcs.next_date( HD80606b.date[(HD80606b.sep>=0.003) & (HD80606b.cr>=10**-6)].min(), HD80606b.t, date_after=True )
    date_end = funcs.next_date( HD80606b.date[(HD80606b.sep>=0.003) & (HD80606b.cr>=10**-6)].max(), HD80606b.t, date_after=True )
    print "Start date when previous condition is met: ", astro.julian_to_calendar( date_begin )
    print "End date when previous condition is met: ", astro.julian_to_calendar( date_end )

    date_obs = np.linspace(date_begin, date_end, 100)
    HD80606b_obs = c_planet(date=date_obs, t=111.436, tperi=2454424.857, a=0.449, e=0.93366, i=89.285, w=300.651, distance=58.4, radius=0.921, albedo_geo=0.5, albedo_scat=0.43)
    HD80606b_obs.compute(silent=True)

    plt.figure(5)
    plt.clf()
    HD80606b.plot_diag(x='date', y='cr', log='y', markers='nrp',label='HD80606b', overplot_fig=5)
    HD80606b_obs.plot_diag(x='date', y='cr', markers='', log='y', label='Observability', overplot_fig=5, color='r')

    dummy = raw_input("\nPress Enter to continue...")

    print """\n\n\n\n\n
    ################################################################################

    Fig 6: Custom plot

    ################################################################################
    """
    print "Available ploting outputs for HD80606b: ", HD80606b_obs.available_output,"\n\n"
    x = str(raw_input("Type the output parameter to plot on x-axis: "))
    xlog = int(raw_input("Do you want x in log10? (1 yes, 0 no): "))
    y = str(raw_input("Type the output parameter to plot on y-axis: "))
    ylog = int(raw_input("Do you want y in log10? (1 yes, 0 no): "))
    z = str(raw_input("Type the output parameter to plot on color-axis (empty is no color-axis): "))
    if z=='': z=None

    HD80606b.plot_diag(x=x, y=y, z=z, markers='trpn', log='x'*xlog+'y'*ylog, label='My custom plot')


    print "\n\n\n\n\nFor any further testing, type 'c_planet?'"


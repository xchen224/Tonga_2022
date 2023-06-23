import os, sys, json
import numpy as np
import datetime, pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate, stats, optimize, integrate
from sklearn.linear_model import LinearRegression
import pandas as pd
import pingouin as pg

import mylib.MLS_lib as MLS
import mylib.GCpylib as Gpy
from plot_libs import *
   
# Boltzmann constant
# k = 1.38e-23 # m2kg/s2/K  

def lutweights():
   return sorted([78.6, 75.2, 73.3, 71.9, 62.0, 53.2, 46.9, 42.0]) 
   
####
def pres2alt( p_prof, Tprof ):
   '''
   deltaP/P = - deltaz * (Mair * g / (R*T))
   '''
   p0   = 1013* 100
   allp = np.append( np.array([p0]), p_prof*100 ) 
   Mair = 0.02896
   g    = 9.8
   R    = 8.31
   alt  = np.full( (len(allp)), np.nan)
   alt[0] = 0
   
   for il in range(1,len(allp)):
      deltaP = allp[il] - allp[il-1]
      midP   = (allp[il] + allp[il-1])/2
      midT   = Tprof[il-1]
      deltaz = deltaP / midP * (R * midT) / (Mair * g) * (-1)
      if deltaz < 0:
         sys.exit('Delta z < 0 !!!')
      alt[il] = alt[il-1] + deltaz
      
   return alt[1:]      


#####
def rho_h2so4( T, W ):
   '''
   reference: Cathrine et al., 1998, density and surface tension of aqueous H2SO4 at low temperature
   https://pubs.acs.org/doi/pdf/10.1021/je980013g
   rho = sum(sum(a(i,j) * w^i * (T - 273.15)^j))
   '''
   a      = np.empty( (5,11) )
   a[0,:] = [999.8426, 547.2659, 526.295*10, -621.3958e2, 409.0293e3, -159.6989e4, 385.7411e4, -580.8064e4, 530.1976e4, 
             -268.2616e4, 576.4288e3]
   a[1,:] = [334.5402e-4, -530.0445e-2, 372.0445e-1, -287.767, 127.0854e1, -306.2836e1, 408.3714e1, -284.4401e1, 809.1053, 
             0, 0]
   a[2,:] = [-569.1304e-5, 118.7671e-4, 120.1909e-3, -406.4638e-3, 326.971e-3, 136.6499e-3, -192.7785e-3, 0, 0, 
             0, 0]
   a[3,:] = [0, 599e-6, -414.8594e-5, 111.9488e-4, -137.7435e-4, 637.3031e-5, 0, 0, 0, 
             0, 0]
   a[4,:] = [0, 0, 119.7973e-7, 360.7768e-7, -263.3585e-7, 0, 0, 0, 0, 
             0, 0]    
      
   Tc = T -273.15
       
   rho = 0          
   for i in range(a.shape[1]):
      for j in range(a.shape[0]):
         rho = rho + a[j,i] * (W ** i) * (Tc ** j)                  
   
   # g/cm3
   return rho/1e3

#####
def intp_LUT_wh2so4(iwh2so4, inlut, vars = ['AE1','kext1']):

   weights = lutweights()
   outlut  = dict()
   outlut['allr'] = inlut['allr'] 
   outlut['allv'] = inlut['allv'] 
   
   for var in vars:
        lut = np.full( (inlut[var].shape[0],inlut[var].shape[1]), np.nan )
        if iwh2so4 < weights[0]:
            lut  =  inlut[var][:,:,0]
        elif iwh2so4 > weights[-1]:
            lut =  inlut[var][:,:,-1]  
        else:
            for ir in range(len(inlut['allr'])):
              for iv in range(len(inlut['allv'])):
                wlut = inlut[var][ir,iv,:]
                f2   = interpolate.interp1d( weights, wlut, fill_value='extrapolation' )
                lut[ir, iv] = f2(iwh2so4)
        
        outlut[var] = lut
        
   return  outlut           

def intp_LUT_meanv(ix, lutdata, **kwargs):

    intp = dict()
    
    if 'vars' in kwargs.keys():
       vars = kwargs['vars']
    else:
       vars = list( lutdata.keys())   
       
    for var in vars:
        if len(lutdata[var].shape) == 2:
           intpy = []
           for iv in [3]: #range(len(lutdata['allv'])):
              if kwargs['ret']:
                 newy = np.interp( ix, lutdata[var][:,iv], lutdata['allr'] )
              else:   
                 newy = np.interp( ix, lutdata['allr'], lutdata[var][:,iv] )
              intpy.append( newy ) 
           intp[var] = np.mean( np.array(intpy) )              
        else:
           sys.exit('wh2so4 data needs to be input!')   
    
    return intp       

###
def quad_point(Rmin, Rmax):

   # Define the order of the Gaussian quadrature rule
   order = 10

   # Use the leggauss function to get the quadrature points and weights
   x, w = np.polynomial.legendre.leggauss(order)

   # Transform the quadrature points and weights to the integration limits
   x_transformed = 0.5*(Rmax-Rmin)*x + 0.5*(Rmax+Rmin)
   w_transformed = 0.5*(Rmax-Rmin)*w

   return x_transformed, w_transformed

   
#####
def read_Mie_LUT( ispec, largeLUT=False ):

   # read Mie LUT

   #specs = ['SULF','BC','OC']
   wavs  = [675, 997, 1020]
   #weights = [40, 70, 73, 81]
   weights   = sorted([78.6, 75.2, 73.3, 71.9, 62.0, 53.2, 46.9, 42.0])

   output  = dict()

   allext = dict()
   allQext= dict()
   for iw in range(len(wavs)):
      if largeLUT:
         data = np.load('data/'+ispec+'_'+str(wavs[iw])+'nm_large.npz')
      else:   
         data = np.load('data/'+ispec+'_'+str(wavs[iw])+'nm.npz')
      allext[wavs[iw]] = data['ext']
      allQext[wavs[iw]]= data['Qext']
      if iw == 0:
        output['vol0'] = data['vol']

   output['kext1'] = 0.75 * allQext[997] 
   output['kext2'] = 0.75 * allQext[1020]
   output['AE1'] = - (np.log(allext[675] / allext[997]) / np.log(675 / 997)) 
   output['AE2'] = - (np.log(allext[675] / allext[1020]) / np.log(675 / 1020)) 

   output['allr'] = data['reff']
   output['allv'] = data['veff']
   
   return output
   
####
def m_freepath( T, P ):
   '''
   mean free path of air at temperature T (K) and pressure P (hPa)
   Chapter 9 in ACP (Seinfeld): Eq. 9.6
   '''
   
   #M = 28.96 # g/mol
   #R = 8.3 # J/mol·K
   
   # https://www.omnicalculator.com/physics/kinematic-viscosity-of-air
   mu = 1.458e-6 * np.sqrt( T ** 3 ) / (T + 110.4)  # kg/m/s
   
   #lamda = 2 * mu / (P * 100 * np.sqrt( 8 * M * 1e-3 / np.pi / R / T )) # m
   
   # Dr. Jun Wang dissertation: A.5
   lamda0 = 0.0665 # um
   lamda  = lamda0 * (1013 / P ) * (T/293.15) * ((1 + 110/293.15)/(1 + 110/T))
   
   return mu, lamda
   
def slip_cor( lamda, R ):
   '''
   slip correction factor for particle size with R radius (um)
   Chapter 9 in ACP (Seinfeld): Eq. 9.34
   '''
   
   Cc = 1 + lamda/R * (1.257 + 0.4 * np.exp( - 1.1 * R / lamda))
   
   return Cc
   
def vel_g( T, P, R ):
   '''
   settling velocity due to gravity for particle size with R radius
   Chapter 9 in ACP (Seinfeld): Eq. 9.42
   '''
   rho = 1.83 # g/cm3 H2SO4
   g   = 9.8 # m/ s2
   
   mu, lamda = m_freepath( T, P )
   Cc        = slip_cor( lamda, R )
   
   vt = 2/9 * ((R * 1e-6) **2) * rho * 1e3 * g * Cc / mu # m/s
   
   return vt

def lognor_dis( ri, reff, veff):
   '''
   single-mode PSD following lognormal distribution defined by reff and veff (refer. Xu et al. 2015)
   '''
   
   # reff = rg*exp(5/2*ln2(sigma))
   # veff = exp(ln2(sigma)) - 1
   # dn(r)/dr = 1/(sqrt(2pi) * r * ln(sigma)) * exp(-0.5 * (((lnr - lnrg)/ln(sigma))**2) )
   
   lnsig2 = np.log(veff + 1)
   lnsig  = np.sqrt( lnsig2 )
   rg     = reff / np.exp( 2.5 * lnsig2 )
   dndr   = 1/( np.sqrt(2*np.pi) * ri * lnsig ) * np.exp( -0.5 * (((np.log(ri) - np.log(rg))**2)/lnsig2) )
   
   return dndr

def intg_nr( rmin, rmax, reff, veff):
   return integrate.quad( lognor_dis, rmin, rmax, args=(reff, veff))[0]   

####
def fKn_alpha(alpha=1, **kwargs):
   '''
   Chapter 12 in ACP (Seinfeld): Eq. 12.43
   ''' 
   
   T = kwargs.get('T', None)
   P = kwargs.get('P', None)
   R = kwargs.get('R', None)   
   
   if ((R is None) or (P is None)) or (R is None):
      Kn = kwargs.get('Kn', None)
   else:   
      mu, lamda = m_freepath( T, P )
      Kn = lamda/R
   
   if Kn is not None:
      f = 0.75*alpha*(1+Kn) / (Kn*Kn + Kn + 0.283*Kn*alpha + 0.75*alpha)
   else:
      sys.exit('There is no Kn value!') 
   if 'Kn' in kwargs.keys():
      return f
   else:      
      return f, Kn
  
def cond_dmdt( Ri, T, P, pp ):
   '''
   Chapter 13 in ACP (Seinfeld): Eq. 13.3
   ''' 
   R   = 8.3
   Mi  = 98 # g/mol
   kb = 1.38e-23  # m2kg/s2/K 
   
   fKn, Kn = fKn_alpha(alpha=1, T=T, P=P, R=Ri)
   epp = MLS.es(T, gas='H2SO4')
   
   Di = 0.1*1e-4 # m2/s #kb * T * Cc / (6 * np.pi * mu * Ri)
   dmdt = 2* np.pi * 2 * Ri * 1e-6 * Di * Mi / R / T * fKn * (pp - epp) * 100  # g/s
   dmdt = dmdt * 3600 * 24 # g/d
   #print(f' - Di:{Di*1e10},')
   #sys.exit()
   
   return dmdt
   
def cond_dvdt( Ri, T, P, pp ):
   
   rho = 1.83 # g/cm3
   dmdt = cond_dmdt( Ri, T, P, pp ) 
   dvdt = dmdt / rho * 1e12 # um3/d
   print(f' - dmdt:{np.min(dmdt), np.max(dmdt)}')
   if dmdt[10] > 0:
      print('Condensation')
   else:
      print('Evaporation')   
   
   return dvdt  
   
def cond_drdt( Ri, T, P, pp ):
   '''
   Chapter 13 in ACP (Seinfeld): Eq. 13.3
   ''' 
   R   = 8.3
   Mi  = 98 # g/mol
   kb = 1.38e-23  # m2kg/s2/K 
   rho = 1.83 # g/cm3  
   
   fKn = fKn_alpha(alpha=1, T=T, P=P, R=Ri)
   epp = MLS.es(T, gas='H2SO4')
   mu, lamda = m_freepath( T, P )
   Cc = slip_cor( lamda, R )
   
   b0 = 2 * kb * Mi / (3 * np.pi * mu * R * rho) # cm3/s/Pa
   
   drdt = b0 * Cc * fKn * (pp - epp) * 100 / (Ri*Ri) * 1e12  # um/s
   
   return drdt


####
def intgnr_didt(Ri, T, P, pp, reff, veff, oldvol, var):
      
   if var == 'V': 
   
      for ir in range(len(Ri)-1):
         rpoint, wpoint = quad_point(Ri[ir], Ri[ir+1])
         if ir == 0:
            rp, wp = rpoint, wpoint
         else:
            rp = np.append( rp, rpoint )
            wp = np.append( wp, wpoint )   
      
      dndr   = lognor_dis( rp, reff, veff )   
      v00   = 4/3 * np.pi * (rp **3)
      oldvol2= np.sum( wp*v00*dndr ) / np.sum( wp*dndr )
      dvdt  = cond_dvdt( rp, T, P, pp )
      irvar = dvdt + v00
      intgy = np.sum( wp*irvar*dndr ) # new V
      vol0  = intgy / np.sum( wp*dndr ) * (oldvol / oldvol2)
      
      dmdt = cond_dmdt( rp, T, P, pp )
      intgm = np.sum( wp*dmdt*dndr ) 
      #print(f' - intgm:{intgm}')
      #sys.exit()
#    elif var == 'R':
#       irvar = cond_dRdt( Ri, T, P, pp ) + Ri   
#       iry3  = (irvar**3) * dndr
#       iry2  = (irvar**2) * dndr
#       intgy = integrate.simpson( iry3, Ri ) / integrate.simpson( iry2, Ri ) # new Reff
   
   
   return vol0, intgm
   
def update_pp( dmdt, P, T ):
   
   # dmdt: g/m3
   Mi = 98 # g/mol
   R = 8.31 
   nair = P * 100 / R / T # mol/m3
   dvmr = dmdt / Mi / nair
   print(f' - dvmr:{dvmr}, nair:{nair}, dmdt:{dmdt}')
   dpp  = MLS.VMRtoP('H2SO4', dvmr, P)
   #pp1  = pp0 - dpp
   
   return dpp
  
#####
def expdecay( t, a ):
   return np.exp(-a*t)
   
def fit_expdecay( t, y ): 
   popt, pcov = optimize.curve_fit(expdecay, t, y, bounds=([0],[np.inf]) )
   return popt
   
####
#def efunc(x, a, b, c):
#   return a * (b ** (x - c))
   
def efunc(x, a, b, c):
   return a*(x**3) + b*(x**2) + c*x
   
def fit_e( x, y ):
   popt, pcov = optimize.curve_fit(efunc, x, y)
      
   return popt  
   
####
def fitfunc(X, a, b, c, d):
   return a * X[0,:] + b * X[1,:] + c * X[2,:] + d
   
def fit_each( X, y, use_bd=True ):
   if use_bd:
      popt, pcov = optimize.curve_fit(fitfunc, X, y, bounds=([-np.inf, 0, -np.inf, -np.inf], [0, np.inf, np.inf, np.inf]))
      #popt, pcov = optimize.curve_fit(fitfunc, X, y, bounds=([-np.inf, 0, -np.inf], [0, np.inf, np.inf]))
   else:   
      popt, pcov = optimize.curve_fit(fitfunc, X, y)
      
   return popt   

####
def beta( R, Cc, T, rho, kb, mu ):
   '''
   coagulation coefficient (cm3/s) R: m
   Chapter 13 in ACP (Seinfeld): Table 13.1
   '''

   mi = 4/3 * np.pi * (R**3) * rho
   ci = (8 * kb * T / np.pi / mi) ** (1/2)
   Di = kb * T * Cc / (6 * np.pi * mu * R)
   li = 8 * Di / (np.pi * ci)
   gi = 1/(6*R*li) * ( (2*R + li)**3 - (4*(R**2) + li*li)**(3/2) ) - 2*R
   beta0 = 1/ ( 4*R/(4*R+2*np.sqrt(2)*gi) + 4*Di/(R*np.sqrt(2)*ci) )
   
   return beta0
   
def coag_coef( T, P, R ):
   '''
   coagulation coefficient (cm3/s) 
   Chapter 13 in ACP (Seinfeld): Table 13.1
   '''
   kb = 1.38e-23 # m2kg/s2/K  
   rho = 1.83e-3 # kg/m3 
   
   mu, lamda = m_freepath( T, P )
   Cc = slip_cor( lamda, R )
   
   R0 = R * 1e-6 # um to m
   beta0 = beta( R0, Cc, T, rho, kb, mu )
   
   K0 = 8/3 * kb * T * Cc / mu
   K = K0 * beta0 * 1e6 # cm3/s
   K = K * 3600 * 24 # cm3/d
   
   return K, Cc
   
def char_time( K, N0 ):
   return 2 / K / N0

def Ntime( N0, t, K ):
   tauc = char_time( K, N0 )
   Nt   = N0 / (1 + (t/tauc))
   
   return Nt, tauc 

def coag_Nt( K, N0, t):
   return N0 / (1 + t * K * N0/2)

def coag_t2( Reff_t1, Nt1, T, P ):

   Kt1, Cc = coag_coef( T, P, Reff_t1 )
   print('K', Kt1, Cc)
   Nt2 = Nt1 - 1/2 * Kt1 * (Nt1**2)  
   Reff_t2 = Reff_t1 * ((Nt1/Nt2) ** (1/3))
   
   return Reff_t2, Nt2
   
def power_func(t, a, b, c):
   return a - b * (t ** (-c))

def power2_func(t, a, b):
   return a * (b ** t)
   
def exp_func(t, a, b, c):
   return a - b * np.exp(-c*t)

def fit_power(t, y):
   valid = np.logical_and( ~np.isnan(t), ~np.isnan(y) )
   popt, pcov = optimize.curve_fit(exp_func, t[valid], y[valid], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
   return popt

def fit_power2(t, y):
   valid = np.logical_and( ~np.isnan(t), ~np.isnan(y) )
   popt, pcov = optimize.curve_fit(power2_func, t[valid], y[valid], bounds=([0, 0], [np.inf, np.inf]))
   return popt
   
def allprocess_t1( Reff_t0, Nt0, T, P, **kwargs ):

   '''
   combine all processes together
   parameters: a: nucleation/transport; b: condensation/evaporation; frac: hygroscopic growth factor
               t: number of days from settling start
               
   '''
   Vt0 = 4/3 * np.pi * (Reff_t0**3) * Nt0
   
   Nt  = Nt0
   Vt  = Vt0
   Rt  = Reff_t0
   
   # 1. nucleation / transport:
   if 'a' in kwargs.keys():
      Nt = Nt + kwargs['a']
      Vt = Vt + kwargs['a'] * (4/3) * np.pi * (Rt**3)
      Rt = (Vt / Nt * (3/4) / np.pi) ** (1/3)
      #print('NUC:', Rt, Vt, Nt)
   
   # 2. coagulation:
   if 'Kt' in kwargs.keys():
      Kt = kwargs['Kt']
   else:   
      Kt, Cc = coag_coef( T, P, Rt )
      #Kt = Kt*10
   #print('K', Kt)
   
   if kwargs['do_coag']:   
     Nt0 = Nt
     Rt0 = Rt
     Nt = Nt - 1/2 * Kt * (Nt**2)  
     Rt = Rt0 * ((Nt0/Nt) ** (1/3))
     if np.abs(Nt0/Nt -1) < 1e-6:
       print('check coagulation:', Nt0, Nt)
   
   # 3. condensation / evaporation:
   if 'b' in kwargs.keys():
      Rt = Rt + kwargs['b'] / Rt
      #Rt = Rt + kwargs['b'] / (Rt**3)
      Vt = 4/3 * np.pi * (Rt**3) * Nt
   
   # 4. hygroscopic growth:
   if 'frac' in kwargs.keys():
      #print('before:', Rt, kwargs['frac']  )
      Rt = Rt * kwargs['frac'][1] / kwargs['frac'][0] 
      Vt = 4/3 * np.pi * (Rt**3) * Nt
      #print('HYG:', Rt, Vt, Nt)
      
   # 5. gravitational settling:
   if 't' in kwargs.keys():  
      if 'vg' not in kwargs.keys(): 
         sys.exit('Miss vg data for gravitational settling calculation')
      else:   
         Rs = kwargs['Rs']
         veff = kwargs['veff']
         R_lim = np.interp( 1/kwargs['t'], kwargs['vg'], Rs )
         norm_n = intg_nr( Rs[0], Rs[-1], Rt, veff )
         rem_n  = intg_nr( R_lim, Rs[-1], Rt, veff )
         if (Rt < Rs[-1]):
           frac   = rem_n/norm_n   
           deltaN = Nt * frac
           Nt     = Nt - deltaN             
#          Rtg    = efunc( R_lim, *kwargs['pars'] )     
#          if Rt >= Rtg:
#             Rt = Rtg
#          Vt = 4/3 * np.pi * (Rt**3) * Nt
           Vt     = Vt - 4/3 * np.pi * ((R_lim*1)**3) * deltaN
           Rt     = (Vt / Nt * (3/4) / np.pi) ** (1/3)


   # 6. transport:
   if 'c' in kwargs.keys():
      Nt = Nt - kwargs['c']
      Vt = Vt - kwargs['c'] * (4/3) * np.pi * (Rt**3)
      Rt = (Vt / Nt * (3/4) / np.pi) ** (1/3)
             
   
   return Rt, Nt, Vt, Kt
   

########   
def allprocess_t2( Reff_t0, Nt0, Vt0, T, P, rho, **kwargs ):

   '''
   combine all processes together, involving observed V
   parameters: a: nucleation/transport; b: condensation/evaporation; frac: hygroscopic growth factor
               t: number of days from settling start
               
   '''
   
   #rho     = 1.83 # g/cm3
   lutdata = read_Mie_LUT('SULF', largeLUT=False)
   wlut    = lutdata
      
   sf      = 1e-12 * 1e9 # cm-3 to um-3 to um-2   
   intp    = intp_LUT_meanv(Reff_t0, wlut, vars=['vol0'], ret=False)
   intpv0  = intp['vol0']
#    Vt0     = intpv0 * Nt0 * sf
   #print('check intp0:', intpv0, Reff_t0, wlut['vol0'])
   #sys.exit()
   
   Nt  = Nt0
   Vt  = Vt0
   Rt  = Reff_t0
   vol = intpv0
   deltap = 0
   
   # 1. nucleation / transport:
   if kwargs['do_nuc']:
    if 'a' in kwargs.keys():
      Nt = Nt + kwargs['a']
      #Vt = Vt + kwargs['a'] * vol * sf
      Vt = Vt + kwargs['a'] * 4/3 * np.pi * (0.01 **3) * sf
      vol = Vt / Nt / sf
      intp= intp_LUT_meanv(vol, wlut, vars=['vol0'], ret=True)
      Rt  = intp['vol0']
      #deltam = kwargs['a'] * 4/3 * np.pi * (0.01 **3) * rho  * 1e-12 * 1e6
      #deltap = update_pp( deltam, P, T )
      #print(f' - NUC: Rt={Rt}, Vt={Vt}, Nt={Nt}, vol={vol}, deltap={deltap}')
      
    else:
      if 'amass' in kwargs.keys():
         Va   = kwargs['amass'] / rho * 1e12 / 1e18 * 1e9 # cm3/m3 to um3/um2
      elif 'Va' in kwargs.keys(): 
         Va   = kwargs['Va']
        
      Nt = Nt + Va / (4/3 * np.pi * (0.01 **3)) / sf
      Vt = Vt + Va 
      vol = Vt / Nt / sf
      intp= intp_LUT_meanv(vol, wlut, vars=['vol0'], ret=True)
      Rt  = intp['vol0']   
      
      
    print(f' - NUC: Rt={Rt}, Vt={Vt}, Nt={Nt}, vol={vol}')
      #sys.exit()
   
   # 2. coagulation:
   if 'Kt' in kwargs.keys():
      Kt = kwargs['Kt']
   else:   
      Kt, Cc = coag_coef( T, P, Rt )
      #Kt = Kt*10
   #print('K', Kt)
   
   if kwargs['do_coag']:   
     Nt0 = Nt
     Rt0 = Rt
     Nt = Nt - 1/2 * Kt * (Nt**2)  
     vol = Vt / Nt / sf
     intp= intp_LUT_meanv(vol, wlut, vars=['vol0'], ret=True)
     Rt  = intp['vol0']
     #print('check intp:', vol, Rt, wlut['vol0'])
     #sys.exit()
   
   # 3. condensation / evaporation:
   if 'b' in kwargs.keys():
      Rt = Rt + kwargs['b'] / Rt
      #Rt = Rt + kwargs['b'] / (Rt**3)
      intp    = intp_LUT_meanv(Rt, wlut, vars=['vol0'], ret=False)
      vol     = intp['vol0']
      Vt      = vol * Nt * sf
   elif 'Vb' in kwargs.keys():   
      Vt      = Vt + kwargs['Vb']
      vol     = Vt / Nt / sf
      intp= intp_LUT_meanv(vol, wlut, vars=['vol0'], ret=True)
      Rt  = intp['vol0']
   elif 'do_phycond' in kwargs.keys():
      if kwargs['do_phycond']:   
         oldvol = vol
         usepp  = kwargs['pp'] - deltap
         vol, intgm = intgnr_didt(kwargs['Rs'], T, P, usepp, Rt, kwargs['veff'], oldvol, 'V')
         deltaV = vol * Nt * sf - Vt
         print( f'- deltaV:{deltaV}, oldVt:{Vt}')
         Vt  = vol * Nt * sf
         intp= intp_LUT_meanv(vol, wlut, vars=['vol0'], ret=True)
         Rt  = intp['vol0']
         deltap = update_pp( intgm*Nt*1e6, P, T )
         
   
   # 4. hygroscopic growth:
   if 'frac' in kwargs.keys():
      #print('before:', Rt, kwargs['frac']  )
      Rt = Rt * kwargs['frac'][1] / kwargs['frac'][0] 
      intp    = intp_LUT_meanv(Rt, wlut, vars=['vol0'], ret=False)
      vol     = intp['vol0']
      Vt      = vol * Nt * sf
      #print('HYG:', Rt, Vt, Nt)
      
   # 5. gravitational settling:
   if 't' in kwargs.keys():  
      if 'vg' not in kwargs.keys(): 
         sys.exit('Miss vg data for gravitational settling calculation')
      else:   
         Rs = kwargs['Rs']
         veff = kwargs['veff']
         R_lim = np.interp( 1/kwargs['t'], kwargs['vg'], Rs )
         norm_n = intg_nr( Rs[0], Rs[-1], Rt, veff )
         rem_n  = intg_nr( R_lim, Rs[-1], Rt, veff )
         if (Rt < Rs[-1]):
           frac   = rem_n/norm_n   
           deltaN = Nt * frac
           Nt     = Nt - deltaN             
#          Rtg    = efunc( R_lim, *kwargs['pars'] )     
#          if Rt >= Rtg:
#             Rt = Rtg
#          Vt = 4/3 * np.pi * (Rt**3) * Nt
           Vt     = Vt - 4/3 * np.pi * ((R_lim*1)**3) * deltaN
           Rt     = (Vt / Nt * (3/4) / np.pi) ** (1/3)


   # 6. transport:
   if 'c' in kwargs.keys():
      Nt = Nt - kwargs['c']
      Vt = Vt - kwargs['c'] * vol * sf
      vol = Vt / Nt / sf
      intp= intp_LUT_meanv(vol, wlut, vars=['vol0'], ret=True)
      Rt  = intp['vol0']
   elif 'Vc' in kwargs.keys():
      Vt = Vt - kwargs['Vc'] 
      Nt = Nt - kwargs['Vc'] / (4/3*np.pi*(8**3)) / sf
      vol = Vt / Nt / sf
      intp= intp_LUT_meanv(vol, wlut, vars=['vol0'], ret=True)
      Rt  = intp['vol0'] 
   elif 'Rc' in kwargs.keys():
      Rt  = Rt - kwargs['Rc']
      intp    = intp_LUT_meanv(Rt, wlut, vars=['vol0'], ret=False)
      vol     = intp['vol0']
      Vt = vol * Nt * sf   
             
   print(f' - Rt:{Rt}, Nt:{Nt}, Vt:{Vt}, Kt:{Kt}, deltap:{deltap}')
   return Rt, Nt, Vt, Kt, deltap

####
def fit_Rlim_Reff(Rs, reff, veff, plt=False):

   R_lim = Rs[Rs > 0.02]
   nr    = np.empty( (len(reff),len(Rs)) )
   for ir in range(len(reff)):
      nr[ir,:] = lognor_dis( Rs, reff[ir], veff ) 

   cor_reff = np.full( len(R_lim), np.nan )   
   for ir0 in range(len(R_lim)):
      idx = np.argmin( np.abs(Rs - R_lim[ir0]) )
      iall = np.where( nr[:,idx] < 1e-5 )[0]
      if len(iall)>0:
         deltanr = nr[iall,idx] - nr[iall,idx-1]
         ii = iall[np.where(deltanr < 0)[0][-1]]
         cor_reff[ir0] = reff[ii]   
   
   pltidx = (~np.isnan(cor_reff)) & (cor_reff < 1) 
   # polynomial
   pars = fit_e( R_lim[pltidx], cor_reff[pltidx] )
   fity = efunc( R_lim[pltidx], *pars )
   
   if plt:
     fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
     axes.scatter( R_lim[pltidx], cor_reff[pltidx], c='r', s=3, label='cal' )
     axes.plot( R_lim[pltidx], fity, lw=0.8, color='k', label='fit: %.2fx2 + %.2fx + %.2f' % tuple(pars))
     axes.legend(fontsize='x-small')

# linear
# paths, slope, intercept, min_all, max_all = sca.scatter_plot(axes, 
#              R_lim[pltidx], cor_reff[pltidx],
#              fig=None, mcolor='r', label_p = 'upper left', prt_mean=True, prt_rmse=False, \
#              linearregress_line=True, one2one_line=False, alpha=1.0, )  

     axes.set_xlabel( r'r$_{grav}$ ($\mu$m)' )
     axes.set_ylabel( r'R$_{eff}$ ($\mu$m)' )
     plt.tight_layout()
     plt.savefig( 'img/Rlim_vs_Reff_gravity.png', dpi=300 )
     plt.close()    
   
   return pars  


####               
def plt_timeseries_fillx( axes, xdata, ydata, **kwargs ):

   title      = kwargs.get('title', "")
   ylims      = kwargs.get('ylims', [0, 1])
   xlims      = kwargs.get('xlims', [xdata[0], xdata[-1]])
   xticks     = kwargs.get('xticks', np.arange(xlims[0],xlims[1],(xlims[1]-xlims[0])/10))
   yticks     = kwargs.get('yticks', np.arange(ylims[0],ylims[1],(ylims[1]-ylims[0])/10))
   xticklabel = kwargs.get('xticklabel', None)
   ylabel     = kwargs.get('ylabel', "")
   xlabel     = kwargs.get('xlabel', "")
   color      = kwargs.get('color', 'r')
   label      = kwargs.get('label', None)
   setaxis    = kwargs.get('setaxis', True)
   xminor     = kwargs.get('xminor', 1)
   ls         = kwargs.get('ls', '-')
   pltline    = kwargs.get('pltline', True)
   alpha      = kwargs.get('alpha', 0.3)
   
 
   if pltline:
     if label is None:
       axes.plot( xdata, ydata, color=color, lw=1.25, ls=ls, alpha=0.9 )
     else:   
       axes.plot( xdata, ydata, color=color, lw=1.25, ls=ls, alpha=0.9, label=label )
   if ('y1' in kwargs.keys()):
     if ('y2' in kwargs.keys()):    
        axes.fill_between( xdata, kwargs['y1'], kwargs['y2'], facecolor=color, edgecolor=None, alpha=alpha )
     else:
        if label is None:
           axes.fill_between( xdata, ydata, kwargs['y1'], facecolor=color, edgecolor=None, alpha=0.9 ) 
        else:
           axes.fill_between( xdata, ydata, kwargs['y1'], facecolor=color, edgecolor=None, alpha=0.9, label=label )      
    
   if setaxis:
     if ylims[0] > ylims[1]:
         axes.set_yscale('log')   
         axes.invert_yaxis()  
         axes.yaxis.set_minor_locator(ticker.FixedLocator([1,2,3,5,7,10,20,30,50,70,100,150,200,300]))  
         axes.yaxis.set_minor_formatter(ticker.FormatStrFormatter("%.0f"))
         axes.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
     else:
         axes.set_yticks( yticks )    
     axes.set_ylim(ylims) 
     axes.set_xlim(xlims)
     axes.set_ylabel(ylabel)
     axes.set_xlabel(xlabel)
   #axes.ticklabel_format(style='sci',axis='x',scilimits=(0,0))  
     if len(title) > 0:
        axes.set_title( title )   
     if xticks is not None:
        axes.set_xticks( xticks )
     if xminor is not None:   
        axes.xaxis.set_minor_locator(ticker.MultipleLocator(xminor))
     if xticklabel is not None:
        axes.set_xticklabels( xticklabel )
   
   return  axes 

def comb_Xy( allxdata, allTanom, lagcorr, deltalag, nidx=1 ):
    '''
    allWV: original timeseries of WV 
    Note: not test for other nidx
    
    '''
    X = np.array([])
    maxtime = []
    idxarr = dict()
    
    for key in lagcorr.keys():
      maxlag = np.nanargmax(np.abs(lagcorr[key]) )
    
      if maxlag < int(nidx/2):
        idxarr[key] = [i*deltalag for i in range(nidx)]
      else:
        idxarr[key] = [(maxlag+i-int(nidx/2))*deltalag for i in range(nidx)]
      maxtime.append( max(idxarr[key]) )     
      print(key, maxlag, idxarr[key])
       
    #X = allaer[maxtime:]
    allmaxt = max(maxtime)
    nt      = len(allTanom) - allmaxt
    
    for key in lagcorr.keys():
      #print('Comb X', key)         
      for i in range(len(idxarr[key])):
        istart = allmaxt - idxarr[key][i] # pay attention   
        if len(X) == 0:
          X = allxdata[key][istart:istart+nt]
        else:   
          X = np.vstack( (X, allxdata[key][istart:istart+nt]) )
    
    if 'N2O' in allxdata.keys():
       X = np.vstack( (X, allxdata['N2O'][allmaxt:allmaxt+nt]) )      

    y = np.reshape( allTanom[allmaxt:allmaxt+nt], (-1,1) )
    
    return X, y, allmaxt, nt


def partial_corr( X, y, addN2O=False ):

   '''
   calculate the partial correlation between y and each x in matrix X removing the influence of rest x
   X: H2O, aer and N2O
   '''
   
   p_obj = dict()
   
   if addN2O:
      nWV = 3
   else:   
      nWV = 2
   
   if X.shape[1] > nWV:
      WVnum = ['H2O'+str(i+1) for i in range(X.shape[1]-1)]
   else:
      WVnum = ['H2O']
         
   if addN2O:      
      df = pd.DataFrame( np.hstack((X,y)), columns=WVnum+['aer','N2O','y'] ) 
      N2Oname = ['N2O']
   else:
      df = pd.DataFrame( np.hstack((X,y)), columns=WVnum+['aer','y'] )    
      N2Oname = []
   #print(df)
   
   # partial corr between Tanom and AER removing WV
   p_obj['Aer'] = pg.partial_corr(data=df, x='aer', y='y', covar=WVnum+N2Oname)
        
   # partial corr between Tanom and WV removing AER
   if len(WVnum) == 1:
      p_obj['H2O'] = pg.partial_corr(data=df, x='H2O', y='y', covar=['aer']+N2Oname)   
   
   if addN2O:  
      p_obj['N2O'] = pg.partial_corr(data=df, x='N2O', y='y', covar=WVnum+['aer'])   
      
   print('corr mat:', df.pcorr())
   
   return p_obj   



def partial_corr_r( X, y, covar='H2O' ):

   '''
   calculate the partial correlation between y and each x in matrix X removing the influence of rest x
   X: H2O, aer and N2O
   '''
   
   p_obj = dict()
   
   if X.shape[1] > 2:
      sys.exit('More than two X factors')
      
   if covar == 'H2O':
      x0 = X[:,1]
      x1 = X[:,0]
   else:
      x0 = X[:,0]
      x1 = X[:,1]   
      
   rx = stats.pearsonr(x0, x1)[0]
   ry0 = stats.pearsonr(x0, y)[0]    
   ry1 = stats.pearsonr(x1, y)[0] 
   
   # r_abc = (r_ab - r_ac*r_bc)/sqrt(1-r_ac^2)/sqrt(1-r_bc^2)
   pr = (ry0 - rx*ry1)/(np.sqrt(1 - rx**2)*np.sqrt(1 - ry1**2))
   
   normr = [rx, ry0, ry1]
            
   return pr, normr   


def total_corr( X, y ):

   '''
   calculate the total correlation between y and each x in matrix X, as well as the correlation between each x
   X: H2O, aer and N2O
   '''
   
   corr = dict()
   
   if X.shape[1] > 2:
      xname = ['H2O','Aer','N2O']
   else:
      xname = ['H2O','Aer']   
      
   for ic in range(X.shape[1]):
      savename0 = 'T&'+xname[ic]
      corr[savename0] = stats.pearsonr(y, X[:,ic])[0]
      for ic2 in range(ic+1, X.shape[1]):
         savename1 = xname[ic]+'&'+xname[ic2]  
         corr[savename1] = stats.pearsonr(X[:,ic2], X[:,ic])[0]
            
   return corr   


def read_MLS_gas( gasname, iy = 'v4.2noQC_' ):  

   # 3. read MLS WV
   gas = gasname #['H2O','Temperature']
   profname = []
   for igas in gas:
      if igas == 'H2O':
         profname.append('H2O')
      else:
         profname.append('Tprof')
   numday = dict()
   didx   = dict()
   conzm  = dict()
   allzm  = dict()
   
   mmname  = ['Dec']
   usemm   = np.arange(11)   
   for imm in usemm:
     mmtime = datetime.datetime(2022, imm+1, 1)
     strtime = mmtime.strftime('%b')
     mmname.append(strtime)
   usemm = np.append(np.array([11]), usemm)
   
   for im in range(len(usemm)):
     imm = usemm[im]
     imonth   = mmname[im]

     with open('./data/MLS_gas_daily_mean_zm_'+iy+imonth+'.json', 'r') as f:
       idata = json.load(f)
     zmprof = np.load( './data/MLS_prof_daily_mean_zm_'+iy+imonth+'.npz' )

     for i, igas in enumerate(gas):
       # connect all times
       if im == 0:
          numday[igas] = []
       numday[igas].append( idata[igas]['useday'][-1] + 1 )

       if im == 0:
         didx[igas] = np.array(idata[igas]['useday'])   
         conzm[igas]  = zmprof[profname[i]]
         
       else:
         didx[igas] = np.append( didx[igas], sum(numday[igas][:im]) + np.array(idata[igas]['useday']) )
         conzm[igas]  = np.concatenate( (conzm[igas], zmprof[profname[i]]), axis=0 )  

   glevs = dict()
   for igas in gas:
      if igas == 'Temperature':
         glevs[igas] = zmprof['lev_T']
      else:
         glevs[igas] = zmprof['lev_'+igas]  
      allzm[igas] = np.full( (sum(numday[igas]), len(zmprof['clats']), len(glevs[igas])), np.nan )
      allzm[igas][didx[igas],:,:] = conzm[igas]

   clats = zmprof['clats']
   nday  = sum(numday[igas])

   # check ndays
   print(igas, glevs[igas])
   
   return allzm, glevs, nday, clats

def read_intp_wh2so4(lats, alt, label='v4.2noQC_'):

   wh2so4_data = np.load('data/MLS_wh2so4_'+label+'dzm_DectoNov.npz')
   Pprof  = wh2so4_data['levT']
   useP   = wh2so4_data['uselev']
   lidx   = np.where((Pprof >= useP[-1]) & (Pprof <= useP[0]))[0]

   # interp in altitude
   xx = np.arange(wh2so4_data['wh2so4'].shape[0])
   intp_wh2so4 = np.full( ( len(xx), len(lats), len(alt)), np.nan )

   for idd in range(len(xx)):
     for ic in range(len(lats)):
        iwh2so4 = wh2so4_data['wh2so4'][idd,ic,:]
        if len(np.where(~np.isnan(iwh2so4))[0]) > 0:
          iTprof = wh2so4_data['Tprof'][idd,ic,:]
          valid  = np.where(~np.isnan(iTprof[lidx]))[0]
          Hprof  = np.full( len(lidx), np.nan )
          Hprof[valid]  = pres2alt( Pprof[lidx][valid], iTprof[lidx][valid] ) * 1e-3 # km
          val1   = (~np.isnan(Hprof))
          val2   = (~np.isnan(iwh2so4))
          val    = np.where((val1 == True) & (val2 == True) )[0]
          if len(val) == 0:
             print('before', idd, lats[ic], iTprof[lidx], Pprof[lidx], Hprof, iwh2so4)
             continue
          f  = interpolate.interp1d(Hprof[val], iwh2so4[val], kind='linear')
          uselidx = np.where( (alt>= Hprof[val][0]) & (alt<= Hprof[val][-1]) )[0]
          newdata = f(alt[uselidx]) 
          intp_wh2so4[idd,ic,uselidx] = newdata
          intp_wh2so4[idd,ic,alt>Hprof[-1]] = newdata[-1]
          #if lats[ic] == -40:
          #   print('check:', idd, ic, iTprof, Hprof,intp_wh2so4[idd,ic,:])
   
   return intp_wh2so4

def read_wh2so4( label='v4.2noQC_' ):
   
   wh2so4_data = np.load('data/MLS_wh2so4_'+label+'dzm_DectoNov.npz')
   
   return wh2so4_data
   
def intp_daywh2so4(zidx, wh2so4_data):

   Pprof  = wh2so4_data['levT']
   useP   = wh2so4_data['uselev']
   lidx   = np.where((Pprof >= useP[-1]) & (Pprof <= useP[0]))[0]
   wh2so4 = wh2so4_data['wh2so4']
   print(f' - wh2so4 shape:{wh2so4.shape}, nlevel:{len(useP)}')

   # region average
   wh2so4_2d = np.nanmean( wh2so4_data['wh2so4'][:,zidx,:], axis=1 )
   T_2d      = np.nanmean( wh2so4_data['Tprof'][:,zidx,:], axis=1 )
   nday      = wh2so4_2d.shape[0]
   xday      = np.arange(nday)
   new_wh2so4= np.array([])
   new_T     = np.array([])
   validil   = []
   validil2  = []
   
   # interp in day
   for il in range(len(useP)):
      valid = np.where( ~np.isnan(wh2so4_2d[:,il]) )[0]
      if len(valid) == 0:
         continue
      else:   
         if len(new_wh2so4) == 0:
            new_wh2so4 = np.reshape( np.interp( xday, xday[valid], wh2so4_2d[valid,il] ), (-1,1) )
         else:
            intp  =  np.reshape( np.interp( xday, xday[valid], wh2so4_2d[valid,il] ), (-1,1) )
            new_wh2so4 = np.hstack( (new_wh2so4, intp) )  
         validil.append( il )
         
   for il in range(len(Pprof)):
      valid = np.where( ~np.isnan(T_2d[:,il]) )[0]
      if len(valid) == 0:
         continue
      else:   
         if len(new_T) == 0:
            new_T = np.reshape( np.interp( xday, xday[valid], T_2d[valid,il] ), (-1,1) )
         else:
            intp  =  np.reshape( np.interp( xday, xday[valid], T_2d[valid,il] ), (-1,1) )
            new_T = np.hstack( (new_T, intp) )  
         validil2.append( il )
         
   wh2so4_P = useP[validil] 
   T_P      = Pprof[validil2]     
    
   return  new_wh2so4, wh2so4_P, T_P, new_T    
          
   
def intp_altwh2so4(alt, iwh2so4, iwh2so4_T, wh2so4_P, T_P):

   Pprof  = T_P
   useP   = wh2so4_P
   lidx   = np.where((Pprof >= useP[-1]) & (Pprof <= useP[0]))[0]
   
   # interp in altitude
   intp_wh2so4 = np.full( len(alt), np.nan )

   if len(np.where(~np.isnan(iwh2so4))[0]) > 0:
        iTprof = iwh2so4_T
        valid  = np.where(~np.isnan(iTprof[lidx]))[0]
        Hprof  = np.full( len(lidx), np.nan )
        Hprof[valid]  = pres2alt( Pprof[lidx][valid], iTprof[lidx][valid] ) * 1e-3 # km
        val1   = (~np.isnan(Hprof))
        val2   = (~np.isnan(iwh2so4))
        val    = np.where((val1 == True) & (val2 == True) )[0]
        if len(val) == 0:
             print('before', iTprof[lidx], Pprof[lidx], Hprof, iwh2so4)
        f  = interpolate.interp1d(Hprof[val], iwh2so4[val], kind='linear')
        uselidx = np.where( (alt>= Hprof[val][0]) & (alt<= Hprof[val][-1]) )[0]
        newdata = f(alt[uselidx]) 
        intp_wh2so4[uselidx] = newdata
        intp_wh2so4[alt>Hprof[-1]] = newdata[-1]
          #if lats[ic] == -40:
          #   print('check:', idd, ic, iTprof, Hprof,intp_wh2so4[idd,ic,:])
   else:
        intp_wh2so4 = np.full( len(alt), np.nan )
   
   return intp_wh2so4

###############

def read_Tanom( start_t, end_t, savestr, ttype='daily'):

   # read in period start_t ~ end_t
   # start_t: [yy, mm, dd]
   
   data0    = np.load( './data/MLS_Tprofanom_'+savestr+'_'+ttype+'_mean.npz' )
   levs0     = data0['Tlevs'] #

   starttime = datetime.datetime(start_t[0],start_t[1],start_t[2])
   endtime   = datetime.datetime(end_t[0],end_t[1],end_t[2])
   yystart   = datetime.datetime(2005,1,1)
   sxidx   = int( (starttime - yystart).total_seconds() / 3600 / 24 )
   exidx   = int( (endtime - yystart).total_seconds() / 3600 / 24 )
 

   return data0, sxidx, exidx

def comb_TaerWV( removeN2O=True ):

   ####################
   # daily in 2022
   #####################

   times = ['202112']
   for itt in range(11):
      times.append( '2022{:0>2d}'.format(itt+1) )
   print(times)

   im_nday = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
   nday    = 31
   mm_nday = [31]
   mmname  = ['Dec']
   usemm   = np.arange(11)   
   for imm in usemm:
     mmtime = datetime.datetime(2022, imm+1, 1)
     strtime = mmtime.strftime('%b')
     mmname.append(strtime)
     nday += im_nday[imm]
     mm_nday.append( im_nday[imm] )
   usemm = np.append(np.array([11]), usemm)

   allzm  = dict()

   # 1. read OMPS-LP aer ext     
   vars    = ['zmb5']
   for i, it in enumerate(times):
      data  = np.load('data/omps_aer_region_meanprof_'+it+'.npz')     
      dayidx  = data['dayidx']
      if it == times[0]:
        key = vars[0]
        if len(data[key].shape) == 2:
           allzm['aer'] = np.full( (nday,data[key].shape[1]), np.nan )
        else:
           allzm['aer'] = np.full( (nday,data[key].shape[1],data[key].shape[2]), np.nan )   
      else:
         dayidx += sum(mm_nday[:i]) 
    
      allzm['aer'][dayidx,:] = data[vars[0]]      
         
   bands = data['bands']
   nb    = len(bands)
   alt   = data['alt']
   clats = data['lat_bin']
   print('nday:',nday)

   # 2. read MLS Tanom
   savestr  = 'detrend'
   data0, sxidx, exidx = read_Tanom( [2021,12,1], [2022,12,1], savestr, ttype='daily')
   levs0     = data0['Tlevs'] #

   # remove dynamics impact
   if removeN2O:
      data1    = np.load( './data/MLS_Tprofanom_N2Ocorr_daily_mean.npz' )
      levs1    = data1['Tlevs']

   # read N2O as well
   else:
      data1    = np.load( './data/MLS_N2Oprofanom_detrend_daily_mean.npz' )
      levs1    = data1['Tlevs'] #

   if len(levs0) > len(levs1):
      print('The number of vertical levels of T is larger than N2O!')
      levs = levs1
      lidx1 = np.arange(len(levs1))
      lidx0 = np.array([ np.argmin(np.abs(levs0 - il)) for il in levs])

   elif len(levs0) < len(levs1):
      print('The number of vertical levels of T is smaller than N2O!')
      levs = levs0
      lidx0 = np.arange(len(levs0))
      lidx1 = np.array([ np.argmin(np.abs(levs1 - il)) for il in levs])
            
   else:
      levs = levs1
      lidx0 = np.arange(len(levs1))
      lidx1 = lidx0
      
   nlat    = len(clats)
   nlev    = len(levs)
   
   if removeN2O:
      N2OTprof = data1['deltaT'][:,:,lidx1] 
      detr_prof= data0['Tprof'][:,:,lidx0] - N2OTprof
      allzm['Tanom'] = detr_prof[sxidx:exidx,:,:]
   else:
      allzm['N2O']= data1['Tprof'][sxidx:exidx,:,lidx1] * 1e9
      allzm['Tanom'] = data0['Tprof'][sxidx:exidx,:,lidx0]
   print('idx:', sxidx, exidx)


   # 3. read MLS WV
   iy = 'v4.2noQC_'
   gas = ['H2O','Temperature']
   profname = ['H2O','Tprof']
   numday = dict()
   didx   = dict()
   conzm  = dict()

   for im in range(len(usemm)):
     imm = usemm[im]
     imonth   = mmname[im]

     with open('./data/MLS_gas_daily_mean_zm_'+iy+imonth+'.json', 'r') as f:
       idata = json.load(f)
     zmprof = np.load( './data/MLS_prof_daily_mean_zm_'+iy+imonth+'.npz' )

     for i, igas in enumerate(gas):
       # connect all times
       if im == 0:
          numday[igas] = []
       numday[igas].append( idata[igas]['useday'][-1] + 1 )

       if im == 0:
         didx[igas] = np.array(idata[igas]['useday'])   
         conzm[igas]  = zmprof[profname[i]]
         
       else:
         didx[igas] = np.append( didx[igas], sum(numday[igas][:im]) + np.array(idata[igas]['useday']) )
         conzm[igas]  = np.concatenate( (conzm[igas], zmprof[profname[i]]), axis=0 )  

   glevs = dict()
   for igas in gas:
      if igas == 'Temperature':
         glevs[igas] = zmprof['lev_T']
      else:
         glevs[igas] = zmprof['lev_'+igas]  
      allzm[igas] = np.full( (sum(numday[igas]), len(zmprof['clats']), len(glevs[igas])), np.nan )
      allzm[igas][didx[igas],:,:] = conzm[igas]


   # check ndays
   print('check nday for aer, Tanom, WV data:', allzm['aer'].shape, allzm['Tanom'].shape, allzm['H2O'].shape)
   print(levs, glevs['H2O'])
   
   return allzm, glevs, levs, nday, mmname, clats, alt

#######

def smo_timeseries( allzm, nday, smo_day = 5):
   # 5 days smooth

   alldata = dict()
   for var in allzm.keys():
      usedata = allzm[var]
      ishape = len(usedata.shape)
      if ishape > 3:
         data10d = np.full( (smo_day, nday-smo_day+1, usedata.shape[1], usedata.shape[2], usedata.shape[3]), np.nan )
      else:    
         data10d = np.full( (smo_day, nday-smo_day+1, usedata.shape[1], usedata.shape[2]), np.nan )  
        
      for i in range(smo_day):
         data10d[i,:] = usedata[i:(nday-smo_day+i+1),:]
      
      deltad = int(smo_day/2) 
      alldata[var] = np.full( allzm[var].shape, np.nan )  
      alldata[var][:deltad,:] = allzm[var][:deltad,:]
      alldata[var][deltad:nday-deltad,:] = np.nanmean(data10d, axis=0)
      alldata[var][nday-deltad:,:] = allzm[var][nday-deltad:,:]
      
   return alldata   
   
##############

def omps_pres( aer, alt, Tprof, Tpres ):

    invalid = np.where(np.isnan(Tprof))[0]
    valid   = np.where(~np.isnan(Tprof))[0]
    if len( invalid ) > 0 and len( valid ) > 0:
        #print(idd, useMLS[idd,:])
        intpMLS = MLS.intp_lev_ip( Tpres[valid], Tpres, Tprof[valid] )
    elif len( invalid ) == 0:
        intpMLS = Tprof      
    altT = pres2alt( Tpres, intpMLS ) * 1e-3  #km
    func = interpolate.interp1d(alt, aer  )
    intpaer = func( altT )

    return intpaer, altT  

#############

def Gaussian_PDF(x, mean, std):

    normx = (x - mean)/std
    C  = 1 / (std * np.sqrt(2 * np.pi))
    
    return C * np.exp( -1/2 * (normx**2) )


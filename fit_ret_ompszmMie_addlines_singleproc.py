import os, sys, json
import numpy as np
import datetime, pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator

import mylib.MLS_lib as MLS
import Tonga_ana as Ta
from plot_libs import *

#######
def interplut( iobs, lut, allr ):
   delta = lut[1:] - lut[:-1]
   ii = np.where(delta < 0)[0][0]
   jj = np.where(lut > 0)[0][-1]
   if (ii < jj + 1):
     if ii > 0: 
       newlut = lut[(ii-1):(jj+1)]
       newr   = allr[(ii-1):(jj+1)]   
     else:
       newlut = lut[:(jj+1)]
       newr   = allr[:(jj+1)]
   else:
       radii = np.nan    
   if iobs < np.min(newlut):
      radii = newr[-1]
   elif iobs > np.max(newlut):
      radii = newr[0]
   else:
      #print(newlut, iobs)
      i_end   = np.where((newlut - iobs) <= 0)[0][0]
      i_start = np.where((newlut - iobs) > 0)[0][-1]
      frac    = np.abs(iobs - newlut[i_start]) / np.abs(newlut[i_end] - newlut[i_start])
      radii   = newr[i_start] + frac * (newr[i_end] - newr[i_start])
      
   return radii   
   
####

               
def plt_timeseries_fillx( axes, xdata, ydata, y1, y2, **kwargs ):

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
 
   if label is None:
      axes.plot( xdata, ydata, color=color, lw=1.25, alpha=0.9 )
   else:   
      axes.plot( xdata, ydata, color=color, lw=1.25, alpha=0.9, label=label )
   axes.fill_between( xdata, y1, y2, facecolor=color, edgecolor=None, alpha=0.3 )
   if setaxis:
     axes.set_ylim(ylims) 
     axes.set_xlim(xlims)
     axes.set_ylabel(ylabel)
     axes.set_yticks( yticks )
     axes.set_xlabel(xlabel)
   #axes.ticklabel_format(style='sci',axis='x',scilimits=(0,0))  
     if len(title) > 0:
        axes.set_title( title )   
     if xticks is not None:
        axes.xaxis.set_minor_locator(ticker.MultipleLocator(xminor))
        axes.set_xticks( xticks )
     if xticklabel is not None:
        axes.set_xticklabels( xticklabel )
   
   return  axes 


####################
# 2022
#####################

months     = np.append( np.array([11]), np.arange(11) )
cho_lat  = [[-30,0]] #,[-55,-40]]

# retrieval results
# read OMPS-LP
retomps = np.load('data/omps_aerret_v4.2noQC_dzm_DectoNov.npz')
nday    = retomps['nday']
alt     = retomps['alt']
clats   = retomps['clats']
vars    = []
for key in retomps.keys():
   if key not in ['alt','nday','clats']:
      vars.append(key)

# read sage3
retsage = np.load('data/SAGE3_aerret_v4.2noQC_dzm_DectoNov.npz')
Salt    = retsage['alt']

# check ndays
if (retsage['nday'] != retomps['nday']):
   print( 'The number of days of OMPS-LP and SAGE3 for 2022 does not match!')
   print(retsage['nday'],retomps['nday'])
   

alldata = dict()
alldata['omps'] = dict()
alldata['sage'] = dict()
print(vars)

#######
# n days average
smo_day = 1
for var in vars:
   ori_data = retsage[var]
   null_arr = np.full_like( ori_data[-1:,:], np.nan )
   print(null_arr.shape, ori_data.shape)
   usedata = np.concatenate( (ori_data, null_arr), axis=0 )
   ishape = len(usedata.shape)
   if ishape > 3:
      data10d = np.full( (smo_day, int(nday /smo_day), usedata.shape[1], usedata.shape[2], usedata.shape[3]), np.nan )
   else:    
      data10d = np.full( (smo_day, int(nday /smo_day), usedata.shape[1], usedata.shape[2]), np.nan )  
        
   for i in range(smo_day):
      data10d[i,:] = usedata[i:nday-(nday%smo_day):smo_day,:]
   alldata['sage'][var] = np.nanmean(data10d, axis=0)
   
   # average all variance
   alldata['sage'][var] = np.nanmean(alldata['sage'][var], axis=3)

# OMPS-LP
for var in vars:
   ishape = len(retomps[var].shape)
   if ishape > 3:
      data10d = np.full( (smo_day, int(nday /smo_day), retomps[var].shape[1], retomps[var].shape[2], retomps[var].shape[3]), np.nan )
   else:    
      data10d = np.full( (smo_day, int(nday /smo_day), retomps[var].shape[1], retomps[var].shape[2]), np.nan )  
        
   for i in range(smo_day):
      data10d[i,:] = retomps[var][i:nday-(nday%smo_day):smo_day,:]
   alldata['omps'][var] = np.nanmean(data10d, axis=0)
   
   # average all variance
   #alldata['omps'][var] = np.nanmean(alldata['omps'][var], axis=3)
   
   alldata['omps'][var] = alldata['omps'][var][:,:,:,3]
   
nday = int(nday /smo_day)
xmt  = int(5/smo_day)

#############
# 10 days smooth
alldata1 = Ta.smo_timeseries( alldata['omps'], nday, smo_day = 5)

# SAGE MM data
retsage = np.load('data/SAGE3_aerret_v4.2noQC_mzm_DectoNov.npz')
alldata2 = dict()
for var in vars:
  alldata2[var] = retsage[var][:,:,:,3]
  
##################
# read MLS temperature
##################

allzm, glevs, nday, clats = Ta.read_MLS_gas( ['Temperature'], iy = 'v4.2noQC_' )
xday     = np.arange(nday)

##################
# read MLS RH
##################
with open('data/MLS_hygbeta_timeseries_DectoNovb.pickle', 'rb') as handle:
    hygdata = pickle.load(handle)

################
# read MLS wh2so4
##################
wh2so4data = Ta.read_wh2so4( label='v4.2noQC_' )

##################
# 1. gravitational settling velocity

stratidx = np.where( (glevs['Temperature'] >=7) & (glevs['Temperature'] <= 100) )[0]
Pi       = glevs['Temperature'][stratidx]
Rs       = np.append( np.arange(0.01,1,0.005), np.arange(1,10,0.1) )
Rs       = np.append( np.arange(0.001,0.01,0.001), Rs)
colors   = plt.cm.jet( np.linspace(0,1, len(Pi)) )
vg       = np.full( (len(cho_lat),nday,len(Pi),len(Rs)), np.nan )
alt_MLS  = np.full( (len(cho_lat),nday,len(Pi)), np.nan )

fig, axs = plt.subplots(nrows=len(cho_lat), ncols=3, figsize=(12, 4*len(cho_lat)))
for ic in range(len(cho_lat)):
  zidx   = np.where((clats >= cho_lat[ic][0]) & (clats <= cho_lat[ic][1]))[0]
  
  for idd in range(nday):
    Ti = np.nanmean( allzm['Temperature'][idd,zidx,:], axis=0 )
    
    valid = np.where( np.isnan(Ti[stratidx]) == False )[0]
    validTi = Ti[stratidx][valid]
    
    # altitude
    alti = Ta.pres2alt( Pi[valid], validTi )
    alt_MLS[ic,idd,valid] = alti / 1e3 # km
    
    for il in range(len(validTi)):
       vg[ic,idd,valid[il],:] = Ta.vel_g( validTi[il], Pi[valid[il]], Rs ) * 1e-3 * 3600 * 24 # convert from m/s to km/day

  
#sys.exit()  
##############      
# 2. fit Rlim and Reff
 
reff = np.arange(0.002,1.01,0.002)
veff = 0.25 #0.275 # mean value for MIE LUT
Rs2  = np.linspace(0.01, 10.0, 100)

pars = Ta.fit_Rlim_Reff(Rs, reff, veff, plt=False)         

#sys.exit()
###############
# timeseries
ndenmax = [30,10]
deltanden = [5,1]
cho_alt = np.arange(22.5,24,1)[::-1]
par0      = dict()
pare      = dict()
endid     = dict()
for key in ['cond','trans']:
   par0[key] = np.empty( len(cho_alt) )
   pare[key] = np.empty( len(cho_alt) )
par0['nuc'] = [150, 4, 5, 15]
pare['nuc'] = [2, 0.95, 0.9, 0.9]
par0['coag'] = [0.1, 0.08, 0.05, 0.1]
pare['coag'] = [0.6, 0.58, 0.52, 1.1]
endid['cond'] = [120, 80, 90, 90]
endid['coag'] = [90, 75, 75, 90]
Kcoef = [1.5, 1., 1, 1]
Kcoef2 = [1, 0.9, 1, 0.8]
nuccoef = [1, 1., 1, 1.2]
nuccoef2 = [1, 1., 1, 1]
transcoef = [1., 1., 0.5, 1]
transcoef2 = [0.6, 1.5, 3, 0.8]


# factor separation approach (Stein and Alpert, 1993)
sceflag = dict()
fitdata = dict()
procs = ['Condensation','Nucleation','Coagulation','Hygroscopic growth']
procs2 = ['Cond&Coag', 'Hyg&Coag', 'Nuc&Coag', 'Nuc&Cond','Hyg&Cond','Nuc&Hyg']
procs3 = ['Nuc&Cond&Coag','Nuc&Cond&Hyg','Nuc&Coag&Hyg','Hyg&Cond&Coag']
procs4 = ['Nuc&Cond&Coag&Hyg']
allprocs = procs + procs2 + procs3 + procs4 + ['All']
labelname2 = ['Nucleation', 'Condensation', 'Coagulation']
flags = ['do_coag','do_nuc','do_cond','do_RH','do_trans']
vars = ['Nt','Rt','Vt']

for key in allprocs+['OMPS-LP','SAGE III/ISS']:
   sceflag[key] = dict()
   fitdata[key] = dict()
   for key2 in flags:
      sceflag[key][key2] = False

sceflag['Condensation']['do_cond'] = True
sceflag['Nucleation']['do_nuc'] = True
sceflag['Coagulation']['do_coag'] = True
sceflag['Hygroscopic growth']['do_RH'] = True

for key in procs2 + procs3 + procs4 + ['All']:
   if ('Nuc' in key) or (key == 'All'):
      sceflag[key]['do_nuc'] = True
   if ('Cond' in key) or (key == 'All'):
      sceflag[key]['do_cond'] = True   
   if ('Hyg' in key) or (key == 'All'):
      sceflag[key]['do_RH'] = True 
   if ('Coag' in key) or (key == 'All'):
      sceflag[key]['do_coag'] = True    
   if ('Trans' in key) or (key == 'All'):
      print(key)
      sceflag[key]['do_trans'] = True            

for ic in range(len(cho_lat)):

   zidx   = np.where((clats >= cho_lat[ic][0]) & (clats <= cho_lat[ic][1]))[0]
   wh2so4_2d, wh2so4_P, T_P, T_2d = Ta.intp_daywh2so4(zidx, wh2so4data)

   # aerosol in each layer
   fig, axes = plt.subplots(nrows=len(cho_alt), ncols=2, figsize=(13,2.8*len(cho_alt)))
    
   Kt   = np.full((nday,len(cho_alt)), np.nan)
   for key in allprocs+['OMPS-LP','SAGE III/ISS']:
      for var in vars:
         fitdata[key][var] = np.full((nday,len(cho_alt)), np.nan)
         
   J_Jc = np.full((nday,len(cho_alt)), np.nan)
   Kn   = np.full((nday,len(cho_alt)), np.nan)
   usepp= np.full((nday,len(cho_alt)), np.nan)
   usend = 150
   obsdvdt = np.full((nday,len(cho_alt)), 0.0)
   obsdRdt = np.full((nday,len(cho_alt)), 0.0)
   obsdNdt = np.full((usend,len(cho_alt)), 0.0)
   Vmaxid  = []
   Rmaxid  = []
   
      
   for il in range(len(cho_alt)):
      lidx   = np.argmin(np.abs(alt - cho_alt[il]))
      Slidx  = np.argmin(np.abs(Salt - alt[lidx]))
      
      # 1. Obs
      # a. numden
      mean1   = np.nanmean(alldata1['numden'][:,zidx,lidx],axis=1) * 1e12
      std1    = np.nanstd(alldata1['numden'][:,zidx,lidx],axis=1) * 1e12
      didx    = np.where( np.isnan(mean1) )[0]
      print('nvalid day:', mean1[didx], mean1[didx+1], mean1[didx-1])
      mean1[mean1>100] = np.nan
      mean1[didx-1] = np.nan
      mean1[didx+1] = np.nan
      dn_dt   = mean1[1:] - mean1[:-1]
      maxid   = np.nanargmax(mean1)
      print('max:', ic, maxid, np.nanmax(mean1), mean1[maxid-5:maxid+20], xmt)
      fitdata['OMPS-LP']['Nt'][:,il] = mean1
      #if il == 2:
      #  sys.exit()

      axes[il,0] = Ta.plt_timeseries_fillx( axes[il,0], xday, mean1, color='k', ls='--', label='Obs',
                 xticks=xday[::xmt*6], xlims=[xday[0],xday[-1]], xticklabel=[datetime.datetime(2022,i+1,1).strftime('%b') for i in months], 
                 ylims=[-ndenmax[ic],ndenmax[ic]], yticks=np.arange(-ndenmax[ic],ndenmax[ic]+deltanden[ic],deltanden[ic]), 
                 ylabel=r'Number density (cm$^{-3}$)',xlabel='Days after 1 Dec 2021', xminor=xmt, 
                 title=r'OMPS-LP particles evolution at {:.0f} km in latitudes {:d}$^\circ$-{:d}$^\circ$'.format(
                 cho_alt[il], cho_lat[ic][0], cho_lat[ic][1]) )
      Smean1 = np.nanmean(alldata2['numden'][:,zidx,Slidx],axis=1) * 1e12
      axes[il,0].scatter( xday[15:nday:30], Smean1, s=20, marker='>', edgecolors='r', c='r', label='SAGE III/ISS')  
      fitdata['SAGE III/ISS']['Nt'][:len(Smean1),il] = Smean1                     
      for ix in [20, 43, 45]:
         axes[il,0].axvline( x=ix, color='k', linewidth=0.8, linestyle='--')                 

      # b. radii
      mean2   = np.nanmean(alldata1['radii'][:,zidx,lidx],axis=1)
      std2    = np.nanstd(alldata1['radii'][:,zidx,lidx],axis=1)
      didx    = np.where( np.isnan(mean2) )[0]
      print('nvalid day:', didx)
      mean2[didx-1] = np.nan
      mean2[didx+1] = np.nan
      mean2[160:165] = np.nan
      std2[160:165]  = np.nan
      #obsdRdt[:-1,il] = mean2[1:] - mean2[:-1]
      obsdRdt[:,il] = mean2 - np.nanmax(mean2)
      Rmaxid.append( np.nanargmax(mean2) )
      fitdata['OMPS-LP']['Rt'][:,il] = mean2
      
      axes[il,1] = Ta.plt_timeseries_fillx( axes[il,1], xday, mean2, color='k', ls='--', label='Obs',
                 xticks=xday[::xmt*6], xlims=[xday[0],xday[-1]], xticklabel=[datetime.datetime(2022,i+1,1).strftime('%b') for i in months], 
                 ylims=[0,0.8], yticks=np.arange(0.,0.81,0.1), 
                 ylabel=r'Number density (cm$^{-3}$)',xlabel='Days after 1 Dec 2021', xminor=xmt, 
                 title=r'OMPS-LP particles evolution at {:.0f} km in latitudes {:d}$^\circ$-{:d}$^\circ$'.format(
                 cho_alt[il], cho_lat[ic][0], cho_lat[ic][1]) )
      Smean2 = np.nanmean(alldata2['radii'][:,zidx,Slidx],axis=1)
      fitdata['SAGE III/ISS']['Rt'][:len(Smean2),il] = Smean2    
      

                             
      for ix in [20, 43, 45]:
         axes[il,1].axvline( x=ix, color='k', linewidth=0.8, linestyle='--')  

      # c. cvol
      mean3   = np.nanmean(alldata1['cvol'][:,zidx,lidx],axis=1) 
      std3    = np.nanstd(alldata1['cvol'][:,zidx,lidx],axis=1)
      mean3[160:165] = np.nan
      std3[160:165]  = np.nan
      print('max cvol:', np.nanmax(mean3))
      Vmaxid.append( np.nanargmax( mean3 ) )
      fitdata['OMPS-LP']['Vt'][:,il] = mean3
            
      # 2. single process fit
      # 2a. exp dacay fit condensation
      obsdvdt[44:-1,il] = mean3[45:] - mean3[44:-1]
      flag = (obsdvdt[45:usend,il] < 1e-5)
      obsdvdt[45:usend,il][flag] = np.nan
      par, valid, idx, y0 = MLS.fit_decay(xday[:usend], obsdvdt[:usend,il], endi=endid['cond'][il]) 
      par0['cond'][il] = y0
      pare['cond'][il] = par[0]
      if il > 1:
         pare['cond'][il] = par[0] * 1.1
      
      # 2b. exp dacay fit coagulation K
      # N(t) = N0 / (1 + t * KN0/2)
      obsdNdt[maxid:usend,il] = ((mean1[maxid] / mean1[maxid:usend]) - 1 ) * 2 / mean1[maxid] / (xday[maxid:usend] - xday[maxid])
      if il == 3:
         par, valid, coagidx, y0 = MLS.fit_decay(xday[61:usend], obsdNdt[61:usend,il], endi=endid['coag'][il], maxt0=True) 
      else:   
         par, valid, coagidx, y0 = MLS.fit_decay(xday[:usend], obsdNdt[:usend,il], endi=endid['coag'][il]) 
      par0['coag'][il] = y0
      pare['coag'][il] = par[0]      
      
      # 2c. exp decay fit nucleation rate
      fitK = par0['coag'][il]*np.exp(-pare['coag'][il]*(xday[45:]-coagidx))
      nuc_dndt = dn_dt[45:] + fitK[:-1] / 2 * (mean1[45:-1] ** 2) 
      nuc_dndt[nuc_dndt < 0.1] = np.nan
      nuc_dndt[8:18] = np.nan
      par, valid, nucidx, y0 = MLS.fit_decay(xday[45:-1], nuc_dndt, endi=40, starti=0) 
      par0['nuc'][il] = y0 * nuccoef[il]
      pare['nuc'][il] = par[0] * nuccoef2[il]
      
      par0['coag'][il] = par0['coag'][il]*Kcoef[il]
      pare['coag'][il] = pare['coag'][il]*Kcoef2[il]

      # 2d. exp decay fit transport
      usefitdv = -obsdvdt[Vmaxid[il]:,il]
      usefitdv[usefitdv < 0.03e-4] = np.nan
      par, valid, transidx, y0 = MLS.fit_decay(xday[Vmaxid[il]:], usefitdv, endi=330-Vmaxid[il], maxt0=False) 
      if il == 0:
         transidx = transidx - 12
      par0['trans'][il] = y0 * transcoef[il]
      pare['trans'][il] = par[0] * transcoef2[il] 

      # fitted coag K
      #Rrat = (mean2[maxid:]/mean2[maxid]) ** 3
      #Kt[maxid:,il] = (Rrat-1) * 2 / mean1[maxid]/(xday[maxid:]-maxid)
      
      # 3. all processes fit
      
      for ik, sce_key in enumerate( allprocs ):
      
        Rt   = np.full(nday, np.nan)
        Nt   = np.full(nday, np.nan)
        Vt   = np.full(nday, np.nan)
        deltapp = np.full(nday, np.nan)
      
        Rt[:45] = mean2[44]
        Nt[:45] = mean1[44]  #np.nanmax(mean1)*2
        Vt[:45] = mean3[44]
        deltapp[44] = 0.0
#       Rt[0] = mean2[4]
#       Nt[0] = np.nanmax(mean1)*2
        usey = hygdata['beta']['mstd'][:50,lidx]
        valid = ~np.isnan(usey)
        if len(np.where(valid==False)[0]) > 0:
          intpy = np.interp( xday[:50][~valid], xday[:50][valid], usey[valid] )
          hygdata['beta']['mstd'][np.where(valid==False)[0],lidx] = intpy
          print(intpy)
        
      
      
        for idd in range(45,nday):
        
           print(f'- Start day {idd} {alt[lidx]}')
           # temperature and pressure
           Ti = np.nanmean( allzm['Temperature'][idd,zidx,:], axis=0 )
           valid = np.where( np.isnan(Ti[stratidx]) == False )[0]
           validTi = Ti[stratidx][valid]
           validPi = Pi[valid]
           Tii = np.interp( alt[lidx], alt_MLS[ic,idd,valid], validTi )
           Pii = np.exp( np.interp( alt[lidx], alt_MLS[ic,idd,valid], np.log(validPi) ) ) 
        #print('check pressure:',idd, Pii, alt[lidx])

           # wh2so4
           intpiwh2so4 = Ta.intp_altwh2so4(cho_alt[::-1], wh2so4_2d[idd,:], T_2d[idd,:], wh2so4_P, T_P)
           print(f'intpiwh2so4:{intpiwh2so4}')
                
           # rho
           irho = Ta.rho_h2so4( Tii, intpiwh2so4[::-1][il]*0.01 )
           print(f'irho:{irho}')
        #sys.exit()
                      
           # RH
           beta_RH = hygdata['beta']['mstd'][idd,lidx]
        
           if np.isnan(beta_RH):
              beta_RH = hygdata['beta']['mstd'][idd-1,lidx]
              hygdata['beta']['mstd'][idd,lidx] = beta_RH
           #print( 'after', beta_RH, hygdata['beta']['max'][(idd-1):(idd+1),lidx] )
           
           beta_RH2 = [hygdata['beta']['mstd'][idd-1,lidx],beta_RH]
        #print(idd, beta_RH2)
           if len(np.where(np.isnan(beta_RH2))[0] ) > 0:
              sys.exit()      
              
           if sceflag[sce_key]['do_cond']:
              Vb = par0['cond'][il]*np.exp(-pare['cond'][il]*(idd-44))
           else:
              Vb = 0 
              
           if sceflag[sce_key]['do_RH']:
              usebeta = beta_RH2
           else:
              usebeta = [1., 1.]  
              
           if sceflag[sce_key]['do_trans']:   
             if idd < transidx:
               Vc = 0.
             else:
               Vc = par0['trans'][il]*np.exp(-pare['trans'][il]*(idd-transidx) )    
           else:
             Vc = 0.            
           
           # all fit   
           Rt[idd], Nt[idd], Vt[idd], Kt[idd,il], deltapp[idd] = Ta.allprocess_t2( Rt[idd-1], Nt[idd-1], Vt[idd-1], Tii, Pii, irho,
                                    do_coag=sceflag[sce_key]['do_coag'], do_nuc=sceflag[sce_key]['do_nuc'],
                                    Kt=par0['coag'][il]*np.exp(-pare['coag'][il]*(idd-coagidx)),
                                    a=par0['nuc'][il]*np.exp(-pare['nuc'][il]*(idd-nucidx)),
                                    #Va=np.nanmax(mean3)*a0*np.exp(-a0*(idd-44)),
                                    #amass=amass*nuc_frac,
                                    frac=usebeta, 
                                    #b=par0['cond'][il]*np.exp(-pare['cond'][il]*(idd-44)), #1e-3*(idd-44)**(-0.3), #1e-4*(np.log(idd-44)/np.log(20))
                                    Vb=Vb,
                                    #do_phycond=True, Rs=Rs2, pp=usepp[idd,il], veff=veff, 
                                    Vc=Vc,
                                    #c=Nt[44]*0.5*np.exp((idd-44)*(-0.08)),
                                    #t=idd-44, vg=vgl, Rs=Rs, veff=veff, pars=pars, 
                                    )

        fitdata[sce_key]['Nt'][:,il] = Nt
        fitdata[sce_key]['Rt'][:,il] = Rt
        fitdata[sce_key]['Vt'][:,il] = Vt

      
        # plot
        colors = plt.cm.jet( np.linspace(0,1,len(allprocs)) )

        if sce_key in procs or sce_key == 'Nuc&Coag':
           axes[il,0] = Ta.plt_timeseries_fillx( axes[il,0], xday, fitdata[sce_key]['Nt'][:,il], color=colors[ik],label=sce_key,
                 xticks=xday[::xmt*6], xlims=[xday[0],xday[-1]], xticklabel=[datetime.datetime(2022,i+1,1).strftime('%b') for i in months], 
                 ylims=[0,ndenmax[ic]], yticks=np.arange(0,ndenmax[ic]+deltanden[ic],deltanden[ic]), 
                 ylabel=r'Number density (cm$^{-3}$)',xlabel='Days after 1 Dec 2021', xminor=xmt,
                 title=r'OMPS-LP particles evolution at {:.1f} km in latitudes {:d}$^\circ$-{:d}$^\circ$'.format(
                 cho_alt[il], cho_lat[ic][0], cho_lat[ic][1]) )
           axes[il,0].legend(ncol=2,fontsize='x-small')           
 
        if sce_key in allprocs:  #procs2 + procs3 + procs4 + ['All']: 
           axes[il,1] = Ta.plt_timeseries_fillx( axes[il,1], xday, fitdata[sce_key]['Rt'][:,il], color=colors[ik],label=sce_key,
                 xticks=xday[::xmt*6], xlims=[xday[0],xday[-1]], xticklabel=[datetime.datetime(2022,i+1,1).strftime('%b') for i in months], 
                 ylims=[0.,0.8], yticks=np.arange(0.,0.81,0.2), ylabel=r'R$_{eff}$ ($\mu$m)', 
                 xlabel='Days after 1 Dec 2021', xminor=xmt,
                 title=' ' )
                 #r'OMPS-LP particles evolution at {:.1f} km in latitudes {:d}$^\circ$-{:d}$^\circ$'.format(
                 #cho_alt[il], cho_lat[ic][0], cho_lat[ic][1]) )
           if il > 0:      
              axes[il,1].legend(ncol=5, fontsize='x-small',loc='upper center', bbox_to_anchor=(0.5, 1.5),facecolor='w',frameon=0)      
   
#          
   plt.tight_layout()
   plt.savefig( 'img/omps&SAGE3_retaer_singleproc_timeseries_intpwh2so4_DectoNovb_addlines_reg'+str(ic+1)+'.png', dpi=300 )
   plt.close()  
   
   
   # contribution
   fig, axes = plt.subplots(nrows=len(cho_alt), ncols=2, figsize=(15,2.5*len(cho_alt)))
   colors = ['r','g','b','c']
   
   for il in range(len(cho_alt)):
    Ncontri_all = np.full( nday, 0.0 )
    Rcontri_all = np.full( nday, 0.0 )
    for ik, key in enumerate(procs):
      iRcontri = np.abs( fitdata[key]['Rt'][:,il] - fitdata[key]['Rt'][44,il] )
      if key == 'Coagulation':
         iNcontri = np.abs( fitdata['Nuc&Coag']['Nt'][:,il] - fitdata['Nucleation']['Nt'][:,il] )
      else:
         iNcontri = np.abs( fitdata[key]['Nt'][:,il] - fitdata[key]['Nt'][44,il] )
      Ncontri_all = Ncontri_all + iNcontri
      Rcontri_all = Rcontri_all + iRcontri
    
    for ik, key in enumerate(procs):
      if key == 'Coagulation':
         iNcontri = np.abs( fitdata['Nuc&Coag']['Nt'][:,il] - fitdata['Nucleation']['Nt'][:,il] )
      else:   
         iNcontri = np.abs( fitdata[key]['Nt'][:,il] - fitdata[key]['Nt'][44,il] ) 
      #mincontri = np.nanmin( iNcontri/Ncontri_all*100 )
      #maxcontri = np.nanmax( iNcontri/Ncontri_all*100 )
      mincontri = np.nanmean( iNcontri[44:55]/Ncontri_all[44:55]*100 )
      maxcontri = np.nanmean( iNcontri[55:]/Ncontri_all[55:]*100 )
      axes[il,0] = Ta.plt_timeseries_fillx( axes[il,0], xday, iNcontri/Ncontri_all*100, color=colors[ik],
                 label=procs[ik]+':{:.1f}%-{:.1f}%'.format(mincontri, maxcontri),
                 xticks=xday[::xmt*6], xlims=[xday[0],xday[-1]], xticklabel=[datetime.datetime(2022,i+1,1).strftime('%b') for i in months], 
                 ylims=[0,100], yticks=np.arange(0,101,20), 
                 ylabel='N contribution (%)',xlabel='Days after 1 Dec 2021', xminor=xmt,
                 title=r'OMPS-LP particles evolution at {:.1f} km in latitudes {:d}$^\circ$-{:d}$^\circ$'.format(
                 cho_alt[il], cho_lat[ic][0], cho_lat[ic][1]) )
      axes[il,0].legend()           
      
      iRcontri = np.abs( fitdata[key]['Rt'][:,il] - fitdata[key]['Rt'][44,il] )
      mincontri = np.nanmin( iRcontri/Rcontri_all*100 ) 
      maxcontri = np.nanmax( iRcontri/Rcontri_all*100 ) 
      axes[il,1] = Ta.plt_timeseries_fillx( axes[il,1], xday, iRcontri/Rcontri_all*100, color=colors[ik],
                 label=procs[ik]+':{:.1f}%-{:.1f}%'.format(mincontri, maxcontri),
                 xticks=xday[::xmt*6], xlims=[xday[0],xday[-1]], xticklabel=[datetime.datetime(2022,i+1,1).strftime('%b') for i in months], 
                 ylims=[0,100], yticks=np.arange(0,101,20), ylabel='R$_{eff}$ contribution (%)', 
                 xlabel='Days after 1 Dec 2021', xminor=xmt,
                 title=r'OMPS-LP particles evolution at {:.1f} km in latitudes {:d}$^\circ$-{:d}$^\circ$'.format(
                 cho_alt[il], cho_lat[ic][0], cho_lat[ic][1]) )
      axes[il,1].legend(ncol=len(procs))      
   
#          
   plt.tight_layout()
   plt.savefig( 'img/omps&SAGE3_retaer_proccontri_timeseries_intpwh2so4_DectoNovb_addlines_reg'+str(ic+1)+'.png', dpi=300 )
   plt.close()  
     

   #########################     
   # plot partition
   useprocs = ['Nucleation','Hygroscopic growth','Hyg&Coag','Hyg&Cond&Coag']
   procname = ['Nucleation','Hygroscopic growth','Coagulation','Condensation','Transport']
   #colors = plt.cm.rainbow( np.linspace(0, 1, len(procname)))
   colors = ['r','g','b','c','m']
   cho_tidx = [110, 290]

   fig, axes = plt.subplots(nrows=len(cho_alt), ncols=1, figsize=(7,3*len(cho_alt)))
   for il in range(len(cho_alt)):
   
     allcontri = np.full( nday, 0.0 )
     contriline = dict()
     for ik in range(len(procname)):
      if ik < 2:
         baseline = np.full( nday, fitdata[useprocs[ik]]['Rt'][44,il])
      else:
         baseline = fitdata[useprocs[ik-1]]['Rt'][:,il]   
         
      if procname[ik] == 'Transport':
          contriline[procname[ik]] = np.abs( fitdata['Nuc&Cond&Coag&Hyg']['Rt'][:,il] - fitdata['All']['Rt'][:,il] ) 
      else:     
          contriline[procname[ik]] = np.abs( fitdata[useprocs[ik]]['Rt'][:,il] - baseline )   
      allcontri = allcontri + contriline[procname[ik]]
     
     for ik in range(len(procname)):
      if ik < 2:
         baseline = np.full( nday, fitdata[useprocs[ik]]['Rt'][44,il] )
         upperline = fitdata[useprocs[ik]]['Rt'][:,il]
      else:
         if procname[ik] == 'Transport':  # transport
            baseline = fitdata['Nucleation']['Rt'][:,il]  
            upperline = baseline - contriline[procname[ik]]
         else:
            baseline = fitdata[useprocs[ik-1]]['Rt'][:,il]
            upperline = fitdata[useprocs[ik]]['Rt'][:,il]      
         
      perc = contriline[procname[ik]] / allcontri * 100
           
      if ik == 0:   
         axes[il] = Ta.plt_timeseries_fillx( axes[il], xday, baseline, y1=upperline, color=colors[ik], label=procname[ik],
                 xticks=xday[::xmt*6], xlims=[xday[0],xday[-1]], xticklabel=[datetime.datetime(2022,i+1,1).strftime('%b') for i in months], 
                 ylims=[-0.1,0.8], yticks=np.arange(0,0.81,0.2), ylabel=r'R$_{eff}$ ($\mu$m)', 
                 xlabel='Days after 1 Dec 2021', xminor=xmt, pltline=False,
                 title=r'OMPS-LP particles evolution at {:.1f} km in latitudes {:d}$^\circ$-{:d}$^\circ$'.format(
                 cho_alt[il], cho_lat[ic][0], cho_lat[ic][1]) )
      else:   
         axes[il].fill_between( xday, baseline, upperline, facecolor=colors[ik], edgecolor=None, alpha=0.5,
                                label=procname[ik] ) 
                                
      # add arrows                          
      for it in range(len(cho_tidx)):
        tid = cho_tidx[it]
        perctext = '{:.1f}%'.format(perc[tid]) 
        imin = min([baseline[tid], upperline[tid]])
        imax = max([baseline[tid], upperline[tid]]) 
        if procname[ik] == 'Hygroscopic growth':
            xpos = xday[tid+30]
        else:
            xpos = xday[tid] 
        if perc[tid] > 0.1:       
            axes[il].annotate( perctext, xy=(xpos, (imin+imax)/1.7), xytext=(xpos, (imin+imax)/2.1, ) )
      #axes[il].plot( xday, fitdata[key]['Rt'][:,il], color='k', lw=0.5 )         
     
     axes[il].axhline( y=fitdata['OMPS-LP']['Rt'][44,il], color='k', lw=1.5, ls=':', label='Background' ) 
     axes[il].plot( xday, fitdata['OMPS-LP']['Rt'][:,il], color='k', lw=1.5, label='OMPS-LP' ) 
     axes[il].scatter( xday[15:nday:30], fitdata['SAGE III/ISS']['Rt'][:len(Smean2),il], s=20, marker='>', edgecolors='k', c='k', label='SAGE III/ISS')  
     axes[il].plot( xday, fitdata['All']['Rt'][:,il], color='k', lw=1.5, ls='--', label='All fitted' ) 
     axes[il].legend(ncol=3, fontsize='x-small', loc='lower left', )  #bbox_to_anchor=(0.5, 0.05)
      
   plt.tight_layout()
   plt.savefig( 'img/omps&SAGE3_retaer_particontri_timeseries_intpwh2so4_DectoNovb_addlines_reg'+str(ic+1)+'.png', dpi=300 )
   plt.close()  
           
           
      
      

 
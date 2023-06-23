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
#alldata2 = Ta.smo_timeseries( alldata['sage'], nday, smo_day = 5)

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
    
    if ic == 0:
     for ilso2 in range(len(SO2lev)):
       llidx = np.argmin( np.abs(Pi[valid]-SO2lev[ilso2]) )
       if np.abs(Pi[valid][llidx]-SO2lev[ilso2]) < SO2lev[ilso2]*0.01:
          SO2alt[idd,ilso2] = alt_MLS[ic,idd,valid][llidx]
    
    for il in range(len(validTi)):
       vg[ic,idd,valid[il],:] = Ta.vel_g( validTi[il], Pi[valid[il]], Rs ) * 1e-3 * 3600 * 24 # convert from m/s to km/day

print('check SO2 alt:', SO2alt[45,:], SO2lev)   
#sys.exit()  
##############      

reff = np.arange(0.002,1.01,0.002)
veff = 0.25 #0.275 # mean value for MIE LUT
Rs2  = np.linspace(0.01, 10.0, 100)
       

#sys.exit()
###############
# timeseries
ndenmax = [20,10]
deltanden = [3,1]
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


for ic in range(len(cho_lat)):

   zidx   = np.where((clats >= cho_lat[ic][0]) & (clats <= cho_lat[ic][1]))[0]
   wh2so4_2d, wh2so4_P, T_P, T_2d = Ta.intp_daywh2so4(zidx, wh2so4data)

   # aerosol mass [20-24 km]
   fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8,2.5))
   cvoltime = np.nanmean(alldata1['cvol'][:,zidx,:],axis=1) 
   lii      = np.where((alt >= 20) & (alt < 24))[0]
   allcvol  = np.nansum( cvoltime[:,lii], axis=1 )  * 1.83 * 1e-12 # g/um2
   allcvol[allcvol < 1e-16] = np.nan
   allcvol[160:165] = np.nan
   axs = Ta.plt_timeseries_fillx( axs, xday, allcvol, color='r', 
                 xticks=xday[::xmt*6], xlims=[xday[0],xday[-1]], xticklabel=[datetime.datetime(2022,i+1,1).strftime('%b') for i in months], 
                 ylims=[0,np.nanmax(allcvol)], #yticks=np.arange(-ndenmax[ic],ndenmax[ic]+deltanden[ic],deltanden[ic]), 
                 ylabel=r'sulfate mass (g$\mu$m$^{-2}$)',xlabel='Days after 1 Dec 2021', xminor=xmt, 
                 title=r'OMPS-LP sulfate particles mass at 20-24 km in latitudes {:d}$^\circ$-{:d}$^\circ$'.format(
                 cho_lat[ic][0], cho_lat[ic][1]) )
                             
   for ix in [20, 43, 45]:
       axs.axvline( x=ix, color='k', linewidth=0.8, linestyle='--')     
   plt.tight_layout()
   plt.savefig( 'img/omps&SAGE3_ret_allaermass_timeseries_intpwh2so4_DectoNovb_reg'+str(ic+1)+'.png', dpi=300 )
   plt.close()  

   
   # aerosol in each layer
   fig, axs = plt.subplots(nrows=len(cho_alt), ncols=1, figsize=(8,2.5*len(cho_alt)))
   axes   = axs.flatten()
    
   Kt   = np.full((nday,len(cho_alt)), np.nan)
   saveRt = np.full((nday,len(cho_alt)), np.nan)
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
      #if il == 2:
      #  sys.exit()
      
      axes[il] = Ta.plt_timeseries_fillx( axes[il], xday, mean1, color='r', label='OMPS-LP N',
                 xticks=xday[::xmt*6], xlims=[xday[0],xday[-1]], xticklabel=[datetime.datetime(2022,i+1,1).strftime('%b') for i in months], 
                 ylims=[-ndenmax[ic],ndenmax[ic]], yticks=np.arange(-ndenmax[ic],ndenmax[ic]+deltanden[ic],deltanden[ic]), 
                 ylabel=r'Number density (cm$^{-3}$)',xlabel='Days after 1 Dec 2021', xminor=xmt, 
                 title=r'OMPS-LP particles evolution at {:.0f} km in latitudes {:d}$^\circ$-{:d}$^\circ$'.format(
                 cho_alt[il], cho_lat[ic][0], cho_lat[ic][1]) )
      
      #Smean1 = np.nanmean(alldata2['numden'][:,zidx,Slidx],axis=1) * 1e12
      #axes[il].scatter( xday[15:nday:30], Smean1, s=20, marker='>', edgecolors='r', c='r', label='SAGE III/ISS N')                       
      for ix in [20, 43, 45]:
         axes[il].axvline( x=ix, color='k', linewidth=0.8, linestyle='--')                 
      axes[il].yaxis.label.set_color('r') 
      axes[il].tick_params(axis='y', colors='r')  
      axes[il].spines['left'].set_color('r')   


      # b. radii
      #ax2     = axes[il].twinx()
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
      
#       ax2 = Ta.plt_timeseries_fillx( ax2, xday, mean2, color='b', xminor=xmt, xlims=[xday[0],xday[-1]],
#                  xticks=xday[::xmt*6], ylims=[0.,0.55], yticks=np.arange(0.,0.51,0.1), ylabel=r'R$_{eff}$ ($\mu$m)',
#                  xticklabel=[datetime.datetime(2022,i+1,1).strftime('%b') for i in months] )
# 
#       Smean2 = np.nanmean(alldata2['radii'][:,zidx,Slidx],axis=1)
#       ax2.scatter( xday[15:nday:30], Smean2, s=20, marker='>', edgecolors='b', c='b')  

      # c. cvol
      ax3     = axes[il].twinx()
      mean3   = np.nanmean(alldata1['cvol'][:,zidx,lidx],axis=1) 
      std3    = np.nanstd(alldata1['cvol'][:,zidx,lidx],axis=1)
      mean3[160:165] = np.nan
      std3[160:165]  = np.nan
      print('max cvol:', np.nanmax(mean3))
      Vmaxid.append( np.nanargmax( mean3 ) )
      
      ax3 = Ta.plt_timeseries_fillx( ax3, xday, mean3*1e4, color='g',  xminor=xmt, xlims=[xday[0],xday[-1]],
                 xticks=xday[::xmt*6], ylims=[0.,10], yticks=np.arange(0.,11,2), ylabel=r'V (10$^{-4}$$\mu$m$^3$$\mu$m$^{-2}$)',
                 label='OMPS-LP V',)
      Smean3 = np.nanmean(alldata2['cvol'][:,zidx,Slidx],axis=1)
      ax3.scatter( xday[15:nday:30], Smean3*1e4, s=20, marker='>', edgecolors='g', c='g', label='SAGE III/ISS V')    
      
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
         
      #print(cho_alt[il], par0['cond'][il], pare['cond'][il], idx, mean3[44])
      #sys.exit()   
      
      # 2b. exp dacay fit coagulation K
#       obsdNdt[44:,il] = (mean1[44:usend] - mean1[45:usend+1]) / ( mean1[44:usend] ** 2 )
#       flag = (obsdNdt[:,il] < 0.0)
#       obsdNdt[:,il][flag] = np.nan
      # N(t) = N0 / (1 + t * KN0/2)
      obsdNdt[maxid:usend,il] = ((mean1[maxid] / mean1[maxid:usend]) - 1 ) * 2 / mean1[maxid] / (xday[maxid:usend] - xday[maxid])
      if il == 3:
         par, valid, coagidx, y0 = MLS.fit_decay(xday[61:usend], obsdNdt[61:usend,il], endi=endid['coag'][il], maxt0=True) 
      else:   
         par, valid, coagidx, y0 = MLS.fit_decay(xday[:usend], obsdNdt[:usend,il], endi=endid['coag'][il]) 
      par0['coag'][il] = y0
      pare['coag'][il] = par[0]      
      
      # 2c. exp decay fit nucleation rate
      #Ncoag[45:,il] = Ta.coag_Nt( par0['coag'][il]*np.exp(-pare['coag'][il]*(xday[45:]-coagidx)), mean1[maxid], xday[45:]-xday[45] )
      #print(Ncoag[:,il])
      #sys.exit()
      #Nnuc[:,il]  = mean1 + Ncoag[:,il]
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
      
      #print(cho_alt[il], par0['trans'][il], pare['trans'][il], transidx)
      

      # fitted coag K
      #Rrat = (mean2[maxid:]/mean2[maxid]) ** 3
      #Kt[maxid:,il] = (Rrat-1) * 2 / mean1[maxid]/(xday[maxid:]-maxid)
      
      # 3. all processes fit
      Rt   = np.full(nday, np.nan)
      Nt   = np.full(nday, np.nan)
      Vt   = np.full(nday, np.nan)
      deltapp = np.full(nday, np.nan)
      
      Rt[44] = mean2[44]
      Nt[44] = mean1[44]  #np.nanmax(mean1)*2
      Vt[44] = mean3[44]
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
        
        # estimate Kn
        J_Jc[idd,il], Kn[idd,il] = Ta.fKn_alpha(alpha=1, T=Tii, P=Pii, R=0.25)  
      
        # vg
        vgl = []
        valid = np.where(np.isnan(alt_MLS[ic,idd,:]) == False)[0]
        for irr in range(len(Rs)):
           vgl.append( np.interp( alt[lidx], alt_MLS[ic,idd,valid], vg[ic,idd,valid,irr]) )
        
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
           
        # nucleation rate from SO2 decay
        llidx = np.argmin( np.abs( SO2alt[idd,:] - alt[lidx]) ) 
        if idd < SO2tidx[llidx]:
           deltapp[idd] = deltapp[idd-1]
           Rt[idd], Nt[idd], Vt[idd] = Rt[idd-1], Nt[idd-1], Vt[idd-1]
           continue


        nvalid = len( np.where( ~np.isnan(dndt[idd,:]) )[0] )
        if nvalid > 1:
             idndt = np.interp( alt[lidx], SO2alt[idd,:], dndt[idd,:] ) 
        elif nvalid == 1:
             idndt = dndt[idd,llidx] 
        else:
             idndt = 0
             print('All nan data for day ', idd, SO2tidx[llidx])  

           
        # H2SO4 partial pressure interpolation
        ipp = np.exp( np.interp( alt[lidx], SO2alt[idd,:], np.log(pp[idd,:]) )  )  
        usepp[idd,il] = ipp - deltapp[idd-1]
        print( f' - ipp:{ipp}, usepp:{usepp[idd,il]}' )

        if idd < transidx:
           Vc = 0
           Rc = 0
        else:
           Vc = par0['trans'][il]*np.exp(-pare['trans'][il]*(idd-transidx) )  
           #Vc = par0['trans'][il]*0.25*((idd-transidx)**(pare['trans'][il]) )  
             
           
        # all fit   
        Rt[idd], Nt[idd], Vt[idd], Kt[idd,il], deltapp[idd] = Ta.allprocess_t2( Rt[idd-1], Nt[idd-1], Vt[idd-1], Tii, Pii, irho,
                                    do_coag=True, do_nuc=True,
                                    Kt=par0['coag'][il]*np.exp(-pare['coag'][il]*(idd-coagidx)),
                                    a=par0['nuc'][il]*np.exp(-pare['nuc'][il]*(idd-nucidx)),
                                    #Va=np.nanmax(mean3)*a0*np.exp(-a0*(idd-44)),
                                    #amass=amass*nuc_frac,
                                    frac=beta_RH2, 
                                    #b=par0['cond'][il]*np.exp(-pare['cond'][il]*(idd-44)), #1e-3*(idd-44)**(-0.3), #1e-4*(np.log(idd-44)/np.log(20))
                                    Vb=par0['cond'][il]*np.exp(-pare['cond'][il]*(idd-44)),
                                    #do_phycond=True, Rs=Rs2, pp=usepp[idd,il], veff=veff, 
                                    Vc=Vc,
                                    #c=Nt[44]*0.5*np.exp((idd-44)*(-0.08)),
                                    #t=idd-44, vg=vgl, Rs=Rs, veff=veff, pars=pars, 
                                    )


      saveRt[:,il] = Rt
      if il == 0:
         print(cho_alt[il], Vt[transidx-2:transidx+3], Rt[transidx-2:transidx+3])
         #sys.exit()
         
      # plot
      axes[il] = Ta.plt_timeseries_fillx( axes[il], xday, Nt, color='r',ls='--', label='fitted N',
                 xticks=xday[::xmt*6], xlims=[xday[0],xday[-1]], xticklabel=[datetime.datetime(2022,i+1,1).strftime('%b') for i in months], 
                 ylims=[0,ndenmax[ic]], yticks=np.arange(0,ndenmax[ic]+1,deltanden[ic]), 
                 ylabel=r'N (cm$^{-3}$)',xlabel='Days after 1 Dec 2021', xminor=xmt, 
                 title=r'Particles evolution at {:.1f} km in latitudes {:d}$^\circ$-{:d}$^\circ$'.format(
                 cho_alt[il], cho_lat[ic][0], cho_lat[ic][1]) )
                 
                     

#       ax2 = Ta.plt_timeseries_fillx( ax2, xday, Rt, color='b', ls='--',xminor=xmt, xlims=[xday[0],xday[-1]], 
#                  xticks=xday[::xmt*6], ylims=[0.,0.55], yticks=np.arange(0.,0.51,0.1), ylabel=r'R$_{eff}$ ($\mu$m)',
#                  xticklabel=[datetime.datetime(2022,i+1,1).strftime('%b') for i in months] )
# 
#       ax2.yaxis.label.set_color('b') 
#       ax2.tick_params(axis='y', colors='b')  
#       ax2.spines['right'].set_color('b')    
      
      
      ax3 = Ta.plt_timeseries_fillx( ax3, xday, Vt*1e4, color='g', ls='--',label='fitted V', xminor=xmt, xlims=[xday[0],xday[-1]], 
                 xticks=xday[::xmt*6], ylims=[0.,10], yticks=np.arange(0.,11,2), ylabel=r'V (10$^{-4}$$\mu$m$^3$$\mu$m$^{-2}$)',
                 xticklabel=[datetime.datetime(2022,i+1,1).strftime('%b') for i in months] )  
                                
      ax3.fill_between( xday, Vt*1e4, y2=Vt[44]*1e4, where=(xday<=xday[transidx]), facecolor='g', edgecolor=None, alpha=0.2 )
      ax3.annotate( 'Condensation', xy=(xday[int(transidx/2)], Vt[int(transidx/2)]*1e4), fontname='Arial',
                    xytext=(xday[int(transidx/2)], Vt[int(transidx/2)]*1e4/2), color='g' ) 
      ax3.fill_between( xday, Vt*1e4, y2=Vt[44]*1e4, where=(xday>=xday[transidx]), facecolor='g', edgecolor=None, alpha=0.1 )     
      ax3.annotate( 'Transport', xy=(xday[int(transidx+5)], Vt[int(transidx+5)]*1e4),fontname='Arial',
                    xytext=(xday[int(transidx+5)], Vt[int(transidx+5)]*1e4/2.5), color='g' )      
      ax3.yaxis.label.set_color('g') 
      ax3.tick_params(axis='y', colors='g')  
      ax3.spines['right'].set_color('g')   
      #ax3.spines['right'].set_position(('axes',1.12))
      ax3.legend(loc='upper right', bbox_to_anchor=(0.8, 0.8, 0.2, 0.2), fontsize='small',frameon=False)   
      
      axes[il].fill_between( xday, Nt, y2=Nt[44], facecolor='r', edgecolor=None, alpha=0.6 )
      axes[il].annotate( 'Coagulation (27.1-46.8%) + Nucleation (72.9-53.2%)', xy=(xday[55], 1.5), xytext=(xday[55], 0.5), fontname='Arial',color='r' )
      axes[il].legend(loc='upper right', bbox_to_anchor=(0.6, 0.8, 0.2, 0.2), fontsize='small', frameon=False)  
#          
   plt.tight_layout()
   plt.savefig( 'img/omps&SAGE3_retaer_coag+nuc+cond+hyg+trans2_timeseries_intpwh2so4_DectoNovb_addlines_reg'+str(ic+1)+'.png', dpi=300 )
   plt.close()  
   
   #sys.exit()
   
   

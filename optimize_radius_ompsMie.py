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

import mylib.ompslimb as omps
import mylib.GCpylib as Gpy

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
def pres2alt( p_prof, Tprof ):
   '''
   deltaP/P = - deltaz * (Mair * g / (R*T))
   '''
   p0   = 1013* 100
   allp = np.append( np.array([p0]), p_prof ) 
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
               
###########################
# read Mie LUT

specs = ['SULF','BC','OC']
wavs  = [675, 997, 1020]
#weights = [40, 70, 73, 81]
weights   = sorted([78.6, 75.2, 73.3, 71.9, 62.0, 53.2, 46.9, 42.0])

AE1  = dict()
AE2  = dict()
kext1 = dict()
kext2 = dict()
vol0 = dict()
for ispec in specs:
   allext = dict()
   allQext= dict()
   for iw in range(len(wavs)):
      data = np.load('data/'+ispec+'_'+str(wavs[iw])+'nm.npz')
      allext[wavs[iw]] = data['ext']
      allQext[wavs[iw]]= data['Qext']
      if iw == 0:
         vol0[ispec] = data['vol']
      
   AE1[ispec] = - (np.log(allext[675] / allext[997]) / np.log(675 / 997)) 
   AE2[ispec] = - (np.log(allext[675] / allext[1020]) / np.log(675 / 1020)) 
   usereff    = np.reshape( data['reff'], (len(data['reff']), 1, 1) )
   shape3d = allQext[997].shape 
   kext1[ispec] = 0.75 * allQext[997] 
   kext2[ispec] = 0.75 * allQext[1020]

allr = data['reff']
allv = data['veff']

####################
# 10 years
####################

# read omps 10 year aerosol profile
vars  = [ 'b'+str(i) for i in range(6) ]
vars += ['AE','zmSAOD']
alt_levels = ['10-15','15-20','20-25','25-30']
vars += ['zmAE'+str(i+1) for i in range(len(alt_levels))]

years   = np.arange(2014,2023)
im_nday = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
mms     = np.arange(12)
alltimes= []
mm_nday = []
s_time  = [ datetime.datetime.strptime(str(years[0])+'Jan01', '%Y%b%d') ]

smk_stime = datetime.datetime(2019,12,25)
smk_etime = datetime.datetime(2020,6,25)
smk_sday = (smk_stime - s_time[0]).total_seconds()/(3600*24)
smk_eday = (smk_etime - s_time[0]).total_seconds()/(3600*24)

nday    = 0
for iyy in years:
  if iyy == 2022:
     usemm = np.arange(2)
  else:
     usemm = mms   
  for imm in usemm:
     mmtime = datetime.datetime(iyy, imm+1, 1)
     strtime = mmtime.strftime('%Y%m')
     alltimes.append( strtime )
     nday += im_nday[imm]
     mm_nday.append( im_nday[imm] )

# read mean profile
pltdata0 = dict()   
for i, it in enumerate(alltimes):
   data = np.load('data/omps_aer_region_meanprof_'+it+'.npz')
   dayidx  = data['dayidx']
   if i == 0:
      for key in vars:
         pltdata0[key] = np.full( (nday,data[key].shape[1]), np.nan )
   else:
      dayidx += sum(mm_nday[:i]) 
      
   for key in vars:         
      pltdata0[key][dayidx,:] = data[key]  
         
bands = data['bands']
nb    = len(bands)
alt   = data['alt']
lats  = data['lat_bin']
print('nday:',nday)
   
# 10 days average
pltdata = dict()
for key in vars: 
   pltdata[key] = pltdata0[key]
#    data10d = np.full( (10, int(nday /10), pltdata0[key].shape[1]), np.nan )     
#    for i in range(10):
#       data10d[i,:,:] = pltdata0[key][i:nday-(nday%10):10,:]
#    pltdata[key] = np.nanmean(data10d, axis=0)
   print(key, pltdata[key].shape)
#nday = int(nday /10)  
print('nday adjust:', nday)   

# -----------------------
# optimization

nlev = pltdata['AE'].shape[1]
ret  = np.full( (nday, nlev, len(allv), len(weights)), np.nan )
# for idd in range(nday):
#    for il in range(nlev):
#       iobs = pltdata['AE'][idd,il]
#       
#       if ~np.isnan(iobs):
#        for iv in range(len(allv)):
#          for iw in range(len(weights)):
#             if (idd >= smk_sday) and (idd <= smk_eday):
#             #lut = AE1['OC'][:,iv]
#                lut = AE1['SULF'][:,iv,iw]
#             else:
#                lut = AE1['SULF'][:,iv,iw] 
#          
#             ret_ir = interplut( iobs, lut, allr )
#             ret[idd,il,iv,iw] = ret_ir
# 
# # 10-day smooth
# ret10d = np.full( (10, int(nday /10), ret.shape[1], ret.shape[2], ret.shape[3]), np.nan )     
# for i in range(10):
#     ret10d[i,:,:,:,:] = ret[i:nday-(nday%10):10,:,:,:]
# pltret = np.nanmean(ret10d, axis=0)
# nday   = int(nday/10)
#          
# # -----------------------
# # plot
# years   = np.arange(2014,2023)
# xticks  = np.arange(0, nday, 365/10)
# fig, axes = plt.subplots( nrows=len(weights), ncols=1, figsize=(12,3*len(weights)) ) #6
# cmap  = plt.cm.gist_ncar
# xday  = np.arange(nday)
# xx, yy = np.meshgrid( xday, alt )
# #levels= np.arange(vmin, vmax+0.05, 0.05)
# vmin  = 0.02
# vmid  = 0.08
# vmax  = 0.38
# levels0= np.arange(vmin, vmid, 0.02)
# levels1= np.arange(vmid, vmax+0.01, 0.02)
# levels = np.append( levels0, levels1)
# #levels= np.arange(vmin, vmax+0.1, 0.1)
# print('levels:', levels)
# # create new colormap
# newcolors = np.vstack((cmap(np.linspace(0, 1.0, 64))[0:-15:3,:],
#                        cmap(np.linspace(0, 1.0, 64))[-15:,:]))
# # newcolors = np.vstack((newcolors,
# #                        cmap(np.linspace(0.8, 1, 32))))
# print('cmap', cmap(np.linspace(0, 0.9, 8)))
# print(cmap(np.linspace(0., 1.0, 8)))
# newcmp = colors.ListedColormap(newcolors, name='cmbgist')
# 
# for iw in range(len(weights)):
#   img  = axes[iw].contourf( xx, yy, np.nanmean(pltret[:,:,:,iw],axis=2).T, levels, vmin=vmin, vmax=vmax, cmap=plt.cm.get_cmap(newcmp, len(levels) - 1) )  
#   axes[iw].contour(xx, yy, np.nanmean(pltret[:,:,:,iw],axis=2).T, img.levels, colors='k', linewidths=0.25)
# # for ie in range(len(t_erup)):
# #    axes.axvline( x=t_erup[ie], color='w', linewidth=1.0, linestyle='--')
#   print('ext max:', np.nanmax(np.nanmean(pltret[:,:,:,iw],axis=2)) )
#    #axes.contour(Mxx, Myy, data['MLSdata'].T* sf, img.levels, colors='k', linewidths=0.75)
#   cbar = fig.colorbar(img, ax=axes[iw], extend='both',shrink=0.98, pad=0.01, ticks=levels) 
#   cbar.ax.set_yticklabels(['{:.2f}'.format(x) for x in levels])
#   cbar.ax.set_ylabel( r'Effective Radius ($\mu$m)' ) 
#   axes[iw].set_ylim([15,30]) 
#   axes[iw].set_xticks(xticks)
#   axes[iw].set_xticklabels(years)
#   axes[iw].set_ylabel('Altitude (km)')  
# #axes.set_xlabel('Days after 1 Dec 2021')   
#   axes[iw].set_title('OMPS-LP effective radius for {:d}% H2SO4'.format(weights[iw]))  
# plt.tight_layout()
# plt.savefig( 'img/omps_radii_timeseries_10y.png', dpi=300 )
# plt.close() 

####################
# 2022
#####################

times = ['202112']
for itt in range(11):
   times.append( '2022{:0>2d}'.format(itt+1) )
print(times)
s_time= [ datetime.datetime.strptime(itime+'01', '%Y%m%d') for itime in times]
alt_levels = ['10-15','15-20','20-25','25-30']
# vars  = [ 'b'+str(i) for i in range(6) ]
# vars += ['AE','zmSAOD']
# vars += ['zmAE'+str(i+1) for i in range(len(alt_levels))]
vars = ['zmAE','zmb5']

im_nday = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
nday    = 31
mm_nday = [31]
usemm   = np.arange(11)   
for imm in usemm:
     mmtime = datetime.datetime(2022, imm+1, 1)
     strtime = mmtime.strftime('%Y%m')
     nday += im_nday[imm]
     mm_nday.append( im_nday[imm] )
print(mm_nday)
     
# read mean profile
pltdata = dict()
for i, it in enumerate(times):
   data = np.load('data/omps_aer_region_meanprof_'+it+'.npz')
#    if it == times[0]:
#       for key in vars:
#          pltdata[key] = data[key]
#       nday  = data['b0'].shape[0]   
#    else:
#       for key in vars:      
#          pltdata[key] = np.vstack( (pltdata[key], data[key]))
#       nday  += data['b0'].shape[0]   
      
   dayidx  = data['dayidx']
   if it == times[0]:
      for key in vars:
        if len(data[key].shape) == 2:
           pltdata[key] = np.full( (nday,data[key].shape[1]), np.nan )
        else:
           pltdata[key] = np.full( (nday,data[key].shape[1],data[key].shape[2]), np.nan )   
   else:
      dayidx += sum(mm_nday[:i]) 
      
   for key in vars:         
      pltdata[key][dayidx,:] = data[key]     

bands = data['bands']
nb    = len(bands)
alt   = data['alt']
lats  = data['lat_bin']
print('nday:',nday)    

# eruption dates
spec_days = ['20211220','20220113','20220115']
t_erup  = []
for idd in spec_days:
    iday = datetime.datetime.strptime(idd, '%Y%m%d')
    ix   = round((iday - s_time[0]).total_seconds() / 3600 /24)
    t_erup.append( ix )

# ----------------------
# regional mean
region = [-30,20]
lidx   = (lats >= region[0]) & (lats <= region[-1])
# for idd in xday:
#   for ic in range(len(lats)):
#     zmtrpp = pltdata['zm_trp'][idd,ic]
#     flag = (alt < zmtrpp)
#     pltdata['zmAE'][idd,ic,flag] = np.nan  
pltdata['b5'] = np.nanmean( pltdata['zmb5'][:,lidx,:], axis=1 )
pltdata['AE'] = np.nanmean( pltdata['zmAE'][:,lidx,:], axis=1 )  


# -----------------------
# optimization

nlev = pltdata['AE'].shape[1]
ret2  = np.full( (nday, nlev, len(allv), len(weights)), np.nan )
# for idd in range(nday):
#    for il in range(nlev):
#       iobs = pltdata['AE'][idd,il]
#       
#       if ~np.isnan(iobs):
#        for iv in range(len(allv)):
#          for iw in range(len(weights)):
#             if (idd >= smk_sday) and (idd <= smk_eday):
#             #lut = AE1['OC'][:,iv]
#                lut = AE1['SULF'][:,iv,iw]
#             else:
#                lut = AE1['SULF'][:,iv,iw] 
#          
#             ret_ir = interplut( iobs, lut, allr )
#             ret2[idd,il,iv,iw] = ret_ir
# 
# 
# # -----------------
# # plot    
# fig, axes = plt.subplots( nrows=len(weights), ncols=1, figsize=(2.3*len(times),3*len(weights)) ) #6
# cmap  = plt.cm.gist_ncar
# xday  = np.arange(nday)
# xx, yy = np.meshgrid( xday, alt )
# #levels= np.arange(vmin, vmax+0.05, 0.05)
# #levels= np.arange(vmin, vmax+0.1, 0.1)
# vmin  = 0.02
# vmid  = 0.08
# vmax  = 0.4
# levels0= np.arange(vmin, vmid, 0.02)
# levels1= np.arange(vmid, vmax+0.01, 0.02)
# levels = np.append( levels0, levels1)
# #levels= np.arange(vmin, vmax+0.1, 0.1)
# print('levels:', levels)
# # create new colormap
# newcolors = np.vstack((cmap(np.linspace(0, 1.0, 64))[0:-15:3,:],
#                        cmap(np.linspace(0, 1.0, 64))[-15:,:]))
# newcmp = colors.ListedColormap(newcolors, name='cmbgist')
# 
# for iw in range(len(weights)):
#   img  = axes[iw].contourf( xx, yy, np.nanmean(ret2[:,:,:,iw],axis=2).T, levels, extend='both',vmin=vmin, vmax=vmax, cmap=plt.cm.get_cmap(newcmp, len(levels) - 1) )  
#   axes[iw].contour(xx, yy, np.nanmean(ret2[:,:,:,iw],axis=2).T, img.levels, colors='k', linewidths=0.25)
#   for ie in range(len(t_erup)):
#      axes[iw].axvline( x=t_erup[ie], color='w', linewidth=1.0, linestyle='--')
#   print('ext max:', np.nanmax(np.nanmean(ret2[:,:,:,iw],axis=2)) )
#    #axes.contour(Mxx, Myy, data['MLSdata'].T* sf, img.levels, colors='k', linewidths=0.75)
#   cbar = fig.colorbar(img, ax=axes[iw], shrink=0.98, pad=0.01, ticks=levels) 
#   cbar.ax.set_yticklabels(['{:.2f}'.format(x) for x in levels])
#   cbar.ax.set_ylabel( r'Effective Radius ($\mu$m)' ) 
#   axes[iw].set_ylim([15,30]) 
#   axes[iw].set_yticks(np.arange(16,31,2))
#   axes[iw].set_xticks(xday[::5])
#   axes[iw].set_ylabel('Altitude (km)')  
#   axes[iw].set_xlabel('Days after 1 Dec 2021')   
#   axes[iw].set_title('OMPS-LP effective radius for {:d}% H2SO4'.format(weights[iw]))   
# plt.tight_layout()
# plt.savefig( 'img/omps_radii_timeseries_DectoAprb.png', dpi=300 )
# plt.close()      


####################
# 2022: change H2SO4 weight
#####################

vars  = [ 'b'+str(i) for i in range(6) ]
vars += ['AE','zmSAOD']
alt_levels = ['10-15','15-20','20-25','25-30']
vars += ['zmAE'+str(i+1) for i in range(len(alt_levels))]

# -----------------------
# read wh2so4
wh2so4_data = np.load('data/MLS_wh2so4_v4.2noQC_DectoNov.npz')
mTprof = np.nanmean(wh2so4_data['Tprof'], axis=1)
Pprof  = wh2so4_data['levT']
useP   = wh2so4_data['uselev']
lidx   = np.where((Pprof >= useP[-1]) & (Pprof <= useP[0]))[0]
Hprof  = pres2alt( Pprof[lidx], mTprof[lidx] ) * 1e-3 # km
print(Hprof.shape, wh2so4_data['wh2so4'].shape)
print(alt, Hprof[0], Hprof[-1])

# interp in altitude
xx = np.arange(wh2so4_data['wh2so4'].shape[1])
intp_wh2so4 = np.full( (len(alt), len(xx)), np.nan )
altidx      = np.where((alt >= Hprof[0]) & (alt <= Hprof[-1]))[0]
intpalt     = alt[altidx]
for idd in range(len(xx)):
  if len(np.where(np.isnan(wh2so4_data['wh2so4'][:,idd]))[0]) == 0:
   f  = interpolate.interp1d(Hprof, wh2so4_data['wh2so4'][:,idd], kind='linear')
   newdata = f(intpalt) 
   intp_wh2so4[altidx,idd] = newdata
   intp_wh2so4[alt>Hprof[-1],idd] = newdata[-1]
   
print('intp max:', np.nanmax(intp_wh2so4), newdata.shape, len(intpalt))
print(Hprof, len(np.where(np.isnan(intp_wh2so4[alt>=Hprof[0],:]))[0]), len(np.where(np.isnan(wh2so4_data['wh2so4']))[0]))

# -----------------------
# optimization (interp)

nlev = pltdata['AE'].shape[1]
ret3  = np.full( (nday, nlev, len(allv)), np.nan )
retnumden= np.full( (nday, nlev, len(allv)), np.nan )
retvol   = np.full( (nday, nlev, len(allv)), np.nan )
useil = np.where(alt >= Hprof[0])[0]
#print(np.where(np.isnan(intp_wh2so4[useil,:])))

for idd in range(nday):
   for il in range(len(useil)):
      lidx = useil[il]
      iobs = pltdata['AE'][idd,lidx]
      iaod = pltdata['b5'][idd,lidx] * (alt[1] - alt[0])
      iwh2so4 = intp_wh2so4[lidx,idd]
      if np.isnan(iobs) or np.isnan(iwh2so4):
         print('lut',idd, il, iobs, iwh2so4)
      
      if ~np.isnan(iobs) and ~np.isnan(iwh2so4):
       for iv in range(len(allv)):
         lut = np.full( AE1['SULF'].shape[0], np.nan )
         lut2 = np.full( kext1['SULF'].shape[0], np.nan )
         if iwh2so4 < weights[0]:
            lut  =  AE1['SULF'][:,iv,0]
            lut2 = kext1['SULF'][:,iv,0]
         elif iwh2so4 > weights[-1]:
            lut =  AE1['SULF'][:,iv,-1]  
            lut2 = kext1['SULF'][:,iv,-1] 
         else:
            for ir in range(AE1['SULF'].shape[0]):
               wlut = AE1['SULF'][ir,iv,:]
               f2   = interpolate.interp1d( weights, wlut, fill_value='extrapolation' )
               lut[ir] = f2(iwh2so4)
               
               wlut2 = kext1['SULF'][ir,iv,:]
               f3   = interpolate.interp1d( weights, wlut2, fill_value='extrapolation' )
               lut2[ir] = f3(iwh2so4)
               
                  
         print('lut',iv, lut)
         ret_ir = interplut( iobs, lut, allr )
         ret3[idd,lidx,iv] = ret_ir
         
         # retrieve volume concentration
         f4 = interpolate.interp1d( allr, lut2, fill_value='extrapolation' )
         irkext = f4( ret_ir ) 
         cvol   = iaod * ret_ir / irkext
         retvol[idd,lidx,iv] = cvol # um3/um2
         
         # retrieve number density from ext profile
         lutvol = vol0['SULF'][:,iv]
         f5 = interpolate.interp1d( allr, lutvol, fill_value='extrapolation' )
         irvol  = f5( ret_ir )
         retnumden[idd,lidx,iv] = cvol / irvol / ((alt[1] - alt[0]) * 1e9) # um-3


# -----------------
# plot    
fig, axes = plt.subplots( nrows=1, ncols=1, figsize=(12,3) ) #6
cmap  = plt.cm.gist_ncar
xday  = np.arange(nday)
xx, yy = np.meshgrid( xday, alt )
#levels= np.arange(vmin, vmax+0.05, 0.05)
#levels= np.arange(vmin, vmax+0.1, 0.1)
vmin  = 0.02
vmid  = 0.08
vmax  = 0.44
levels0= np.arange(vmin, vmid, 0.02)
levels1= np.arange(vmid, vmax+0.01, 0.02)
levels = np.append( levels0, levels1)
#levels= np.arange(vmin, vmax+0.1, 0.1)
print('levels:', levels)
# create new colormap
newcolors = np.vstack((cmap(np.linspace(0, 1.0, 64))[0:-8:4,:],
                       cmap(np.linspace(0, 1.0, 64))[-8:,:]))
newcmp = colors.ListedColormap(newcolors, name='cmbgist')

img  = axes.contourf( xx, yy, np.nanmean(ret3,axis=2).T, levels, extend='both',vmin=vmin, vmax=vmax, cmap=plt.cm.get_cmap(cmap, len(levels) - 1) )  
axes.contour(xx, yy, np.nanmean(ret3,axis=2).T, img.levels, colors='k', linewidths=0.25)
for ie in range(len(t_erup)):
     axes.axvline( x=t_erup[ie], color='w', linewidth=1.0, linestyle='--')
print('ext max:', np.nanmax(np.nanmean(ret3,axis=2)) )
   #axes.contour(Mxx, Myy, data['MLSdata'].T* sf, img.levels, colors='k', linewidths=0.75)
cbar = fig.colorbar(img, ax=axes, shrink=0.98, pad=0.01, ticks=levels[::2]) 
cbar.ax.set_yticklabels(['{:.2f}'.format(x) for x in levels[::2]])
cbar.ax.set_ylabel( r'Effective Radius ($\mu$m)' ) 
axes.set_ylim([17,30]) 
axes.set_yticks(np.arange(18,31,2))
#axes.set_xticks(xday[::5])
axes.set_xticks(xday[::30])
axes.set_xticklabels([itime.strftime('%b') for itime in s_time])
axes.xaxis.set_minor_locator(ticker.MultipleLocator(5))
axes.set_ylabel('Altitude (km)')  
axes.set_xlabel('Days after 1 Dec 2021')   
axes.set_title('OMPS-LP effective radius' )   
plt.tight_layout()
plt.savefig( 'img/omps_radii_timeseries_intpwh2so4_DectoNovb.png', dpi=300 )
plt.close()      

# -------------
fig, axes = plt.subplots( nrows=1, ncols=1, figsize=(12,3) ) #6
cmap  = plt.cm.gist_ncar
xday  = np.arange(nday)
xx, yy = np.meshgrid( xday, alt )
#levels= np.arange(vmin, vmax+0.05, 0.05)
#levels= np.arange(vmin, vmax+0.1, 0.1)
vmin  = 0
vmid  = 0.01
vmax  = 0.1
levels0= np.arange(vmin, vmid, 0.001)
levels1= np.arange(vmid, vmax+0.01, 0.02)
levels = np.append( levels0, levels1)
#levels= np.arange(vmin, vmax+0.01, 0.01)

# create new colormap
newcolors = np.vstack((cmap(np.linspace(0, 1.0, 64))[0:-15:3,:],
                       cmap(np.linspace(0, 1.0, 64))[-15:,:]))
newcmp = colors.ListedColormap(newcolors, name='cmbgist')

img  = axes.contourf( xx, yy, np.nanmean(retvol,axis=2).T, levels, extend='both', 
                      vmin=vmin, vmax=vmax, cmap=plt.cm.get_cmap(cmap, len(levels) - 1) ) 

axes.contour(xx, yy, np.nanmean(retvol,axis=2).T, img.levels, colors='k', linewidths=0.25)
for ie in range(len(t_erup)):
     axes.axvline( x=t_erup[ie], color='w', linewidth=1.0, linestyle='--')
print('ext max:', np.nanmax(np.nanmean(retvol,axis=2)) )
   #axes.contour(Mxx, Myy, data['MLSdata'].T* sf, img.levels, colors='k', linewidths=0.75)
cbar = fig.colorbar(img, ax=axes, shrink=0.98, pad=0.01, ticks=levels)
cbar.ax.set_yticklabels(['{:.3f}'.format(x) for x in levels])
cbar.ax.set_ylabel( r'Volume conc ($\mu$m$^3$/$\mu$m$^2$)' ) 
axes.set_ylim([17,30]) 
axes.set_yticks(np.arange(18,31,2))
#axes.set_xticks(xday[::5])
axes.set_xticks(xday[::30])
axes.set_xticklabels([itime.strftime('%b') for itime in s_time])
axes.xaxis.set_minor_locator(ticker.MultipleLocator(5))
axes.set_ylabel('Altitude (km)')  
axes.set_xlabel('Days after 1 Dec 2021')   
axes.set_title('OMPS-LP sulfate number density' )   
plt.tight_layout()
plt.savefig( 'img/omps_cvol_timeseries_intpwh2so4_DectoNovb.png', dpi=300 )
plt.close()      

# -------------
fig, axes = plt.subplots( nrows=1, ncols=1, figsize=(12,3) ) #6
add_cnum = True
cmap  = plt.cm.jet
xday  = np.arange(nday)
xx, yy = np.meshgrid( xday, alt )
#levels= np.arange(vmin, vmax+0.05, 0.05)
#levels= np.arange(vmin, vmax+0.1, 0.1)
vmin  = 0
vmid  = 10
vmax  = 50
levels0= np.arange(vmin, vmid, 2)
levels1= np.arange(vmid, vmax+10, 10)
levels = np.append( levels0, levels1)
#levels = np.append( levels, np.array([100,200,400,800,5000]) )
#levels= np.arange(vmin, vmax+1, 10)
#lev_exp = np.arange(5)
#levels = np.power(10, lev_exp)
print('levels:', levels)
# create new colormap
newcolors = np.vstack((cmap(np.linspace(0, 1.0, 64))[0:-8,:],
                       cmap(np.linspace(0, 1.0, 64))[-8:,:]))
newcmp = colors.ListedColormap(newcolors, name='cmbgist')
#newcmp.set_bad('lightgray',1)
usecmp = plt.cm.get_cmap('jet')(np.linspace(0,1, len(levels) - 1))
cmap, norm = colors.from_levels_and_colors(levels, usecmp)
usenumden = np.nanmean(retnumden,axis=2)* 1e12
usenumden[usenumden >50] = np.nan
img  = axes.contourf( xx, yy, usenumden.T, levels, extend='max', 
                      norm=norm, cmap=cmap ) 
                      #norm=colors.LogNorm(vmin=1, vmax=1e4), cmap=cmap)
#img.cmap.set_under(usecmp[0])
#img.cmap.set_over(usecmp[-1])
#img.cmap.set_bad('lightgray')
axes.contour(xx, yy, usenumden.T, img.levels, colors='k', linewidths=0.25)
for ie in range(len(t_erup)):
     axes.axvline( x=t_erup[ie], color='w', linewidth=1.0, linestyle='--')
print('ext max:', np.nanmax(np.nanmean(retnumden,axis=2)* 1e12) )
   #axes.contour(Mxx, Myy, data['MLSdata'].T* sf, img.levels, colors='k', linewidths=0.75)
cbar = fig.colorbar(img, shrink=0.98, pad=0.07, ticks=levels[::2]) #
cbar.ax.set_yticklabels(['{:.2f}'.format(x) for x in levels[::2]])
cbar.ax.set_ylabel( r'Number density (cm$^{-3}$)' ) 
axes.set_ylim([17,30]) 
axes.set_yticks(np.arange(18,31,2))
#axes.set_xticks(xday[::5])
axes.set_xticks(xday[::30])
axes.set_xticklabels([itime.strftime('%b') for itime in s_time])
axes.xaxis.set_minor_locator(ticker.MultipleLocator(5))
axes.set_ylabel('Altitude (km)')  
axes.set_xlabel('Days after 1 Dec 2021')   
axes.set_title('OMPS-LP sulfate number density' )   

# total column number   
if add_cnum:
   usenumden[:,alt < 18] = np.nan
   usenumden[:,alt > 30] = np.nan
   retcnum = np.nansum( usenumden * (alt[1] - alt[0]) * 1e5, axis=1 ) / 1e6
   retcnum[retcnum < 1] = np.nan
   axs = axes.twinx()
   axs.plot(xday, retcnum, color='magenta', lw=1.5, ls='-')
   wvmin = np.nanmin(retcnum)
   wvmax = np.nanmax(retcnum)
   yticks= np.arange(wvmin, wvmax+(wvmax - wvmin)/10, (wvmax - wvmin)/10)
   axs.set_yticks( yticks )
   axs.set_yticklabels(['{:.1f}'.format(iy) for iy in yticks])
   axs.set_xlim([0,xday[-1]])
   axs.set_xticks(xday[::30])
   axs.set_ylabel(r'Strat. Col. particle number (10$^6$ cm$^{-2}$)' )
   axs.yaxis.label.set_color('magenta')
   axs.tick_params(axis='y', colors='magenta')

plt.tight_layout()
plt.savefig( 'img/omps_numden_timeseries_intpwh2so4_DectoNovb.png', dpi=300 )
plt.close()      


# --------------------------
# read MLS RH%
mons   = [i for i in range(1,12)]
times2 = ['Dec']
for im in mons:
   imdate = datetime.datetime( 2022, im, 1 )
   monname = imdate.strftime( '%b' )
   times2.append( monname )
#times2 = ['2mons','Feb','Mar','Apr']
vars2  = ['RH']
iy     = 'v4.2noQCb_'

numday      = []
useday      = dict()
idata       = dict()
data0 = np.load('./data/MLS_RHprof_daily_mean_hemis_Feb.npz')
for it, im in enumerate(times2): 
     with open('./data/MLS_RH_daily_mean_hemis_'+iy+im+'.json', 'r') as f:
        idata[it] = json.load(f)
     numday.append( idata[it]['useday'][-1] + 1 )
     useday[it]   = np.array(idata[it]['useday'])
     
# read RH vs growth factor
with open('./data/h2so4_RH_vs_beta.json', 'r') as f:
    pardata = json.load(f)  
useRH   = np.array( [0] + pardata['RH'] )
usebeta = np.array( [1.0] + pardata['beta'] )
func = interpolate.interp1d(useRH, usebeta, kind='linear',fill_value='extrapolate')
print('read beta:', useRH, usebeta)

# define  
nday     = sum(numday)   
xday     = np.arange(nday) 
RH_prof  = dict()
intp_RHprof = dict()
intp_beta   = dict()

for stats in ['max','median','mean','90th','75th','mstd']:  
   if stats != 'mstd':   
      RH_prof[stats]     = np.full(len(data0['lev_RH']), np.nan)
   intp_beta[stats]   = np.full((len(xday),len(alt)), np.nan)
   intp_RHprof[stats] = np.full((len(xday),len(alt)), np.nan)
   

# match T and RH lev
RHpres = data0['lev_RH']
Tpres  = data0['lev_T']
useP   = wh2so4_data['uselev']
lidx   = np.where((RHpres >= useP[-1]) & (RHpres <= useP[0]))[0]
     
for it, im in enumerate(times2): 
     data = np.load('./data/MLS_RHprof_daily_mean_hemis_'+iy+im+'.npz')    
     print(list(data.keys()), data['RHhigh'].shape, len(data['highidx']), data['highidx'])
     for key in vars2: 
        if len(data[key+'high'].shape) == 2:
           endday =  sum(numday[:it])
              
           # find the daily maximum for each layer
           for idd in range(len(idata[it]['npixels'])):
              startid = 0
              if idd > 0:
                 startid = sum(idata[it]['npixels'][:idd])   
              inp = idata[it]['npixels'][idd]
              if im == times[0]:
                 idx = idata[it]['useday'][idd]
              else:
                 idx = endday+idata[it]['useday'][idd]
              RH_prof['max']      = np.nanmax( data[key+'all'][startid:startid+inp,:], axis=0 )
              RH_prof['median']   = np.nanmedian( data[key+'all'][startid:startid+inp,:], axis=0 )
              RH_prof['90th']     = np.nanpercentile( data[key+'all'][startid:startid+inp,:], 90, axis=0 )
              RH_prof['mean']     = np.nanmean( data[key+'all'][startid:startid+inp,:], axis=0 )
              RH_prof['75th']     = np.nanpercentile( data[key+'all'][startid:startid+inp,:], 75, axis=0 )
                  
              
              intp_T    = Gpy.intp_lev_ip( Tpres, RHpres, data['Tprof'][idd,:] )
              RHalt     = pres2alt( RHpres[lidx], intp_T[lidx] ) * 1e-3 # km
              
              for stats in RH_prof.keys():
                 valid = np.where(~np.isnan( RH_prof[stats] ))
                 print(valid[0],len(RHalt),)
                 print(len(alt),len(RH_prof[stats][lidx]), intp_RHprof[stats].shape)
                 intp_RHprof[stats][idx,:] = Gpy.intp_hgt_ip( RHalt, alt, RH_prof[stats][lidx] )
                 intp_RHprof[stats][idx,:][intp_RHprof[stats][idx,:] > 100] = 100
                 intp_RHprof[stats][idx,:][intp_RHprof[stats][idx,:] < 0] = 0
                 intp_beta[stats][idx,:] = func( intp_RHprof[stats][idx,:] )
              #intp_RHprof['mstd'][idx,:] = intp_RHprof['mean'][idx,:]   
              #intp_beta['mstd'][idx,:] = intp_beta['mean'][idx,:]   
                 #print(intp_RHprof[stats][idx,:], intp_beta[stats][idx,:])
              
              # mean + 10std
              if idata[it]['useday'][idd] in data['highidx']:
                 print(idata[it]['useday'][idd], data['highidx'])
                 iid = np.where( np.array(data['highidx']) == idata[it]['useday'][idd] )[0][0]
                 print('check high idx:', iid, data['highidx'][iid], idata[it]['useday'][idd], data['RHhigh'][iid,:])   
                 valid2 = np.where(~np.isnan( data['RHhigh'][iid,:] ))
                 intp_RHprof['mstd'][idx,:] = Gpy.intp_hgt_ip( RHalt, alt, data['RHhigh'][iid,:][lidx] )
                 if it == 2:
                    print(intp_RHprof['mstd'][idx,:], idx, data['RHhigh'][iid,:][lidx], alt, RHalt)
              else:
                 intp_RHprof['mstd'][idx,:] = Gpy.intp_hgt_ip( RHalt, alt, data['RHlow'][idd,:][lidx] )
                 
              intp_RHprof['mstd'][idx,:][intp_RHprof['mstd'][idx,:] > 100] = 100
              intp_RHprof['mstd'][idx,:][intp_RHprof['mstd'][idx,:] < 0] = 0
              intp_beta['mstd'][idx,:] = func( intp_RHprof['mstd'][idx,:] )   
 
# save 
savedata = dict()
savedata['RH'] = intp_RHprof
savedata['beta'] = intp_beta
savedata['alt']  = alt
with open('data/MLS_hygbeta_timeseries_DectoNovb.pickle', 'wb') as handle:
    pickle.dump(savedata, handle, protocol=pickle.HIGHEST_PROTOCOL)

              
# -----------------
# plot    

for stats in intp_RHprof.keys():
  fig, axes = plt.subplots( nrows=3, ncols=1, figsize=(12,9) ) #6
  cmap  = plt.cm.gist_ncar
  xday  = np.arange(nday)
  xx, yy = np.meshgrid( xday, alt )
  cmap  = plt.cm.gist_ncar
  vmin  = 0.06
  vmid  = 0.2
  vmax  = 0.47
  levels0= np.arange(vmin, vmid-0.01, 0.02)
  levels1= np.arange(vmid, vmax+0.01, 0.02)
  levels = np.append( levels0, levels1)   
  print(levels)
  newcolors = np.vstack((cmap(np.linspace(0, 1.0, 64))[0:-8:4,:],
                       cmap(np.linspace(0, 1.0, 64))[-8:-6,:]))
  newcmp = colors.ListedColormap(newcolors, name='cmbgist')

  #bgreff = np.nanmean( np.nanmean(ret3,axis=2)[:31,:], axis=0, keepdims=True ) / np.nanmean( intp_beta[stats][:31,:], axis=0, keepdims=True )
  bgreff = np.nanmean( np.nanmean(ret3,axis=2)[:10,:], axis=0, keepdims=True ) / np.nanmean( intp_beta[stats][:10,:], axis=0, keepdims=True )
  hygreff = np.tile( bgreff, (len(xday),1) ) * intp_beta[stats]
  img  = axes[1].contourf( xx, yy, hygreff.T, levels, extend='both',vmin=vmin, vmax=vmax, cmap=plt.cm.get_cmap(cmap, len(levels) - 1) )  
  axes[1].contour(xx, yy, hygreff.T, img.levels, colors='k', linewidths=0.25)
  for ie in range(len(t_erup)):
     axes[1].axvline( x=t_erup[ie], color='w', linewidth=1.0, linestyle='--')
  print('ext max:', np.nanmax(np.nanmean(ret3,axis=2)) )
   #axes.contour(Mxx, Myy, data['MLSdata'].T* sf, img.levels, colors='k', linewidths=0.75)
  cbar = fig.colorbar(img, ax=axes[1], shrink=0.98, pad=0.01, ticks=levels[::2]) 
  cbar.ax.set_yticklabels(['{:.2f}'.format(x) for x in levels[::2]])
  cbar.ax.set_ylabel( r'Effective Radius ($\mu$m)' ) 
  axes[1].set_ylim([17,30]) 
  axes[1].set_yticks(np.arange(18,31,2))
  #axes.set_xticks(xday[::5])
  axes[1].set_xticks(xday[::30])
  axes[1].set_xticklabels(times2)
  axes[1].xaxis.set_minor_locator(ticker.MultipleLocator(5))
  axes[1].set_ylabel('Altitude (km)')  
  axes[1].set_xlabel('Days after 1 Dec 2021')   
  axes[1].set_title('OMPS-LP effective radius' )   
  
  # %RH
  #levels= np.arange(0, 80, 2)
  newcolors = np.vstack((cmap(np.linspace(0, 1.0, 64))[0:20:5,:],
                       cmap(np.linspace(0, 1.0, 64))[20:,:]))
  newcmp = colors.ListedColormap(newcolors, name='cmbgist')
  if stats == 'max':
     levels= [0,1,2,3,4,5,8,11,15,20,25,30,40]
  else:   
     levels= [0,1,2,3,4,5,8,11,15]
  intp_RHprof[stats][intp_RHprof[stats] < 0] = 0.0
  
  img  = axes[0].contourf( xx, yy, intp_RHprof[stats].T, levels, vmin=levels[0], vmax=levels[-1],
                extend='max',cmap=plt.cm.get_cmap(newcmp, len(levels) - 1) ) 
  lw = 0.25    
  axes[0].contour(xx, yy, intp_RHprof[stats].T, img.levels, colors='k', linewidths=lw)
  for ii in range(len(t_erup)):
      axes[0].axvline( x=t_erup[ii], color='w', linewidth=1.0, linestyle='--')
  cbar = fig.colorbar(img, ax=axes[0], shrink=0.98, pad=0.01, ticks=levels) 
  cbar.ax.set_yticklabels(['{:.0f}'.format(x) for x in levels])
  cbar.ax.set_ylabel( 'RH (%)' )    
  #axes[0].set_xticks(xday[::5])
  axes[0].set_xticks(xday[::30])
  axes[0].set_xticklabels(times2)
  axes[0].xaxis.set_minor_locator(ticker.MultipleLocator(5))
  axes[0].set_ylim([17,30]) 
  axes[0].set_yticks(np.arange(18,31,2))
  axes[0].set_ylabel('Altitude (km)')  
  axes[0].set_xlabel('Days after 1 Dec 2021')  
  axes[0].set_title('MLS daily '+stats+' relative humidity')    
    
  cmap  = plt.cm.gist_ncar
  xday  = np.arange(nday)
  xx, yy = np.meshgrid( xday, alt )
#levels= np.arange(vmin, vmax+0.05, 0.05)
#levels= np.arange(vmin, vmax+0.1, 0.1)
  vmin  = 1
  vmax  = 1.4
  #levels0= np.arange(vmin, vmid, 0.02)
  #levels1= np.arange(vmid, vmax+0.01, 0.02)
  #levels = np.append( levels0, levels1)
  levels= np.arange(vmin, vmax+0.03, 0.03)
  print('levels:', levels)
# create new colormap
  newcolors = np.vstack((cmap(np.linspace(0, 1.0, 64))[0:-15:3,:],
                       cmap(np.linspace(0, 1.0, 64))[-15:,:]))
  newcmp = colors.ListedColormap(newcolors, name='cmbgist')

  img  = axes[2].contourf( xx, yy, intp_beta[stats].T, levels, extend='both',vmin=vmin, vmax=vmax, cmap=plt.cm.get_cmap(cmap, len(levels) - 1) )  
  axes[2].contour(xx, yy, intp_beta[stats].T, img.levels, colors='k', linewidths=0.25)
  for ie in range(len(t_erup)):
     axes[2].axvline( x=t_erup[ie], color='w', linewidth=1.0, linestyle='--')
  print('ext max:', np.nanmax(intp_beta[stats]) )
   #axes.contour(Mxx, Myy, data['MLSdata'].T* sf, img.levels, colors='k', linewidths=0.75)
  cbar = fig.colorbar(img, ax=axes[2], shrink=0.98, pad=0.01, ticks=levels[::2]) 
  cbar.ax.set_yticklabels(['{:.2f}'.format(x) for x in levels[::2]])
  cbar.ax.set_ylabel( 'Particle growth factor' ) 
  axes[2].set_ylim([17,30]) 
  axes[2].set_yticks(np.arange(18,31,2))
  #axes.set_xticks(xday[::5])
  axes[2].set_xticks(xday[::30])
  axes[2].set_xticklabels(times2)
  axes[2].xaxis.set_minor_locator(ticker.MultipleLocator(5))
  axes[2].set_ylabel('Altitude (km)')  
  axes[2].set_xlabel('Days after 1 Dec 2021')   
  axes[2].set_title('OMPS-LP particle growth factor' )   
  
  plt.tight_layout()
  plt.savefig( 'img/MLS_hygradii_timeseries_'+stats+'_DectoNovb.png', dpi=300 )
  plt.close()      
  
  
  # hygreff start from different time
  deltay = 10
  nbin   = 6
  fig, axes = plt.subplots( nrows=nbin, ncols=1, figsize=(12,18) ) #6
  xday  = np.arange(nday)
  xx, yy = np.meshgrid( xday, alt )
  cmap  = plt.cm.gist_ncar
  vmin  = 0.06
  vmid  = 0.2
  vmax  = 0.47
  levels0= np.arange(vmin, vmid-0.01, 0.02)
  levels1= np.arange(vmid, vmax+0.01, 0.02)
  levels = np.append( levels0, levels1)   
  print(levels)
  newcolors = np.vstack((cmap(np.linspace(0, 1.0, 64))[0:-8:4,:],
                       cmap(np.linspace(0, 1.0, 64))[-8:-6,:]))
  newcmp = colors.ListedColormap(newcolors, name='cmbgist')

  for idd in range(nbin):
     bgreff2 = np.nanmean(ret3,axis=2)[:30+idd*deltay,:]
     afreff = np.tile( bgreff2[-1:,:], (len(xday)-(30+idd*deltay), 1) ) * intp_beta[stats][(30+idd*deltay):,:]
     #print(idd, afreff, bgreff2)
     hygreff2 = np.concatenate( (bgreff2, afreff), axis=0 )
     img  = axes[idd].contourf( xx, yy, hygreff2.T, levels, extend='both',vmin=vmin, vmax=vmax, cmap=plt.cm.get_cmap(cmap, len(levels) - 1) )  
     axes[idd].contour(xx, yy, hygreff2.T, img.levels, colors='k', linewidths=0.25)
     for ie in range(len(t_erup)):
        axes[idd].axvline( x=t_erup[ie], color='w', linewidth=1.0, linestyle='--')
     print('ext max:', np.nanmax(hygreff2) )
     cbar = fig.colorbar(img, ax=axes[idd], shrink=0.98, pad=0.01, ticks=levels[::2]) 
     cbar.ax.set_yticklabels(['{:.2f}'.format(x) for x in levels[::2]])
     cbar.ax.set_ylabel( r'Effective Radius ($\mu$m)' ) 
     axes[idd].set_ylim([17,30]) 
     axes[idd].set_yticks(np.arange(18,31,2))
  #axes.set_xticks(xday[::5])
     axes[idd].set_xticks(xday[::30])
     axes[idd].set_xticklabels(times2)
     axes[idd].xaxis.set_minor_locator(ticker.MultipleLocator(5))
     axes[idd].set_ylabel('Altitude (km)')  
     axes[idd].set_xlabel('Days after 1 Dec 2021')   
     axes[idd].set_title('OMPS-LP effective radius' )   

  plt.tight_layout()
  plt.savefig( 'img/MLS_hygradii_diffstart_timeseries_'+stats+'_DectoNovb.png', dpi=300 )
  plt.close()      
  
#   if stats == 'max':
#      sys.exit()

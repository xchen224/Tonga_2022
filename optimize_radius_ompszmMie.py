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

specs = ['SULF'] #,'BC','OC']
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
      data = np.load('data/'+ispec+'_'+str(wavs[iw])+'nm_large.npz')
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


####################
# 2022
#####################

times = ['202112']
for itt in range(11):
   times.append( '2022{:0>2d}'.format(itt+1) )
print(times)
s_time= [ datetime.datetime.strptime(itime+'01', '%Y%m%d') for itime in times]
vars  = [ 'b'+str(i) for i in range(6) ]
vars += ['AE','zmSAOD','zmb5','zmAE']
alt_levels = ['10-15','15-20','20-25','25-30']
vars += ['zmAE'+str(i+1) for i in range(len(alt_levels))]

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
nlat  = len(lats)
print('nday:',nday)    

# eruption dates
spec_days = ['20211220','20220113','20220115']
t_erup  = []
for idd in spec_days:
    iday = datetime.datetime.strptime(idd, '%Y%m%d')
    ix   = round((iday - s_time[0]).total_seconds() / 3600 /24)
    t_erup.append( ix )

# -----------------------
# optimization

nlev = pltdata['AE'].shape[1]
ret2  = np.full( (nday, nlev, len(allv), len(weights)), np.nan )



####################
# 2022: change H2SO4 weight
#####################

vars  = [ 'b'+str(i) for i in range(6) ]
vars += ['AE','zmSAOD','zmb5']
alt_levels = ['10-15','15-20','20-25','25-30']
vars += ['zmAE'+str(i+1) for i in range(len(alt_levels))]

# -----------------------
# read wh2so4
wh2so4_data = np.load('data/MLS_wh2so4_v4.2noQC_dzm_DectoNov.npz')
Pprof  = wh2so4_data['levT']
useP   = wh2so4_data['uselev']
lidx   = np.where((Pprof >= useP[-1]) & (Pprof <= useP[0]))[0]

# interp in altitude
xx = np.arange(wh2so4_data['wh2so4'].shape[0])
intp_wh2so4 = np.full( ( len(xx), len(lats), len(alt)), np.nan )
#altidx      = np.where((alt >= 15) & (alt <= 30))[0]
#intpalt     = np.ma.filled( alt[altidx], fill_value=np.nan )
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
   
print('intp max:', np.nanmax(intp_wh2so4), newdata.shape, len(alt))
print(Hprof, len(np.where(np.isnan(intp_wh2so4[:,20,alt>=Hprof[0]]))[0]), len(np.where(np.isnan(wh2so4_data['wh2so4']))[0]))

# -----------------------
# optimization (interp)

nlat = len(lats)
nlev = pltdata['zmAE'].shape[2]
ret3  = np.full( (nday, nlat, nlev, len(allv)), np.nan )
retnumden= np.full( (nday, nlat, nlev, len(allv)), np.nan )
retvol   = np.full( (nday, nlat, nlev, len(allv)), np.nan )
#print(np.where(np.isnan(intp_wh2so4[useil,:])))

for idd in range(nday):
  for ic in range(nlat):
    iprof_obs = pltdata['zmAE'][idd,ic,:]
    iprof_aod = pltdata['zmb5'][idd,ic,:] * (alt[1] - alt[0])
    iprof_wh2so4 = intp_wh2so4[idd,ic,:]
    
    flag  = np.logical_and( iprof_aod > 3e-4, ~np.isnan(iprof_wh2so4) )
    useil = np.where( flag==True )[0]
    print( idd, ic, iprof_aod, iprof_wh2so4, useil)

    for il in range(len(useil)):
      lidx = useil[il]
      iobs = iprof_obs[lidx]
      iaod = iprof_aod[lidx]
      iwh2so4 = iprof_wh2so4[lidx]
      if np.isnan(iobs) or np.isnan(iwh2so4):
         print('lut',idd, il, iobs, iwh2so4)
      
      if ~np.isnan(iobs):
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
               
                  
         ret_ir = interplut( iobs, lut, allr )
         ret3[idd,ic,lidx,iv] = ret_ir
         
         # retrieve volume concentration
         f4 = interpolate.interp1d( allr, lut2, fill_value='extrapolation' )
         if ~np.isnan(iaod):
           irkext = f4( ret_ir ) 
           cvol   = iaod * ret_ir / irkext
           retvol[idd,ic,lidx,iv] = cvol # um3/um2
         
           # retrieve number density from ext profile
           lutvol = vol0['SULF'][:,iv]
           f5 = interpolate.interp1d( allr, lutvol, fill_value='extrapolation' )
           irvol  = f5( ret_ir )
           retnumden[idd,ic,lidx,iv] = cvol / irvol / ((alt[1] - alt[0]) * 1e9) # um-3


# save retrieval results
np.savez('data/omps_aerret_v4.2noQC_dzm_DectoNov.npz', radii=ret3, cvol=retvol, numden=retnumden, 
          alt=alt, nday=nday, clats=lats)
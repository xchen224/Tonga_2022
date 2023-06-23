import os, sys
import datetime
from scipy import stats

sys.path.append('/Users/xchen224/anaconda3/envs/xcpy/lib/python3.7/site-packages/mylib/')
from plot_libs import *
import MLS_lib as MLS

def detrend_linear( xday, ydata ):
    '''
    remove the trend of y timeseries
    '''
   
    # linear regression
    nanidx = np.where(~np.isnan(ydata))[0]
    if len(nanidx) > 0:
         slope, intercept, correlation, p_value_slope, std_error = stats.linregress(xday[nanidx], ydata[nanidx])
         regy = (slope * xday) + intercept
         deltay = ydata - regy
    else:
         deltay = np.full( len(ydata), np.nan ) 
         
    return deltay, regy       
   
   
############
gas     = ['N2O']#['O3']'T'

# daily
days    = np.arange(365)
# monthly
#days    = np.arange(12) 
years   = np.arange(2005,2023)

profname = 'prof' #'Tprof'

datadir  = '/Dedicated/jwang-data/shared_satData/MLS/L2/'+gas[0]+'/'
files    = glob.glob( datadir + '/2005/MLS_'+profname+'_zonal_*.npz' )
data0    = np.load( files[0] )
levs     = data0['levs']
clats    = data0['clats']
nlat    = len(clats)
nlev    = len(levs)

# data0    = np.load( datadir + '/2022/MLS_Tprof_zonal_2022d150.npz' )
# print(data0['Tprof'].shape, data0['Tprof'][30,:], clats[30])
# sys.exit()


# read data of each day in the region
all_dprof = np.full( (len(years), len(days), nlat, nlev), np.nan)
ave_dprof = np.full( (1, len(days), nlat, nlev), np.nan)
for i, iday in enumerate(days):

   y_dprof = np.full( (len(years), nlat, nlev), np.nan)
   
   for j, yy in enumerate(years):
   
      # daily mean
      if len(days) > 300:
         itime = datetime.datetime.strptime('{:.0f}d{:0>3d}'.format(yy, iday+1), "%Yd%j")
         if MLS.leap_year( yy ) == 1:
           if iday > 58:
              itime = datetime.datetime.strptime('{:.0f}d{:0>3d}'.format(yy, iday+2), "%Yd%j")
           
         time_str = itime.strftime("%Yd%j")
         
      # monthly mean
      else:
         itime = datetime.datetime.strptime('{:.0f}{:0>2d}'.format(yy, iday+1), "%Y%m")
         time_str = itime.strftime("%Y%m")   
      print('start ', time_str)
      
      # read daily data
      filename = glob.glob( datadir + '/'+str(yy)+'/MLS_'+profname+'_zonal_'+ time_str + '.npz' )
      if len(filename) == 0:
         continue
         
      data     = np.load( filename[0] )
      y_dprof[j,:,:] = data['Tprof']
      all_dprof[j,i,:,:] = data['Tprof']
      
   ave_dprof[0,i,:,:] = np.nanmean( y_dprof, axis=0 )

print(np.where(np.isnan(ave_dprof))[3])
anom_dprof = all_dprof - np.tile( ave_dprof, (len(years),1,1,1) )
anom_dprof3D = np.reshape( anom_dprof, (-1,nlat,nlev) )

# plot anomaly time series at different pressures
cho_pres = [68, 31, 15]

fig, axes = plt.subplots(nrows=len(cho_pres), ncols=1, figsize=(8,6))
axs       = axes.flatten()
if 'T' in profname:
  vmin = -5
  vmax = 5.1
  sf   = 1
  levels = np.arange(vmin,vmax,1)
else:
  vmin = -20
  vmax = 21
  sf   = 1e9
  levels = np.arange(vmin,vmax,4)  
cmap = plt.cm.bwr
cmap.set_bad('w')
xday = np.arange(anom_dprof3D.shape[0])
if len(days) > 300:
  xticks = xday[::365] 
  suffix = ''
  savename = 'daily'
else:
  xticks = xday[::12]  
  suffix = '_mon'
  savename = 'monthly'
      
for il in range(len(cho_pres)):
      ilev = np.argmin(np.abs(levs - cho_pres[il]))
      
      MLS.plot_timeseries( axs[il], xday, clats, anom_dprof3D[:,:,ilev]*sf, 
                              cbar=True, xticks=xticks, xticklabel=np.arange(2005,2023),
                              vmin=vmin, vmax=vmax, cmap=cmap, levels=levels,
                              title='T anomaly at {:.0f}hPa'.format(levs[ilev]), 
                              cbar_ylabel= 'T (K)')                                            
                                          
plt.tight_layout() 
plt.savefig('img/MLS_'+gas[0]+'anom_ilevmap'+suffix+'_allyears.png',format='png', dpi=300 )      
plt.close()   

# detrend anomaly
alt    = ((levs <= 101) & (levs >= 5))
uselev = levs[alt]
detr_prof = np.full( anom_dprof3D.shape, np.nan)

# split into two periods for N2O: 2005-2009, 2010-2022
if gas[0] == 'N2O': 
   yymid = 2010
   xmid  = (2010 - years[0]) * len(days)
   
for ilat in range(nlat):

   useprof   = anom_dprof3D[:,ilat, alt]
   fig, axes = plt.subplots(nrows=len(uselev), ncols=1, figsize=(9,1.3*len(uselev)))
   
   for il in range(len(uselev)):
   
      axes[il].plot( xday, useprof[:,il], color='k', label='raw data' )
      
      # split into two periods:
      if gas[0] == 'N2O': # and uselev[il] > 50:
         xday0 = xday[:xmid]
         useprof0 = useprof[:xmid,il]
         xday1 = xday[xmid:]
         useprof1 = useprof[xmid:,il]

         # detrend
         deltay0, regy0 = detrend_linear( xday0, useprof0 )
         deltay1, regy1 = detrend_linear( xday1, useprof1 )
         deltay = np.append( deltay0, deltay1 )
         
         axes[il].plot(xday0, regy0, color='r', label='regression ({:.1f}hPa)'.format(uselev[il]))   
         axes[il].plot(xday1, regy1, color='r') 
         
      # single period:   
      else:     
         deltay, regy = detrend_linear( xday, useprof[:,il] )  
         axes[il].plot(xday, regy, color='r', label='regression ({:.1f}hPa)'.format(uselev[il]))  
         
      if il == 0:
         deltay2d = deltay
      else:
         deltay2d = np.vstack( (deltay2d, deltay) )   
        
      axes[il].set_ylabel( 'T anomaly (K)' )
      axes[il].legend(loc='upper right', ncol=2)
      axes[il].set_xticks(xticks)
      axes[il].set_xticklabels(np.arange(2005,2023))
      axes[il].axhline(y = 0, color = 'k', lw=0.2, linestyle = '--')

   plt.tight_layout() 
   if clats[ilat]%10 == 0:
      plt.savefig('img/MLS_'+gas[0]+'anom_linear'+suffix+'_allyears_lat{:d}.png'.format(clats[ilat]), format='png', dpi=300 )      
   plt.close()   
   
   detr_prof[:,ilat,alt] = deltay2d.T

# save            
np.savez('./data/MLS_'+gas[0]+'profanom_detrend_'+savename+'_mean.npz', Tprof=detr_prof, Tlevs=levs, 
         uselev=uselev,clats=clats)
         

 

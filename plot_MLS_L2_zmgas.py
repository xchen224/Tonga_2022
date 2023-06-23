import os, sys
import datetime
from matplotlib.ticker import FuncFormatter, FormatStrFormatter, FixedLocator

from plot_libs import *
import MLS_lib as MLS

############
#gas     = ['Temperature','SO2','H2O','O3']
gas     = ['Temperature','H2O','O3'] 
gasname = dict()
gasname['Temperature'] = 'T'
gasname['H2O'] = r'H$_2$O'
gasname['SO2'] = r'SO$_2$'
gasname['O3'] = r'O$_3$'
Mgas    = dict()
Mgas['SO2'] = 64
Mgas['H2O'] = 18
Mgas['O3'] = 48
xres = dict()
alongres = dict()
xres['H2O'] = 7 #km
alongres['H2O'] = 165
xres['SO2'] = 6 #km
alongres['SO2'] = 170
xres['O3'] = 6 #km
alongres['O3'] = 170

m_gas = dict()
dprof_gas = dict()
allprof_gas = dict()
zm_gas    = dict()
vars  = ['mass','DU','mVMR','peakp','useday','npixels','mass-bg']



#SO2/standard/'
#region  = [0,-165,-35,-5]
#plt_region = [0,195, -35,-5]
region  = [-180,180,-35,-5]
plt_region = [0,360, -35,-5]
days1   = np.arange(335,366)
n1      = len(days1)
#days    = np.append( days1, np.arange(1,32) )
#days    = np.arange(1,32)
days    = days1
yys     = np.empty(len(days))
#yys[:n1]  = 2021
#yys[n1:]  = 2022 #np.arange(13, 20)
yys[:]  = 2021
byys    = np.empty( len(days) )
byys[yys==2021] = 2017
byys[yys==2022] = 2018
vol_pos = [184.62, -20.57]


clats     = np.arange(-70,71,2)
#clats     = np.arange(region[2],region[3]+1,2)
lat_left  = clats - 1
lat_right = clats + 1


im_nday = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
mons  = np.arange(12)[11:]
 
for im in range(len(mons)):

 midx = mons[im]
 end_day = sum(im_nday[:midx+1])
 mmdate  = datetime.datetime( 2022, midx+1, 1)
 mmname  = mmdate.strftime('%b')
 print('Start '+mmname)
 
 if midx == 0:
    start_day = 1
 else:
    start_day = sum(im_nday[:midx])+1
 days = np.arange(start_day, end_day+1)
 print('days:', days)
 yys  = np.empty(len(days))
 yys[:] = 2021 
 zprof_gas = dict()
 levs_gas  = dict()
 for key in gas[1:]:
   m_gas[key] = dict.fromkeys(vars)
   zprof_gas[key] = np.array([])
   for var in vars:
      m_gas[key][var] = []
 m_gas['Temperature'] = dict()
 m_gas['Temperature']['npixels'] = []
 m_gas['Temperature']['useday'] = []
 
 # read data of each day in the region
 for i, iday in enumerate(days):
#for iday in [27]:
#   i = 40 

   itime = datetime.datetime.strptime('{:.0f}d{:0>3d}'.format(yys[i], iday), "%Yd%j")
   time_str = itime.strftime("%Y-%m-%d")

   for igas in gas:
      if igas == 'SO2':
         datadir = '/Dedicated/jwang-data/shared_satData/MLS/L2/'+igas+'/standard'
      elif igas == 'O3' or igas == 'Temperature':
         datadir = '/Dedicated/jwang-data/shared_satData/MLS/L2/'+igas+'/{:.0f}'.format(yys[i])   
      elif igas == 'H2O':
         datadir = '/Dedicated/jwang-data/shared_satData/MLS/L2/'+igas+'/v4.2'      
      else:
         datadir = '/Dedicated/jwang-data/shared_satData/MLS/L2/'+igas
         
      if igas == 'SO2':
         DU_max = 4
         VMR_max = 50
         DU_cho = 1.5
         ppb_max = 60
         ppb_min = -5
         sf      = 1e9
         unit    = 'ppbv'
      elif igas == 'H2O':
         DU_max = 1000
         VMR_max = 15
         DU_cho  = 300
         ppb_max = 20
         ppb_min = 0
         sf      = 1e6
         unit    = 'ppmv'   
      elif igas == 'O3':
         DU_max = 250
         VMR_max = 10
         DU_cho  = 100
         ppb_max = 15
         ppb_min = 0
         sf      = 1e6
         unit    = 'ppmv'         
    
      if igas == 'O3':
         uppres = 1
      else:
         uppres = 10
         
      filename = glob.glob( datadir + '/MLS-Aura_L2GP-'+igas+'_v*-*-*_{:.0f}d{:0>3d}.he5'.format(yys[i], iday) )
      if igas == 'H2O':
         QC = False
      else:
         QC = True      
      data     = MLS.read_ifile( filename[0], usetime=None, QC=QC )
      levs     = data['lev']
      
      if igas == 'Temperature':
         tropp0 = data['WMOtropp']
         Tprof0 = data['Temperature']
         levs_gas['Temperature'] = levs
         print(data.keys(), tropp0)
         print('Tprof0:', len(np.where(~np.isnan(Tprof0))[0]))
         #sys.exit()
         continue
       
      # select region
      idx      = MLS.select_region( region, data['lat'], data['lon'] )
      Mlons     = data['lon'][idx]
      Mlats     = data['lat'][idx]
      times     = data['time'][idx]
      gasdata   = data[igas][idx,:]
      gasdata[gasdata<0] = 0.0
      tropp0[tropp0 > 130] = np.nan
      tropp    = tropp0[idx]
      #print('tropopause:', tropp[0], gasdata[0,:])
      print(igas, data['lev'])
      
      
      if len(data['lon']) != len(tropp0):   
         print('npixel:', len(data['lon']), len(tropp0))  
         continue   

      # plot two layer gas vmr map
      fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8,5))
      axs       = axes.flatten()
      vmin = 4
      vmax = ppb_max - 2
      SO2  = data[igas][idx,:]*sf
      cmap = plt.cm.rainbow
      cmap.set_bad('w',1)
      
      if igas == 'O3':
         pres_cho = [38, 10]
         vmin = 0
      elif igas == 'H2O':   
         pres_cho = [68,21,4.6]
      else:
         pres_cho = [100,68,46]   
      
      
      # zonal mean profile 
      levs_gas[igas] = levs
      zm_tropp = [] 
      for iz in range(len(clats)):
            zidx      = ((data['lat'] > lat_left[iz]) & (data['lat'] <= lat_right[iz]))
            zm_prof   = np.nanmean( data[igas][zidx,:], axis=0 ) 
            zm_tropp.append( np.nanmean( tropp0[zidx] ) )
            if iz == 0 and len(zprof_gas[igas]) == 0:
               zprof_gas[igas] = zm_prof 
            else:
               zprof_gas[igas] = np.vstack((zprof_gas[igas],zm_prof))
               
      m_gas[igas]['useday'].append( i )
      m_gas[igas]['npixels'].append( len(idx) )
   

   
   print('len idx:', len(data['lon']), len(tropp0))   
   if len(data['lon']) == len(tropp0):      

      m_gas['Temperature']['useday'].append( i )
      m_gas['Temperature']['npixels'].append( len(idx) )
      print('Temperature:',len(np.where(~np.isnan(Tprof0[idx,:]))[0]) ) 
      
      for iz in range(len(clats)):
            zidx      = ((data['lat'] > lat_left[iz]) & (data['lat'] <= lat_right[iz]))
            zm_prof   = np.nanmean( Tprof0[zidx,:], axis=0 ) 
            if iz == 0 and i == 0:
               zprof_gas['Temperature'] = zm_prof 
            else:
               zprof_gas['Temperature'] = np.vstack((zprof_gas['Temperature'],zm_prof))

            
 with open('./data/MLS_gas_daily_mean_zm_v4.2noQC_'+mmname+'.json', 'w') as f:
   json.dump(m_gas, f)    

  
 # daily mean profile
 ave_prof = dict()
 for igas in gas: 
   nday = len(m_gas[igas]['useday'])
   ave_prof[igas] = np.reshape( zprof_gas[igas], (nday, len(clats), len(levs_gas[igas])) )
#print(ave_prof['H2O'][:,10:20,10:15])
 
 np.savez('./data/MLS_prof_daily_mean_zm_v4.2noQC_'+mmname+'.npz', H2O=ave_prof['H2O'], 
         Tprof=ave_prof['Temperature'], O3=ave_prof['O3'], lev_O3=levs_gas['O3'],
         lev_H2O=levs_gas['H2O'], lev_T=levs_gas['Temperature'], clats=clats) 
            
    
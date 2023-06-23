import os, sys
import datetime
from matplotlib.ticker import FuncFormatter, FormatStrFormatter, FixedLocator

sys.path.append('/Users/xchen224/anaconda3/envs/xcpy/lib/python3.7/site-packages/mylib/')
from plot_libs import *
import MLS_lib as MLS
import SAGE3_lib as Slib

############
#gas     = ['Temperature','SO2','H2O','O3']
gas     = ['Temperature','SO2','H2O']
gasname = dict()
gasname['Temperature'] = 'T'
gasname['H2O'] = r'H$_2$O'
gasname['SO2'] = r'SO$_2$'
gasname['O3'] = r'O$_3$'
Mgas    = dict()
Mgas['SO2'] = 64
Mgas['H2O'] = 18
Mair = 28.96
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
zprof_gas = dict()
allprof_gas = dict()
maxprof_gas = dict()
levs_gas  = dict()
zm_gas    = dict()
gasdata   =dict()
vars  = ['useday','npixels','np_high','np_low']
for key in gas[1:]:
   m_gas[key] = dict.fromkeys(vars)
   for var in vars:
      m_gas[key][var] = []
m_gas['Temperature'] = dict()
m_gas['Temperature']['npixels'] = []

usebg = True


#SO2/standard/'
#region  = [0,-165,-35,-5]
#plt_region = [0,195, -35,-5]
region  = [-180,180,-35,-5]
plt_region = [0,360, -35,-5]
days1   = np.arange(335,366)
n1      = len(days1)
#days    = np.append( days1, np.arange(1,32) )
#days    = days1
days    = np.arange(1,32)
yys     = np.empty(len(days))
#yys[:n1]  = 2021
#yys[n1:]  = 2022 #np.arange(13, 20)
yys[:]  = 2022
byys    = np.empty( len(days) )
byys[yys==2021] = 2017
byys[yys==2022] = 2018
vol_pos = [184.62, -20.57]


#clats     = np.arange(-70,1,2)
clats     = np.arange(region[2],region[3]+1,2)
lat_left  = clats - 1
lat_right = clats + 1

# read background RH
if usebg:
   bgdata = np.load('./data/MLS_RHprof_daily_mean_hemis_Dec.npz')
   bgRH   = bgdata['RHall']
   bgmRH  = np.nanmean( bgRH, axis=0 )
   bgsRH  = np.nanstd( bgRH, axis=0 )


   
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
 allprof_gas = dict()
 for key in gas[1:]:
   m_gas[key] = dict.fromkeys(vars)
   for var in vars:
      m_gas[key][var] = []
 m_gas['Temperature'] = dict()
 m_gas['Temperature']['npixels'] = []
 mprof_high = []
 mprof_low  = []
 didx = []

 # read data of each day in the region
 for i, iday in enumerate(days):
#for iday in [27]:
#   i = 40 

   itime = datetime.datetime.strptime('{:.0f}d{:0>3d}'.format(yys[i], iday), "%Yd%j")
   time_str = itime.strftime("%Y-%m-%d")

   for igas in gas:
      if igas == 'SO2':
         datadir = '/Dedicated/jwang-data/shared_satData/MLS/L2/'+igas+'/standard'
      elif igas == 'Temperature':
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
         Tprof00 = data['Temperature']
         levs_gas['Temperature'] = levs
         print(data.keys(), tropp0)
         print('Tprof0:', Tprof00.shape, len(np.where(~np.isnan(Tprof00))[0]))
         #sys.exit()
         continue
      else:
         levs_gas[igas] = levs   
       
      # select region
      idx      = MLS.select_region( region, data['lat'], data['lon'] )
      Mlons     = data['lon'][idx]
      Mlats     = data['lat'][idx]
      times     = data['time'][idx]
      gasdata[igas]   = data[igas][idx,:]
      #gasdata[gasdata<0] = 0.0
      tropp0[tropp0 > 130] = np.nan
      #tropp    = tropp0[idx]
      #print('tropopause:', tropp[0], gasdata[0,:])
      if igas == 'H2O':
         locidx = idx
        
      if abs(len(data['lon']) - len(tropp0)) > 1:   
         print('npixel:', len(data['lon']), len(tropp0))  
         continue   
      elif abs(len(data['lon']) - len(tropp0)) != 0:
         if igas == 'H2O':
            if len(data['lon']) < len(tropp0):
               tropp0 = tropp0[:-1] 
               Tprof0 = Tprof00[:-1]
            else:
               tropp0 = np.append(tropp0, tropp0[-1:])   
               Tprof0 = np.append(Tprof0, Tprof00[-1:]) 
         else:
            Tprof0 = Tprof00
      else:
         Tprof0 = Tprof00                 
      tropp    = tropp0[idx]      
       
               
      if (Tprof0.shape[0] != len(data['lon'])):
         if igas == 'H2O':
            print('The pixels of T do not match with '+ igas)
            continue
         else:
            print('The pixels of T do not match with '+ igas)   
      elif (Tprof0.shape[1] != data[igas].shape[1]):  
         if igas == 'H2O':
            sys.exit( 'The vertical levels of T and '+igas+' are not the same!' )
         else:
            print( 'The vertical levels of T and '+igas+' are not the same!' )   
         
   if gasdata['SO2'].shape[0] != gasdata['H2O'].shape[0]:
      print('SO2 and H2O have different locations at iday = ', iday)
      continue     
 
   # 1. remove H2O reaction with SO2
   # SO2 /H2O plume location
   alt0  = ((levs_gas['SO2'] <= 101) & (levs_gas['SO2'] >= 9.9))
   SO2alt = gasdata['SO2'][:,alt0]
   iloc = np.where( np.nanmax(SO2alt,axis=1) > 50*1e-9 )[0]
   print('iloc:', iloc, np.nanmax(SO2alt,axis=1)[iloc])
   
   # match levs
   useSO2   = np.full((gasdata['H2O'].shape[0], len(levs_gas['H2O'])), 0.0)
   if len(iloc) > 0:
     for ip in range(len(iloc)):
       pidx   = iloc[ip]
       useSO2[pidx,:] = MLS.intp_lev_ip( levs_gas['SO2'], levs_gas['H2O'], gasdata['SO2'][pidx,:] )
   useSO2[useSO2<0] = 0.0
   #print(useSO2[iloc,:])
   
   # remove
   restH2O = gasdata['H2O'] - useSO2  
   #restH2O[iloc,:] = gasdata['H2O'][iloc,:]     
   
   # 2. calculate RH from rest H2O
   RHdata = MLS.VMR2RH( restH2O, Tprof0[locidx,:], levs_gas['H2O'])   

   
   # plot gas profile
   SO2 = RHdata
   fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(3,3.5))
   npoint = SO2.shape[0]
   peakp  = np.full(npoint, np.nan)
   dayave = np.nanmean(SO2, axis=0)
   daystd = np.nanstd(SO2, axis=0)
   alt  = ((levs <= 51) & (levs >= 9.9)) # H2O
      #alt  = ((levs <= 201) & (levs >= 9.9))  # SO2
   uselev = levs_gas['H2O'][alt]
   np_low = 0
   np_high= 0
   for ip in range(npoint):
         maxv = np.nanmax(SO2[ip,alt])
         if ~np.isnan(maxv):  
           maxl = np.nanargmax(SO2[ip,alt])
           if (maxv > 6): #((maxv > 10) and (uselev[maxl] > 12)):
            np_high += 1
            axes.plot( SO2[ip,:], data['lev'], 'k-', linewidth=0.5 )
           else:
            np_low  += 1
            axes.plot( SO2[ip,:], data['lev'], color='lightgray', alpha=0.7, linewidth=0.5 )   
            
#          if ~np.isnan(maxv):
#             hidx = np.nanargmax(SO2[ip,alt])
#          
#             if (daystd[alt][hidx] > 3) and ((maxv - dayave[alt][hidx]) > daystd[alt][hidx]*3):
#                axes.plot( SO2[ip,:], data['lev'], 'k-', linewidth=0.5 )
#                peakp[ip] = uselev[hidx]
#             else:
#                axes.plot( SO2[ip,:], data['lev'], 'lightgray-', linewidth=0.5 )    
#          else:   
#             axes.plot( SO2[ip,:], data['lev'], 'k-', linewidth=0.5 )
      
          
      #axes.errorbar( dayave, data['lev'], xerr=daystd, yerr=None, color='w', alpha=0.4 )
                   
      #m_gas[igas]['peakp'].append( np.nanmean( peakp ) )    

   #axes.plot( , data['alt'], 'ro-', linewidth=1.5 ) 
   axes.plot( [0.0], [np.nan], 'k-', label='N = '+str(np_high))
   axes.plot( [0.0], [np.nan], color='lightgray', alpha=0.7, label='N = '+str(np_low))   
   axes.legend(frameon=False)
   axes.set_xlabel(gasname[igas]+' '+unit)
   axes.set_ylim([10,50])
   axes.invert_yaxis()
   axes.set_yscale('log')
      #axes.yaxis.set_minor_locator(FixedLocator([1,2,3,5,7,10,20,30,50,70,100]))
   axes.yaxis.set_minor_locator(FixedLocator([5,7,10,20,30,50,70,100,130,170,200]))
   axes.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
   axes.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
   axes.set_ylabel('Pressure (hPa)')  
   axes.set_xlim([0, 40])
   axes.set_title( time_str )

   plt.tight_layout() 
      #plt.savefig('img/MLS_RH_prof_{:.0f}d{:0>3d}.png'.format(yys[i], iday), format='png', dpi=300 )      
   plt.close()   
      
   # 3.    
   # mean profile for large values
   if usebg:
         fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(3,3.5))
         npoint = SO2.shape[0]
         alt  = ((levs_gas['H2O'] <= 51) & (levs_gas['H2O'] >= 9.9)) # H2O
         alt2  = ((levs_gas['H2O'] <= 101) & (levs_gas['H2O'] >= 0.99)) # H2O
         uselev = levs_gas['H2O'][alt]
         np_low = 0
         np_high= 0
         bgprof = bgmRH[alt] + bgsRH[alt]*10
         for ip in range(npoint):
              # 1. background +- std
#             delta = SO2[ip,alt] - bgprof
#             nl_high = len(np.where(delta > 0)[0])
#             if nl_high > 0:  
#               #print('high: ', delta, uselev) 
#               if np_high == 0:
#                  high_prof = SO2[ip:ip+1,:]
#               else:
#                  high_prof = np.vstack((high_prof, SO2[ip:ip+1,:])  ) 

            # 2. H2O max vmr > thresh
            maxv = np.nanmax(restH2O[ip,alt2]) * 1e6
            if (maxv > 10): #((maxv > 10) and (uselev[maxl] > 12)):
               if np_high == 0:
                 high_prof = SO2[ip:ip+1,:]
               else:
                 high_prof = np.vstack((high_prof, SO2[ip:ip+1,:])  ) 
               np_high += 1
               print('maxv:', maxv, i, iday)
               axes.plot( SO2[ip,:], levs_gas['H2O'], 'r-', linewidth=0.5 )
            else:
              if np_low == 0:
                 low_prof = SO2[ip:ip+1,:]
              else:
                 low_prof = np.vstack((low_prof, SO2[ip:ip+1,:])  ) 
              np_low  += 1
              axes.plot( SO2[ip,:], levs_gas['H2O'], color='lightgray', alpha=0.7, linewidth=0.5 )   
         
         print(np_high, np_low)
         m_gas[igas]['np_high'].append( np_high )
         m_gas[igas]['np_low'].append( np_low )
         
         if np_high > 0:
            print(high_prof.shape, np_high)   
            axes.plot( np.nanmean(high_prof,axis=0), levs_gas['H2O'], 'b-', label='mean for red')
         axes.plot( [0.0], [np.nan], 'r-', label='N = '+str(np_high))
         axes.plot( [0.0], [np.nan], color='lightgray', alpha=0.7, label='N = '+str(np_low))   
         axes.legend(frameon=False)
         axes.set_xlabel('RH (%)')
         axes.set_ylim([1,70])
         axes.invert_yaxis()
         axes.set_yscale('log')
      #axes.yaxis.set_minor_locator(FixedLocator([1,2,3,5,7,10,20,30,50,70,100]))
         axes.yaxis.set_minor_locator(FixedLocator([5,7,10,20,30,50,70,100,130,170,200]))
         axes.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
         axes.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
         axes.set_ylabel('Pressure (hPa)')  
         axes.set_xlim([0, 40])
         axes.set_title( time_str )

         plt.tight_layout() 
         if midx == 0:
            plt.savefig('img/MLS_RH_2prof_{:.0f}d{:0>3d}.png'.format(yys[i], iday), format='png', dpi=300 )      
         plt.close()
         
         # save mean anomaly profile
         print(low_prof)
         if i == 0:
           mprof_low  = np.nanmean(low_prof, axis=0)
         else:
           mprof_low  = np.vstack( (mprof_low, np.nanmean(low_prof, axis=0)) )
           
         if np_high > 0:
           print('high day:', iday, np_high, high_prof.shape)
           didx.append( i )
           if (len(mprof_high) == 0 ):
              mprof_high = np.nanmean(high_prof,axis=0)  
           else: 
              mprof_high = np.vstack( (mprof_high,  np.nanmean(high_prof,axis=0)) )  
           print( 'high RH prof:', iday, np.nanmean(high_prof,axis=0))    
         
           
#       # plot mean+-std gas profile
#       fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(3,3.5))
#       npoint = SO2.shape[0]
#       
#    #sys.exit()
#       peakp  = np.full(npoint, np.nan)
#       dayave = np.nanmean(SO2, axis=0)
#       daystd = np.nanstd(SO2, axis=0)
#       alt  = ((levs <= 51) & (levs >= 9.9)) # H2O
#       #alt  = ((levs <= 201) & (levs >= 9.9))  # SO2
#       uselev = levs[alt]
#       axes    = Slib.plt_sage_fillprof( axes, dayave, data['lev'], dayave - daystd, dayave + daystd,
#                     setaxis=False, color='r')            
#       axes.set_xlabel('RH (%)')
#       axes.set_ylim([10,50])
#       axes.invert_yaxis()
#       axes.set_yscale('log')
#       #axes.yaxis.set_minor_locator(FixedLocator([1,2,3,5,7,10,20,30,50,70,100]))
#       axes.yaxis.set_minor_locator(FixedLocator([5,7,10,20,30,50,70,100,130,170,200]))
#       axes.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
#       axes.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
#       axes.set_ylabel('Pressure (hPa)')  
#       axes.set_xlim([0, 20])
#       axes.set_title( time_str )
# 
#       plt.tight_layout() 
#       #plt.savefig('img/MLS_RH_mprof_std_{:.0f}d{:0>3d}.png'.format(yys[i], iday), format='png', dpi=300 )      
#       plt.close()   
      
      # zonal mean profile 
#       zm_tropp = [] 
#       RHall = MLS.VMR2RH( data[igas], Tprof0, levs)  
#       for iz in range(len(clats)):
#             zidx      = ((data['lat'] > lat_left[iz]) & (data['lat'] <= lat_right[iz]))
#             zm_prof   = np.nanmean( RHall[zidx,:], axis=0 ) 
#             zm_tropp.append( np.nanmean( tropp0[zidx] ) )
#             if iz == 0:
#                zprof_gas[igas] = zm_prof
#             else:
#                zprof_gas[igas] = np.vstack((zprof_gas[igas],zm_prof))
# 
# 
#       # plot zonal mean
#       fig2, axes = plt.subplots( nrows=1, ncols=1, figsize=(6,3) )
#       cmap = plt.cm.rainbow
#       xx, yy = np.meshgrid( clats, levs ) 
#       
#       if igas == 'H2O':
#          vmin = 0
#          vmax = 30
#       else:
#          vmin = 0  
#          vmax = ppb_max/2 
#          
#       delta  = (vmax - vmin)/10.   
#       levels = np.arange(vmin, vmax+delta, delta)   
#       
#       #img  = axes.pcolormesh( xx, yy, zprof_gas[igas].T, shading='nearest', cmap=cmap, vmin=vmin, vmax=ppb_max/2 ) 
#       img  = axes.contourf( xx, yy, zprof_gas[igas].T, levels, vmin=vmin, vmax=vmax, 
#                             cmap=plt.cm.get_cmap(cmap, len(levels) - 1), extend='max' )  
#       print('ext lat, max:', np.nanmax(zprof_gas[igas]) )
#       axes.contour(xx, yy, zprof_gas[igas].T, img.levels, colors='k', linewidths=0.75)
#       cbar = fig2.colorbar(img, ax=axes,shrink=0.98, pad=0.03, ticks=levels) 
#       if igas == 'H2O':
#          cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in levels])
#       else:   
#          cbar.ax.set_yticklabels(['{:.0f}'.format(x) for x in levels])
#       cbar.ax.set_ylabel( 'RH (%)' ) 
#       axes.set_ylim([1,100]) 
#       axes.invert_yaxis()  
#       axes.set_yscale('log')
#       axes.set_ylabel('Pressure (hPa)')   
#       axes.set_xticks(clats[::10])
#       axes.set_xlabel(r'Latitude ($\circ$)')   
#       axes.set_title(time_str+' '+igas)  
#       plt.tight_layout()
#       #plt.savefig( 'img/MLS_RH_zonalprof_{:.0f}d{:0>3d}.png'.format(yys[i], iday), dpi=300 )
#       plt.close()         
#       
       
   # save mean anomaly profile
   if i == 0:
       allprof_gas['H2O'] = RHdata
   else:
         #dprof_gas[igas] = np.vstack( (dprof_gas[igas], dm_prof) ) 
       allprof_gas['H2O'] = np.vstack( (allprof_gas['H2O'], RHdata) )              
            
#       if igas == 'H2O':
#          ip = np.nanargmax(gas_STL['mean_VMR'])
#          print('pressure:', levs)
#          print('H2O prof', ip, Mlons[ip], Mlats[ip], gasdata[ip,:], tropp[ip]) 
#          sys.exit()  
      
   # calculate gas mass using regional mean VMR
   m_gas['H2O']['useday'].append( i )
   m_gas['H2O']['npixels'].append( len(locidx) )
   m_gas['Temperature']['npixels'].append( len(locidx) )
      
   print('len idx:', len(data['lon']), len(tropp0))   
   
   # save mean anomaly profile
   if i == 0:
         #dprof_gas['Temperature'] = dm_prof
         allprof_gas['Temperature'] = Tprof0[locidx,:]
   else:
         #dprof_gas['Temperature'] = np.vstack( (dprof_gas['Temperature'], dm_prof) ) 
         allprof_gas['Temperature'] = np.vstack( (allprof_gas['Temperature'], Tprof0[locidx,:]) ) 
            

 for igas in ['H2O']: 
   #m_gas[igas]['zm'] = zm_gas[igas]
   with open('./data/MLS_RHI_daily_mean_hemis_v4.2noQCb_'+mmname+'.json', 'w') as f:
       json.dump(m_gas[igas], f)    
  
  
 # daily mean profile
 ave_prof = dict()
 std_prof = dict()
 for igas in ['H2O','Temperature']: 
  for idd in range(len(m_gas[igas]['npixels'])):
   startid = 0
   if idd > 0:
      startid = sum(m_gas[igas]['npixels'][:idd])   
   inp = m_gas[igas]['npixels'][idd]
   if idd == 0:
      ave_prof[igas] = np.nanmean( allprof_gas[igas][startid:startid+inp,:], axis=0 )
      std_prof[igas] = np.nanstd( allprof_gas[igas][startid:startid+inp,:], axis=0 )
   else:
      ave_prof[igas] = np.vstack((ave_prof[igas], np.nanmean( allprof_gas[igas][startid:startid+inp,:], axis=0 )))
      std_prof[igas] = np.vstack((std_prof[igas], np.nanstd( allprof_gas[igas][startid:startid+inp,:], axis=0 )))     
 
 print('mprof_high', mprof_high.shape)  
 if len(mprof_high.shape) == 1:
    mprof_high = np.reshape(mprof_high,(1,len(mprof_high)))   
  #if igas == 'Temperature':
  #   print(ave_prof[igas][:10,:] )     
 np.savez('./data/MLS_RHIprof_daily_mean_hemis_v4.2noQCb_'+mmname+'.npz', RHprof=ave_prof['H2O'], RHall=allprof_gas['H2O'],
         RHhigh=mprof_high, RHlow=mprof_low, highidx=didx,
         Tprof=ave_prof['Temperature'], lev_RH=levs_gas['H2O'], lev_T=levs_gas['Temperature']) 
            
    
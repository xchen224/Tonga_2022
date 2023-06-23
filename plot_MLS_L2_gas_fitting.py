import os, sys
import datetime
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('/Users/xchen224/anaconda3/envs/xcpy/lib/python3.7/site-packages/mylib/')
from plot_libs import *
import MLS_lib as MLS

############
gas     = ['Temperature','SO2','H2O'] #['SO2','H2O','O3']
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
zprof_gas = dict()
allprof_gas = dict()
levs_gas  = dict()
zm_gas    = dict()
vars  = ['mass','DU','mVMR','peakp','useday','npixels','mass-bg','mass-bg-mVMR','mVMR-bg','mcol_air']
for key in gas[1:]:
   m_gas[key] = dict.fromkeys(vars)
   dprof_gas[key] = np.array([])
   for var in vars:
      m_gas[key][var] = []
m_gas['Temperature'] = dict()
m_gas['Temperature']['npixels'] = []


#SO2/standard/'
#region  = [0,-165,-35,-5]
#plt_region = [0,195, -35,-5]
region  = [-180,180,-35,-5]
plt_region = [0,360, -35,-5]
days1   = np.arange(335,366)
n1      = len(days1)
days    = np.append( days1, np.arange(1,32) )
#days    = np.arange(32,60)
yys     = np.empty(len(days))
yys[:n1]  = 2021
yys[n1:]  = 2022 #np.arange(13, 20)
#yys[:]  = 2022
byys    = np.empty( len(days) )
byys[yys==2021] = 2017
byys[yys==2022] = 2018
vol_pos = [184.62, -20.57]


#clats     = np.arange(-70,1,2)


# read monthly mean fitting background in 2021-2022
data = np.load('./data/MLS_fitting_L3_prof_monthly_mean_all_vol.npz')
bg_vmr    = dict()
bg_zm     = dict() 
bg_lev    = dict()
if np.max(days) > 360:
   bg_vmr['H2O'] = data['regH2O'][:,:2]
   bg_vmr['SO2'] = data['regSO2'][:,:2]
   bg_zm['H2O']  = data['zmH2O'][:,:2]
   bg_zm['SO2']  = data['zmSO2'][:,:2]
else:
   bgid = 2
   bg_vmr['H2O'] = data['regH2O'][:,bgid]
   bg_vmr['SO2'] = data['regSO2'][:,bgid]
   bg_zm['H2O']  = data['zmH2O'][:,bgid]
   bg_zm['SO2']  = data['zmSO2'][:,bgid]
bg_lev['H2O'] = data['levH2O']
bg_lev['SO2'] = data['levSO2']
clats     = data['zmlat']
lat_left  = clats - 1
lat_right = clats + 1
print('background:', bg_vmr['H2O'].shape, np.nanmax(bg_vmr['H2O']), np.nanargmax(bg_vmr['H2O'], axis=0), bg_vmr['SO2'].shape,)
#sys.exit()

# read data of each day in the region
for i, iday in enumerate(days):
#for iday in [27]:
#   i = 40 

   itime = datetime.datetime.strptime('{:.0f}d{:0>3d}'.format(yys[i], iday), "%Yd%j")
   time_str = itime.strftime("%Y-%m-%d")

   for igas in gas:
      if igas == 'SO2':
         datadir = '/Dedicated/jwang-data/shared_satData/MLS/L2/'+igas+'/standard'
      elif (igas == 'O3') or (igas == 'Temperature'):
         datadir = '/Dedicated/jwang-data/shared_satData/MLS/L2/'+igas+'/{:.0f}'.format(yys[i])   
      elif igas == 'H2O':
         datadir = '/Dedicated/jwang-data/shared_satData/MLS/L2/'+igas+'/v4.2'   
      else:
         datadir = '/Dedicated/jwang-data/shared_satData/MLS/L2/'+igas
         
      if igas == 'SO2':
         DU_max = 4
         VMR_max = 50
         DU_cho = 1.5
         ppb_max = 100
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
    
      if (igas == 'O3') or (igas == 'H2O'):
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
      #print(igas, data['lev'])
      
      
      if len(data['lon']) != len(tropp0):   
         print('npixel:', len(data['lon']), len(tropp0))  
         continue   

      # plot two layer gas vmr map
      fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,4))
      axs       = axes.flatten()
      vmin = 4
      vmax = ppb_max - 2
      SO2  = data[igas][idx,:]*sf
      cmap = plt.cm.rainbow
      cmap.set_bad('w',1)
      
      if igas == 'O3':
         pres_cho = [38, 10]
         vmin = 0
      else:   
         pres_cho = [100, 68]
      
      for il in range(len(pres_cho)):
      
         ipres = pres_cho[il]
         ilev   = np.argmin( np.abs(levs - ipres) )

         m, ms = MLS.plot_location_map( Mlons, Mlats, SO2[:,ilev], fig=fig, 
                              ax=axs[il], region=plt_region, cbar=True, alpha=1.0,
                              vmin=vmin, vmax=vmax, markersize=10, lon_0=180,cmap=cmap,
                              deltalat=20, deltalon=40,
                              title=time_str+ ' ' + igas + ' at {:.0f}hPa'.format(levs[ilev]), 
                              cbar_ylabel=  'VMR ('+unit+')')    
         x, y = m(vol_pos[0], vol_pos[1])  # transform coordinates
         axs[il].scatter(x, y, 15, marker='*', color='k')                                        
                                          
      plt.tight_layout() 
      if igas == 'O3':
         plt.savefig('img/MLS_'+igas+'_ilevmap_{:.0f}d{:0>3d}.png'.format(yys[i], iday), format='png', dpi=300 )      
      plt.close()   

      # plot gas profile
      fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4,3.5))
      npoint = SO2.shape[0]
      
   #sys.exit()
      peakp  = np.full(npoint, np.nan)
      dayave = np.nanmean(SO2, axis=0)
      daystd = np.nanstd(SO2, axis=0)
      alt  = ((levs <= 101) & (levs >= 9.9))
      uselev = levs[alt]
      for ip in range(npoint):
         maxv = np.nanmax(SO2[ip,alt])  
         if ~np.isnan(maxv):
            hidx = np.nanargmax(SO2[ip,alt])
         
            if (daystd[alt][hidx] > 3) and ((maxv - dayave[alt][hidx]) > daystd[alt][hidx]*3):
               axes.plot( SO2[ip,:], data['lev'], 'r-', linewidth=0.75 )
               peakp[ip] = uselev[hidx]
            else:
               axes.plot( SO2[ip,:], data['lev'], 'k-', linewidth=0.5 )    
         else:   
            axes.plot( SO2[ip,:], data['lev'], 'k-', linewidth=0.5 )
          
      axes.errorbar( dayave, data['lev'], xerr=daystd, yerr=None, color='w', alpha=0.4 )
                   
      m_gas[igas]['peakp'].append( np.nanmean( peakp ) )    

   #axes.plot( , data['alt'], 'ro-', linewidth=1.5 ) 
      axes.set_xlabel(igas+' '+unit)
      axes.set_ylabel('Pressure (hPa)') 
      axes.set_ylim([10,100])
      axes.invert_yaxis()
      axes.set_xlim([ppb_min, ppb_max*2])
      axes.set_title( time_str )

      plt.tight_layout() 
      if igas == 'O3':
         plt.savefig('img/MLS_'+igas+'_prof_{:.0f}d{:0>3d}.png'.format(yys[i], iday), format='png', dpi=300 )      
      plt.close()   
      
      # zonal mean profile 
      zm_tropp = [] 
      for iz in range(len(clats)):
            zidx      = ((data['lat'] > lat_left[iz]) & (data['lat'] <= lat_right[iz]))
            zm_prof   = np.nanmean( data[igas][zidx,:], axis=0 ) 
            zm_tropp.append( np.nanmean( tropp0[zidx] ) )
            if iz == 0:
               zprof_gas[igas] = zm_prof * sf
            else:
               zprof_gas[igas] = np.vstack((zprof_gas[igas],zm_prof* sf))           
               
      # plot zonal mean
      fig2, axes = plt.subplots( nrows=1, ncols=1, figsize=(6,3) )
      cmap = plt.cm.rainbow
      xx, yy = np.meshgrid( clats, levs ) 
      
      if igas == 'H2O':
         vmin = 3
      else:
         vmin = 0   
         
      delta  = (ppb_max/2 - vmin)/10.   
      levels = np.arange(vmin, ppb_max/2+delta, delta)   
      
      #img  = axes.pcolormesh( xx, yy, zprof_gas[igas].T, shading='nearest', cmap=cmap, vmin=vmin, vmax=ppb_max/2 ) 
      img  = axes.contourf( xx, yy, zprof_gas[igas].T, levels, vmin=vmin, vmax=ppb_max/2, cmap=plt.cm.get_cmap(cmap, len(levels) - 1) )  
      print('ext lat, max:', np.nanmax(zprof_gas[igas]) )
      axes.contour(xx, yy, zprof_gas[igas].T, img.levels, colors='k', linewidths=0.75)
      cbar = fig2.colorbar(img, ax=axes,shrink=0.98, pad=0.03, ticks=levels) 
      if igas == 'H2O':
         cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in levels])
      else:   
         cbar.ax.set_yticklabels(['{:.0f}'.format(x) for x in levels])
      cbar.ax.set_ylabel( 'VMR ('+unit+')' ) 
      axes.set_ylim([10,100]) 
      axes.invert_yaxis()  
      axes.set_yscale('log')
      axes.set_ylabel('Pressure (hPa)')   
      axes.set_xticks(clats[::10])
      axes.set_xlabel(r'Latitude ($\circ$)')   
      axes.set_title(time_str+' '+igas)  
      plt.tight_layout()
      if igas == 'O3':
         plt.savefig( 'img/MLS_'+igas+'_zonalprof_{:.0f}d{:0>3d}.png'.format(yys[i], iday), dpi=300 )
      plt.close()         
      
      # plot gas peak pressure
      if len(np.where(np.isnan(peakp)==False)[0]) > 0: 
         fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7,2))
         vmin = np.min(uselev)
         vmax = np.max(uselev)
         cmap = plt.cm.rainbow
         cmap.set_bad('w',1)

         m, ms = MLS.plot_location_map( Mlons, Mlats, peakp, fig=fig, 
                              ax=axes, region=plt_region, cbar=True, alpha=1.0,
                              vmin=vmin, vmax=vmax, markersize=10, lon_0=180,cmap=plt.cm.get_cmap('Set1', len(uselev)),
                              deltalat=20, deltalon=40,
                              title=time_str+ ' ' + igas + ' pressure with peak VMR',
                              cbar_ylabel=  'Pressure (hPa)')    
         x, y = m(vol_pos[0], vol_pos[1])  # transform coordinates
         axs[il].scatter(x, y, 15, marker='*', color='k')                                        
                                          
         plt.tight_layout() 
         #plt.savefig('img/MLS_'+igas+'_peakp_{:.0f}d{:0>3d}.png'.format(yys[i], iday), format='png', dpi=300 )      
         plt.close()   


      # ----------------------------------
      #inputvmr = gasdata - np.tile( bg_vmr[igas], (gasdata.shape[0],1) )
      inputvmr0 = gasdata
      area    = xres[igas] * alongres[igas] * 1e6
      gas_STL0 = MLS.cal_gas_STmass( inputvmr0, levs, tropp, area=area, Mgas=Mgas[igas], uppres=uppres )
      print(igas, 'DU', np.nanmax(gas_STL0['col_DU']), 'VMR', np.nanmax(gas_STL0['mean_VMR']), 
            'mass', gas_STL0['mass'] )
            
      if yys[i] == 2021:
         bg_prof = bg_vmr[igas][:,0]
      elif (yys[i] == 2022) and (len(days) > 40):
         bg_prof = bg_vmr[igas][:,1]
      else:
         bg_prof = bg_vmr[igas]
         
      use_prof = np.full(len(levs), np.nan) 
      for il0 in range(len(bg_lev[igas])):   
         bgilev = bg_lev[igas][il0]
         for il1 in range(len(levs)):
            ilev = levs[il1]
            if np.abs(bgilev - ilev) < 1e-4:
               use_prof[il1] = bg_prof[il0] / sf
               break
               
      #print(use_prof, bg_prof, bg_lev[igas], levs)
      #sys.exit()                  
            
      inputvmr = data[igas][idx,:] - np.tile( use_prof, (gasdata.shape[0],1) )      
      dm_prof  = np.nanmean(inputvmr,axis=0)
      #print(use_prof, data[igas][idx,:])
      #usebgvmr = bg_vmr[igas][i,:]
      #usebgvmr[usebgvmr < 0] = 0.0
      #usevmr   = gasdata - np.tile( usebgvmr, (gasdata.shape[0],1) ) 
      #usevmr[usevmr < 0] = 0.0
      gas_STL  = MLS.cal_gas_STmass( inputvmr, levs, tropp, area=area, Mgas=Mgas[igas], uppres=uppres )
            
      # save mean anomaly profile
      if len(dprof_gas[igas]) == 0:
         dprof_gas[igas] = dm_prof
         levs_gas[igas]  = levs
         allprof_gas[igas] = data[igas][idx,:]
      else:
         dprof_gas[igas] = np.vstack( (dprof_gas[igas], dm_prof) ) 
         allprof_gas[igas] = np.vstack( (allprof_gas[igas], data[igas][idx,:]) )              
            
#       if igas == 'H2O':
#          ip = np.nanargmax(gas_STL['mean_VMR'])
#          print('pressure:', levs)
#          print('H2O prof', ip, Mlons[ip], Mlats[ip], gasdata[ip,:], tropp[ip]) 
#          sys.exit()  
      
      # calculate gas mass using regional mean VMR
      R_earth = 6.371e6 #m
      region_area = 2*np.pi*(R_earth**2)*(np.sin(region[3]*np.pi/180)-np.sin(region[2]*np.pi/180)) * (plt_region[1] - plt_region[0])/360.
      gas_mass = np.nanmean(gas_STL['col_conc']) * Mgas[igas] * 1e-12 * region_area #- bg_mass[igas][i]
      if gas_mass < 0:
         gas_mass = 0.0
      gas_mass0 = np.nanmean(gas_STL0['col_conc']) * Mgas[igas] * 1e-12 * region_area #- bg_mass[igas][i]
      if gas_mass0 < 0:
         gas_mass0 = 0.0   
      mass_vmr =  np.nanmean(gas_STL['mean_VMR']) * np.nanmean(gas_STL0['col_air']) * region_area * Mgas[igas] * 1e-12
      if mass_vmr < 0:
         mass_vmr = 0.0        
      
      # save mean
      m_gas[igas]['mass'].append( gas_mass0 )
      m_gas[igas]['mass-bg'].append( gas_mass )
      m_gas[igas]['mass-bg-mVMR'].append( mass_vmr )
      m_gas[igas]['DU'].append( np.nanmean(gas_STL0['col_DU']) )
      m_gas[igas]['mVMR'].append( np.nanmean(gas_STL0['mean_VMR']) )
      m_gas[igas]['mVMR-bg'].append( np.nanmean(gas_STL['mean_VMR']) )
      m_gas[igas]['mcol_air'].append( np.nanmean(gas_STL0['col_air']) )
      m_gas[igas]['useday'].append( i )
      m_gas[igas]['npixels'].append( len(idx) )
   
      # plot strat. gas column map
      fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,4))
      axs       = axes.flatten()
      vmin = 0
      vmax = DU_max
      cmap = plt.cm.rainbow
      cmap.set_bad('w',1)

      Mlons[Mlons < 0] += 360

      m, ms = MLS.plot_location_map( Mlons, Mlats, gas_STL0['col_DU'], fig=fig, 
                              ax=axs[0], region=plt_region, cbar=True, alpha=1.0,
                              vmin=vmin, vmax=vmax, markersize=10, lon_0=180,cmap=cmap,
                              deltalat=20, deltalon=40,
                              title=time_str + ' '+igas + '\n Strat. mass = {:.4f}Tg'.format(gas_mass), 
                              cbar_ylabel= igas+' STL (DU)')    
      x, y = m(vol_pos[0], vol_pos[1])  # transform coordinates
      axs[0].scatter(x, y, 15, marker='*', color='k')                                        
      
      vmin = 0
      vmax = VMR_max

      m, ms = MLS.plot_location_map( Mlons, Mlats, gas_STL0['mean_VMR']*sf, fig=fig, 
                              ax=axs[1], region=plt_region, cbar=True, alpha=1.0,
                              vmin=vmin, vmax=vmax, markersize=10, lon_0=180,cmap=cmap,
                              deltalat=20, deltalon=40,
                              title=time_str+ ' ' + igas, 
                              cbar_ylabel= igas+' VMR ('+unit+')')    
      x, y = m(vol_pos[0], vol_pos[1])  # transform coordinates
      axs[1].scatter(x, y, 15, marker='*', color='k')                                        
      plt.tight_layout() 
      if igas == 'O3':
         plt.savefig('img/MLS_'+igas+'_STLmap_{:.0f}d{:0>3d}.png'.format(yys[i], iday), format='png', dpi=300 )      
      plt.close()   
   
   print('len idx:', len(data['lon']), len(tropp0))               

for igas in gas: 
   with open('./data/MLS_'+igas+'_fitting_daily_mean_hemis_v4.2noQC_2mons.json', 'w') as f:
       json.dump(m_gas[igas], f)    

   
# save anomaly daily mean profile
np.savez('./data/MLS_fitting_anomaly_prof_daily_mean_hemis_v4.2noQC_2mons.npz', SO2=dprof_gas['SO2'], H2O=dprof_gas['H2O'], 
         lev_SO2=levs_gas['SO2'], lev_H2O=levs_gas['H2O']) 

print('anomaly:', m_gas['SO2']['mass-bg-mVMR'])
print(m_gas['H2O']['mass-bg-mVMR'])       
  
         
# plot mean profile
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(3.5,4))
# colors    = plt.cm.jet( np.linspace(0,1,len(days)) )
# for ii, iday in enumerate(days):
#    axes.plot( m_prof[ii], data['alt'], c=colors[ii], linewidth=1.5, label='Jan. '+str(iday) ) 
# axes.set_xlabel('SO2 ppbv')
# axes.set_ylabel('Pressure (hPa)') 
# axes.set_ylim([10,150])
# axes.invert_yaxis()
# axes.set_xlim([-1, 10])
# plt.tight_layout() 
# plt.savefig('img/MLS_NRT_meanprof_5d.png', format='png', dpi=300 )      
# plt.close()     
#    
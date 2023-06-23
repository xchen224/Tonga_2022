import os, sys
import numpy as np
import h5py
from netCDF4 import Dataset
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.basemap import Basemap, maskoceans
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from matplotlib.ticker import MultipleLocator

import mylib.resample_parallel as res

#
# read_ompslim(): Read OMPS-Limb aerosol extinction profile
# revise for v2.0 OMPS-LP-AER L2 data by Xi Chen 2020-10-22
def read_ompslim_v2( filename, bound=None, QC=False ):

   # filename: the OMPS-LP filename
   #    bound: Boundary = 'None' or [lon1,lon2,lat1,lat2]

   # open the h5 file
   h5 = h5py.File(filename,'r')

   # Variables to read 
   grp0 = 'ProfileFields'
   var0 = ['Altitude','Wavelength']
   grp1 = 'GeolocationFields'
   var1 = ['Longitude', 'Latitude', 'CloudHeight', 'SingleScatteringAngle', 'SolarZenithAngle','ResidualFlag'] #'Reflectance675', 
   grp2 = 'ProfileFields'
   var2 = ['RetrievedExtCoeff', 'ExtCoeffError', 'Residual'] #'Residual675']
   var3 = ['TotalColumnStratosphericAerosol']
   grp3 = 'AncillaryData'
   var4 = ['TropopauseAltitude']

   data0 = dict()
   for var in var0:
         data0[var] = h5[grp0][var][:]
   for var in var1:
         data0[var] = h5[grp1][var][:]
   for var in var2+var3:
         data0[var] = h5[grp2][var][:] 
   for var in var4:
         data0[var] = h5[grp3][var][:]        

   # Quality Control: 
   # https://snpp-omps.gesdisc.eosdis.nasa.gov/data/SNPP_OMPS_Level2/OMPS_NPP_LP_L2_AER_DAILY.2/doc/README.OMPS_NPP_LP_L2_AER_DAILY_v2.1.pdf
   if QC:
      data0['RetrievedExtCoeff'][data0['RetrievedExtCoeff'] < 1e-5] = -999.0
      maskflag = (data0['ResidualFlag'] != 0)
      for il in range(len(data0['Altitude'])):  
         data0['RetrievedExtCoeff'][:,:,:,il][maskflag] = -999.0
      data0['TotalColumnStratosphericAerosol'][maskflag] = -999.0


   data = dict() 
   # if no bound set
   if ( bound is None ):
      for var in data0.keys():
         data[var] = data0[var]

   # else, only read data in the boundary [lon1,lon2,lat1,lat2]
   else:
      # check elements of 'bound'
      if ( len(bound) != 4 ):
         sys.exit("read_ompslim: Error 1, wrong bound (ompslimb.py)") 

      # Read lat, lon
      lon1, lon2 = bound[0], bound[1]
      lat1, lat2 = bound[2], bound[3]
      #print(lon1, lon2,lat1, lat2)
      lon = data0[var1[0]][:]
      lat = data0[var1[1]][:]

      # Find the index within the bound
      if lon1 < lon2:
         idx = np.where( (lon > lon1) & (lon < lon2)  & (lat > lat1) & (lat < lat2) )
      else:
         idx = np.where( ((lon > lon1) | (lon < lon2))  & (lat > lat1) & (lat < lat2) )   

      # Read grp1
      data = dict()
      for var in var0:
         data[var] = data0[var][:]
      for var in var1:
         data[var] = data0[var][:][idx]
         
      # Read group 3
      for var in var4:
         data[var] = data0[var][:][idx]   
      
      # Read group 2
      for var in var2:
         for j in range(len(data['Wavelength'])):
            for i in range(len(data['Altitude']) ):
               if i==0 and j==0:
                  xx = data0[var][:,:,j,i][idx]
               else:
                  xx = np.vstack( (xx,data0[var][:,:,j,i][idx]) ) 
         data[var] = np.transpose(xx)
         
      for var in var3:
         for j in range(len(data['Wavelength'])):
            if j==0:
                  xx = data0[var][:,:,j][idx]
            else:
                  xx = np.vstack( (xx,data0[var][:,:,j][idx]) ) 
         data[var] = np.transpose(xx)   
         

   # Close the h5 file
   h5.close()

   #for key in data.keys():
   #  print(key, data[key][:].shape)

   return data

#####
def read_ompsnm_l2( filename, bound=None ):

   # filename: the OMPS-NMSO2-L2 filename
   #    bound: Boundary = 'None' or [lon1,lon2,lat1,lat2]

   # open the h5 file
   h5 = h5py.File(filename,'r')

   # Variables to read 
   grp1 = 'GeolocationData'
   var1 = ['Longitude','Latitude','LatitudeCorner','LongitudeCorner','UTC_CCSDA_A'] #'Reflectance675', 
   grp2 = 'ScienceData'
   var2 = ['ColumnAmountSO2_STL','PixelQualityFlags'] #centered at 16 km

   data = dict()
   for var in var1:
      data[var] = h5[grp1][var][:]
   for var in var2:
      data[var] = h5[grp2][var][:]  
      
   # quality control
   badip = (data['PixelQualityFlags'] > 0)
   data['ColumnAmountSO2_STL'][badip] = np.nan  

   # region [lon1,lon2,lat1,lat2]
   if bound is not None:
      # check elements of 'bound'
      if ( len(bound) != 4 ):
         sys.exit("read_ompslim: Error 1, wrong bound (ompslimb.py)") 

      # Read lat, lon
      lon1, lon2 = bound[0], bound[1]
      lat1, lat2 = bound[2], bound[3]
      #print(lon1, lon2,lat1, lat2)
      lon = h5[grp1][var1[0]][:]
      lat = h5[grp1][var1[1]][:]
      #print('lon, lat', lon.shape, lat.shape)

      # Find the index within the bound
      if lon1 < lon2:
         idx = ( (lon > lon1) & (lon < lon2)  & (lat > lat1) & (lat < lat2) )
      else:   
         idx = ( ((lon > lon1) | (lon < lon2))  & (lat > lat1) & (lat < lat2) )
         
      #print('idx', idx, lon[idx], lat[idx])  

      data['index'] = idx

   # Close the h5 file
   h5.close()

   #for key in data.keys():
   #  print(key, data[key][:].shape)

   return data

#####
def read_ompsnm_pca( filename, bound=None ):

   # filename: the OMPS-NMSO2-PCA filename
   #    bound: Boundary = 'None' or [lon1,lon2,lat1,lat2]

   # open the h5 file
   h5 = h5py.File(filename,'r')

   # Variables to read 
   grp1 = 'GEOLOCATION_DATA'
   var1 = ['Longitude', 'Latitude','LatitudeCorner','LongitudeCorner', 'UTC_CCSDS_A'] #'Reflectance675', 
   grp2 = 'SCIENCE_DATA'
   var2 = ['ColumnAmountSO2_STL','ColumnAmountSO2_TRM','ColumnAmountSO2_TRU','Flag_SO2'] #centered at 18 km

   data = dict()
   for var in var1:
      data[var] = h5[grp1][var][:]
   for var in var2:
      data[var] = h5[grp2][var][:]  
      
   # quality control
   badip = (data['Flag_SO2'] == 0)
   for key in data.keys():
     if 'ColumnAmountSO2' in key:
        data[key][badip] = np.nan  

   # region [lon1,lon2,lat1,lat2]
   if bound is not None:
      # check elements of 'bound'
      if ( len(bound) != 4 ):
         sys.exit("read_ompslim: Error 1, wrong bound (ompslimb.py)") 

      # Read lat, lon
      lon1, lon2 = bound[0], bound[1]
      lat1, lat2 = bound[2], bound[3]
      #print(lon1, lon2,lat1, lat2)
      lon = h5[grp1][var1[0]][:]
      lat = h5[grp1][var1[1]][:]
      #print('lon, lat', lon.shape, lat.shape)

      # Find the index within the bound
      if lon1 < lon2:
         idx = ( (lon > lon1) & (lon < lon2)  & (lat > lat1) & (lat < lat2) )
      else:   
         idx = ( ((lon > lon1) | (lon < lon2))  & (lat > lat1) & (lat < lat2) )
         
      #print('idx', idx, lon[idx], lat[idx])  

      data['index'] = idx

   # Close the h5 file
   h5.close()

   #for key in data.keys():
   #  print(key, data[key][:].shape)

   return data
   
####
def plot_masked_data( lon, lat, data,
                      proj='eqdc', area_thresh=1000,
                      mapwidth=3000000, mapheight=1800000,
                      maplat0=46.5, maplon0=-123.5,
                      region=[-180,180,-90,90], laea=None,
                      ccoast='0.2', cgrid='0.2', lon_0=0,
                      grid=True, lonlabel=[0,0,0,1], latlabel=[1,0,0,0],
                      cbar=True, cbar_ticks=None, fig=0, cbar_ylabel=None,
                      tracks=None, ctrack='red', cbar_ticklabel=None,
                      title='',  cb_ext=False, delta_lat=10, delta_lon=20,
                      bstate=True,
                      **kwargs ):

   #-- determine if there is an ax argument
   if 'ax' in kwargs.keys():
      ax = kwargs['ax']
   else:
      ax = None

   #-- Start a map projection
   if proj == 'cyl':
      m = Basemap( ax=ax, projection=proj, resolution='h',
                   area_thresh=area_thresh, lon_0=lon_0,
                   llcrnrlat=region[2], urcrnrlat=region[3],
                   llcrnrlon=region[0], urcrnrlon=region[1] )
   elif proj == 'eqdc':
      m = Basemap( ax=ax, projection=proj, resolution='h',
                   area_thresh=area_thresh,
                  width=mapwidth,height=mapheight,
                   lat_0=maplat0,lon_0=maplon0 )
   elif proj == 'ortho':
      m = Basemap( ax=ax, projection=proj, resolution='i',
                   area_thresh=area_thresh,
                   lat_0=maplat0,lon_0=maplon0 )
   elif proj == 'splaea' :
      if laea is None:
         laea    = {'lat_ts':-60,'lat_0':-90,'lon_0':-90, 'bdlat':-0.1}
      m = Basemap( ax=ax, projection=proj, resolution='h', 
                   round=True, boundinglat=laea['bdlat'], \
                   lat_ts=laea['lat_ts'], lat_0=laea['lat_0'], lon_0=laea['lon_0'] ) 
   elif proj == 'laea':
      m = Basemap( ax=ax, projection=proj, resolution='h', 
                   width=laea['width'],height=laea['height'], \
                   lat_ts=laea['lat_ts'], lat_0=laea['lat_0'], lon_0=laea['lon_0'] ) 

   #-- draw costlines and map grid
   m.drawmapboundary(fill_color = 'w')
   m.drawcoastlines(linewidth=0.25,color=ccoast)
   m.drawcountries(linewidth=0.1,color='gray')
   if bstate:
      m.drawstates(linewidth=0.1,color='gray')
   if grid:
      if proj == 'cyl':
         m.drawparallels( np.arange(-90,90,delta_lat), color=cgrid, labels=latlabel, fontsize=10, linewidth=0.4 )
         if lon_0 == 0:
            m.drawmeridians( np.arange(-180,180,delta_lon), color=cgrid, labels=lonlabel, fontsize=10, linewidth=0.4 )
         else:
            m.drawmeridians( np.arange(0,360,delta_lon), color=cgrid, labels=lonlabel, fontsize=10, linewidth=0.4 )   
      else:
         m.drawparallels( np.arange(-90,90,delta_lat), color=cgrid, labels=latlabel, linewidth=0.2 )
         m.drawmeridians( np.arange(-180,180,delta_lon), color=cgrid, labels=lonlabel, linewidth=0.2 )

   #-- plot data on the map
   msk_lon = np.array(lon)
   msk_lat = np.array(lat)
   msk_dat = np.ma.masked_invalid(data)
   
   invalid0 = np.where( np.isnan(lat) )[0]
   invalid1 = np.where( np.isnan(lon) )[0]
   
   if (len(invalid0) > 0) or (len(invalid1) > 0):
      print( 'There are NAN values in lat or lon!!!')
      cm = m.pcolor(np.ma.masked_invalid(msk_lon), np.ma.masked_invalid(msk_lat), msk_dat, latlon=True, **kwargs)
   else:   
      cm = m.pcolormesh(msk_lon, msk_lat, msk_dat, latlon=True, **kwargs)

   #-- draw satellite tack
   if tracks is not None:
     for it in range(len(tracks)):
       xt, yt = m(tracks[it]['lon'][:], tracks[it]['lat'][:])
       m.plot( xt, yt, color=ctrack, linewidth=1.0 )

   #-- draw color bar
   if cbar:
      if cb_ext:
         cb = m.colorbar(cm,location='right',extend='min', shrink=0.98, pad=0.03,
	                    extendfrac=0.1, extendrect=True, ticks=cbar_ticks)
      else:                  
         cb = m.colorbar(cm,location='right',shrink=0.98, pad=0.03, aspect=100,ticks=cbar_ticks)
      if cbar_ticklabel is not None:  
         cb.ax.set_yticklabels(cbar_ticklabel) #, fontsize=14
      if cbar_ylabel is not None:
         cb.ax.set_ylabel(cbar_ylabel,rotation=-90, va="bottom") #, fontsize=14   
 
   #-- show image
   if 'ax' in kwargs.keys():
      ax.set_title( title )
   else: 
      plt.title( title )
   #plt.show()

   return m, cm
   
######
def footprint_leap( ax, lons, lats, laea=None, alpha=0.5, cmap=None, shading='auto', \
                    color_coded=False, ccolors=None, cminmax=None, scat=True, \
                    cat_lons=None, cat_lats=None, latlabel=[0,0,0,0], lonlabel=[0,0,0,0],
                    ):

   '''
   Function to plot OMPS-LP data on map with Lambert Azimuthal Equal Area Projection.
   '''

   if laea is None:
      #laea = {'proj':'nplaea', 'lat_ts':60,'lat_0':90,'lon_0':-90, 'bdlat':30}
      laea = {'proj':'splaea', 'lat_ts':-60,'lat_0':-90,'lon_0':-90, 'bdlat':-0.1}

   # Draw the map
   if laea['proj'] == 'nplaea' or laea['proj'] == 'splaea' :
      m = Basemap( ax=ax, resolution='c', round=True, \
                   projection=laea['proj'], boundinglat=laea['bdlat'], \
                   lat_ts=laea['lat_ts'], lat_0=laea['lat_0'], lon_0=laea['lon_0'] )
   else:
      m = Basemap( ax=ax, resolution='c', \
                   projection=laea['proj'], width=laea['width'],height=laea['height'], \
                   lat_ts=laea['lat_ts'], lat_0=laea['lat_0'], lon_0=laea['lon_0'] )

   m.drawcoastlines(linewidth=0.6)
   m.drawparallels(np.arange(-80.,81.,20.),labels=latlabel, linewidth=0.4)
   m.drawmeridians(np.arange(0.,360.,60.),labels=lonlabel, linewidth=0.4)
   #m.fillcontinents(color='lightgray',lake_color='white', zorder=0)

   # Draw the OMPS data
   sx, sy = m(lons,lats)
   msk_lon = lons #np.ma.masked_invalid(lons)
   msk_lat = lats #np.ma.masked_invalid(lats)
   msk_dat = np.ma.masked_invalid(ccolors)
   #print('plot data:', msk_dat)

   if color_coded:
      if cmap is None:
         cmap=plt.cm.jet
         cmap.set_under("w")
      else:
         cmap=cmap   
         cmap.set_bad('w')
      
      if scat:

         cax = m.scatter( sx, sy, s=3, lw=0, c=ccolors, alpha=alpha, cmap=cmap, vmax=cminmax[1], vmin=cminmax[0] )
      else:
         cax = m.pcolormesh(msk_lon, msk_lat, msk_dat, latlon=True, shading=shading, cmap=cmap, vmax=cminmax[1], vmin=cminmax[0]) 
      # if colorbar:     
#          cb = fig.colorbar(cax, ticks=[0,cminmax[1]/2,cminmax[1]], orientation='horizontal')
#          cb.ax.set_xlabel(cbar_label)
   else:
      if scat:
         cax = m.scatter( sx, sy, s=3, lw=0)
      
      else:
         cax = m.pcolormesh(msk_lon, msk_lat, msk_dat, latlon=True, shading=shading, cmap=cmap, vmax=cminmax[1], vmin=cminmax[0]) 


   return cax, m


def footprint_on_map( ax, proj='eqdc', area_thresh=1000,
                      mapwidth=3000000, mapheight=1800000,
                      maplat0=46.5, maplon0=-123.5,
                      region=[-180,180,-90,90],
                      ccoast='0.2', cgrid='0.2', lon_0=0,
                      grid=True, lonlabel=[0,0,0,1], latlabel=[1,0,0,0],
                      delta_lat=10, delta_lon=20, title=None, nlegend=0,
                      ctitle=None, stitle=None, cbar_ticks=None, markersize=0,
                      bstate=True, line_lons=None, line_lats=None,
                      color_coded=False, ccolors=None, scat_lons=None, scat_lats=None,
                      cbar=False, cbar_ticklabel=None, cbar_ylabel=None, **kwargs  ):

      #-- Start a map projection
   if proj == 'cyl':
      m = Basemap( ax=ax, projection=proj, resolution='h',
                   area_thresh=area_thresh, lon_0=lon_0,
                   llcrnrlat=region[2], urcrnrlat=region[3],
                   llcrnrlon=region[0], urcrnrlon=region[1] )
   elif proj == 'eqdc':
      m = Basemap( ax=ax, projection=proj, resolution='h',
                   area_thresh=area_thresh,
                  width=mapwidth,height=mapheight,
                   lat_0=maplat0,lon_0=maplon0 )
   elif proj == 'ortho':
      m = Basemap( ax=ax, projection=proj, resolution='i',
                   area_thresh=area_thresh,
                   lat_0=maplat0,lon_0=maplon0 )

   #-- draw costlines and map grid
   m.drawmapboundary(fill_color = 'w')
   m.drawcoastlines(linewidth=0.25,color=ccoast)
   m.drawcountries(linewidth=0.1,color='gray')
   if bstate:
      m.drawstates(linewidth=0.1,color='gray')
   if grid:
      if proj == 'cyl':
         m.drawparallels( np.arange(-90,90,delta_lat), color=cgrid, labels=latlabel, fontsize=10, linewidth=0.4 )
         if lon_0 == 0:
            m.drawmeridians( np.arange(-180,180,delta_lon), color=cgrid, labels=lonlabel, fontsize=10, linewidth=0.4 )
         else:
            m.drawmeridians( np.arange(0,360,delta_lon), color=cgrid, labels=lonlabel, fontsize=10, linewidth=0.4 )   
      else:
         m.drawparallels( np.arange(-90,90,delta_lat), color=cgrid, labels=latlabel, linewidth=0.2 )
         m.drawmeridians( np.arange(-180,180,delta_lon), color=cgrid, labels=lonlabel, linewidth=0.2 )

   if ( (line_lons is not None) and (line_lats is not None) ): 
      sx, sy = m(line_lons, line_lats)
      m.plot(sx,sy,'bo-', markersize=3, alpha=0.7)
      
   ms = None
   if ( (scat_lons is not None) and (scat_lats is not None) ): 
      sx1, sy1 = m(scat_lons,scat_lats)
      #m.plot(sx1,sy1,'-r')
      if 'marker' in kwargs.keys():
         ms = m.scatter(sx1,sy1, markersize, **kwargs)
      else:   
         ms = m.scatter(sx1,sy1, **kwargs)    
      
      #-- draw color bar
      if cbar:
        cb = m.colorbar(ms,location='right',shrink=0.98, pad=0.03, ticks=cbar_ticks)
        if cbar_ticklabel is not None:  
           cb.ax.set_yticklabels(cbar_ticklabel) 
        if cbar_ylabel is not None:
           cb.ax.set_ylabel(cbar_ylabel,rotation=-90, va="bottom")   
      
      #-- legend for color
      if (nlegend >= 1) and (cbar is False):
         legend1 = ax.legend(*ms.legend_elements(num=5),
                    loc="upper left", title=ctitle)
         ax.add_artist(legend1)    
      
      if (nlegend == 2) and ('s' in kwargs.keys() or 'marker' in kwargs.keys()):
         l1 = ax.scatter([],[], s=2, linewidths=0, c=ms.cmap([0]))
         l2 = ax.scatter([],[], s=5, linewidths=0, c=ms.cmap([0]))
         l3 = ax.scatter([],[], s=10, linewidths=0, c=ms.cmap([0]))
         l4 = ax.scatter([],[], s=20, linewidths=0, c=ms.cmap([0]))
         

         labels = ['0.02', '0.05', '0.10', '0.20      '+stitle]

         leg = ax.legend([l1, l2, l3, l4], labels, ncol=4, frameon=True, fontsize='x-small',
                          loc = 9, bbox_to_anchor=(0.1, 1.1, 0.8, 0.2), )    
           
   ax.set_title(title)        
      
   return m, ms 

def draw_footprint( lons, lats, ax, bound=None, color_coded=False, ccolors=None, cminmax=None, cat_lons=None, cat_lats=None ):

   #fig = plt.figure( figsize=(8,6) )
   #ax1 = fig.add_subplot(111)
 
   if color_coded:
      cmap=plt.cm.jet
      cmap.set_under("k")
      colors = np.zeros(np.size(lats))
      
      cax = ax.scatter( lons, lats, s=20, c=ccolors, cmap=cmap, vmax=cminmax[1], vmin=cminmax[0] )
      #cbar = fig.colorbar(cax, ticks=[0,0.002,0.004,0.006], orientation='horizontal')
   else:
      ax.scatter( lons, lats, s=20)

   if ( bound is None ):
      latlim = [-90, 90]
      lonlim = [ -180, 180]
   else:
      latlim = bound[2:4]
      lonlim = bound[0:2]
    
   if ( (cat_lons is not None) and (cat_lats is not None) ):
      ax.plot(cat_lons,cat_lats,'-r')   

   ax.set_ylim( latlim )
   ax.set_xlim( lonlim )
   ax.set_aspect(1)
   #plt.tight_layout()
   #plt.savefig( 'img/omps_fp_'+imgtitle+'.png', dpi=200 )
   #plt.close()

def draw_profile( z, var, ax, style='--bo', linewidth=2, min_point=None):
 
   index_invalid = np.where(var<0)
   nvalid = np.size(var) - np.size(index_invalid)
   #print( nvalid )
   if ( (min_point is not None) and (nvalid < min_point ) ):
      var[:] = np.nan
      return False
   else:  
      var[index_invalid] = np.nan
      ax.plot(var,z, style, linewidth=linewidth, markersize=3)
      return True

    
############   
def plt_spleae_6bd_iday( ext3D, alt, lon, lat, bands, fname, saveplt=True):
      '''
      All six bands at different altitudes
      '''

      imgtitle = 'omps_spleae-' + fname.split('_')[-2]
      nb = len(bands)

      #fig = plt.figure( figsize=(10,10) )
      #ax1   = fig.add_axes([0.05,0.6,0.4,0.3])
      #ax1cb = fig.add_axes([0.10,0.57, 0.3,0.02])
      #ax2   = fig.add_axes([0.55,0.6,0.4,0.3])
      #ax2cb = fig.add_axes([0.60,0.57, 0.3,0.02])
      #ax3 = fig.add_axes([0.10,0.2,0.3,0.3])
#       if fname in fnames1:
#          data = omps.read_ompslim_v2( fpath1+fname, bound=bound )
#       else:
#          data = omps.read_ompslim_v2( fpath2+fname, bound=bound )  

      fig1, ax1 = plt.subplots( nrows=3, ncols=6, figsize=(15,6) )
      plt.subplots_adjust(left=0.04)
      # ax1   = fig.add_axes([0.05,0.6,0.4,0.3])
      axcb1 = fig1.add_axes([0.92,0.7, 0.01,0.25])
      # ax2   = fig.add_axes([0.55,0.6,0.4,0.3])
      axcb2 = fig1.add_axes([0.92,0.4, 0.01,0.25])
      axcb3 = fig1.add_axes([0.92,0.1, 0.01,0.25])
      # ax3 = fig.add_axes([0.10,0.2,0.3,0.3])
      
      # different altitudes
      idx3 = np.where( (alt<=25) & (alt>20) )[0]
      idx1 = np.where( (alt<=15) & (alt>10) )[0]
      idx2 = np.where( (alt<=20) & (alt>15) )[0]
      idx4 = np.where( (alt<=30) & (alt>25) )[0]
      aod1 = np.nanmean(ext3D[:,:,idx1], axis=2)
      aod2 = np.nanmean(ext3D[:,:,idx2], axis=2)
      aod3 = np.nanmean(ext3D[:,:,idx3], axis=2)
      aod4 = np.nanmean(ext3D[:,:,idx4], axis=2)

      #ax3.set_ylim([15,35])
      #ax3.set_xlim([0,0.0030])
      #ax3.set_ylabel('Altitude (km)')
      #ax3.set_xlabel(r'OMPS-LP ExtCoef (km$^{-1}$)')
      #ax3.ticklabel_format(style='sci',axis='x',scilimits=(0,0))

      for ib in range(nb):
         if ib == 0:
            latlabel = [0,0,0,0]
         lonlabel = [0,0,0,1]   
         cax1, m = footprint_leap( ax1[0,ib], \
                           lon, \
                           lat, \
                           laea=None, \
                           color_coded=True, \
                           latlabel = latlabel, lonlabel=lonlabel, \
                           ccolors=aod2[:,ib], \
                           cminmax=[0,0.005] )
      #fig.colorbar(cax1, cax=ax1cb, ticks=[0,0.0003,0.0006], orientation='horizontal')
      #ax1.set_title('ExtCoef (km$^{-1}$) between 20-25 km')

         cax2, m = footprint_leap( ax1[1,ib], \
                           lon, \
                           lat, \
                           laea=None, \
                           color_coded=True, \
                           latlabel = latlabel, lonlabel=lonlabel, \
                           ccolors=aod3[:,ib], \
                           cminmax=[0,0.001] )
                           
         cax3, m = footprint_leap( ax1[2,ib], \
                           lon, \
                           lat, \
                           laea=None, \
                           color_coded=True, \
                           latlabel = latlabel, lonlabel=lonlabel, \
                           ccolors=aod4[:,ib], \
                           cminmax=[0,0.0005] )                  
                           
         ax1[0,ib].set_title('{:d}'.format(int(bands[ib]))+' nm')

      cb = fig1.colorbar(cax1, cax=axcb1, ticks=[0,0.0025,0.005,0.010], orientation='vertical')
      cb.ax.set_ylabel('15-20 km ExtCoef (km$^{-1}$)',rotation=-90, va="bottom")  
      cb = fig1.colorbar(cax2, cax=axcb2, ticks=[0,0.0005,0.0010,0.005], orientation='vertical')
      cb.ax.set_ylabel('20-25 km ExtCoef (km$^{-1}$)',rotation=-90, va="bottom") 
      cb = fig1.colorbar(cax3, cax=axcb3, ticks=[0,0.00025,0.0005,0.0010], orientation='vertical')
      cb.ax.set_ylabel('25-30 km ExtCoef (km$^{-1}$)',rotation=-90, va="bottom") 


#plt.tight_layout()
      if saveplt:
         plt.savefig( 'img/'+imgtitle+'.png', dpi=300 )
      plt.close()
      
      return aod1, aod2, aod3, aod4
      
###########
def plt_spleae_1bd_iday( aod2, aod3, aod4, lon, lat, laea, fname):

      fig1, ax1 = plt.subplots( nrows=1, ncols=3, figsize=(9,3) )
      plt.subplots_adjust(left=0.08, bottom=0.3, wspace=0.38)
      axcb1 = fig1.add_axes([0.09,0.2, 0.2,0.015])
      axcb2 = fig1.add_axes([0.4,0.2, 0.2,0.015])
      axcb3 = fig1.add_axes([0.71,0.2, 0.2,0.015])
      latlabel = [0,0,0,1]
      lonlabel = [1,0,0,0]   
      
      cax1, m = footprint_leap( ax1[0], \
                lon, \
                lat, \
                laea=laea, \
                color_coded=True, alpha = 0.8, \
                latlabel = latlabel, lonlabel=lonlabel, \
                ccolors=aod2[:,5], \
                cminmax=[0,0.005] )
                           
      ax1[0].set_title('15-20 km' )
      x, y = m(vol_pos[0], vol_pos[1])  # transform coordinates
      ax1[0].scatter(x, y, 25, marker='^', color='r')  

      cax2, m = footprint_leap( ax1[1], \
                lon, \
                lat, \
                laea=laea, \
                color_coded=True, alpha = 0.8,  \
                latlabel = latlabel, lonlabel=lonlabel, \
                ccolors=aod3[:,5], \
                cminmax=[0,0.001] )
                           
      ax1[1].set_title('20-25 km' )
      ax1[1].scatter(x, y, 25, marker='^', color='r')
      xt, yt = m(150, 20) 
#       ax1[1].annotate(fname.split('_')[-2].replace('m',' '), (xt, yt),
#             xytext=(0.8, 0.9), textcoords='axes fraction',
#             fontsize=12, horizontalalignment='left', verticalalignment='top')
      
      cax3, m = footprint_leap( ax1[2], \
                lon, \
                lat, \
                laea=laea, \
                color_coded=True, alpha = 0.8,  \
                latlabel = latlabel, lonlabel=lonlabel, \
                ccolors=aod4[:,5], \
                cminmax=[0,0.0005] )
                           
      ax1[2].set_title('25-30 km' )
      ax1[2].scatter(x, y, 25, marker='^', color='r')

      cb = fig1.colorbar(cax1, cax=axcb1, ticks=[0,0.0025,0.0050], orientation='horizontal')
      cb.ax.set_xlabel('ExtCoef at {:d} nm '.format(int(bands[5])) + r'(km$^{-1}$)' ) #,rotation=-90, va="bottom") 
      cb = fig1.colorbar(cax2, cax=axcb2, ticks=[0,0.0005,0.0010], orientation='horizontal')
      cb.ax.set_xlabel('ExtCoef at {:d} nm '.format(int(bands[5])) + r'(km$^{-1}$)' ) #,rotation=-90, va="bottom") 
      cb = fig1.colorbar(cax3, cax=axcb3, ticks=[0,0.00025,0.0005], orientation='horizontal')
      cb.ax.set_xlabel('ExtCoef at {:d} nm '.format(int(bands[5])) + r'(km$^{-1}$)' ) #,rotation=-90, va="bottom") 
    
#plt.tight_layout()
      imgtitle = 'omps_leae-' + fname.split('_')[-2]
      plt.savefig( 'img/'+imgtitle+'_997.png', dpi=300 )
      plt.close()
      
      return 

#########
def plt_ext_prof_dm( ext3D, alt, bands, clats, lats, lat_right, lat_left, fname, saveplt=True ):
      '''
      plot extinction profile at 15-30 km
      '''
      fig2, axes = plt.subplots( nrows=2, ncols=3, figsize=(8,6) )
      ax2 = axes.flatten()
      style='-k' 
      maxext = []
      ymax   = [0.46, 0.35, 0.15, 0.15, 0.1, 0.1]
      zonal_prof = dict.fromkeys(list(bands)) 
      dm_prof = dict()
      nb = len(bands)

      for ib in range(nb):
         for i in range(ext3D.shape[0]):
            valid = draw_profile( alt, ext3D[i,ib,:], ax2[ib], \
                    style=style, linewidth=0.75) #, min_point=10 )
            maxext.append( np.nanmax(ext3D[i,ib,:]) )
            
         # save mean profile 
         dm_prof[ib] = np.nanmean(ext3D[:,ib,:],axis=0) 
            
         # zonal mean      
         for iz in range(len(clats)):
            zidx = ((lats > lat_left[iz]) & (lats <= lat_right[iz]))
            zm_prof   = np.nanmean( ext3D[zidx,ib,:], axis=0 )
            if iz == 0:
               zonal_prof[bands[ib]] = zm_prof
            else:
               zonal_prof[bands[ib]] = np.vstack((zonal_prof[bands[ib]],zm_prof))
                  
         print(bands[ib],'max ext:', np.nanmax(maxext) )   
         # valid2 = omps.draw_profile( alt, np.nanmean(ib_extprof,axis=0), ax2[ib], \
#                     style='-r', linewidth=1.5)
                    
         ax2[ib].set_ylim([10,35])
         ax2[ib].set_xlim([0,ymax[ib]])
         ax2[ib].set_ylabel('Altitude (km)')
         ax2[ib].set_xlabel(r'ExtCoef (km$^{-1}$)')
    #ax2[ib].ticklabel_format(style='sci',axis='x',scilimits=(0,0))  
         ax2[ib].set_title( '{:d}'.format(int(bands[ib]))+' nm' )      

      plt.tight_layout()
      if saveplt:
         plt.savefig( 'img/omps_aer_prof_' + fname.split('_')[-2]+'.png', dpi=300 )
      plt.close()
      
      return dm_prof, zonal_prof
      
##########
def plt_ext_prof_zm( zonal_prof, alt, clats, bands, fname, varname='ext', subfig=6, saveplt=True ):
      '''
      plot zonal mean extinction profile for each day
      '''
      ymax   = [0.004, 0.003, 0.0015, 0.001, 0.0008, 0.004]
      ydel   = [0.001, 0.001, 0.0005, 0.0005, 0.0002, 0.0004]
      itime  = datetime.strptime( fname.split('_')[-2], '%Ym%m%d')
      timestr= itime.strftime( '%Y-%m-%d')
      if subfig > 1:
        fig2, axs = plt.subplots( nrows=3, ncols=2, figsize=(9,6.5) )
        axes = axs.flatten()
        cmap = plt.cm.rainbow
   #cmap  = plt.cm.jet
        xx, yy = np.meshgrid( clats, alt )
        nb    = len(bands)

        for ib in range(nb):   
         img  = axes[ib].pcolormesh( xx, yy, zonal_prof[bands[ib]].T, shading='nearest', cmap=cmap, vmin=0, vmax=ymax[ib] ) 
   #img  = axes.contourf( Mxx, Myy, data['MLSdata'].T * sf, levels, vmin=0, vmax=vmax, cmap=plt.cm.get_cmap(cmap, len(levels) - 1) )  
         print('ext lat, max:', np.nanmax(zonal_prof[bands[ib]]) )
   #axes.contour(Mxx, Myy, data['MLSdata'].T* sf, img.levels, colors='k', linewidths=0.75)
         cbar = fig2.colorbar(img, ax=axes[ib],shrink=0.98, pad=0.03, ticks=np.arange(0, ymax[ib]+ydel[ib], ydel[ib])) 
         cbar.ax.set_yticklabels(['{:.4f}'.format(x) for x in np.arange(0, ymax[ib]+ydel[ib], ydel[ib])])
         cbar.ax.set_ylabel( r'ExtCoef (km$^{-1}$)' ) 
         axes[ib].set_ylim([15,35]) 
         axes[ib].set_yticks(np.arange(16,35,2)) 
         axes[ib].set_xticks(clats[::20])
         axes[ib].xaxis.set_minor_locator(MultipleLocator(5))
         #axes[ib].yaxis.set_minor_locator(MultipleLocator(1))
         axes[ib].set_ylabel('Altitude (km)')  
         axes[ib].set_xlabel(r'Latitude ($\circ$)')   
         axes[ib].set_title('{:d}'.format(int(bands[ib]))+' nm')  
         
      else:
        fig2, axes = plt.subplots( nrows=1, ncols=1, figsize=(7,3.5) )
   #cmap  = plt.cm.jet
        xx, yy = np.meshgrid( clats, alt )

        if varname == 'ext':
           vmax  = 4.8 #2.11
           vmin  = 0.3
           levels= np.arange(vmin, vmax+0.3, 0.3)
           levels= np.append(np.array([0]), levels)
           cmap = plt.cm.jet
           #img  = axes.pcolormesh( xx, yy, zonal_prof[bands[-1]].T * 1e4, shading='nearest', cmap=cmap, vmin=0, vmax=ymax[-1]*1e4 ) 
           img  = axes.contourf( xx, yy, zonal_prof[bands[-1]].T * 1e3, levels, extend='max', 
                                 vmin=levels[0], vmax=levels[-1], cmap=plt.cm.get_cmap(cmap, len(levels) - 1) )  
           print('ext lat, max:', np.nanmax(zonal_prof[bands[-1]]* 1e3) )
           axes.contour(xx, yy, zonal_prof[bands[-1]].T * 1e3, img.levels, colors='k', linewidths=0.25)
           cbar = fig2.colorbar(img, ax=axes,shrink=0.98, pad=0.03, ticks=levels) 
           cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in levels])
           cbar.ax.set_ylabel( r'ExtCoef (10$^{-3}$ km$^{-1}$)' ) 
           axes.set_title('{:d}'.format(int(bands[-1]))+' nm '+timestr)
           axes.set_ylim([17,30]) 
           axes.set_yticks(np.arange(18,31,2)) 
        else:
           cmap  = plt.cm.gist_ncar
           vmin  = 1.2
           vmid  = 2.0
           vmax  = 3.3
           levels0= np.arange(vmin, vmid, 0.05)
           levels1= np.arange(vmid, vmax+0.18, 0.18)
           levels = np.append( levels0, levels1)
           # create new colormap
           newcolors = np.vstack((cmap(np.linspace(0, 1.0, 64))[0:-15:3,:],
                       cmap(np.linspace(0, 1.0, 64))[-15:,:]))
           newcmp = colors.ListedColormap(newcolors, name='cmbgist')            
           img  = axes.pcolormesh( xx, yy, zonal_prof.T, shading='nearest', cmap=plt.cm.get_cmap(newcmp, len(levels) - 1), vmin=vmin, vmax=vmax ) 
   #img  = axes.contourf( Mxx, Myy, data['MLSdata'].T * sf, levels, vmin=0, vmax=vmax, cmap=plt.cm.get_cmap(cmap, len(levels) - 1) )  
           print('ext lat, max:', np.nanmax(zonal_prof) )
   #axes.contour(Mxx, Myy, data['MLSdata'].T* sf, img.levels, colors='k', linewidths=0.75)
           cbar = fig2.colorbar(img, ax=axes,shrink=0.98, pad=0.03, ticks=levels[::2]) 
           cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in levels[::2]])
           cbar.ax.set_ylabel( 'AE ({:.0f}-{:.0f}nm)'.format(bands[2], bands[-1]) ) 
           axes.set_title('AE '+ timestr)
           axes.set_ylim([15,35]) 
           axes.set_yticks(np.arange(16,35,2)) 
        axes.set_xticks(np.arange(-60,61,20))
        axes.xaxis.set_minor_locator(MultipleLocator(5))
        #axes.yaxis.set_minor_locator(MultipleLocator(1))
        axes.set_ylabel('Altitude (km)')  
        axes.set_xlabel(r'Latitude ($\circ$)')   
        
           
      plt.tight_layout()
      if saveplt:
         if varname == 'ext':
            plt.savefig( 'img/omps_aer_zonalmprof_' + str(subfig) + '_' + fname.split('_')[-2]+'.png', dpi=300 )
         else:
            plt.savefig( 'img/omps_AE_zonalmprof_' + fname.split('_')[-2]+'.png', dpi=300 )   
      plt.close() 
      
      return  

##########
def plt_combext_prof_zm( zonal_prof1, zonal_prof2, alt, clats, bands, fname, saveplt=True ):
    '''
    plot combined zonal mean extinction profile for each day
    '''

    fig2, axes = plt.subplots( nrows=1, ncols=1, figsize=(6.,3.5) )
    xx, yy = np.meshgrid( clats, alt )
    itime  = datetime.strptime( fname.split('_')[-2], '%Ym%m%d')
    timestr= itime.strftime( '%Y-%m-%d')

    vmax  = 4.8 #2.11
    vmin  = 0
    levels= np.arange(vmin, vmax+0.3, 0.3)
    #levels= np.append(np.array([0]), levels)
    cmap = plt.cm.jet
    #cmap.set_under('w')
           #img  = axes.pcolormesh( xx, yy, zonal_prof[bands[-1]].T * 1e4, shading='nearest', cmap=cmap, vmin=0, vmax=ymax[-1]*1e4 ) 
    img  = axes.contourf( xx, yy, zonal_prof1[bands[-1]].T * 1e3, levels, extend='max', 
                    vmin=levels[0], vmax=levels[-1], cmap=plt.cm.get_cmap(cmap, len(levels) - 1) )  
    print('ext lat, max:', np.nanmax(zonal_prof1[bands[-1]]* 1e3) )
    ctlevels = np.arange(0.6, 1.5, 0.2)
    flag = (zonal_prof1[bands[-1]] < 3e-4)
    zonal_prof2[flag] = np.nan
    ctimg = axes.contour(xx, yy, zonal_prof2.T, levels=ctlevels, colors='k', linewidths=0.6)
    axes.clabel(ctimg, ctimg.levels, inline=True, fontsize=8, fmt='%.1f')
    cbar = fig2.colorbar(img, ax=axes,shrink=0.98, pad=0.03, ticks=levels) 
    cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in levels])
    cbar.ax.set_ylabel( r'$\beta_{997}$ (10$^{-3}$ km$^{-1}$)' ) 
    axes.set_title('{:d}'.format(int(bands[-1]))+' nm '+timestr)
    axes.set_ylim([17,30]) 
    axes.set_xlim([-70,40])
    axes.set_yticks(np.arange(18,31,2)) 
    axes.set_xticks(np.arange(-60,41,20))
    axes.xaxis.set_minor_locator(MultipleLocator(5))
        #axes.yaxis.set_minor_locator(MultipleLocator(1))
    axes.set_ylabel('Altitude (km)')  
    axes.set_xlabel(r'Latitude ($\circ$)')   
        
    plt.tight_layout()
    if saveplt:
        plt.savefig( 'img/omps_combaer_zonalmprof_' + fname.split('_')[-2]+'.png', dpi=300 )
    plt.close() 
      
    return  


##########
def plt_SAOD_spleae_iday( SAOD, bands, vol_pos, fname, saveplt=True):
      '''
      plot total strat. AOD
      '''
      fig2, axes = plt.subplots( nrows=2, ncols=3, figsize=(8,7) )
      plt.subplots_adjust(bottom=0.25, hspace=0.00001)
      latlabel = [0,0,0,1]
      lonlabel = [1,0,0,0]   
      cmax     = [0.1, 0.1, 0.05, 0.05, 0.02, 0.02]
      nb       = len(bands)
      
      for ib in range(nb):   
         ic = ib // 2
         ir = ib % 2
         cax, m = footprint_leap( axes[ir,ic], \
                np.array(data['Longitude']), \
                np.array(data['Latitude']), \
                laea=None, \
                color_coded=True, \
                latlabel = latlabel, lonlabel=lonlabel, \
                ccolors=SAOD[:,ib], 
                cminmax=[0,cmax[ib]] )  
                
         axes[ir,ic].set_title( '{:d}'.format(int(bands[ib]))+' nm' ) 
         x, y = m(vol_pos[0], vol_pos[1])  
         axes[ir,ic].scatter(x, y, 25, marker='^', color='r')     
    
         if ir == 1:  
            pos1  = axes[ir,ic].get_position()
            pos1b = [pos1.xmax+pos1.width*0.22*(ic-5.5), pos1.y0-pos1.height*0.8, pos1.width, pos1.height*0.04]
            ax_cb = fig2.add_axes( pos1b )
            cb = fig2.colorbar(cax, cax=ax_cb, ticks=[0,cmax[ib]/2,cmax[ib]], orientation='horizontal')
            cb.ax.set_xlabel('Strat. AOD') #,rotation=-90, va="bottom") 
       

      plt.tight_layout()
      if saveplt:
         plt.savefig( 'img/omps_SAOD_spleae_' + fname.split('_')[-2]+'.png', dpi=300 )
      plt.close()   
      
      return    
      
      
########
def plt_AE_prof_iday( ext3D, alt, bands, fname, saveplt=True):
      '''
      plot AE profile (675-997nm)
      '''
      fig4, axes = plt.subplots( nrows=1, ncols=2, figsize=(4.5,3) )
      style='-k' 
      maxext = []
      
      
      AE_allprof = - (np.log(ext3D[:,2,:] / ext3D[:,-2,:]) / np.log(bands[2] / bands[-2]))
      AE_allprof[AE_allprof < 0] = np.nan
      imeanAE   = np.nanmean( AE_allprof, axis=0 )
      istdAE    = np.nanstd( AE_allprof, axis=0 )
      print('AE prof:', AE_allprof.shape, np.nanmax(AE_allprof), np.nanmin(AE_allprof))

      for i in range(ext3D.shape[0]):
            valid = draw_profile( alt, AE_allprof[i,:], axes[0], \
                    style=style, linewidth=0.75) #, min_point=10 )
      axes[0].set_ylim([10,35])
      axes[0].set_xlim([0,15])
      axes[0].set_ylabel('Altitude (km)')
      axes[0].set_xlabel('AE ({:.0f}-{:.0f}nm)'.format(bands[2], bands[-2]))   
      axes[0].set_title( fname.split('_')[-2]+' AE', fontsize=10 )   
           
      # mean 
      axes[1].plot(imeanAE, alt, marker='o', markersize=2,color='r') 
      axes[1].fill_betweenx(alt, imeanAE-istdAE, imeanAE+istdAE, color='r', alpha=0.3, ec=None)     
      axes[1].set_ylabel('Altitude (km)')              
      axes[1].set_ylim([10,35])
      axes[1].set_xlim([0,5])
      #axes.set_ylabel('Altitude (km)')
      axes[1].set_xlabel('AE ({:.0f}-{:.0f}nm)'.format(bands[2], bands[-2]))
    #ax2[ib].ticklabel_format(style='sci',axis='x',scilimits=(0,0))  
      axes[1].set_title( 'mean AE', fontsize=10 )      

      plt.tight_layout()
      if saveplt:
         plt.savefig( 'img/omps_AE_prof_' + fname.split('_')[-2]+'.png', dpi=300 )
      plt.close()
      
      return imeanAE, istdAE, AE_allprof
      
######
def plt_spleae_AE_iday( ext3D, alt, lon, lat, bands, laea, fname, saveplt=True):
      '''
      AE (675-997nm) at different altitudes
      '''
      AE_allprof = - (np.log(ext3D[:,2,:] / ext3D[:,-2,:]) / np.log(bands[2] / bands[-2]))
      AE_allprof[AE_allprof < 0] = np.nan 
      
      # different altitudes
      idx3 = np.where( (alt<=25) & (alt>20) )[0]
      idx1 = np.where( (alt<=15) & (alt>10) )[0]
      idx2 = np.where( (alt<=20) & (alt>15) )[0]
      idx4 = np.where( (alt<=30) & (alt>25) )[0]
      AE1 = np.nanmean(AE_allprof[:,idx1], axis=1)
      AE2 = np.nanmean(AE_allprof[:,idx2], axis=1)
      AE3 = np.nanmean(AE_allprof[:,idx3], axis=1)
      AE4 = np.nanmean(AE_allprof[:,idx4], axis=1)
      
      # plot
      imgtitle = 'omps_AE_spleae-' + fname.split('_')[-2]
      nb = len(bands)

      fig1, ax1 = plt.subplots( nrows=1, ncols=3, figsize=(9,3) )
      plt.subplots_adjust(left=0.08, bottom=0.3, wspace=0.38)
      axcb1 = fig1.add_axes([0.09,0.2, 0.2,0.015])
      axcb2 = fig1.add_axes([0.4,0.2, 0.2,0.015])
      axcb3 = fig1.add_axes([0.71,0.2, 0.2,0.015])
      latlabel = [0,0,0,1]
      lonlabel = [1,0,0,0]   
      
      cax1, m = footprint_leap( ax1[0], \
                lon, \
                lat, \
                laea=laea, \
                color_coded=True, alpha = 0.8, \
                latlabel = latlabel, lonlabel=lonlabel, \
                ccolors=AE2, \
                cminmax=[2.0,2.5] )
                           
      ax1[0].set_title('15-20 km' )

      cax2, m = footprint_leap( ax1[1], \
                lon, \
                lat, \
                laea=laea, \
                color_coded=True, alpha = 0.8,  \
                latlabel = latlabel, lonlabel=lonlabel, \
                ccolors=AE3, \
                cminmax=[1.5,2.0] )
                           
      ax1[1].set_title('20-25 km' )

      
      cax3, m = footprint_leap( ax1[2], \
                lon, \
                lat, \
                laea=laea, \
                color_coded=True, alpha = 0.8,  \
                latlabel = latlabel, lonlabel=lonlabel, \
                ccolors=AE4, \
                cminmax=[1.0,1.5] )
                           
      ax1[2].set_title('25-30 km' )

      cb = fig1.colorbar(cax1, cax=axcb1, ticks=[2.0,2.25,2.5], orientation='horizontal')
      cb.ax.set_xlabel('AE ({:.0f}-{:.0f}nm)'.format(bands[2], bands[-2]) ) #,rotation=-90, va="bottom") 
      cb = fig1.colorbar(cax2, cax=axcb2, ticks=[1.5,1.75,2.0], orientation='horizontal')
      cb.ax.set_xlabel('AE ({:.0f}-{:.0f}nm)'.format(bands[2], bands[-2]) ) #,rotation=-90, va="bottom") 
      cb = fig1.colorbar(cax3, cax=axcb3, ticks=[1.0,1.25,1.5], orientation='horizontal')
      cb.ax.set_xlabel('AE ({:.0f}-{:.0f}nm)'.format(bands[2], bands[-2]) ) #,rotation=-90, va="bottom") 



#plt.tight_layout()
      if saveplt:
         plt.savefig( 'img/'+imgtitle+'.png', dpi=300 )
      plt.close()
      
      return AE1, AE2, AE3, AE4
            
######
def fnames_in_path( fpath, prefix=None ):
    fnames = []
    for fname in os.listdir(fpath):
        if prefix is None:
           if ( fname.startswith("OMPS-NPP_LP-L2-AER-DAILY_v2.") and fname.endswith(".h5")):
               fnames.append(fpath+fname)
        else:
           if ((fname.startswith("OMPS-NPP_LP-L2-AER-DAILY_v2.0_"+prefix)) or (fname.startswith("OMPS-NPP_LP-L2-AER-DAILY_v2.1_"+prefix)) ) \
              and (fname.endswith(".h5")):
               fnames.append(fpath+fname)       
    return sorted(fnames)

def fnames17sep():
   fpath = '/Dedicated/jwang-data/xxu69/data/OMPS/OMPS-Limb/201709/'
   fnames = fnames_in_path( fpath )
   print( fnames )
   sys.exit('ccc')
   fnames = \
         ['OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0901_2017m0902t201246.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0902_2017m0903t235941.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0903_2017m0904t125814.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0904_2017m0905t111147.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0905_2017m0906t132608.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0906_2017m0907t131233.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0907_2017m0908t152659.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0908_2017m0909t151222.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0909_2017m0911t030055.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0910_2017m0911t123931.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0911_2017m0912t110055.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0912_2017m0913t114152.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0913_2017m0914t120906.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0914_2017m0915t145754.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0915_2017m0916t172444.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0916_2017m0917t121556.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0917_2017m0918t102739.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0918_2017m0919t131855.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0919_2017m0920t124354.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0920_2017m0921t154112.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0921_2017m0922t130719.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0922_2017m0924t010332.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0923_2017m0925t005559.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0924_2017m0926t060327.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0925_2017m0926t232755.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0926_2017m0927t110627.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0927_2017m0928t133837.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0928_2017m0929t125120.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0929_2017m1001t004708.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2017m0930_2017m1002t004050.h5']
   return [fpath+fnames[i] for i in range(len(fnames)) ]

######
def fnames16sep():
   fpath = '/Dedicated/jwang-data/xxu69/data/OMPS/OMPS-Limb/201609/'
   fnames = \
         ['OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0901_2017m0131t205758.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0902_2017m0131t205807.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0903_2017m0131t205759.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0904_2017m0131t205800.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0905_2017m0131t205748.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0906_2017m0131t205749.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0907_2017m0131t205750.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0908_2017m0131t205800.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0909_2017m0131t205752.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0910_2017m0131t205808.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0911_2017m0131t205752.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0912_2017m0131t205802.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0913_2017m0131t205759.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0914_2017m0131t205802.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0915_2017m0131t205801.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0916_2017m0131t205754.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0917_2017m0131t205808.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0918_2017m0131t205754.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0919_2017m0131t205815.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0920_2017m0131t205804.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0921_2017m0131t205813.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0922_2017m0131t205838.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0923_2017m0131t205813.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0924_2017m0131t210458.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0925_2017m0131t205802.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0926_2017m0131t210509.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0927_2017m0131t205817.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0928_2017m0131t210506.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0929_2017m0131t205809.h5',
          'OMPS-NPP_LP-L2-AER675-DAILY_v1.0_2016m0930_2017m0131t205807.h5']
   return [fpath+fnames[i] for i in range(len(fnames)) ]

######
def find_ref_loc( lats, lons ):
	'''
	find the reference latitude and longitude for albert equal area projection
	'''
	lat_0 = np.mean(lats)
	lon_0 = np.mean(lons)
	lat_1 = np.nanmin(lats) - 0.2 * abs(np.nanmin(lats))
	lat_2 = np.nanmax(lats) + 0.2 * abs(np.nanmax(lats))

	if lat_1 <=-90:
		lat_1 = -90
			
	if lat_2 >=90:
		lat_2 = 90
			
	print(' - lat 1:', lat_1, 'lat 2:', lat_2, ' Ratio', abs(lat_1/lat_2))
			
	while (abs(lat_1/lat_2) <=1.1) & (abs(lat_1/lat_2) >= 0.9):
		print( ' - Improprate reference lat')
		lat_1 = lat_1 - abs(lat_1)*0.1
				
		print(' - lat 1:', lat_1, 'lat 2:', lat_2)
		
	return [lat_0, lat_1, lat_2, lon_0]

######
def cal_omps_NM_area_test( lat_bnds, lon_bnds, valid_idx ):		
    '''
    calculate OMPS NM pixel area using Albers equal-area conic projection based on pixel lat, lon
    latitude range: [-5,5] to reduce the uncertainty from shape variation, average along track pixels
    valid_idx: the index of pixels locate in the domain
    '''
    from shapely.geometry import Polygon

    idx_2d   = np.where( valid_idx == True ) 
    min_col  = np.min(idx_2d[1])
    max_col  = np.max(idx_2d[1])
    s_row = np.min( np.where( valid_idx[:,min_col] == True ) )
    e_row = np.max( np.where( valid_idx[:,max_col] == True ) )
    print('valid row:', s_row, e_row)
    #print('check lon,lat:', lon_bnds[s_row:e_row+1,0,0], lat_bnds[s_row:e_row+1,0,0])

    ncol    = lon_bnds.shape[1]
    rowidxs = np.arange(s_row, e_row+1)
    area2D  = np.full( (len(rowidxs),ncol), np.nan )  
    
    m_row   = int( (e_row + s_row)/2 )+1
    lon_0   = ( lon_bnds[m_row, int(ncol/2), 0] + lon_bnds[m_row, int(ncol/2), 1] )/2. 
    #print('lon_0:',lon_bnds[m_row, int(ncol/2), :]) 
    
    for ir in range(len(rowidxs)):
       irow = rowidxs[ir]
    
       for ic in range(ncol):       
       
          # calculate pixel area
          lon_arr = lon_bnds[irow,ic,:]
          lat_arr = lat_bnds[irow,ic,:]
          #print('corner:', lon_arr, lat_arr, lon_0)
       
          # project the standard mesh grid on the alber equal area coordinates system
          x, y = res.alber_equal_area(lon_arr, lat_arr, lat_0 = -20, lat_1 = -35, lat_2 = -5, lon_0 = 180)
			
          cCord = [
    		 (x[3], y[3]),\
    		 (x[2], y[2]),\
    	     (x[1], y[1]),\
    	     (x[0], y[0])
    	    ]	
    	    
          #print(cCord)  
          cPolygon = Polygon(cCord)	
          area2D[ir,ic] = cPolygon.area	# m2
          #print(ip, 'area', area[ix,iy]*1e-6)
           
    return area2D, rowidxs   	

######
def cal_omps_NM_area( lat_bnds, lon_bnds, valid_idx ):		
    '''
    calculate OMPS NM pixel area using Albers equal-area conic projection based on pixel lat, lon
    latitude range: [-5,5] to reduce the uncertainty from shape variation, average along track pixels
    valid_idx: the index of pixels locate in the domain
    '''
    from shapely.geometry import Polygon

    idx_2d   = np.where( valid_idx == True )  
    area2d   = np.full( valid_idx.shape, np.nan )
    
    for ip in range(len(idx_2d[0])):
          irow = idx_2d[0][ip]
          ic   = idx_2d[1][ip]
          #print('index:', irow, ic)
    
          # calculate pixel area
          lon_arr = lon_bnds[irow,ic,:]
          lat_arr = lat_bnds[irow,ic,:]
          #print('corner:', lon_arr, lat_arr)
       
          # project the standard mesh grid on the alber equal area coordinates system
          x, y = res.alber_equal_area(lon_arr, lat_arr, lat_0 = -20, lat_1 = -35, lat_2 = -5, lon_0 = 180)
			
          cCord = [
    		 (x[3], y[3]),\
    		 (x[2], y[2]),\
    	     (x[1], y[1]),\
    	     (x[0], y[0])
    	    ]	
    	    
          #print(cCord)  
          cPolygon = Polygon(cCord)	
          area2d[irow,ic] = cPolygon.area	# m2
          #print(ip, 'area', area2d[irow,ic]*1e-6)
           
    return area2d 	

#######
def cal_SO2_mass( DU, lat_bnds, lon_bnds, area ):
    '''
    calculate the SO2 mass from SO2 column concentration (DU)
    OMPS spatial resolution is 50*50km
    area: 1D array, length = ncolumn
    '''
    
    output   = dict()
    DU[DU<0.] = np.nan
    NA = 6.022e23
    molmass = 64 # g/mol
        
    # simple assumption: not accurate!!!
    # area = 50 * 50 * 1e6 # pixel area: m2 
    if len(area.shape) == 1:
       area[area > 2e10] = np.nan
       area = np.tile( area, (DU.shape[0],1) )
    
    conc = DU * 2.69e16 # molecules per cm2
    pixelmol  = conc * area * 1e4 / NA
    output['mass'] = np.nansum( pixelmol * molmass ) / 1e12 # Tg
    
    flag      = np.ones( DU.shape )
    flag[np.isnan(DU)] = 0
    output['area'] = np.nansum( flag * area ) * 1e-6 # km2
    
    return output

##########    
def maskland( Lons, Lats ):

   #-- Get land/ocean mask
   invalid    = np.isnan(Lons)
   masked_lon = np.ma.masked_invalid(Lons)
   masked_lat = np.ma.masked_invalid(Lats)
   ocean = np.ones( Lons.shape )
   land  = np.ones( Lons.shape )
   ocean[invalid] = np.nan
   land[invalid] = np.nan
   msk_ocean = maskoceans( masked_lon, masked_lat, ocean, resolution='h', grid=2.5 )
   msk_land  = np.ma.masked_array( land, mask=~msk_ocean.mask )
   

   return msk_land
   

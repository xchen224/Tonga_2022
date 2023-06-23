import os, sys
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator
from scipy import interpolate
from scipy.interpolate import interp1d
import matplotlib.colors as colors
from scipy import interpolate, stats, optimize, integrate

sys.path.append('/Users/xchen224/anaconda3/envs/xcpy/lib/python3.7/site-packages/mylib/')
from plot_libs import *
      
####
def toYearFraction(date):
   year = date.year
   startOfThisYear = datetime(year=year, month=1, day=1)
   startOfNextYear = datetime(year=year+1, month=1, day=1)

   yearElapsed  = (date - startOfThisYear).total_seconds()
   yearDuration = (startOfNextYear - startOfThisYear).total_seconds()
   fraction = yearElapsed/yearDuration

   return date.year + fraction

###
def leap_year( yy ):
   if yy%100 == 0:
      if yy%400 == 0:
         return 1
      else:
         return 0
   elif yy%4 == 0:
      return 1
   else:
      return 0    

####
def toDay3d( date ):

   yy = int(date[:4])
   mm = int(date[4:6])
   dd = int(date[6:])
   
   if yy%100 == 0:
      if yy%400 == 0:
         days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
      else:
         days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]   
   elif yy%4 == 0:
      days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
   else:
      days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] 
      
   numday = sum(days[:(mm - 1)]) + dd
   
   return numday         

###
def intp_lev_ip( ori_pres, out_pres, iprof ):
   '''
   linear interpolation along log(p)
   ori_pres: 1D GC level grids
   out_pres: 1D MLS level grids
   '''
   
   # sort x
   inputx = np.sort(np.log(ori_pres))
   idx    = np.argsort( np.log(ori_pres) )
   inputy = iprof[idx]
     
   f = interp1d( inputx, inputy, fill_value='extrapolate' )
   
   out_x  = np.sort( np.log(out_pres) )
   oidx   = np.argsort( np.log(out_pres) )
   bkidx  = np.argsort( oidx )
   out_y  = f( out_x )
   
   #print('intp pres:', inputx, inputy, out_x, out_y)
   #sys.exit()
   return out_y[bkidx]
      
####
def plot_location_map( lon, lat, time, 
                      proj='cyl', area_thresh=1000,
                      mapwidth=3000000, mapheight=1800000,
                      maplat0=46.5, maplon0=-123.5, lon_0=0,
                      region=[-180,180,-90,90], cmap=plt.cm.rainbow, 
                      ccoast='0.2', cgrid='0.2', vmin=0, vmax=366,
                      grid=True, lonlabel=[0,0,0,1], latlabel=[1,0,0,0],
                      cbar=True, cbar_ticks=None, fig=0, cbar_ylabel=None,
                      tracks=None, ctrack='red', cbar_ticklabel=None,
                      title='',  markersize=2, edgecolor=False, alpha=0.5,
                      deltalat=30, deltalon=45, bstate=True,
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

   #-- draw costlines and map grid
   m.drawmapboundary(fill_color = 'w')
   m.drawcoastlines(linewidth=0.3,color=ccoast)
   m.drawcountries(linewidth=0.1,color='gray')
   #m.drawstates(linewidth=0.1,color='gray')
   if bstate:
      m.drawstates(linewidth=0.1,color='gray')
   if grid:
      m.drawparallels( np.arange(-90,91,deltalat), color=cgrid, labels=latlabel, linewidth=0.4 )
      if lon_0 == 0:
         m.drawmeridians( np.arange(-180,181,deltalon), color=cgrid, labels=lonlabel, linewidth=0.4 )
      else:
         m.drawmeridians( np.arange(0,361,deltalon), color=cgrid, labels=lonlabel, linewidth=0.4 )   
      
   #-- plot data on the map
   msk_lon = np.ma.masked_invalid(lon)
   msk_lat = np.ma.masked_invalid(lat)      
   if edgecolor:
      if 'linewidths' in kwargs.keys():
         ms = m.scatter(msk_lon, msk_lat, c='', s=markersize, edgecolors=time, alpha=alpha, 
                     latlon=True, vmin=vmin, vmax=vmax,cmap=cmap, linewidths=0.25)
      else:               
         ms = m.scatter(msk_lon, msk_lat, c='', s=markersize, edgecolors=time, alpha=alpha, latlon=True, vmin=vmin, vmax=vmax,cmap=cmap, **kwargs)
   else:
      ms = m.scatter(lon, lat, c=time, s=markersize, alpha=alpha, latlon=True, vmin=vmin, vmax=vmax,cmap=cmap, **kwargs)
   
   #-- draw color bar
   if cbar:
      cb = m.colorbar(ms,location='right',shrink=0.98, pad=0.03, ticks=cbar_ticks)
      if cbar_ticklabel is not None:  
         cb.ax.set_yticklabels(cbar_ticklabel) 
      if cbar_ylabel is not None:
         cb.ax.set_ylabel(cbar_ylabel) #,rotation=-90, va="bottom")   
 
   #-- show image
   if 'ax' in kwargs.keys():
      ax.set_title( title )
   else: 
      plt.title( title )
   #plt.show()

   return m, ms

###
def plot_timeseries( axes, x, y, z, **kwargs):
    
    title      = kwargs.get('title', "")
    ylims      = kwargs.get('ylims', [np.nanmin(y), np.nanmax(y)])
    xlims      = kwargs.get('xlims', [np.nanmin(x), np.nanmax(x)])
    if xlims is not None:
       xticks     = kwargs.get('xticks', np.arange(xlims[0],xlims[1],(xlims[1]-xlims[0])/10))
       xticklabel = kwargs.get('xticklabel', xticks)
    else:
       xticks     = kwargs.get('xticks', [])
       xticklabel = kwargs.get('xticklabel',[])   
    yticks     = kwargs.get('yticks', [])
    yticklabel = kwargs.get('yticklabel',[])
    if 'norm' in kwargs.keys():
       norm    = kwargs['norm']
       levels  = kwargs['levels']
    else:   
       vmin       = kwargs.get('vmin', np.nanmin(z))
       vmax       = kwargs.get('vmax', np.nanmax(z))
       levels     = kwargs.get('levels', np.arange(vmin,vmax,(vmax-vmin)/20))
    ylabel     = kwargs.get('ylabel', "")
    xlabel     = kwargs.get('xlabel', "")
    cmap       = kwargs.get('cmap', plt.cm.jet)
    cbar       = kwargs.get('cbar', True)
    clabel     = kwargs.get('cbar_ylabel', "")
    padrat     = kwargs.get('padrat', 0.2)
    set_xminor = kwargs.get('set_xminor', False)
    plt_lines  = kwargs.get('plt_lines', False)
    extend     = kwargs.get('extend', 'neither')
    cbarwd     = kwargs.get('cbarwd', '2%')
    logcolor   = kwargs.get('logcolor', False)
    cbticks    = kwargs.get('cbticks', levels)
    xminor     = kwargs.get('xminor', 1)
    useax      = kwargs.get('useax',False)
    usepcol    = kwargs.get('usepcol',False)
    
    xx, yy = np.meshgrid( x, y )
    if usepcol: 
      if 'norm' in kwargs.keys():
         img = axes.pcolormesh( xx, yy, z.T, shading='nearest', cmap=cmap, norm=norm)
      else:
         img = axes.pcolormesh( xx, yy, z.T, shading='nearest', cmap=cmap, vmin=vmin, vmax=vmax) 
    else:  
      if logcolor:
       img  = axes.contourf( xx, yy, z.T , levels, norm=colors.LogNorm(), cmap=cmap, extend=extend )
      else:   
       if 'norm' in kwargs.keys():
          img  = axes.contourf( xx, yy, z.T , levels, norm=norm, cmap=cmap, extend=extend ) 
       else:   
          img  = axes.contourf( xx, yy, z.T , levels, vmin=vmin, vmax=vmax, cmap=cmap, extend=extend ) 
    if plt_lines:
       axes.contour(xx, yy, z.T , img.levels, colors='k', linewidths=0.25)
    axes.set_xticks(xticks)
    axes.set_xticklabels(xticklabel)
    if xlims is not None:
       axes.set_xlim(xlims) 
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel) 
    axes.set_title(title) 
    if set_xminor:
       axes.xaxis.set_minor_locator(MultipleLocator(xminor))
    if len(yticks) > 0:
       axes.set_yticks(yticks)
       if len(yticklabel) > 0:
          axes.set_yticklabels(yticklabel)
    
    
    if cbar:
         if useax:
            cb = plt.colorbar(img, ax=axes, shrink=0.98, ticks=cbticks, pad=padrat) 
         else:
            divider = make_axes_locatable(axes)
            cax = divider.append_axes("right", size=cbarwd, pad=padrat) 
            cb = plt.colorbar(img, cax=cax, shrink=0.98, ticks=cbticks  )
         deltav = np.min(np.abs(levels[1:]-levels[:-1]))
         if abs(deltav) < 0.001:
            ctlabels = ['{:.4f}'.format(x) for x in cbticks]
         elif abs(deltav) < 0.01:
            ctlabels = ['{:.3f}'.format(x) for x in cbticks]
         elif abs(deltav) < 0.099:
            ctlabels = ['{:.2f}'.format(x) for x in cbticks]
         elif abs(deltav) < 1:
            ctlabels = ['{:.1f}'.format(x) for x in cbticks] 
         else:
            ctlabels = ['{:.0f}'.format(x) for x in cbticks]  
         if 'cbfontsize' in kwargs:   
            cb.ax.set_yticklabels(ctlabels, fontsize=kwargs['cbfontsize'])
         else:   
            cb.ax.set_yticklabels(ctlabels)
         cb.ax.set_ylabel( clabel )    
    if 'Pressure' in ylabel:
         axes.set_yscale('log')   
         axes.invert_yaxis()  
         axes.yaxis.set_minor_locator(FixedLocator([1,2,3,5,7,10,20,30,50,70,100,150,200,300]))  
         axes.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
         axes.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    
    axes.set_ylim(ylims)       
    return
            

###
def plot_prof( axes, x, y, **kwargs):
    
    title      = kwargs.get('title', "")
    ylims      = kwargs.get('ylims', [np.nanmin(y), np.nanmax(y)])
    xlims      = kwargs.get('xlims', [np.nanmin(x), np.nanmax(x)])
    xticks     = kwargs.get('xticks', np.arange(xlims[0],xlims[1],(xlims[1]-xlims[0])/10))
    xticklabel = kwargs.get('xticklabel', xticks)
    yticks     = kwargs.get('yticks', [])
    yticklabel = kwargs.get('yticklabel',[])
    ylabel     = kwargs.get('ylabel', "")
    xlabel     = kwargs.get('xlabel', "")
    clr        = kwargs.get('color', 'r')
    set_xminor = kwargs.get('set_xminor', False)
    lw         = kwargs.get('lw', 1.0)
    label      = kwargs.get('label', '')
    
    axes.plot(x, y, color=clr, lw=lw, label=label)
    axes.set_xticks(xticks)
    axes.set_xticklabels(xticklabel)
    axes.set_ylim(ylims) 
    axes.set_xlim(xlims) 
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel) 
    axes.set_title(title) 
    if set_xminor:
       axes.xaxis.set_minor_locator(MultipleLocator(1))
    if len(yticks) > 0:
       axes.set_yticks(yticks)
       if len(yticklabel) > 0:
          axes.set_yticklabels(yticklabel)
    
    if 'Pressure' in ylabel:
         axes.set_yscale('log')   
         axes.invert_yaxis()  
         axes.yaxis.set_minor_locator(FixedLocator([1,2,3,5,7,10,20,30,50,70,100,150,200,250]))  
         axes.yaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
         axes.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
              
          
    return
                  
###
def select_region( region, lat, lon ):
    '''
    select data in region
    lat and lon array should have the same shape
    '''
    
    if ( len(region) != 4 ):
       sys.exit("select region: Error 1, wrong region definition!") 

    lon1, lon2 = region[0], region[1]
    lat1, lat2 = region[2], region[3]   
       
    if (lon1 < lon2):
       idx = np.where( (lon >= lon1) & (lon <= lon2)  & (lat >= lat1) & (lat <= lat2) )
    else:
       idx = np.where( ((lon >= lon1) | (lon <= lon2))  & ((lat >= lat1) & (lat <= lat2)) )
    
    if len(idx) == 1:
       return idx[0]
    else:
       return idx
    
###
def QC_thresh():
    '''
    The threshold for each product quality control flags
    '''
    output = dict()
    output['SO2'] = {'pressure': [10,215], 'precision': 0, 'status': 0, 'quality': 0.95, 'convergence': 1.03 }
    output['CO']  = {'pressure': [0.001,215], 'precision': 0, 'status': 0, 'quality': 1.5, 'convergence': 1.03 }
    output['H2O'] = {'pressure': [0.001,316], 'precision': 0, 'status': 0, 'quality': 0.7, 'convergence': 2.0 }
    output['O3']  = {'pressure': [0.001,261], 'precision': 0, 'status': 0, 'quality': 1.0, 'convergence': 1.03 }
    output['Temperature']   = {'pressure': [0.00046,261], 'precision': [0,0.7], 'status': 0, 'quality': [0.2,0.9], 'convergence': 1.03 }
    output['IWC'] = {'pressure': [82,215], 'range': 0.1 } #g/m3
    output['IWP'] = {'range': 200 } #g/m2
    output['RHI'] = {'pressure': [0.001,316], 'precision': 0, 'status': 0, 'quality': 0.7, 'convergence': 2.0 }
    output['ClO'] = {'pressure': [1.0,147], 'precision': 0, 'status': 0, 'quality': 1.3, 'convergence': 1.05 }
    output['HCl'] = {'pressure': [0.32,100], 'precision': 0, 'status': 0, 'quality': 1.2, 'convergence': 1.05 }
    output['N2O'] = {'pressure': [0.46,100], 'precision': 0, 'status': 0, 'quality': 0.8, 'convergence': 2.0 }
    
    return output
    
###
def QC_flag( pressure, precision, status, quality, convergence, gasname ):

    if gasname not in QC_thresh().keys():
       sys.exit('Add the thresholds for quality control of ' + gasname)
    else:   
       threshs = QC_thresh()[gasname]   
    
    if gasname == 'ClO':
       rem = status 
    else:   
       rem = status % 2 
    dima = (rem == threshs['status'] )
    if isinstance(threshs['quality'], list):
       dimb1 = (quality > threshs['quality'][0])
       dimb2 = (quality > threshs['quality'][1])
    else:   
       dimb = (quality > threshs['quality'])
    dimc = (convergence < threshs['convergence'] )
    
    if gasname == 'SO2':
       flag_2d = (precision != threshs['precision'])
    else:
       if isinstance(threshs['precision'], list):
          flag_2d = ((precision > threshs['precision'][0]) & (precision <= threshs['precision'][1]) )
       else:   
          flag_2d = (precision > threshs['precision'])   
       #print('1, flag_2d:', flag_2d)   
    
    if isinstance(threshs['quality'], list):    
       dim2a = ((pressure > (threshs['pressure'][0]*0.999)) & (pressure < 83.1))
       dim2b = ((pressure > 99.9) & (pressure < threshs['pressure'][1]*1.003))
       dim1a = np.logical_and( dima, np.logical_and( dimb1, dimc ) )
       dim1b = np.logical_and( dima, np.logical_and( dimb2, dimc ) )
       idx1a = np.where(dim1a == False)[0]
       idx1b = np.where(dim1b == False)[0]
       idx2a = np.where(dim2a == False)[0]
       idx2b = np.where(dim2b == False)[0]
       flag_2d[idx1a,:][:,idx2a] = False
       flag_2d[idx1b,:][:,idx2b] = False
       #print('2, flag_2d:', flag_2d[idx1a,:].shape, flag_2d)
       
    else:         
       dim1  = np.logical_and( dima, np.logical_and( dimb, dimc ) )
       dim2  = ((pressure > threshs['pressure'][0]*0.999) & (pressure < threshs['pressure'][1]*1.003))      
          
       idx1    = np.where(dim1 == False)[0]
       idx2    = np.where(dim2 == False)[0]
       flag_2d[idx1,:] = False
       flag_2d[:,idx2] = False
    
    return flag_2d

def QC_flagIWC( pressure, values, gasname ):

    if gasname not in QC_thresh().keys():
       sys.exit('Add the thresholds for quality control of ' + gasname)
    else:   
       threshs = QC_thresh()[gasname]   
       
    flag_2d = (values <= threshs['range'])  
     
    if gasname == 'IWC':  

       dim2a = ((pressure > (threshs['pressure'][0]*0.999)) & (pressure < threshs['pressure'][1]*1.003))
       idx2a = np.where(dim2a == False)[0]
       flag_2d[:,idx2a] = False
       #print('2, flag_2d:', flag_2d[idx1a,:].shape, flag_2d)
    
    return flag_2d

###
def QC_Tflag( status, quality, convergence ):

    threshs = QC_thresh()['Temperature']   
     
    rem   = status % 2 
    dima = (rem == threshs['status'] )
    if isinstance(threshs['quality'], list):
       dimb1 = (quality > threshs['quality'][0])
       dimb2 = (quality > threshs['quality'][1])
    else:   
       dimb = (quality > threshs['quality'])
    dimc = (convergence < threshs['convergence'] )
                 
    flag_1d  = np.logical_and( dima, np.logical_and( dimb2, dimc ) )
    
    return flag_1d
            
###
def read_l3( filename, usetime=None ):
    '''
    read each L3 .nc monthly (MB) or daily (DB) file
    usetime: selected month (only valid for MB data)
    '''
    
    ifile = filename.split('/')[-1]
    gasname = ifile.split('-')[2].split('_')[0]
    
    #-- decide the reading group
    groups = [' PressureGrid',' ThetaGrid',' PressureZM',' ThetaZM']
    common_vars = ['lat','lon','lat_bnds','lon_bnds','time','time_bnds']
    diff_vars   = ['nvalues','lev','std_dev','value']
    data = dict()

    #-- open the file to read
    ncf = Dataset( filename, 'r' )
    #print(ncf.groups.keys())

    #-- read variables of each group
    for grp in groups:
      grp_name = gasname + grp
      print (grp_name)
      tempgrp = ncf.groups[ grp_name ]
      
      # common variables for each group
      if grp == groups[0]:
         for var in common_vars:
            data[var]  = tempgrp.variables[var][:]
      
      # different variables for each group  
      for var in diff_vars:
         if usetime is None:
            data[var+grp]   = tempgrp.variables[var][:]    
         else:
            itime = int(usetime) - 1
            data[var+grp]   = tempgrp.variables[var][itime,:] 
               
    #-- close file
    ncf.close()

    #-- screen print
    print('read_MLS l3 data: read ', filename)
    print(' ... variables: ', data.keys() )

   
    return data
    
###
def read_ifile( filename, region=[-180,180,-90,90], usetime=None, QC=True ):
    '''
    read each .he5 data at the time in the region 
    usetime (list): [start datetime, end datetime]
    region: [minlon, maxlon, minlat, maxlat]
    '''
        
    output = {}
    data   = {}
    gasname = filename.split('/')[-1].split('-')[2].split('_')[0]
    print('Start read MLS ', gasname, ' product ', filename.split('/')[-1])
    with h5py.File(filename, mode='r') as f:
       dset_var = f['HDFEOS/SWATHS/'+gasname+'/Data Fields/L2gpValue']
       fillvalue= dset_var.attrs['_FillValue']
       var0     = dset_var[:]
       var0[var0 == fillvalue] = np.nan
       dset_lat = f['HDFEOS/SWATHS/'+gasname+'/Geolocation Fields/Latitude'][:]
       dset_lon = f['HDFEOS/SWATHS/'+gasname+'/Geolocation Fields/Longitude'][:]
       dset_lev = f['HDFEOS/SWATHS/'+gasname+'/Geolocation Fields/Pressure'][:]
       dset_time = f['HDFEOS/SWATHS/'+gasname+'/Geolocation Fields/Time']
       time      = dset_time[:]
       precision = f['HDFEOS/SWATHS/'+gasname+'/Data Fields/L2gpPrecision'][:]
       status    = f['HDFEOS/SWATHS/'+gasname+'/Data Fields/Status'][:]
       quality   = f['HDFEOS/SWATHS/'+gasname+'/Data Fields/Quality'][:]
       convergence = f['HDFEOS/SWATHS/'+gasname+'/Data Fields/Convergence'][:]
    
       if gasname == 'Temperature':
          dset_var1 = f['HDFEOS/SWATHS/WMOTPPressure/Data Fields/L2gpValue']
          fillvalue2= dset_var1.attrs['_FillValue']
          WMOtropp  = dset_var1[:]
          WMOtropp[WMOtropp == fillvalue2] = np.nan
       elif gasname == 'IWC':
          dset_var1 = f['HDFEOS/SWATHS/IWP/Data Fields/L2gpValue']
          fillvalue2= dset_var1.attrs['_FillValue']
          IWP  = dset_var1[:]
          IWP[IWP == fillvalue2] = np.nan   
          
    # Quality control
    if QC:
      if gasname == 'IWC':
       use_flag1 = QC_flagIWC( dset_lev, var0, 'IWC' ) 
       var0[use_flag1 == False] = np.nan 
       use_flag2 = QC_flagIWC( dset_lev, IWP, 'IWP' ) 
       IWP[use_flag2 == False] = np.nan  
      else:   
       use_flag = QC_flag( dset_lev, precision, status, quality, convergence, gasname ) 
       var0[use_flag == False] = np.nan  
       
    if gasname == 'Temperature':
       Tflag =  QC_Tflag( status, quality, convergence )

    # Read the region    
    if ( len(region) != 4 ):
       sys.exit("read_ifile: Error 1, wrong region!") 

    lon1, lon2 = region[0], region[1]
    lat1, lat2 = region[2], region[3]   
       
    if (lon1 < lon2):
       idx = np.where( (dset_lon >= lon1) & (dset_lon <= lon2)  & (dset_lat >= lat1) & (dset_lat <= lat2) )[0]
    else:
       idx = np.where( ((dset_lon >= lon1) | (dset_lon <= lon2))  & ((dset_lat >= lat1) & (dset_lat <= lat2)) )[0] 
    #print('region idx:', lon[idx], lat[idx] )
    #print('ext data:', data1['aer_ext'].shape)  
       
    data['lat'] = dset_lat[idx]
    data['lon'] = dset_lon[idx]
    data[gasname] = var0[idx,:]
    data['lev'] = dset_lev
    data['time'] = time[idx] 
    if gasname == 'Temperature':
       data['WMOtropp'] = WMOtropp[idx]
       data['Tflag']    = Tflag[idx]
    elif gasname == 'IWC':
       data['IWP'] = IWP[idx,:]   
     
    # select time   
    if usetime is None:
       for var in data.keys():
          output[var] = data[var]  
    else:
       use_second = []
       for i in range(len(usetime)):
          isecond = (usetime - datetime.datetime(1993,1,1)).total_seconds()
          use_second.append( isecond )
       idx2 = np.where( (data['time'] >= use_second[0]) & (data['time'] <= use_second[1]) )[0]
       if len(idx2) == 0:
          print('Warning: There is no data between ', usetime[0], ' and ', usetime[1] )
       
       output['lat'] = data['lat'][idx2]
       output['lon'] = data['lon'][idx2]
       output[gasname] = data[gasname][idx2,:]
       output['lev'] = data['lev']
       output['time']= data['time'][idx2]
       if gasname == 'Temperature':
          output['WMOtropp'] = data['WMOtropp'][idx2]
          output['Tflag']    = data['Tflag'][idx2]
       elif gasname == 'IWC':
          output['IWP'] = data['IWP'][idx2,:] 
        
    return output
    
      
####
def cal_gas_STC_old( gas_prof, pres_ilev, Mpres, Mhgrid, Mairden ):
   '''
   calculate MLS gas column DU in the stratosphere by interpolating pressure level to 
   altitude level based on MERRA-2 data
   Mpres, Mhgrid: MERRA-2 pressure and height grids for MLS pixel
   pres_ilev: MLS pressure grid (n + 1)
   gas_prof: gas vmr profile (n)
   '''
   
   from scipy import interpolate
   
   #print('MERRA:', Mpres[::-1], Mhgrid[::-1])
   func = interpolate.splrep(Mpres[::-1], Mhgrid[::-1], s=0)
   hh   = interpolate.splev(pres_ilev[::-1], func, der=0)
   h_ilev = hh[::-1]
   boxh   = h_ilev[1:] - h_ilev[:-1] #m
   #print('boxh', boxh)
   
   func2 = interpolate.splrep(Mpres[::-1], Mairden[::-1], s=0)
   inputp = pres_ilev[::-1][:-1]
   output = interpolate.splev(inputp, func2, der=0)
   airden_ilev = output[::-1]#kg/m3
   airden_ilev[airden_ilev < 0] = 1e-10

   
   # vmr to DU
   # dry air number density cm-3
   Mair  = 28.9644 #g/mol
   NA    = 6.022e23
   Nair  = airden_ilev / (Mair * 1e-3) * NA # molecules/m3
   DU    = np.nansum( gas_prof * Nair * boxh * 1e-4 ) / 2.69e16 # molecules per cm2
   
   return DU    

####
def cal_gas_STC_ip( gas_prof, pres_gas, btpres=None ):
   '''
   calculation method follows Appendix A.3 in https://mls.jpl.nasa.gov/data/eos_algorithm_atbd.pdf
   Eq. A50 to Eq. A63, and section 6.2.1 in https://amt.copernicus.org/articles/8/195/2015/
   
   N(j)_minus = 0.102 * NA / (Mair * g) * vmr(j) * pres(j) = 2.18e25 * vmr(j) * pres(j) 
   N(j)_plus  = 0.090 * NA / (Mair * g) * vmr(j) * pres(j) = 1.92e25 * vmr(j) * pres(j)
   N(j)       = 0.192 * NA / (Mair * g) * vmr(j) * pres(j) = 4.12e25 * vmr(j) * pres(j)
   N          = N(a)_plus + N(b)_minus + sum(N(a+1 to b-1))
   a: the lower level, b: the upper level
   units: pres -- hPa; N(j) -- molecules/m2
   (1 DU = 2.687e20 molecules/m2)
   pres_gas: gas pressure levels within the upper and lower level calculated for stratospheric column
   '''
   
   output      = dict()
   
   if pres_gas[0] > pres_gas[1]:
      if btpres is None:
         lowidx  = 0
      else:
         lowidx   = np.where( (pres_gas - btpres) >= 0 )[0][-1]   
      highidx = -1
   else:
      if btpres is None:
         lowidx  = -1
      else:
         lowidx   = np.where( (pres_gas - btpres) >= 0 )[0][0]      
      highidx = 0   
      
   #print('check bottom level:', pres_gas, btpres, lowidx)   
   
   # sum
   Ng_minus     = 2.18e25 * gas_prof[highidx] * pres_gas[highidx]
   Ng_plus      = 1.92e25 * gas_prof[lowidx] * pres_gas[lowidx]
   if btpres is None:
      Ng_eq        = 4.12e25 * gas_prof[1:-1] * pres_gas[1:-1]
      Na_Col       = 2.18e25 * pres_gas[highidx] + 1.92e25 * pres_gas[lowidx] + np.sum( 4.12e25 * pres_gas[1:-1] )
   else:
      if pres_gas[0] > pres_gas[1]:
        Ng_eq        = 4.12e25 * gas_prof[lowidx+1:-1] * pres_gas[lowidx+1:-1]
        Na_Col       = 2.18e25 * pres_gas[highidx] + 1.92e25 * pres_gas[lowidx] + np.sum( 4.12e25 * pres_gas[lowidx+1:-1] )
      else:
        Ng_eq        = 4.12e25 * gas_prof[1:lowidx] * pres_gas[1:lowidx]
        Na_Col       = 2.18e25 * pres_gas[highidx] + 1.92e25 * pres_gas[lowidx] + np.sum( 4.12e25 * pres_gas[1:lowidx] )   

#    if np.isnan(Ng_minus):
#       Ng_minus  = 0.0
#    if np.isnan(Ng_plus):
#       Ng_plus  = 0.0   
   Ng_Col       = Ng_minus + Ng_plus + np.nansum( Ng_eq )
      
   # convert 
   # constants
   NA    = 6.022e23
   output['col_conc'] = Ng_Col / NA
   output['col_DU']   = Ng_Col / 2.687e20
   output['mean_VMR'] = Ng_Col / Na_Col
   output['col_air']  = Na_Col / NA
            
   return output      


####
def cal_gas_STmass_old( gas_prof, pres_gas, WMOtropp, area=None, Mgas=None, ):
   '''
   calculate MLS gas column DU in the stratosphere by calculate deltaH in each layer based on 
   pressure and temperature profile for each pixel
   gas_prof: gas vmr 2D array (npixels, nlevels)
   pres_gas: pressure level (nlevels)
   WMOtropp: WMO tropopause pressure (npixels)
   if calculate gas mass in the region, area (m2) and Mgas (g/mol) should be input
   
   '''
   output = dict()
   
   # check if the total number of pixels of gas profile and temperature profile are the same
   if gas_prof.shape[0] != len(WMOtropp):
      sys.exit('The number of pixels for gas and temperature product are not the same !!!')
      
   midpres_lev = np.exp((np.log(pres_gas[:-1])+np.log(pres_gas[1:]))/2)
   nlev        = len(pres_gas)
   npixel      = len(WMOtropp)
   
   # select pressure index based on tropopause pressure
   WMOtropp[WMOtropp > 130] = np.nan
   tropp_2D = np.tile( WMOtropp.reshape(-1,1), (1, nlev) )
   pres_2D  = np.tile( pres_gas.reshape(1,-1), (npixel, 1))
   gas_prof[(pres_2D - tropp_2D)>=0]     = np.nan
   gas_prof[np.isnan(tropp_2D)]          = np.nan
   
   # Eq1: lnP2 - lnP1 = g/(Ra*T)*(z1 - z2);  Ra = R (J/K/mol) / Mair (kg/mol) = 8.3143 / 28.9644e-3;
   # g = 9.8 m/s2
   # Eq2: P1 - P2  = - air_den * g * (z1 -z2) = air_den * g * boxh  (Pa, kg/m3, m/s2, m)
   # Eq3: Nair = air_den / Mair * NA
   # Eq4: gas_column = vmr * Nair * boxh (unitless, molecules/m3, m)
   # substitute Eq2 and Eq3 to Eq4: gas_column = vmr * (air_den / Mair * NA) * boxh
   #        = vmr * air_den / Mair * NA * (P1 - P2) / air_den / g
   #        = vmr / Mair * NA * (P1 - P2) /g
   
   # constants
   NA    = 6.022e23
   Mair  = 28.9644e-3
   g     = 9.8
   
   delta_pres    = (midpres_lev[:-1] - midpres_lev[1:]) * 100 #Pa
   #print('delta_pres:', delta_pres)
   gas_prof[gas_prof < 0] = 0.0
   delta_pres_2d = np.tile( np.append( np.nan, delta_pres ), (npixel,1) )
   delta_pres_2d[np.isnan(gas_prof[:,:-1])] = np.nan
   air_num       = delta_pres_2d * NA / Mair / g
   gas_ST_conc   = np.nansum( gas_prof[:,:-1] * air_num, axis=1 ) # molecules/m2 
   ST_mean_VMR   = np.nansum( gas_prof[:,:-1] * delta_pres_2d, axis=1 ) / np.nansum( delta_pres_2d, axis=1 )
   
   output['col_conc'] = gas_ST_conc
   output['col_DU']   = gas_ST_conc * 1e-4 / 2.69e16
   output['mean_VMR'] = ST_mean_VMR
   if area is not None:
      output['mass']     = np.nansum( gas_ST_conc * area ) / NA * Mgas * 1e-12 #Tg 
      output['col_air']  = np.nansum( air_num, axis=1 ) / NA

   
      
   return output

####
def cal_gas_STmass( gas_prof, pres_gas, WMOtropp, area=None, Mgas=None, uppres=10):
   '''
   calculation method follows Appendix A.3 in https://mls.jpl.nasa.gov/data/eos_algorithm_atbd.pdf
   Eq. A50 to Eq. A63, and section 6.2.1 in https://amt.copernicus.org/articles/8/195/2015/
   
   N(j)_minus = 0.102 * NA / (Mair * g) * vmr(j) * pres(j) = 2.18e25 * vmr(j) * pres(j) 
   N(j)_plus  = 0.090 * NA / (Mair * g) * vmr(j) * pres(j) = 1.92e25 * vmr(j) * pres(j)
   N(j)       = 0.192 * NA / (Mair * g) * vmr(j) * pres(j) = 4.12e25 * vmr(j) * pres(j)
   N          = N(a)_plus + N(b)_minus + sum(N(a+1 to b-1))
   a: the lowest level, b: the highest level
   units: pres -- hPa; N(j) -- molecules/m2
   (1 DU = 2.687e20 molecules/m2)
   '''
   
   nlev        = len(pres_gas)
   npixel      = len(WMOtropp)
   WMOtropp[WMOtropp > 130] = np.nan
   output      = dict()
   
   # select the level of 10hPa
   highidx  = np.where(np.abs(pres_gas - uppres)< 1e-5)[0][0]
   #print('highidx:', highidx)
   
   Ng_Col = np.full(npixel, np.nan)
   Na_Col = np.full(npixel, np.nan)
   valid_idx = np.where(~np.isnan(WMOtropp))[0]
   
   for ip in valid_idx:
      # select the level of tropopause
      lowidx   = np.where( (pres_gas - WMOtropp[ip]) > 0 )[0][-1]
      #print('lowidx:', ip, lowidx, WMOtropp[ip])
   
      # sum
      Ng_minus     = 2.18e25 * gas_prof[ip,highidx] * uppres
      Ng_plus      = 1.92e25 * gas_prof[ip,lowidx] * WMOtropp[ip]
      Ng_eq        = 4.12e25 * gas_prof[ip,lowidx+1:highidx] * pres_gas[lowidx+1:highidx]
      Ng_Col[ip]   = Ng_minus + Ng_plus + np.nansum( Ng_eq )
      Na_Col[ip]   = 2.18e25 * 10 + 1.92e25 * WMOtropp[ip] + np.sum( 4.12e25 * pres_gas[lowidx+1:highidx] )
      
   # convert 
   # constants
   NA    = 6.022e23
   output['col_conc'] = Ng_Col / NA
   output['col_DU']   = Ng_Col / 2.687e20
   output['mean_VMR'] = Ng_Col / Na_Col
   output['col_air']  = Na_Col / NA
   if area is not None:
      output['mass']  = np.nansum( Ng_Col * area ) / NA * Mgas * 1e-12 #Tg 
      
      
   return output   
   
####
def cal_SPW( wv_prof, pres_gas, botpres=100, uppres=10):
  '''
  calculate SPW (stratospheric precipitable water vapor) from H2O profile
  SPW = 1/(rho_w * g) * integral from P1 to P2 of (prof_w)*d(pres)
  Units: rho_w (1e3 kg/m3); g (9.8 m/s2); pres (Pa); prof_w (g/kg); SPW (mm) 
  '''
   
   
  # 1. convert from volume mixing ratio to mass mixing ratio g/kg: 
  # mmr = mw/mair = (Nw / NA * Mw) / (Nair / NA * Mair) = Nw / Nair * (Mw / Mair) = vmr * (Mw/Mair)
  mmr = wv_prof * (18 / 28.9644e-3)
  
  # 2. integral from 10 ~ 100 hPa: 
  pidx = np.where((pres_gas >= uppres*0.99) & (pres_gas <= botpres*1.01))[0]
  usepres = pres_gas[pidx[0]-1:pidx[-1]+2]
  deltap  = 0.5 * np.abs(usepres[:-2] - usepres[1:-1]) + 0.5 * np.abs(usepres[1:-1] - usepres[2:])
  #print('deltap:', deltap, usepres)

  if len(wv_prof.shape) == 1:
     useprof = mmr[pidx]
     colwv = np.sum(useprof * deltap * 100) 
  else:
     useprof = mmr[:,pidx]
     colwv = np.sum(useprof * np.tile(deltap * 100,(useprof.shape[0],1)), axis=1) 
    
  #print('check levs:', deltap.shape, useprof.shape)
  #print('colwv', useprof, colwv)
  
  SPW   = colwv / (1e3 * 9.8) * 0.1 # convert to cm

  return SPW
  
  
   
####
def VMR2RH( wv_prof, T_prof, pres_gas):
  '''
  calculate the relative humidity (%) from water vapor volume mixing ratio
  wv_prof, T_prof: (npixels,nlevs)
  pres_gas: (nlevs)
  
  references: 1. https://atoc.colorado.edu/~cassano/atoc3050/lecture_notes/chapter04.pdf
  2. https://journals.ametsoc.org/view/journals/apme/57/6/jamc-d-17-0334.1.xml?tab_body=pdf
  ''' 
   
  # 1. convert from volume mixing ratio to mass mixing ratio kg/kg: 
  # mmr = mw/mair = (Nw / NA * Mw) / (Nair / NA * Mair) = Nw / Nair * (Mw / Mair) = vmr * (Mw/Mair)
  mmr = wv_prof * (18 / 28.9644)
  
  # 2. convert from mass mixing ratio to vapor pressure (hPa, same as pres_gas):
  # e = mmr / (epsilon + mmr) * p
  epsilon = 0.622
  pres2D  = np.tile( pres_gas.reshape(1,-1), (wv_prof.shape[0],1) )
  e = mmr / (epsilon + mmr) * pres2D
  
  # 3. calculate saturated vapor pressure (Pa) from temperature profile:
  es = np.exp( 34.494 - 4924.99 / (T_prof - 36.05) ) / ( (T_prof - 168.15) ** 1.57 )
  
  # 4. RH 
  RH = e * 100 /es * 100
  
  
  return RH


####
def VMR2RHI( wv_prof, T_prof, pres_gas):
  '''
  calculate the relative humidity (%) to ice from water vapor volume mixing ratio
  wv_prof, T_prof: (npixels,nlevs)
  pres_gas: (nlevs)
  
  references: 1. https://atoc.colorado.edu/~cassano/atoc3050/lecture_notes/chapter04.pdf
  2. https://journals.ametsoc.org/view/journals/apme/57/6/jamc-d-17-0334.1.xml?tab_body=pdf
  ''' 
   
  # 1. convert from volume mixing ratio to mass mixing ratio kg/kg: 
  # mmr = mw/mair = (Nw / NA * Mw) / (Nair / NA * Mair) = Nw / Nair * (Mw / Mair) = vmr * (Mw/Mair)
  mmr = wv_prof * (18 / 28.9644)
  
  # 2. convert from mass mixing ratio to vapor pressure (hPa, same as pres_gas):
  # e = mmr / (epsilon + mmr) * p
  epsilon = 0.622
  pres2D  = np.tile( pres_gas.reshape(1,-1), (wv_prof.shape[0],1) )
  e = mmr / (epsilon + mmr) * pres2D
  
  # 3. calculate saturated vapor pressure (Pa) from temperature profile:
  es = np.exp( 43.494 - 6545.8 / (T_prof + 4.85) ) / ( (T_prof + 594.85) ** 2 )
  
  # 4. RH 
  RH = e * 100 /es * 100
  
  return RH    
  
####
def wH2SO4toTP_lut():
   '''
   The lut about water vapor weight (%) in sulfuric acid (H2SO4) variation with temperature and
   humidity in UTLS 
   
   reference: 
   https://www.sciencedirect.com/science/article/pii/0021850281900549
   '''
   
   H2Opp = [2,3,5,6]
   T     = [190,195,200,205,210,215,220,225,230,235,240,245,250,255,260]
   
   wH2SO4  = np.full((len(H2Opp),len(T)), np.nan)
   
   wH2SO4[0,:] = [43.45, 53.96, 60.62, 65.57, 69.42, 72.56, 75.17, 77.38, 79.3, 80.99, 82.5, 83.92, 85.32, 86.79, 88.32]
   wH2SO4[1,:] = [37.02, 49.46, 57.51, 63.12, 67.42, 70.85, 73.7, 76.09, 78.15, 79.96, 81.56, 83.02, 84.43, 85.85, 87.33]
   wH2SO4[2,:] = [25.85, 42.46, 52.78, 59.55, 64.55, 68.45, 71.63, 74.29, 76.56, 78.53, 80.27, 81.83, 83.27, 84.67, 86.1]
   wH2SO4[3,:] = [15.38, 39.35, 50.73, 58.11, 63.41, 67.52, 70.83, 73.6, 75.95, 77.98, 79.77, 81.38, 82.84, 84.25, 85.66]
#    wH2SO4[4,1:] = [34.02, 46.93, 55.61, 61.47, 65.94, 69.49, 72.44, 74.93, 77.08, 78.96, 80.63, 82.15, 83.57, 84.97]
#    wH2SO4[5,1:] = [29.02, 43.69, 53.44, 59.83, 64.62, 68.39, 71.48, 74.1, 76.33, 78.29, 80.02, 81.58, 83.03, 84.44]
   
#    newx = H2Opp[4:]
#    f = interpolate.interp1d(H2Opp[:4], wH2SO4[:4,0], fill_value="extrapolate", kind='quadratic')
#    wH2SO4[4:,0] = f(newx)
   
#    
   # interpolate 2D
#    f = interpolate.interp2d(H2Opp, T[1:], wH2SO4[:,1:].T, kind='linear')
#    newmr = f(H2Opp[4:], T[0:1])
#    print(newmr.shape)
#    wH2SO4[4,0] = newmr[0]
#    wH2SO4[5,0] = newmr[1]
   
   
   return wH2SO4, H2Opp, T

  
####
def exp_decay(t, a):
   return np.exp(-t*a)

def exp_decay2p(t, a, b):
   return b*np.exp(-t*a)
   
def power_decay2p(t, a, b):
   return b*(t**(-a))

def exp_decay_deri(t, a):
   return -a*np.exp(-t*a)
   
def fit_decay(t, y, endi=-1, starti=0, maxt0=False, usepower=False):
   
   valid = np.logical_and( ~np.isnan(t[starti:endi]), ~np.isnan(y[starti:endi]) )
   usey  = y[starti:endi][valid]
   uset  = t[starti:endi][valid]
   #print(usey, uset)
   if starti == 0:
     if maxt0:
        y0 = usey[0]
        idx0 = 0
     else:   
        y0 = np.max(usey)
        idx0 = np.argmax(usey)
     idx  = uset[idx0]
     if usepower:
        popt, pcov = optimize.curve_fit(power_decay2p, uset, usey, bounds=([0,0], [np.inf,np.inf]))
     else:   
        popt, pcov = optimize.curve_fit(exp_decay, uset[idx0:]-uset[idx0], usey[idx0:]/y0, bounds=([0], [np.inf]))
   else:
     if usepower:
        popt, pcov = optimize.curve_fit(power_decay2p, uset, usey, bounds=([0,0], [np.inf,np.inf]))
     else: 
        popt, pcov = optimize.curve_fit(exp_decay2p, uset, usey, bounds=([0,0], [np.inf,np.inf]))   
     y0 = popt[1]
     idx = starti
   
   return popt, valid, idx, y0



######
def VMRtoP(gas, vmr_prof, P):
  '''
  Convert H2SO4 VMR to vapor pressure
  '''
  if gas == 'H2O':
     Mgas = 18
  elif gas == 'H2SO4':   
     Mgas = 98 # g/mol 
  
  # 1. convert from volume mixing ratio to mass mixing ratio kg/kg: 
  # mmr = mw/mair = (Nw / NA * Mw) / (Nair / NA * Mair) = Nw / Nair * (Mw / Mair) = vmr * (Mw/Mair)
  mmr = vmr_prof * (Mgas / 28.9644)
  
  # 2. convert from mass mixing ratio to vapor pressure (hPa, same as pres_gas):
  # e = mmr / (epsilon + mmr) * p
  epsilon = 0.622
  e = mmr / (epsilon + mmr) * P
  #print(f'-mmr:{mmr[-100]}, vmr:{vmr_prof[-100]}, P:{P}')
  
  return e

def VMRtoN(vmr, T, P, V=1e3):
  '''
  per m3 air
  '''
  R = 8.31 # J/K/mol or m3Pa/K/mol
  
  nair = P * 100 * V / R / T
  ngas = nair * vmr   # every layer of OMPS (resolution: 1km)
  
  return ngas


def es_func(T, a):

  # at 298 K, 100% wt H2SO4
#   F_F0 = 0
#   L0   = 0
#   Cp0  = 33.2

  # 70% wt H2SO4
  F_F0 = -4838
  L0   = -6158
  Cp0  = 35.34  

  # constant
  R = 1.98726
  Ap = -3.95519
  Bp = -7413.3
  Cp = 7.03045 + np.log(760)
  Dp = 11.61146e-3
  Ep = -2.19062e-6
  
  # fit eq.24
  A  = Ap + 1/R * (Cp0 - 298*a)
  B  = Bp + 1/R * (L0 - 298*Cp0 + 298*298/2*a)
  C  = Cp + 1/R * (Cp0 + (F_F0 - L0)/298)
  D  = Dp - a/(2*R)
  E  = Ep
  
  # partial pressure (mmHg)
  p = np.exp( A * np.log(298/T) + B/T + C + D*T +E*T*T )

  return p
    
def es(T, gas='H2SO4'):
  '''
  saturated vapor pressure for gas at T temperature (hPa)
  Table A-I in reference: https://www.osti.gov/servlets/purl/876220 
  ''' 
  Ti = np.arange(-50,1,-5) # degree C
  ESi = [0.6084e-8, 0.1607e-7, 0.4066e-7, 0.9873e-7, 0.2307e-6, 0.5202e-6, 0.1133e-5, 0.2392e-5,
         0.4898e-5, 0.9745e-5, 0.1887e-4]
  
  sf = 133.3 # 1mmHg = Pa
  
  #ESi = ESi * sf / 100
  
  # fit a for each Ti
  #popt, pcov = optimize.curve_fit(es_func, Ti + 273.15, ESi)
  
  #print(popt)
  
  # 100 wt% H2SO4
  #alpha0 = 0.0509
  
  # 70% wt H2SO4
  alpha0 = 0.0895
   
  output = es_func(T, alpha0) * sf / 100 # hPa
  
  return output

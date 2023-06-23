import os, sys, json
import numpy as np
import datetime, pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate, stats
from sklearn.linear_model import LinearRegression


import mylib.MLS_lib as MLS
from plot_libs import *
import Tonga_ana as Ta



####################
# read all satellite data in 2022
#####################
removeN2O=False
allzm, glevs, levs, nday, mmname, clats, alt = Ta.comb_TaerWV( removeN2O=removeN2O )
if removeN2O:
   lab=''
else:
   lab='N2O'   

# 5 days smooth
alldata = Ta.smo_timeseries( allzm, nday, smo_day = 5)

#####################
# read previous years Tanom
######################
savestr  = 'detrend'
data0, sxidx, exidx = Ta.read_Tanom( [2005,1,1], [2021,12,1], savestr, ttype='daily')
Tlevs0     = data0['Tlevs']
allyearT   = data0['Tprof'][sxidx:exidx,:,:]
   
####################
# scatter plot of Tanom vs. WV or Aer
########################
cho_lat = clats[(clats <= -14) & (clats >= -56)]
plt_lat = np.arange(-56,-15,8)
ndenmax = [4,8]
deltanden = [1,2]
wvmin   = [3, 4]
wvmax   = [21, 10]
deltawv = [3, 1]  
#cho_pres = [68, 46, 32, 21, 15, 10]
xday    = np.arange(nday)

# len(levs) < len(glevs)
# match to levs
glidx = []
if len(levs) < len(glevs['Temperature']):
   for ilev in levs:
      glidx.append( np.argmin(np.abs(glevs['Temperature']-ilev)) )
print('match lev idx:', len(glevs['Temperature'][glidx]), len(levs))
pres_flag = (glevs['Temperature'][glidx] <= 100) & (glevs['Temperature'][glidx] >= 7) 
cho_pres = glevs['Temperature'][glidx][pres_flag]

ompsaer= np.full( alldata['Tanom'].shape, np.nan ) 
alldata['H2Oanom'] = np.full( alldata['H2O'].shape, np.nan )

fig, axes = plt.subplots(nrows=len(cho_pres), ncols=len(cho_lat), figsize=(3.2*len(cho_lat),2.5*len(cho_pres)))
for ic in range(len(cho_lat)):
   zidx   = np.argmin(np.abs(clats - cho_lat[ic]))
#    if cho_lat[ic] <= -40:
#       botpres = 70
#    else:
#       botpres = 100   
#    Mlidx  = np.where((glevs['Temperature'] <= botpres) & (glevs['Temperature'] >= 10)  )[0]
#    Tlevs  = ((levs <= botpres) & (levs >= 10))
   Mlidx = np.where(pres_flag == True)[0]
   #print('check levs:', glevs['Temperature'], levs, glevs['H2O'])
   useMLS = alldata['Temperature'][:,zidx,:][:,glidx][:,Mlidx]
   if ic == 1:
      print( alldata['Temperature'][:,zidx,:][20,:] ) 
      
   for idd in range(nday):
      intpaer, altT = Ta.omps_pres( alldata['aer'][idd,zidx,:] * 1e3, alt, useMLS[idd,:], glevs['Temperature'][glidx][Mlidx] )
        #ompsidx = np.argmin( np.abs(alt - altT[lidxaer] ) )
      if idd == 100:
           print(idd, glevs['Temperature'][glidx][Mlidx], altT)
           #sys.exit()
      ompsaer[idd,zidx,pres_flag] = intpaer   
        #print(ompsaer[idd,zidx,:], len(intpaer), altT)  
    
      
   for il in range(len(cho_pres)):
      lidxT  = np.argmin(np.abs(levs - cho_pres[il]))
      lidxH2O= np.argmin(np.abs(glevs['H2O'] - cho_pres[il]))
      
      # H2O & aer anomaly
      ompsaer[:,zidx,lidxT] = ompsaer[:,zidx,lidxT] - np.nanmean(ompsaer[:40,zidx,lidxT])
      alldata['H2Oanom'][:,zidx,lidxH2O] = alldata['H2O'][:,zidx,lidxH2O] - np.nanmean(alldata['H2O'][:40,zidx,lidxH2O])
      nvalid1 = len(np.where(~np.isnan(alldata['Tanom'][:,zidx,lidxT]))[0])
      nvalid2 = len(np.where(~np.isnan(ompsaer[:,zidx,lidxT]))[0])
      nvalid3 = len(np.where(~np.isnan(alldata['H2Oanom'][:,zidx,lidxH2O]))[0])
      
      
      if ((nvalid1 >= 2) and (nvalid2 >= 2)) and (nvalid3 >= 2): 

         paths, slope, intercept, min_all, max_all = sca.scatter_plot(axes[il,ic], 
             alldata['Tanom'][:,zidx,lidxT], ompsaer[:,zidx,lidxT],
             fig=None, mcolor='b', label_p = 'upper right', prt_mean=False, prt_rmse=False, \
             linearregress_line=True, one2one_line=False, alpha=0.5, )   
         print('slope = '+str(slope)+', intercept = '+str(intercept))   

         ax2     = axes[il,ic].twinx()
         paths2, slope2, intercept2, min_all2, max_all2 = sca.scatter_plot(ax2, 
             alldata['Tanom'][:,zidx,lidxT], alldata['H2Oanom'][:,zidx,lidxH2O] * 1e6,
             fig=None, mcolor='g', label_p = 'lower right', prt_mean=False, prt_rmse=False, \
             linearregress_line=True, one2one_line=False, alpha=0.5, )   
         print('slope = '+str(slope2)+', intercept = '+str(intercept2))   
 
         axes[il,ic].yaxis.label.set_color('b') 
         axes[il,ic].tick_params(axis='y', colors='b')  
         axes[il,ic].spines['left'].set_color('b')   
         axes[il,ic].set_title(r'{:.0f} hPa, {:d}$^\circ$'.format(cho_pres[il], cho_lat[ic]))
         axes[il,ic].set_xlabel(r'$\Delta$T (K)')
         axes[il,ic].set_ylabel(r'$\Delta$AerExt (10$^{-3}$km$^{-1}$)')

         ax2.yaxis.label.set_color('g') 
         ax2.tick_params(axis='y', colors='g')  
         ax2.spines['right'].set_color('g')   
         ax2.set_ylabel(r'$\Delta$H$_2$O VMR (ppmv)')

         
plt.tight_layout()
plt.savefig( 'img/MLS&omps_Tanom_vs_WVaer_scatter.png', dpi=300 )
plt.close()  
   

#######

# ----------------------------
# multiple regression
cho_lat = clats[(clats <= -14) & (clats >= -56)]
plt_lat = np.arange(-56,-15,8)
ndenmax = [4,8]
deltanden = [1,2]
wvmin   = [3, 4]
wvmax   = [21, 10]
deltawv = [3, 1]  
#cho_pres = [68, 46, 32, 21, 15, 10]
xday    = np.arange(nday)
nlag    = 45
deltalag= 1
lagcorr = dict()
pcorr   = dict()
fit_T   = dict()
totalr  = np.full( (len(cho_lat), len(cho_pres), 6), np.nan )

allname = ['H2O','Aer']
if 'N2O' in alldata.keys():
   allname.append( 'N2O' )
labelname = [r'H$_2$O', 'Aer', r'N$_2$O']
   

for key in ['H2O','Aer']:
   lagcorr[key] = np.full( (len(cho_lat), len(cho_pres), nlag), np.nan )
for key in allname:   
   pcorr[key] = np.full( (len(cho_lat), len(cho_pres)), np.nan )
for i in range(len(cho_lat)*len(cho_pres)):   
   fit_T[i] = dict()
fitcorr = np.full( (len(cho_lat), len(cho_pres)), np.nan )

for ic in range(len(cho_lat)):
   
   fig, axes = plt.subplots(nrows=len(cho_pres), ncols=1, figsize=(8,2.5*len(cho_pres)))
   zidx   = np.argmin(np.abs(clats - cho_lat[ic]))
      
   for il in range(len(cho_pres)):
      lidxT  = np.argmin(np.abs(levs - cho_pres[il]))
      lidxH2O= np.argmin(np.abs(glevs['H2O'] - cho_pres[il]))
      
      # valid flag
      flag1 = ~np.isnan(alldata['Tanom'][:,zidx,lidxT])
      flag2 = ~np.isnan(ompsaer[:,zidx,lidxT])
      flag3 = ~np.isnan(alldata['H2Oanom'][:,zidx,lidxH2O])
      flag  = np.logical_and( np.logical_and( flag1, flag2 ), flag3 )
      if 'N2O' in alldata.keys():
         flag4 = ~np.isnan(alldata['N2O'][:,zidx,lidxT])
         flag  = np.logical_and( flag, flag4 )
      
      
      # linear regression
      if len(np.where(flag==True)[0]) > 2: 
      
        # simple lag correlation
        day0 = 31-1
        if len(np.where(flag==True)[0]) > day0: 
           xdata = dict()
           xdata['H2O'] = alldata['H2Oanom'][:,zidx,lidxH2O][flag][day0:]*1e6
           if 'N2O' in alldata.keys():
              xdata['N2O'] = alldata['N2O'][:,zidx,lidxT][flag][day0:]
           xdata['Aer'] = ompsaer[:,zidx,lidxT][flag][day0:]
           ydata = alldata['Tanom'][:,zidx,lidxT][flag][day0:]
           for ilag in range(nlag):
              lagcorr['H2O'][ic,il,ilag] = sca.lagcorr_func( xdata['H2O'], ydata, ilag*deltalag )  
              lagcorr['Aer'][ic,il,ilag] = sca.lagcorr_func( xdata['Aer'], ydata, ilag*deltalag )
              #lagcorr['Aer'][ic,il,ilag] = sca.lagcorr_func( xdata['Aer'], ydata, 0 )    
                   
           # only the current time step
           #X = np.vstack( (ompsaer[:,zidx,lidxT][flag], alldata['H2Oanom'][:,zidx,lidxH2O][flag]*1e6) )
           #y = np.reshape( alldata['Tanom'][:,zidx,lidxT][flag], (-1,1) )
        
           # past time step with maxlag day
           ilagcorr = dict()
           for key in lagcorr.keys():
              ilagcorr[key] = lagcorr[key][ic,il,:]
              
           X, y, allmaxt, nt = Ta.comb_Xy( xdata, ydata, ilagcorr, deltalag, nidx=1 )
           
           # use linear regression     
           #reg = LinearRegression().fit(X.T, y)
           #predy  = reg.predict(X.T)[:,0]
           #print('score, coef:', cho_lat[ic], cho_pres[il], reg.score(X.T, y), reg.coef_, reg.intercept_)
           
           # set up bounds for parameters
           pars = Ta.fit_each( X, y[:,0], use_bd=True )
           predy = Ta.fitfunc( X, *pars )
           
           abs_allfit = 0
           for ik in range(len(allname)):
              #fit_T[ic*len(cho_pres)+il][allname[ik]] = reg.coef_[0][ik] * X[ik,:]
              fit_T[ic*len(cho_pres)+il][allname[ik]] = pars[ik] * X[ik,:]
              abs_allfit = abs_allfit + np.abs( pars[ik] * X[ik,:] )
           abs_allfit = abs_allfit + np.abs( pars[-1] )   
           fit_T[ic*len(cho_pres)+il]['coef'] = pars #reg.coef_[0]
           fit_T[ic*len(cho_pres)+il]['xday'] = xday[flag][day0:][allmaxt:allmaxt+nt]
           fit_T[ic*len(cho_pres)+il]['obs'] = alldata['Tanom'][:,zidx,lidxT]
           #fit_T[ic*len(cho_pres)+il]['interc'] = reg.intercept_[0]
           fit_T[ic*len(cho_pres)+il]['all']    = predy
           fit_T[ic*len(cho_pres)+il]['all_abs']= abs_allfit
      
           axes[il].scatter( xday, alldata['Tanom'][:,zidx,lidxT], s=5, c='r', marker='o', label='Obs')
           pltx   = xday[flag][day0:][allmaxt:allmaxt+nt]
           #pltx   = xday[flag] 
           axes[il].plot( pltx, predy, color='k', label='Fit' )
           axes[il].axhline( y=0, color='k', lw=0.3, ls='-' )  
           axes[il].set_title(r'{:.0f} hPa, {:d}$^\circ$'.format(cho_pres[il], cho_lat[ic]))
           axes[il].xaxis.set_minor_locator(ticker.MultipleLocator(5))
           axes[il].set_xticks(xday[::30])
           axes[il].set_xticklabels( mmname )
           axes[il].set_ylabel('$\Delta$T (K)')  
           
           
           # partial correlation
           corr = Ta.partial_corr( X.T, y, addN2O=~removeN2O )
           print(corr[key]['r'][0])
           for key in pcorr.keys():
              pcorr[key][ic,il] = corr[key]['r'][0]
              
           # total correlation   
           icorr = Ta.total_corr( X.T, y[:,0] )   
           rname = list(icorr.keys())
           for ik in range(len(rname)):
              totalr[ic,il,ik] = icorr[rname[ik]]  
              
           # fit vs obs
           cho_obs = fit_T[ic*len(cho_pres)+il]['obs'][flag][day0:][allmaxt:allmaxt+nt]
           fitcorr[ic,il] = stats.pearsonr(cho_obs, fit_T[ic*len(cho_pres)+il]['all'])[0]   
           
           # residual
           fit_T[ic*len(cho_pres)+il]['res'] = cho_obs - fit_T[ic*len(cho_pres)+il]['all']  
        
   axes[0].legend()
         
   plt.tight_layout()
   if cho_lat[ic] in plt_lat:
      plt.savefig( 'img/MLS&omps_Tanom_timeseries_DectoNovb_fit2_'+str(cho_lat[ic])+'lat_1lag'+lab+'.png', dpi=300 )
   plt.close()  

#sys.exit()
# ---------------------
# plot lag correlation between WV and Tanom
maxcorr = dict()
for key in lagcorr.keys():
   fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12,8))
   lcolors = plt.cm.jet( np.linspace(0,1,len(cho_pres)) )
   axs = axes.flatten()
   maxcorr[key] = np.full( (len(cho_lat), len(cho_pres)), np.nan )
   for ic in range(len(cho_lat)):
     for il in range(len(cho_pres)):
       maxlag = np.argmax(np.abs(lagcorr[key][ic,il,:]) ) 
       maxcorr[key][ic,il] = lagcorr[key][ic,il,maxlag]
       if cho_lat[ic] in plt_lat:  
         zidx = np.argmin( np.abs( plt_lat - cho_lat[ic] ) )
         if np.isnan( maxlag ):
          ilabel = '{:.0f} hPa, NAN'.format(cho_pres[il])
         else:
          ilabel = '{:.0f} hPa, {:d}'.format(cho_pres[il],int(maxlag)* deltalag  )
         axs[zidx].plot( np.arange(0,nlag*deltalag,deltalag), lagcorr[key][ic,il,:], color=lcolors[il], label=ilabel )
         axs[zidx].set_xlabel( 'lag days' )
         axs[zidx].set_ylabel( 'corr' ) 
         axs[zidx].set_title( r'{:d}$^\circ$'.format(cho_lat[ic]))   
         axs[zidx].legend(ncol=3,fontsize='x-small')   
   plt.tight_layout()
   #plt.savefig( 'img/MLS&omps_Tanom_timeseries_DectoNovb_fit'+key+'_nlag1.png', dpi=300 )
   plt.close() 


 
# only total correlation 
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12,8))
vmin = -1
vmax = 1
#levels = np.arange(vmin, vmax+0.1, 0.1)
#cmap = plt.cm.bwr 

Tcorr = dict()
othercorr = dict()
for key in ['name','idx']:
   Tcorr[key] = []
   othercorr[key] = []
   
for i in range(len(rname)):
   if 'T' in rname[i]:
      Tcorr['name'].append( rname[i] )
      Tcorr['idx'].append( i )
   else:
      othercorr['name'].append( rname[i] )
      othercorr['idx'].append( i )
      
for i in range(len(Tcorr['name'])):
      MLS.plot_timeseries( axes[i,0], cho_lat, cho_pres, totalr[:,:,Tcorr['idx'][i]], levels=levels, 
                        norm=norm, set_xminor=True, xminor=1, plt_lines=False,
                        xticks=np.arange(-55,-14,5), xticklabel=np.arange(-55,-14,5), xlims=[-55,-15],
                        xlabel=r'Latitude ($\circ$)', cbar_ylabel='R (total)', padrat=0.1,
                        ylabel='Pressure (hPa)',cmap=cmap, usepcol=True,
                        ylims=[80,10], cbticks=levels[:], cbarwd='2%',
                        title=r'Total correlation coefficient between $\Delta$'+Tcorr['name'][i].split('&')[0]+
                              r' and $\Delta$'+Tcorr['name'][i].split('&')[1])
for i in range(len(othercorr['name'])):
      MLS.plot_timeseries( axes[i,1], cho_lat, cho_pres, totalr[:,:,othercorr['idx'][i]], levels=levels, 
                        norm=norm, set_xminor=True, xminor=1, plt_lines=False,
                        xticks=np.arange(-55,-14,5), xticklabel=np.arange(-55,-14,5), xlims=[-55,-15],
                        xlabel=r'Latitude ($\circ$)', cbar_ylabel='R (total)', padrat=0.1,
                        ylabel='Pressure (hPa)',cmap=cmap, usepcol=True,
                        ylims=[80,10], cbticks=levels[:], cbarwd='2%',
                        title=r'Total correlation coefficient between $\Delta$'+othercorr['name'][i].split('&')[0]+
                              r' and $\Delta$'+othercorr['name'][i].split('&')[1])
plt.tight_layout()
#plt.savefig( 'img/MLS&omps_Tanom_timeseries_DectoNovb_totalcorr_lag1.png', dpi=300 )
plt.close() 
  
# fitted vs. obs correlation
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7,3))
vmin = 0
vmax = 1
#levels0 = np.arange(vmin,-0.4,0.1)
#levels1 = np.arange(-0.4,0.4,0.4)
#levels2 = np.arange(0.4,vmax+0.1,0.1)
levels = np.arange(vmin, vmax+0.05, 0.05)
#cmap = plt.cm.coolwarm
#levels = np.append( np.append(levels0, levels1), levels2 )
usecmp = plt.cm.get_cmap('rainbow')(np.linspace(0,1, len(levels) - 1))
cmap, norm = colors.from_levels_and_colors(levels, usecmp) 

MLS.plot_timeseries( axes, cho_lat, cho_pres, fitcorr, levels=levels, norm=norm, 
                        set_xminor=True, xminor=1, plt_lines=False,
                        xticks=np.arange(-55,-14,5), xticklabel=np.arange(-55,-14,5), xlims=[-55,-15],
                        xlabel=r'Latitude ($\circ$)', cbar_ylabel='R', padrat=0.1,
                        ylabel='Pressure (hPa)',cmap=cmap, usepcol=True,
                        ylims=[80,10], cbticks=levels[::2], cbarwd='2%',
                        title=r'Correlation coefficient between fitted and observed $\Delta$T')


plt.tight_layout()
plt.savefig( 'img/MLS&omps_Tanom_timeseries_DectoNovb_fitcorr_lag1.png', dpi=300 )
plt.close() 



###################

 
# 2. delta T from fitting
colors = ['g','b','c']
ndenmax = [5]
deltanden = [1]
contri   = dict()
eachfit  = dict()
coef = dict()
for key in allname:
   contri[key]   = np.full( (len(cho_lat), len(cho_pres)), np.nan)
   coef[key]     = np.full( (len(cho_lat), len(cho_pres)), np.nan)
   eachfit[key]  = np.full( (len(xday), len(cho_lat), len(cho_pres)), np.nan)
allfit = np.full( (len(xday), len(cho_lat), len(cho_pres)), np.nan)

for ic in range(len(cho_lat)):
   fig, axes = plt.subplots(nrows=len(cho_pres), ncols=1, figsize=(6.3,2.2*len(cho_pres)))
      
   for il in range(len(cho_pres)):
      if len(list(fit_T[ic*len(cho_pres)+il].keys())) == 0:
         print('No data for ', cho_pres[il], cho_lat[ic] )
         continue
      xday0 = np.arange(len(fit_T[ic*len(cho_pres)+il]['obs'])) 
      #axes[il] = Ta.plt_timeseries_fillx( axes[il], xday0, fit_T[ic*len(cho_pres)+il]['obs'], color='r',
      #           xticks=xday[::30], xticklabel=mmname, xminor=5, label='Obs', 
      #           ylims=[-ndenmax[ic],ndenmax[ic]], yticks=np.arange(-ndenmax[ic],ndenmax[ic]+deltanden[ic],deltanden[ic]), 
      #           ylabel=r'$\Delta$T (K)',xlabel='Days after 1 Dec 2021', 
      #           )
      axes[il] = Ta.plt_timeseries_fillx( axes[il], fit_T[ic*len(cho_pres)+il]['xday'], fit_T[ic*len(cho_pres)+il]['all'], color='k',
                 xticks=xday[::30], xticklabel=mmname, xminor=5, label='Fitted', 
                 ylims=[-ndenmax[0],ndenmax[0]], yticks=np.arange(-ndenmax[0],ndenmax[0]+deltanden[0],deltanden[0]), 
                 ylabel=r'$\Delta$T (K)',xlabel='Days after 1 Dec 2021', 
                 )
      axes[il].axhline( y=0, color='k', lw=0.3, ls='-' )  
      allfit[fit_T[ic*len(cho_pres)+il]['xday'],ic,il] = fit_T[ic*len(cho_pres)+il]['all']
      
      for ik in range(len(allname)):
         perc = np.nanmean( np.abs(fit_T[ic*len(cho_pres)+il][allname[ik]]) / fit_T[ic*len(cho_pres)+il]['all_abs'] ) *100
#          if np.abs(cho_pres[il] - 68) < 1 and (cho_lat[ic] > -32):
#             print(cho_lat[ic], allname[ik], fit_T[ic*len(cho_pres)+il]['all_abs'], np.abs(fit_T[ic*len(cho_pres)+il][allname[ik]]))
         contri[allname[ik]][ic,il] = perc
         eachfit[allname[ik]][fit_T[ic*len(cho_pres)+il]['xday'],ic,il] = fit_T[ic*len(cho_pres)+il][allname[ik]]
         coef[allname[ik]][ic,il] = fit_T[ic*len(cho_pres)+il]['coef'][ik]
         
         axes[il] = Ta.plt_timeseries_fillx( axes[il], fit_T[ic*len(cho_pres)+il]['xday'], 
                 fit_T[ic*len(cho_pres)+il][allname[ik]], color=colors[ik], 
                 label=labelname[ik]+': {:.2e}&{:.1f}%'.format(coef[allname[ik]][ic,il],perc),
                 xticks=xday[::30], xticklabel=mmname, xminor=5, ylims=[-ndenmax[0],ndenmax[0]], 
                 yticks=np.arange(-ndenmax[0],ndenmax[0]+deltanden[0],deltanden[0]),
                 ylabel=r'$\Delta$T (K)',xlabel='Days after 1 Dec 2021',  
                 title=r'Evolution at {:.0f} hPa at latitude {:d}$^\circ$'.format(
                 cho_pres[il], cho_lat[ic]) )
             
      axes[il].scatter( xday, fit_T[ic*len(cho_pres)+il]['obs'], s=0.25, c='r', marker='o', label='Obs')
      
      
      #axes[il] = Ta.plt_timeseries_fillx( axes[il], fit_T[ic*len(cho_pres)+il]['xday'], 
      #           allfit, color='k', label='All',
      #           xticks=xday[::30], xticklabel=mmname, xminor=5, ylims=[-ndenmax[ic],ndenmax[ic]], 
      #           yticks=np.arange(-ndenmax[ic],ndenmax[ic]+deltanden[ic],deltanden[ic]), )
                 
      for ix in [20, 43, 45]:
         axes[il].axvline( x=ix, color='k', linewidth=0.8, linestyle='--')         
      axes[il].legend(ncol=3, fontsize='small') 

   plt.tight_layout()
   plt.savefig( 'img/MLS&omps_fitTanom_timeseries_DectoNovb_lines'+lab+'_'+str(cho_lat[ic])+'lat.png', dpi=300 )
   plt.close()  

      
# 3. fitting coefficient
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9.5,4))
axs  = axes.flatten()
colors2 = plt.cm.rainbow(np.linspace(0,1,int(len(cho_lat)/2)))
xmin = [-6,0.,-0.2]
xmax = [0,1.6,0.2]
deltax = [1.,0.2,0.1]
xlabs = ['a','b','c']
unit  = ['(K/ppmv)',r'(K/(10$^{-3}$km$^{-1}$))','(K/ppbv)']
latcolors = ['m', 'orange']

for ic in range(len(reg)):
   zidx = ( cho_lat < reg[ic][0] ) & ( cho_lat >= reg[ic][1] )
   for ik in range(len(allname)):
      idata = coef[allname[ik]][zidx,:]
      mc  = np.nanmean(idata, axis=0)
      stdc = np.nanstd(idata, axis=0)
      axs[ik].plot( mc, cho_pres, marker='o', markersize=3, color=latcolors[ic], label='{:d}$^\circ$S-{:d}$^\circ$S'.format(abs(reg[ic][1]), abs(reg[ic][0]) ), )
      axs[ik].fill_betweenx( cho_pres, x1=mc-stdc, x2=mc+stdc, facecolor=latcolors[ic], alpha=0.2, edgecolor=None)
      axs[ik] = Ta.plt_timeseries_fillx( axs[ik], [0,1], 
                 [np.nan, np.nan], color=latcolors[ic], xlims=[xmin[ik],xmax[ik]],
                 xticks=np.arange(xmin[ik],xmax[ik]+deltax[ik],deltax[ik]), xminor=None,
                 ylims=[70,10], ylabel='Pressure (hPa)',xlabel=xlabs[ik]+' '+unit[ik],  #Determination coefficient 
                 title='Linear coefficient for '+labelname[ik] )
                 
axs[1].legend()      
plt.tight_layout()
plt.savefig( 'img/MLS&omps_fitTanom_coef_lag1_prof_reg.png', dpi=300 )
plt.close()  


# contribution %
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5,4))
axs  = axes.flatten()
colors2 = ['g','b','c']
for ic in range(len(reg)):
   zidx = ( cho_lat < reg[ic][0] ) & ( cho_lat >= reg[ic][1] )
   for ik in range(len(allname)):
      mc  = np.nanmean(contri[allname[ik]][zidx,:], axis=0)
      stdc = np.nanstd(contri[allname[ik]][zidx,:], axis=0)
      axs[ic].plot( mc, cho_pres, marker='o', markersize=3, color=colors2[ik], label=labelname[ik], )
      axs[ic].fill_betweenx( cho_pres, x1=mc-stdc, x2=mc+stdc, facecolor=colors2[ik], alpha=0.1, edgecolor=None)
      axs[ic] = Ta.plt_timeseries_fillx( axs[ic], [0,1], 
                 [np.nan, np.nan], color=colors2[ik], xlims=[-1,80],
                 xticks=np.arange(0,81,20), ylims=[70,10], xminor=10,
                 ylabel='Pressure (hPa)',xlabel=r'Contribution to $\Delta$T (%)',  #Determination coefficient 
                 title='{:d}$^\circ$S-{:d}$^\circ$S'.format(abs(reg[ic][1]), abs(reg[ic][0]) ) )
   axs[ic].legend()
               
plt.tight_layout()
plt.savefig( 'img/MLS&omps_Tanom_contri_lag1_prof_reg.png', dpi=300 )
plt.close()  


# 4. combined obs and fitted of each factor
colors2 = ['g','b','c']
ndenmax = [5,8]
deltanden = [1,2]
for ic in range(len(reg)):
   zidx1 = ( cho_lat < reg[ic][0] ) & ( cho_lat >= reg[ic][1] )
   zidx  = ( clats < reg[ic][0] ) & ( clats >= reg[ic][1] )
   fig, axes = plt.subplots(nrows=len(cho_pres), ncols=1, figsize=(6.5,2.2*len(cho_pres)))
      
   for il in range(len(cho_pres)):
      if len(list(fit_T[ic*len(cho_pres)+il].keys())) == 0:
         print('No data for ', cho_pres[il], cho_lat[ic] )
         continue
      
      lidxT  = np.argmin(np.abs(levs - cho_pres[il]))
      lidxH2O= np.argmin(np.abs(glevs['H2O'] - cho_pres[il]))   

      # a1. Tanom
      mean1   = np.nanmean(alldata['Tanom'][:,zidx,lidxT],axis=1)
      std1    = np.nanstd(alldata['Tanom'][:,zidx,lidxT],axis=1)
      print('max:', ic, mean1.shape, np.nanmax(mean1))
      print(np.nanmax(ompsaer),)
      
      # a2. fitted Tanom
      axes[il].plot( xday, np.nanmean(allfit[:,zidx1,il],axis=1), color='r', ls='--', lw=1.5, label=r'All fitted $\Delta$T' )
      for ik in range(len(allname)):
         axes[il].plot( xday, np.nanmean(eachfit[allname[ik]][:,zidx1,il],axis=1), color=colors2[ik], ls='--',
                 lw=1.5, label=r'$\Delta$T from '+labelname[ik] )
      
      axes[il] = Ta.plt_timeseries_fillx( axes[il], xday, mean1, y1=mean1-std1, y2=mean1+std1, color='r',
                 label='MLS Obs', alpha=0.15,ylabel=r'$\Delta$T (K)' ,
                 ylims=[-ndenmax[ic],ndenmax[ic]], yticks=np.arange(-ndenmax[ic],ndenmax[ic]+deltanden[ic],deltanden[ic]), 
                 xticks=xday[::30], xticklabel=mmname, xminor=5,xlabel='Days after 1 Dec 2021', 
                 title=r'Evolution at {:.0f} hPa in latitudes {:d}$^\circ$S-{:d}$^\circ$S'.format(
                 cho_pres[il], abs(reg[ic][0]), abs(reg[ic][1]))  )
      axes[il].axhline( y=0, color='k', lw=0.3, ls='-' )  
      for ix in [20, 43, 45]:
         axes[il].axvline( x=ix, color='k', linewidth=0.8, linestyle='--')         
      axes[il].yaxis.label.set_color('r') 
      axes[il].tick_params(axis='y', colors='r')  
      axes[il].spines['left'].set_color('r')  
      if il%2 == 0:
         axes[il].legend(loc='upper left',ncol=3,fontsize='small')       

   plt.tight_layout()
   plt.savefig( 'img/MLS&omps_eachfitTanom_timeseries_DectoNovb_lines'+lab+'_reg'+str(ic+1)+'.png', dpi=300 )
   plt.close()  

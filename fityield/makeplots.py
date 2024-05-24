import numpy as np

import sys, glob, os, subprocess
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from scipy import constants
from scipy.stats import chi2

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib 
import matplotlib.ticker as ptick
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

#matplotlib.style.use('ggplot')


import seaborn as sns


fityield_dir=os.environ['HOME']+'/HMP/fityield'

sys.path.insert(0, os.path.abspath(fityield_dir))

from fityield import abundance,yields,fitparams





def plot_abupattern(x_obs0, y_obs0, sig_p, sig_m, flags, x_model, y_model, outfigname, melemlows=[21,22]): 

    cmap = plt.get_cmap("tab10")  
    
    plt.rcParams["font.size"] = 14


    

    fig, ax = plt.subplots(1,1)
    filt = (x_model != 9) & (x_model != 21)
    ymin = np.min(np.append(y_obs0, y_model[filt]))
    ymax = np.max(np.append(y_obs0, y_model))
    y1 = ymin - (ymax - ymin) * 0.1
    y2 = ymax + (ymax - ymin) * 0.2
    x1 = np.min(x_model) - 1
    x2 = np.max(x_model) + 1
    elems = abundance.get_elem_name(x_obs0)
    elemzs = x_obs0

    # Theoretical lower limit
    #for melemlow in melemlows:
    #    mlow = y_model[x_model == melemlow]
    #    xx1 = melemlow - 0.1
    #    xx2 = melemlow + 0.1
    #    yy1 = mlow
    #    yy2 = y2
    #    ax.fill([xx1,xx2,xx2,xx1], [yy1,yy1,yy2,yy2], color = '#DCDCDC', edgecolor = '#DCDCDC')
        
    # Theoretical scatter
    #melemsigs=[11,13]
    #for melemsig in melemsigs:
    #    msig=xfe[znum==melemsig]
    #    xx1=melemsig-0.1
    #    xx2=melemsig+0.1
    #    yy1=msig-0.5
    #    yy2=msig+0.5
    #    main_ax.fill([xx1,xx2,xx2,xx1],[yy1,yy1,yy2,yy2],color='#DCDCDC',edgecolor='#DCDCDC')

        
    ax.plot(x_model, y_model, marker='', linestyle='-', \
            color = cmap(1), linewidth = 2,\
                         label = r"Bestfit")

    filt = (flags == 0) | (flags == 8) | (flags == 4)

    x_obs = x_obs0[filt]
    y_obs = y_obs0[filt]
    asymmetric_err = np.array(list(zip(sig_m[filt],sig_p[filt]))).T
 
    ax.errorbar(x_obs, y_obs, yerr = asymmetric_err, \
                linestyle='', marker='o',\
                mec=cmap(0), mfc=cmap(0), ms=8, ecolor=cmap(0), \
                label='Observed')

    filt = (flags == 1) | (flags == 9)

    if len(sig_m[filt])!=0:
        sigma_ebars=np.zeros(np.size(sig_m[filt]))
        sigma_ebars.fill(0.5)
        ax.errorbar(x_obs0[filt], y_obs0[filt], sigma_ebars, \
                linestyle='', marker='_', \
                mec=cmap(0), mfc=cmap(0), ecolor=cmap(0), elinewidth=2,\
                ms=8, uplims=True)

    filt = (flags == 16)
    if len(sig_m[filt])!=0:
        ax.errorbar(x_obs0[filt], y_obs0[filt], sig_m[filt], \
                linestyle='', marker='*',\
                mec=cmap(0), mfc=cmap(0), ecolor=cmap(0), \
                ms=10, label="3DNLTE")
                

    filt = (flags == 32)
    if len(sig_m[filt])!=0:
        ax.errorbar(x_obs0[filt],y_obs0[filt],sig_m[filt],\
                               linestyle='',marker='x',\
                color=cmap(0),ms=10, label="Assumed")



    for j,elem in enumerate(elems):
        
        if np.mod(j,2) == 0:
            yfrac=0.07
        else:
            yfrac=0.12
        ax.text(elemzs[j],y2-(y2-y1)*yfrac,elem,ha='center')
    
    
    ax.set_ylim(y1,y2)
    ax.set_xlim(x1,x2)
    ax.set_ylabel(r"$\log A$")
    ax.set_xlabel("Z")
    ax.legend(loc=3, prop={"size":11})
    
    plt.savefig(outfigname)
    plt.close()	
    
    #plt.show()
    return





def plot_abupattern_xfe(x_obs0, y_obs0, sig_p, sig_m, flags, feh, feh_err, x_model, y_model, feh_model, outfigname, modellabs = "", melemlows=[21,22]): 


    cmap = plt.get_cmap("tab10")  
    cmap_grad = plt.get_cmap("Oranges_r")   
 
    plt.style.use('seaborn-white')
    plt.rcParams["font.size"] = 14


    xfe_obs0 = abundance.logA2XFe(x_obs0, y_obs0, feh)

    sig_p_xfe = np.sqrt(sig_p**2 + feh_err**2)
    sig_m_xfe = np.sqrt(sig_m**2 + feh_err**2)
 
    fig, ax = plt.subplots(1,1)

    filt = (x_model != 9) & (x_model != 10) & (x_model != 18) & (x_model != 21)
    
    ymin = np.min(xfe_obs0)
    ymax = np.max(xfe_obs0)
    y1 = ymin - (ymax - ymin) * 0.3
    y2 = ymax + (ymax - ymin) * 0.3

    x1 = np.min(x_model[filt]) - 1
    x2 = np.max(x_model[filt]) + 1
    elems = abundance.get_elem_name(x_obs0)
    elemzs = x_obs0

    # Theoretical lower limit
    #for melemlow in melemlows:
    #    mlow = y_model[x_model == melemlow]
    #    xx1 = melemlow - 0.1
    #    xx2 = melemlow + 0.1
    #    yy1 = mlow
    #    yy2 = y2
    #    ax.fill([xx1,xx2,xx2,xx1], [yy1,yy1,yy2,yy2], color = '#DCDCDC', edgecolor = '#DCDCDC')
        
    # Theoretical scatter
    #melemsigs=[11,13]
    #for melemsig in melemsigs:
    #    msig=xfe[znum==melemsig]
    #    xx1=melemsig-0.1
    #    xx2=melemsig+0.1
    #    yy1=msig-0.5
    #    yy2=msig+0.5
    #    main_ax.fill([xx1,xx2,xx2,xx1],[yy1,yy1,yy2,yy2],color='#DCDCDC',edgecolor='#DCDCDC')


    if np.ndim(y_model)==1:
    
        xfe_model = abundance.logA2XFe(x_model, y_model, feh_model)
        ax.plot(x_model, xfe_model, marker='', linestyle='-', \
               color = cmap(1), linewidth = 2,\
                         label = r"Model", alpha=0.7)
    else:
        labs = modellabs.split(",")
        for k, y_mod in enumerate(y_model):
            xfe_model = abundance.logA2XFe(x_model, y_mod, feh_model)
            ax.plot(x_model, xfe_model, marker='', linestyle='-', \
               color = cmap_grad(k/len(y_model)), linewidth = 2,\
                         label = labs[k], alpha=0.7)


    filt = (flags == 0) | (flags == 8) | (flags == 4)

    x_obs = x_obs0[filt]
    y_obs = xfe_obs0[filt]
    asymmetric_err = np.array(list(zip(sig_m_xfe[filt], sig_p_xfe[filt]))).T
 
    ax.errorbar(x_obs, y_obs, yerr = asymmetric_err, \
                linestyle='', marker='o',\
                mec=cmap(0), mfc=cmap(0), ms=8, ecolor=cmap(0), \
                label='Observed', alpha=0.7)

    filt = (flags == 1) | (flags == 9)

    if len(sig_m_xfe[filt])!=0:
        sigma_ebars=np.zeros(np.size(sig_m_xfe[filt]))
        sigma_ebars.fill(0.5)
        ax.errorbar(x_obs0[filt], xfe_obs0[filt], sigma_ebars, \
                linestyle='', marker='_', \
                mec=cmap(0), mfc=cmap(0), ecolor=cmap(0), elinewidth=2,\
                ms=8, uplims=True, alpha=0.7)

    filt = (flags == 16)
    if len(sig_m_xfe[filt])!=0:
        ax.errorbar(x_obs0[filt], xfe_obs0[filt], sig_m_xfe[filt], \
                linestyle='', marker='*',\
                mec=cmap(0), mfc=cmap(0), ecolor=cmap(0), \
                ms=10, label="3DNLTE")
                

    filt = (flags == 32)
    if len(sig_m_xfe[filt])!=0:
        ax.errorbar(x_obs0[filt],xfe_obs0[filt],sig_m_xfe[filt],\
                               linestyle='',marker='x',\
                color=cmap(0),ms=10, label="Assumed")



    for j,elem in enumerate(elems):
        
        if np.mod(j,2) == 0:
            yfrac=0.07
        else:
            yfrac=0.12
        ax.text(elemzs[j],y2-(y2-y1)*yfrac,elem,ha='center')
    
    
    ax.set_ylim(y1,y2)
    ax.set_xlim(x1,x2)
    ax.set_ylabel(r"[X/Fe]")
    ax.set_xlabel("Z")
    ax.legend(loc=3, prop={"size":11})

    plt.tight_layout()    
    plt.savefig(outfigname)
    plt.close()
    
    #plt.show()
    return




def plot_lnlike(lnlikefile, feh):

    mmix0, lfej0, feh0, lnlike0 = np.loadtxt(lnlikefile,delimiter=',',usecols = (0,1,2,3), unpack = True)

    mmix = mmix0[feh0 == feh]
    lfej = lfej0[feh0 == feh]
    lnlike = lnlike0[feh0 == feh]
	
	
    x1 = mmix[lfej == 0.0]
    x2 = lfej[mmix == 0.0]
    X2, X1 = np.meshgrid(x2, x1)

    #X = np.c_[np.ravel(X1), np.ravel(X2)]

    Z = lnlike.reshape(X1.shape)
  
    
    #zmin = np.min(lnlike)
    zmax = np.max(lnlike)
    zmin = zmax - 0.01 * (zmax - np.min(lnlike))
    if zmin >= zmax:
        zmin = np.min(lnlike)
        
    levs = np.arange(zmin, zmax + 1.0, 1.)
    
    fig, ax = plt.subplots(1, 1)


    cm = ax.contourf(X1, X2, Z,levels = levs, cmap = 'Blues')



    
    levs = np.arange(zmin, zmax + 1.0, 2.)
    #ax.contour(X1, X2, Z,levels = levs, colors = 'black')
    ax.set_xlabel(r"$x_{M_{\rm mix}}$")
    ax.set_ylabel(r"$\log f_{ej}$")

    mass = np.float(((lnlikefile.split("Mass"))[1])[0:3]) 
    en = np.float(((lnlikefile.split("Energy"))[1])[0:4])
    ax.set_title(r"Mass=%.0f, E51=%.1f"%(mass, en))
    fig.colorbar(cm, ax = ax)
    
    outfigfile = outdir + "/" + lnlikefile[:-4] + ".png"
    
    plt.savefig(outfigfile)
    
    #plt.show()
    
    return

def plot_total_lnlike(lnlikefiles):
    
    cmap = plt.get_cmap("Oranges")

    plt.style.use('seaborn-white')
    plt.rcParams["font.size"] = 14
 
    mmix_grid = np.arange(0.0, 2.1, 0.1)
    lfej_grid = np.arange(-7.0, 0.1, 0.1)



    X2, X1 = np.meshgrid(lfej_grid, mmix_grid)

    Z = np.zeros_like(X1)

    for lnlikefile in lnlikefiles:

        mmix0, lfej0, mcut0, feh0, lnlike0 = np.loadtxt(lnlikefile,delimiter=',',usecols = (0,1,2, 3,4), unpack = True)

        mmix_grid = np.sort(np.unique(mmix0))
        lfej_grid = np.sort(np.unique(lfej0))
        mcut_grid = np.sort(np.unique(mcut0))
        feh_grid = np.sort(np.unique(feh0))


        for i, mm in enumerate(mmix_grid):
            for j, lf in enumerate(lfej_grid):
                prob = 0.
                for k, mc in enumerate(mcut_grid):
                    d_mc = mcut_grid[1]-mcut_grid[0] if k==0 else mc - mcut_grid[k-1]
                    prob_fe = 0.
                    for l, fe in enumerate(feh_grid):
                        d_fe = feh_grid[1]-feh_grid[0] if l==0 else fe - feh_grid[l-1]

                        filt = (mmix0 < mm+0.01) & (mmix0 > mm-0.01) & \
                                 (lfej0 < lf+0.01) & (lfej0 > lf-0.01) & \
                                 (mcut0 < mc+0.001) & (mcut0 > mc - 0.001) & \
                                 (feh0 < fe + 0.001) & (feh0 > fe - 0.001)
                        lnlike = lnlike0[filt] if len(lnlike0[filt])==1 else 0 if len(lnlike0[filt])==0 else np.nan
                        
                        if np.isnan(lnlike):
                            print("Two many entries for given parameters!")
                            sys.exit()
                        else:
                            prob_fe = prob_fe + np.e**lnlike * d_fe
                    prob = prob + prob_fe * d_mc

                #lnlike = lnlike0[filt]
                #likelihood = np.e**lnlike  
                Z[i, j] = Z[i, j] + prob
               



    #zmin = np.min(lnlike)
    #lZ = np.log(Z)
    #zmax = np.max(lZ)
    #zmin = zmax - 0.01 * (zmax - np.min(lZ))

    zmax = np.max(Z) 
    zmin = np.max(Z) - (np.max(Z)-np.min(Z))*1.0 
    step = (zmax-zmin)/100.


    levs = np.arange(zmin, zmax + step, step)

    fig, ax = plt.subplots(1, 1)
    
    cm = ax.contourf(X1, X2, Z,levels = levs, cmap = cmap)
    #cm = ax.pcolormesh(X1, X2, Z, vmin=0., vmax=1., cmap = cmap)

    levs = np.arange(zmin, zmax + 1.0, 2.)
    #ax.contour(X1, X2, Z,levels = levs, colors = 'black')
    ax.set_xlabel(r"$x_{M_{\rm mix}}$")
    ax.set_ylabel(r"$\log f_{ej}$")

    mass = np.float64(((lnlikefiles[0].split("Mass"))[1])[0:3])
    en = np.float64(((lnlikefiles[0].split("Energy"))[1])[0:4])
    ax.set_title(r"Mass=%.0f, E51=%.1f"%(mass, en))
    fig.colorbar(cm, ax = ax)

    outfigfile = "../figs/Lnlike_M%.0f_E%.0f.png"%(mass, en) 

    plt.savefig(outfigfile)





def plot_fehdist(abundance_datadir, outfigname):

    filelist = glob.glob(abundance_datadir + "/*.csv")

    feh_solar = -4.5 
    ch_solar = -3.57


    fehs = ()
    chs = ()
    
    for file in filelist:

        starname = (((file.split("/"))[-1]).split("_"))[0]
        df = pd.read_csv(file)
        print("The nummber of [X/Fe] data points for ", starname, ": ", len(df[np.isnan(df['A(X)'])==False])-1.)  # Excluding Fe


        filt = df['Species'] == "Fe"
        feh = (df['A(X)'])[filt] - feh_solar

        fehs = np.hstack((fehs, feh))
      
        filt = (df['Species'] == "C") & (np.isnan(df['A(X)'])!=True)
        if len((df['A(X)'])[filt])==0:
            ch = -99.99
        else:
            ch = (df['A(X)'])[filt] - ch_solar
        chs = np.hstack((chs, ch))

    cfes = chs - fehs
    fig, ax = plt.subplots(1, 1)
    ax.hist(fehs, bins=10, label="All %i"%(len(fehs)))
    ax.hist(fehs[cfes>0.7], bins = 10, label = "[C/Fe]>0.7 %i"%(len(fehs[cfes>0.7])))
    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel("N")
    plt.legend()
    plt.savefig(outfigname)
 
    return


def get_bestfit(filelist):

    func = ()
    chi2 = ()
    mass = ()
    energy = ()
    mcut = ()
    mmix = ()
    lfej = ()  
    dof = ()


    for i, file in enumerate(filelist):
        m = float(((((file.split("/"))[-1]).split("Mass"))[1])[0:3])
        en = float(((((file.split("/"))[-1]).split("Energy"))[1])[0:4])
        df = pd.read_csv(file)
        funcval = df['func'][0]
        chi2val = df['chi2'][0]
        mcutval = df['Mcut'][0]
        mmix_val = df['Mmix'][0]
        lfejval = df['lfej'][0]
        dof_val = df['dof'][0]

        mass = np.hstack((mass, m))
        energy = np.hstack((energy, en))
        func = np.hstack((func, funcval))
        chi2 = np.hstack((chi2, chi2val))
        mcut = np.hstack((mcut, mcutval))
        mmix = np.hstack((mmix, mmix_val))
        lfej = np.hstack((lfej, lfejval))
        dof = np.hstack(((dof, dof_val)))

    index_for_maxlnlike = [i for i, x in enumerate(func) if x == max(func)]
    mass_max = mass[index_for_maxlnlike]
    energy_max = energy[index_for_maxlnlike]
    func_max = func[index_for_maxlnlike]
    chi2_max = chi2[index_for_maxlnlike]    
    mcut_max = mcut[index_for_maxlnlike]
    mmix_max = mmix[index_for_maxlnlike]
    lfej_max = lfej[index_for_maxlnlike]
    dof_max = dof[index_for_maxlnlike]

    if (len(mass_max)!=1):
        print("Multiple or no parameters for the maximum likelihood. Check the results!", mass_max)
        sys.exit()
    elif mass_max[0]==100:
        
        print(filelist[index_for_maxlnlike[0]])


    return(mass_max, energy_max, func_max, chi2_max, mcut_max, mmix_max, lfej_max, dof_max)



def plot_bestfit_mass_hist(starlist, input_catalog1, input_catalog2, pmulti_threshold, pval):
    
    plt.style.use('seaborn-white')
    cmap = plt.get_cmap('tab10')

    masses = ()
    w_masses = ()

    ens = ()
    funcs = ()
    chi2s = ()
    mcuts = ()
    mrems = ()
    pmulti_prob = ()

    wsum = 0.0
    histsum = 0.0

    for star in starlist:

        # Check whether the star is multi-enriched: 
        
        df = pd.read_csv(input_catalog1)
        filt = df["Name"]== star.strip()
        if len(filt[filt==True])==0:
            df = pd.read_csv(input_catalog2)
            filt = df["Name"]== star.strip() 
        N_SN = df["MyPred"][filt]
        p_multi = df["xMean"][filt]
        std = df["xStd"][filt]
        if p_multi.values > pmulti_threshold:
            continue

        filelist = glob.glob("../out/" + star.strip() + "/" + star.strip() + "*.txt")
        if len(filelist)<5:
            continue
        mass_max, en_max, func_max, chi2_max, mcut_max, xmmix_max, lfej_max, dof = get_bestfit(filelist)
        
        mco = yields.get_mco(mass_max)
      
        mmix = mcut_max + xmmix_max*(mco - mcut_max)

        mrem = mcut_max + (1.-10**lfej_max)*(mmix - mcut_max)
        #wsum=wsum+chi2.sf(chi2val,dof,loc=0,scale=1)

        w = chi2.sf(chi2_max,dof,loc=0,scale=1)
        wsum = wsum + w

        masses = np.hstack((masses, mass_max))
        w_masses = np.hstack((w_masses, w))
        print(w)

        ens = np.hstack((ens, en_max))
        funcs = np.hstack((funcs, func_max))
        chi2s = np.hstack((chi2s, chi2_max))
        mcuts = np.hstack((mcuts, mcut_max))
        mrems = np.hstack((mrems, mrem))


    wfac = 1.0/wsum
    w_masses = w_masses * wfac
 
 
    # Plot histogram for the progenitor mass

    plt.rcParams["font.size"]=16
    fig, ax = plt.subplots(1, 1)

    xmasses = [11., 13., 15., 25., 40., 100]
    lx=np.log10(xmasses)

    bottom = np.zeros_like(xmasses)

    for i in range(0, 3):
        if i==0:
            if pval==True:
                barval = [ np.sum(w_masses[(masses==xx) & (ens<1.)]) for xx in xmasses ]
            else:
                barval = [ len(masses[(masses==xx) & (ens<1.)]) for xx in xmasses ]
            lab = r"Low-E ($E_{51}<1$)"
            col = cmap(0)
        elif i==1:
            if pval==True:
                barval = [ np.sum(w_masses[(masses==xx) & (ens==1.)]) for xx in xmasses ]
            else:
                barval = [ len(masses[(masses==xx) & (ens==1.)]) for xx in xmasses ]
            lab = r"SN ($E_{51}=1$)"
            col = cmap(2)
        elif i==2:
            if pval==True:
                barval = [ np.sum(w_masses[(masses==xx) & (ens>1.0)]) for xx in xmasses ]
            else:
                barval = [ len(masses[(masses==xx) & (ens>1.0)]) for xx in xmasses ]
            lab = r"HN ($E_{51}>1$)"
            col = cmap(1)

        ax.bar(lx, barval, align ="center", width=0.03, bottom = bottom, label=lab, color=col)
        bottom = barval

    if pval==True:
        ylab = "p-value"
        figname = "../figs/mass_hist_Mcut_weighted_%.1f.png"%(pmulti_threshold)
    else:
        ylab = "N"
        figname = "../figs/mass_hist_Mcut_%.1f.png"%(pmulti_threshold)       
   

    ax.set_xticks(lx)
    ax.set_xticklabels([ "%.0f"%(xx) for xx in xmasses] )
    ax.set_xlabel(r"Progenitor mass [$M_\odot$]")
    ax.set_ylabel(ylab)
    ax.set_xlim(1.0,2.15)
    y1, y2 = ax.get_ylim()
    ax.set_ylim(y1, y2*1.1)    
    plt.tight_layout()
    plt.legend()
    plt.savefig(figname)
    
    #fig, bx = plt.subplots(1, 1)
    #bx.hist(chi2s)
    #plt.savefig("../figs/chi2_hist.png")

    # Plot for the Mcut distribution
    fig2, dx = plt.subplots(1, 3, figsize=(15, 5))
    for i, xx in enumerate([13., 15., 25.]):
        
        mcuts_mass = mcuts[masses==xx]
        #val, bins = np.histogram(mcuts_mass, bins = 5)
         
        #bincent = bins[0:-1] + (bins[2]-bins[1])*0.5
        dx[i].hist(mcuts_mass, label = r"M=%i M$_\odot$"%(xx))
        dx[i].set_xlabel(r"$M_{\rm cut}$")
        dx[i].legend()
    plt.tight_layout()
    plt.savefig("../figs/Mcuts_hist_%.1f.png"%(pmulti_threshold))
    
    # Mrem distribution 
    fig3, cx = plt.subplots(1, 3, figsize=(15, 5))
    for i, xx in enumerate([13., 15., 25.]):

        mrems_mass = mrems[masses==xx]
     
        #val, bins = np.histogram(mcuts_mass, bins = 5)

        #bincent = bins[0:-1] + (bins[2]-bins[1])*0.5
        cx[i].hist(mrems_mass, label = r"M=%i M$_\odot$"%(xx))
        cx[i].set_xlabel(r"$M_{\rm rem}$")
        cx[i].legend()
    plt.tight_layout()
    plt.savefig("../figs/Mrems_hist_%.1f.png"%(pmulti_threshold))

    return



def plot_models(before = True, MF = False, model = "Umeda", isotopes = "Basic"):


    matplotlib.rcParams.update({'font.size': 14})

    # Relevant isotopes

    if isotopes == "Basic":
        isos=['p','he4', 'c12', 'c13','n14','o16','na23','mg24','al27','si28', 'ar36','ca40','cr48','fe52','co55','cu59','ni58','ge64','ni56']

        isolabs=['H','$^4$He', '$^{12}$C', '$^{13}$C','$^{14}$N','$^{16}$O','$^{23}$Na','$^{24}$Mg','$^{27}$Al',\
          '$^{28}$Si', '$^{36}$Ar', '$^{40}$Ca','$^{48}$Cr','$^{52}$Fe','$^{55}$Co','$^{59}$Cu','$^{58}$Ni','$^{64}$Ge--$^{64}$Zn','$^{56}$Ni--$^{56}$Fe']


        cols=['black','silver','red','kahki','gold','mediumblue','limegreen','magenta','orange','aqua','royalblue','yellow','purple','dimgray','lightpink','navy','purple','brown','darkslategray']

        lss=['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-']



        etexts=["M($^{12}$C)","M($^{13}$C)","M($^{14}$N)","M($^{16}$O)","M($^{23}$Na)","M($^{24}$Mg)","M($^{27}$Al)",\
             "M($^{28}$Si)","M($^{36}$Ar)","M($^{40}$Ca)","M($^{48}$Cr--$^{48}$Ti)","M($^{52}$Fe--$^{52}$Cr)","M($^{55}$Co--$^{55}$Mn)",\
             "M($^{59}$Cu--$^{59}$Co)","M($^{58}$Ni)","M($^{64}$Ge--$^{64}$Zn)","M($^{56}$Ni--$^{56}$Fe)"]


    elif isotoes == "O":

    
        isos=['p','he4', 'c12', 'c13','n14','o16','o17','o18','na23','mg24','al27','si28', 'ca40','cr48','fe52','co55','cu59','ni58']

        ## For basic isotopes
        #isos=['p','he4', 'c12', 'n14','o16','na23','mg24','al27','si28', 'ca40','cr48','co55','ni58','ge64','ni56']
        #isolabs=['H','$^4$He', '$^{12}$C', '$^{14}$N','$^{16}$O','$^{23}$Na','$^{24}$Mg','$^{27}$Al','$^{28}$Si', '$^{40}$Ca','$^{48}$Cr','$^{55}$Co','$^{58}$Ni','$^{64}$Ge--$^{64}$Zn','$^{56}$Ni']


        etexts=["M($^{12}$C)","M($^{14}$N)","M($^{16}$O)","M($^{17}$O)","M($^{18}$O)","M($^{23}$Na)",\
             "M($^{24}$Mg)","M($^{27}$Al)","M($^{28}$Si)","M($^{40}$Ca)","M($^{48}$Cr--$^{48}$Ti)","M($^{52}$Fe--$^{52}$Cr)",\
             "M($^{55}$Co--$^{55}$Mn)","M($^{59}$Cu--$^{59}$Co)","M($^{58}$Ni)"]



    # The location of the grid

    ## For Umeda model

    if model == "Umeda":


        path="../../models/Umeda/"
        gridfiles=["13z0E0.51507Mc1.35m5N260Salpha100T6d8SYeMn0.4997","13z0E1.00014Mc1.35m5N260Salpha150SYeMn0.4997",\
             "15z0E0.99873Mc1.35m5N230Salpha150SYeMn0.4997","25z0E0.9982Mc1.45m5N250Salpha150SYeMn0.4997", \
             "40z0E0.98824Mc2.0m5N260Salpha150SYeMn0.4997","100z0E1.109Mc2.0m9Salpha150-100T3d8SYeMn0.4997", \
             "25z0E10.0198Mc1.5m7N250Salpha150SYeMn0.4997","40z0E29.936Mc2.0m5N260Salpha150SYeMn0.4997","100z0E60.794Mc2.0m20Salpha150SYeMn0.4997"]

    elif model == "AccYe":

        ## For the increased Ye model

        path="../../models/AccYe/"
        gridfiles=["13z0E0.51507Mc1.35m5N260Salpha100T6d8SYeAcc","13z0E1.00014Mc1.35m5N260Salpha150SYeAcc",\
             "15z0E0.99873Mc1.35m5N230Salpha150SYeAcc","25z0E0.9982Mc1.45m5N250Salpha150SYeAcc","40z0E0.98824Mc2.0m5N260Salpha150SYeAcc",\
             "100z0E1.109Mc2.0m9Salpha150-100T3d8SYeAcc","25z0E10.0198Mc1.5m7N250Salpha150SYeAcc","40z0E29.936Mc2.0m5N260Salpha150SYeAcc","100z0E60.794Mc2.0m20Salpha150SYeAcc"]



    # Plotting regions
    xmins=[1.3,1.3,1.3,1.7,2.1,2.1,1.7,2.1,2.1]
    xmaxs=[3.0,3.0,4.0,8.0,17.0,48.0,8.0,17.0,48.0]

    

    # Mixing-fallback parameters
    mcutinis=[1.47,1.47,1.41,1.69,2.42,3.63,1.69,2.42,3.63]
    if MF == False:
        mmixouts=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        lfejecs=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        MFstatus = "noMF"
    else:

        mmixouts=[0.1,0.1,0.1,0.3,0.3,0.3,0.3,0.3,0.3]
        lfejecs=np.array([-0.6,-0.6,-0.6,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4])
        MFstatus = "MF"

    if before == True:
        Mejecfilename = "../figs/" + model + '_Mejec_' + MFstatus + '_beforeExpl.eps'
    else:
        Mejecfilename = "../figs/" + model + '_Mejec_' + MFstatus + '.eps'


    mcos=[2.386,2.386,3.024,6.293,13.890,42.002,6.293,13.890,42.002]
    fejecs=10**lfejecs



    masstots=np.zeros((17,np.size(gridfiles)),dtype=np.float64)


    for ig,gridfile in enumerate(gridfiles):

        fig=plt.figure(figsize=(12,8))
        gs=gridspec.GridSpec(2,2, width_ratios=[3,1])

        # Define the name of the figure
        figname='../figs/' + gridfile+'_structure.eps'

        #
        if before == True:
            massfig=gridfile+'_'+"{:0.1f}".format(mcutinis[ig])+'_'+"{:0.1f}".format(mmixouts[ig])+'_'+"{:0.1f}".format(lfejecs[ig])+'_beforeExpl.eps'
        else:
            massfig=gridfile+'_'+"{:0.1f}".format(mcutinis[ig])+'_'+"{:0.1f}".format(mmixouts[ig])+'_'+"{:0.1f}".format(lfejecs[ig])+'.eps'



        # Open the plotting windows
        ax1=fig.add_subplot(gs[0,0])
        ax2=fig.add_subplot(gs[1,0])
        ax3=fig.add_subplot(gs[0,1])
        ax4=fig.add_subplot(gs[1,1])

        #ax1=plt.subplot2grid((2,4),(0,0),colspan=3)
        #ax2=plt.subplot2grid((2,4),(1,0),colspan=3)
        #ax3=plt.subplot2grid((2,4),(0,3))
        #ax4=plt.subplot2grid((2,4),(1,3))


        # Open the grid file
        f=open(path+gridfile,"r")

        # Read the grid file
        lines=f.readlines()

        # For each isotope, read the data


        for ii,iso in enumerate(isos):
            m=()
            ly=()
            ye=()
            tp=()
            mfracs=()

            for i,line in enumerate(lines):
                data=line.split()
                if np.size(data)==0:
                    continue
                if np.size(data)==8:
                    m=np.append(m,np.float64(data[2]))
                    ly=np.append(ly,data[1])
                    ye=np.append(ye,np.float64(data[6]))
                    tp=np.append(tp,np.float64(data[7]))

                else:
                    for k in range(0,7):
                        if(data[k*2]==iso):
                            mfracs=np.append(mfracs,np.float64(data[k*2+1]))
                            break
            a=mfracs.reshape((np.size(ly),2))
            mfracs_ini=np.array(a[:,0])
            mfracs_fin=np.array(a[:,1])

            if iso=='c12':
                if before==1:
                    cfracs=mfracs_ini
                else:
                    cfracs=mfracs_fin
            elif iso=='c13':
                if before==1:
                    c13fracs=mfracs_ini
                else:
                    c13fracs=mfracs_fin
            elif iso=='n14':
                if before==1:
                    nfracs=mfracs_ini
                else:
                    nfracs=mfracs_fin
            elif iso=='o16':
                if before==1:
                    ofracs=mfracs_ini
                else:
                    ofracs=mfracs_fin
            elif iso=='o17':
                if before==1:
                    o17fracs=mfracs_ini
                else:
                    o17fracs=mfracs_fin
            elif iso=='o18':
                if before==1:
                    o18fracs=mfracs_ini
                else:
                    o18fracs=mfracs_fin
            elif iso=='na23':
                if before==1:
                    nafracs=mfracs_ini
                else:
                    nafracs=mfracs_fin
            elif iso=='mg24':
                if before==1:
                    mgfracs=mfracs_ini
                else:
                    mgfracs=mfracs_fin
            elif iso=='al27':
                if before==1:
                    alfracs=mfracs_ini
                else:
                    alfracs=mfracs_fin
            elif iso=='si28':
                if before==1:
                    sifracs=mfracs_ini
                else:
                    sifracs=mfracs_fin
            elif iso=='ar36':
                if before==1:
                    arfracs=mfracs_ini
                else:
                    arfracs=mfracs_fin
            elif iso=='ca40':
                if before==1:
                    cafracs=mfracs_ini
                else:
                    cafracs=mfracs_fin
            elif iso=='cr48':  # -> 48Ti
                if before==1:
                    tifracs=mfracs_ini
                else:
                    tifracs=mfracs_fin
            elif iso=='fe52':  # -> 52Cr
                if before==1:
                    crfracs=mfracs_ini
                else:
                    crfracs=mfracs_fin
            elif iso=='co55':
                if before==1:
                    mnfracs=mfracs_ini
                else:
                    mnfracs=mfracs_fin
            elif iso=='cu59':
                if before==1:
                    cofracs=mfracs_ini
                else:
                    cofracs=mfracs_fin
            elif iso=='ni58':
                if before==1:
                    nifracs=mfracs_ini
                else:
                    nifracs=mfracs_fin
            elif iso=='ge64':
                if before==1:
                    znfracs=mfracs_ini
                else:
                    znfracs=mfracs_fin
            elif iso=='ni56':
                if before==1:
                    fefracs=mfracs_ini
                else:
                    fefracs=mfracs_fin


            ymin=-8.0


            if iso=='c13' or iso=='ar36' or iso=='cu59':
                continue

            # Before explosion
            lmfrac=np.log10((mfracs_ini+1.0e-50))
            ax1.plot(m,lmfrac,color=cols[ii],ls=lss[ii],lw=2,label=isolabs[ii])
            ax1.set_ylim(ymin,0.0)
            ax1.set_xlim(xmins[ig],xmaxs[ig])
            ax1.set_ylabel(r'$\log X$')

            # After explosion
            lmfrac=np.log10((mfracs_fin+1.0e-50))
            ax2.plot(m,lmfrac,color=cols[ii],ls=lss[ii],lw=2,label=isolabs[ii])
            ax2.set_ylim(ymin,0.0)
            ax2.set_xlim(xmins[ig],xmaxs[ig])
            ax2.set_xlabel(r'$M_{r}$[$M_{\odot}$]')
            ax2.set_ylabel(r'$\log X$')


        #ax2.legend(bbox_to_anchor=(1.1,1.0),prop={'size':9})
        ax1.legend(loc=4,prop={'size':9})

        # Ye
        ax3.plot(m,ye)
        ax3.text(xmins[ig]+0.1,0.515,r'$Y_e$')
        ax3.set_xlim(xmins[ig],xmaxs[ig]*0.7)
        ax3.set_ylim(0.495,0.505)
        # Peak temparature
        func=interp1d(tp,m,kind='linear')
        ax4.plot(m,tp)
        ax4.text(xmins[ig]+0.1,13,r'$T_{p}$ [10$^9$K]')
        ax4.set_xlim(xmins[ig],xmaxs[ig]*0.7)
        ax4.set_ylim(0,14)
        ax4.set_xlabel(r'$M_{r}$[$M_{\odot}$]')

        #plt.tight_layout(h_pad=0.2)
        fig.savefig(figname)

        # Burning layers determined by the peak temperature
        print(gridfile)

        m_tp=np.float64(func(5.0))
        print(r"    Tp=5[10$^9$K]","{:0.2f}".format(m_tp))
        m_tp=np.float64(func(4.0))
        print(r"    Tp=4[10$^9$K]","{:0.2f}".format(m_tp))
        m_tp=np.float64(func(3.3))
        print(r"    Tp=3.3[10$^9$K]","{:0.2f}".format(m_tp))
        m_tp=np.float64(func(2.0))
        print(r"    Tp=2.0[10$^9$K]","{:0.2f}".format(m_tp))



        # Mass of ejected isotopes
        fig2,ax=plt.subplots(8,2,figsize=(8,14),sharex=True)
        for k in range(0,16):
            ## Plotting location
            ix=np.int(k/8)
            iy=np.mod(k,8)
            if k==0:
                fracs=cfracs
            elif k==1:
                fracs=c13fracs
            elif k==2:
                fracs=nfracs
            elif k==3:
                fracs=ofracs
            elif k==4:
                fracs=nafracs
            elif k==5:
                fracs=mgfracs
            elif k==6:
                fracs=alfracs
            elif k==7:
                fracs=sifracs
            elif k==8:
                fracs=arfracs
            elif k==9:
                fracs=cafracs
            elif k==10:
                fracs=tifracs
            elif k==11:
                fracs=crfracs
            elif k==12:
                fracs=mnfracs
            elif k==13:
                fracs=cofracs
            elif k==14:
                fracs=nifracs
# For O isotopes
#       elif k==14:
 #           fracs=o17fracs
 #       elif k==15:
 #           fracs=o18fracs

# For Zn and Fe
            elif k==15:
                fracs=znfracs
            elif k==16:
                fracs=fefracs

            mass=()

            for i,mm in enumerate(m):
                if i>=np.size(m)-1:
                    break
                dm=m[i+1]-m[i]
                mmixout=mcutinis[ig]+mmixouts[ig]*(mcos[ig]-mcutinis[ig])
                if mm<mcutinis[ig]:
                    mass=np.append(mass,0.0)
                elif mm>=mcutinis[ig] and mm<=mmixout:
                    mass=np.append(mass,dm*fracs[i]*fejecs[ig])
                elif mm>mmixout:
                    mass=np.append(mass,dm*fracs[i])

            masstot=np.sum(mass)
            masstots[k,ig]=masstot

            ax[iy,ix].plot(m[:-1],mass)
            ax[iy,ix].set_xlim(xmins[ig],xmaxs[ig])
            ax[iy,ix].tick_params(labelsize=9)
            ax[iy,ix].yaxis.offsetText.set_fontsize(9)
            ax[iy,ix].yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
            ax[iy,ix].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            ax[iy,ix].text(xmins[ig]+(xmaxs[ig]-xmins[ig])*0.98,np.max(mass)*0.8,etexts[k]+', Total='+"{:7.1e}".format(masstot)+'$M_{\odot}$',fontsize=9,ha='right')


            if iy==7:
                ax[iy,ix].set_xlabel("$M_r$ [$M_{\odot}$]",fontsize=9)



        tit="M$_{cut}$/M$_{mix}$/f="+"{:0.2f}".format(mcutinis[ig])+'/'+\
                          "{:0.1f}".format(mmixout)+'/'+"{:0.5f}".format(fejecs[ig])

        ax[0,0].set_title(tit,fontsize=10)
        plt.tight_layout(h_pad=0.0)
        plt.savefig(massfig)

    if np.size(gridfiles==1):
        import sys
        sys.exit()

    fig3,ax=plt.subplots(8,2,figsize=(8,14),sharex=True)
    for k in range(0,16):
        ix=np.int(k/8)
        iy=np.mod(k,8)
        x=np.arange(5)
        ax[iy,ix].plot(x,np.log10(masstots[k,:]),'ko:')
        ax[iy,ix].set_xlim(-0.5,4.5)
        ax[iy,ix].tick_params(labelsize=9)
        ax[iy,ix].yaxis.offsetText.set_fontsize(9)
        ax[iy,ix].yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
#    ax[iy,ix].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        ytxt=(np.max(np.log10(masstots[k,:]))-np.min(np.log10(masstots[k,:])))*0.9+np.min(np.log10(masstots[k,:]))
        ax[iy,ix].text(-0.3,ytxt,etexts[k],fontsize=9,ha='left')
        ax[iy,ix].set_ylabel('$\log (M_{ejected} M_\odot)$',fontsize=9)

        if iy==7:
            ax[iy,ix].set_xticks(x)
            ax[iy,ix].set_xticklabels(["15SN","25SN","25HN","40HN","100SN"],rotation=90)

    plt.tight_layout(h_pad=0.0)
    plt.savefig(Mejecfilename)
    #plt.savefig('Mejec.eps')


def plot_all_lnlike():

    with open("starlist_one.txt") as f:
        for line in f:

            bestfitparamfiles = glob.glob("../out/" + line.strip() + "/" + line.strip() + "*.txt")

            if len(bestfitparamfiles)<5:
                print("The bestfit-parameter file does not found for star:  %s "%(line.strip()))
                continue

            mass_max, en_max, func_max, chi2_max, mcut_max, _, _ = get_bestfit(bestfitparamfiles)

            print("Maximum lnlike for %s: %f"%(line.strip(), func_max))


            fig = plt.figure(figsize=(12, 8))
            ax=fig.add_subplot(111,frameon=False)
            plt.rcParams["font.size"] = 15


            for mass in [11.0, 13.0, 15.0, 25.0, 40.0, 100.0]:

                if mass == 11.0:
                    ens = [0.5]
                    ax_pos = [(2, 0)]
                    ax_name = ["ax1"]
                    im = ["im1"]
                    xticklab = [True]
                    yticklab = [True]
                elif mass == 13.0:
                    ens = [0.5, 1.0]
                    ax_pos = [(2, 1), (1, 1)]
                    ax_name = ["ax2", "ax3"]
                    im = ["im2", "im3"]
                    xticklab = [True, False]
                    yticklab = [False, True]
                elif mass == 15.0:
                    ens = [1.0]
                    ax_pos = [(1, 2)]
                    ax_name = ["ax4"]
                    im = ["im4"]
                    xticklab = [True]
                    yticklab = [False]
                elif mass == 25.0:
                    ens = [1.0, 10.0]
                    ax_pos = [(1, 3), (0, 3)]
                    ax_name = ["ax5", "ax6"]
                    im = ["im5", "im6"]
                    xticklab=[True, False]
                    yticklab = [False, True]
                elif mass==40.0:
                    ens = [1.0, 30.0]
                    ax_pos = [(1, 4), (0, 4)]
                    ax_name = ["ax7", "ax8"]
                    im = ["im7", "im8"]
                    xticklab = [True, False]
                    yticklab = [False, False]
                elif mass==100.0:
                    ens = [1.0, 60.0]
                    ax_pos = [(1, 5), (0, 5)]
                    ax_name = ["ax9", "ax10"]
                    im = ["im9", "im10"]
                    xticklab = [True, False]
                    yticklab = [False, False]

                for k, en in enumerate(ens):
                    lnlikefile = "../out/" + line.strip() + "/LnLike_%s_MethodCalcAllLnlike_Grid_Mass%03i_Energy%04.1f.txt"%(line.strip(), mass, en)
                    mmix0, lfej0, mcut0, feh0, lnlike0 = np.loadtxt(lnlikefile,delimiter=',',usecols = (0,1,2,3, 4), unpack = True)
                    # This is tentative !!!
                    feh0 = np.round(feh0, decimals=1)
                    #print(feh0)
                    #print(lnlike0)
                    # Get best-fit feh:
                    bestfitfile = "../out/" + line.strip() + "/%s_MethodCalcAllLnlike_Grid_Mass%03i_Energy%04.1f.txt"%(line.strip(), mass, en)
                    df = pd.read_csv(bestfitfile)
                    feh = (df["[Fe/H]"].values)[0]
                    mcut = (df["Mcut"].values)[0]
                    #print(feh)
                   
              
                    filt = (feh0==feh) & (mcut0==mcut)
                    mmix = mmix0[filt]
                    lfej = lfej0[filt]
                    lnlike = lnlike0[filt]

                    # This is tentative !
                    if len(lnlike)>2000:
                        continue


                    x1 = mmix[lfej == 0.0]
                    x2 = lfej[mmix == 0.0]
                    X2, X1 = np.meshgrid(x2, x1)
                    print(x1, x2, len(lnlike))
                    if len(x1)<10:
                        continue
 
                    Z = lnlike.reshape(X1.shape)

                    zmax = func_max
                    func_min = -1000.
                    zmin = zmax - 0.01 * (zmax - func_min)
                    if zmin >= zmax:
                        zmin = func_min
 
                    levs = np.arange(zmin, zmax + 1.0, 1.)
                    vars()[ax_name[k]] = plt.subplot2grid((3, 6), ax_pos[k])                    
                    
                    
                    #vars()[im[k]] = vars()[ax_name[k]].contourf(X1, X2, Z,levels = levs, cmap = 'Blues')

                    vars()[ax_name[k]].imshow(Z.T,extent=[np.min(x1),np.max(x1), np.min(x2), np.max(x2)],\
                                        aspect='auto',origin='lower',cmap='Blues',\
                                 interpolation='none',vmin=zmin,vmax=zmax)
                    vars()[ax_name[k]].set_xticks([0.0, 1.0])
                    vars()[ax_name[k]].set_xticklabels(["0.0", "1.0"])
                    vars()[ax_name[k]].set_yticks([-6.0, -4.0, -2.0, 0.0])
                    vars()[ax_name[k]].set_yticklabels(["-6", "-4", "-2", "0"])

                    vars()[ax_name[k]].tick_params(length=2,direction='in',pad=0.5)

                    if mass == 100. and en == 60.0:
                        im = vars()[ax_name[k]].imshow(Z.T,extent=[np.min(x1),np.max(x1), np.min(x2), np.max(x2)],\
                                        aspect='auto',origin='lower',cmap='Blues',\
                                 interpolation='none',vmin=zmin,vmax=zmax)
                        #im = vars()[ax_name[k]].contourf(X1, X2, Z,levels = levs, cmap = 'Blues')

         

                    levs = np.arange(zmin, zmax + 1.0, 2.)
                    #ax.contour(X1, X2, Z,levels = levs, colors = 'black')



                    vars()[ax_name[k]].tick_params(bottom=True, labelbottom=xticklab[k], left=True, labelleft=yticklab[k])
                   

              

                    #ax.set_xlabel(r"$x_{M_{\rm mix}}$")
                    #ax.set_ylabel(r"$\log f_{ej}$")

                    #mass = np.float(((lnlikefile.split("Mass"))[1])[0:3])
                    #en = np.float(((lnlikefile.split("Energy"))[1])[0:4])
                    #ax.set_title(r"Mass=%.0f, E51=%.1f"%(mass, en))
                    #fig.colorbar(cm, ax = ax)

            cbaxes=fig.add_axes([0.45,0.2,0.4,0.035])
            cb=plt.colorbar(im, orientation="horizontal",cax=cbaxes,format="%.0f")
            plt.subplots_adjust(hspace=0.1,wspace=0.1,bottom=0.15,left=0.15)

            #ax.spines['bottom'].set_color('none')
            #ax.spines['left'].set_color('none')
            #ax.spines['top'].set_color('none')
            #ax.spines['right'].set_color('none')

            labs=['Low-E','SN','HN']
            ticks=[0.18,0.50,0.82]
            for i,lab in enumerate(labs):
                fig.text(0.05,ticks[i],lab,ha='center',va='center')
            labs=['11', '13','15','25','40','100']
            ticks=np.arange(0.20,1.00,0.12)
            for i,lab in enumerate(labs):
                fig.text(ticks[i],0.05,lab,ha='center',va='center')


            #ax.tick_params(color='none',labelbottom='off',labelleft='off',bottom='off',left='off',top='off',right='off')

            fig.text(0.5,0.0,'Mass [M$_\odot$]',ha='center',va='bottom')
            fig.text(0.0,0.5,'Energy',ha='left',va='center',rotation='vertical')


            outfigfile = "../figs/Lnlike_%s.eps"%(line.strip())

            plt.savefig(outfigfile)


    return()

def make_starlist():

    # List of successful fitting run    
    subprocess.run("grep Process ../../qsub/Log_*.out | awk 'BEGIN{FS=\":\"}{print $1}' > tmp", shell=True)

    # Get starnames
    
    if os.path.isfile("starlist.txt"):
        os.remove("starlist.txt")

    fout = open("starlist.txt", "a")
    with open("tmp") as f:
        for line in f:
            starname = subprocess.run("grep Abundance %s | awk '{print $4}'"%(line.strip()), shell=True, capture_output=True, text=True)
            fout.write(starname.stdout)

    return


def plot_feh_xfe(abundance_datadir, starcatalogs, star_id, outfigname, plot_pmono=False):
 

    plt.style.use('seaborn-white')
    plt.rcParams["font.size"] = 16

    # Get abundance data

    #alldir = "../data/Hartwig22"
    filelist = glob.glob(abundance_datadir + "/*.csv")
    nstar = len(filelist)

    if len(starcatalogs)==2:
        elemnames=['C', 'N', 'O','Na', 'Mg', 'Al','Si', 'Ca','Sc','Ti', 'V', 'Cr','Mn','Co','Ni','Zn']

        solarabund = {"C":-3.57, "N":-4.17, "O":-3.31, "Na":-5.76, "Mg":-4.40, "Al":-5.55, "Si":-4.49, "Ca":-5.66, "Sc":-8.85, \
                     "Ti":-7.05, "V":-8.07,"Cr":-6.36, "Mn":-6.57, "Fe":-4.50, "Co":-7.01, "Ni":-5.78 , "Zn":-7.44}
        nx_ax = 4
        ny_ax = 4


    else:
        elemnames=['C','Na','Mg','Si','Ca', 'Sc', 'Ti', 'Cr','Mn','Co','Ni','Zn']

        solarabund = {"C":-3.57, "Na":-5.76, "Mg":-4.40, "Si":-4.49, "Ca":-5.66, "Sc":-8.85, \
                     "Ti":-7.05, "V":-8.07, "Cr":-6.36, "Mn":-6.57, \
                        "Fe":-4.50, "Co":-7.01, "Ni":-5.78 , "Zn":-7.44}
        nx_ax = 4
        ny_ax = 3

    #elemids=[6,8,11,12,13,14,20,24,25,27,28,30]
    #elemnames=['[C/Fe]','[O/Fe]','[Na/Fe]','[Mg/Fe]','[Al/Fe]','[Si/Fe]','[Ca/Fe]','[Cr/Fe]','[Mn/Fe]','[Co/Fe]','[Ni/Fe]','[Zn/Fe]']
    nelem = len(elemnames)

    xfes = np.zeros((nelem, nstar))
    #xfes_err = np.zeros((nelem, nstar))

    flgs = np.zeros((nelem, nstar))
    pmono = np.full(nstar, -9.99)
    fes = np.zeros(nstar)
    #starcatalog = "../input/EMP_mono_July21_2022/EMP_monomulti_NSNe5_crossmatch.csv"
    
 
    for i, abufile in enumerate(filelist):

        starname = ((abufile.split("/"))[-1]).strip("_abundance.csv")
        print(starname)

        for starcatalog in starcatalogs:
            df_cat = pd.read_csv(starcatalog)
            filt = df_cat[star_id] == starname
            if len(df_cat[filt])==0:
                continue
            else:
                break

        if len(starcatalogs)==2:

            pmono[i] = 1.0 - df_cat["xMean"][filt]
     
        

        df = pd.read_csv(abufile, index_col=0, skiprows=1, names = ('A(X)', 'sigtot', 'flag') )

        fe = df['A(X)']['Fe'] - solarabund["Fe"] 
        fes[i] = fe
       


        for k, elem in enumerate(elemnames):
            xfe = df['A(X)'][elem] - solarabund[elem] - fe 
            #if elem=="V" and xfe>1:
            #    print(starname, "V/Fe", xfe)

            xfes[k,i] = xfe
            flgs[k,i] = df['flag'][elem]


    xx = fes


    # Plot settings 


    if plot_pmono==True:
        cmap = plt.get_cmap('coolwarm')
    else:
        cmap=plt.get_cmap('tab20')

    #models=['15SN','25SN','25HN','40HN','100SN']

    #elemids=[6,8,11,12,13,14,20,24,25,27,28,30]
    #elemnames=['[C/Fe]','[O/Fe]','[Na/Fe]','[Mg/Fe]','[Al/Fe]','[Si/Fe]','[Ca/Fe]','[Cr/Fe]','[Mn/Fe]','[Co/Fe]','[Ni/Fe]','[Zn/Fe]']


    #mss=[4,7,7,12,16]
    #cols=['#669900','#669900','#ff9900','#ff9900','#669900']


    #refnums=[1,2,3,4,5,6,7,8,9]
    #refs=['Yong+13','Cohen+13','Roederer+14','Jacobson+15','Hansen+14','Placco+15','Frebel+15','Melendez+16','Placco+16']

    #mks=['o','v','^','s','*','p','D','+','x']


    axes=()
    subaxis=()
    for k in range(0,nelem):
        axes=np.hstack((axes,"ax" + str(k)))
        subaxis = np.hstack((subaxis, "subax" + str(k)))

    fig=plt.figure(figsize=(18,15))


    gs_master = GridSpec(nrows = ny_ax, ncols = nx_ax, width_ratios = np.array([1] * nx_ax), wspace=0.4, hspace=0.1)
    

    for k in range(0,nelem):

        y_ax, x_ax = np.divmod(k, nx_ax)


        vars()[axes[k]] = plt.subplot(gs_master[y_ax, x_ax])
        if y_ax!=3:
            vars()[axes[k]].tick_params(labelbottom=False)
        if x_ax!=0:
            vars()[axes[k]].tick_params(labelleft=False)         

        vars()[subaxis[k]] = vars()[axes[k]].inset_axes([1.0, 0., 0.28, 1.],sharey = vars()[axes[k]])     
        vars()[subaxis[k]].tick_params(axis="y", labelleft=False)

        if "EMP" in starcatalog:
            x1=-5.2
            x2=-2.8
        else:
            x1 = -4.8
            x2 = -1.8
        vars()[axes[k]].set_xlim(x1,x2)
        #if k<=1:
        #    y1=-1.
        #    y2=3.8
        #if k<=2:
        #    y1=-0.7
        #    y2=3.9
        #else:
        #    y1=-2.0
        #    y2=2.6

        y1 = -0.9
        y2 = 3.9

        vars()[axes[k]].set_ylim(y1,y2)
        vars()[axes[k]].text(x1+(x2-x1)*0.08,y2-(y2-y1)*0.15,elemnames[k])
        if y_ax == 3:
            vars()[axes[k]].set_xlabel('[Fe/H]')
        if x_ax == 0:
            vars()[axes[k]].set_ylabel('[X/Fe]')

        yy = xfes[k, :]
        fs = flgs[k, :]



        if plot_pmono==True:
            sc = vars()[axes[k]].scatter(x=xx, y=yy, c=pmono, cmap = cmap, marker='o', linewidth=2)
        else:
            sc = vars()[axes[k]].scatter(x=xx[fs==0], y=yy[fs==0], color=cmap(1), edgecolor=cmap(0), marker='o', s=15, linewidth=2, alpha=0.7)
            #sc = vars()[axes[k]].errorbar(x=xx[fs==1], y=yy[fs==1], yerr=np.array([0.2]*len(yy[fs==1])), \
            #        linestyle="", uplims = True, color=cmap(1))

            binwidth = 0.2
            bins = np.arange(-1., 4. + binwidth, binwidth)
            vars()[subaxis[k]].hist(yy[fs==0], bins=bins, orientation = 'horizontal',color=cmap(0))


    #axlist = [ vars()[axes[k]] for k in range(1, nelem+1)]
    #cbar = fig.colorbar(sc, ax = axlist.ravel().tolist())
    plt.savefig(outfigname)




def plot_abupattern_xfe_multimodels(datafile, paramfile, vary = "Mcut"):

    Zstart= 6
    Zend = 30
    
    starname, species, Znums, obs, sigma_ref, flags = \
                abundance.read_monoenriched(Zstart ,Zend, datafile)


    # Asymetrical uncertainties
    feherr = 0.1   # Assumed uncertainty in [Fe/H]
    sig_p, sig_m = abundance.get_Uncertainty_Znum(Znums, feherr)


    # Observed [Fe/H] and the [Fe/H] range for the model
    fehobs = (obs[Znums==26])[0] - (-4.5)

    # Get the best-fit model parameters
    df = pd.read_csv(paramfile)
    mass = df['Mass']
    en = df['E51']


    mmix = df['Mmix'][0]
    lfej = df['lfej'][0]
    mcut = df['Mcut'][0]
    fehmodel = df['[Fe/H]'][0]



    Znums_model = np.arange(Zstart, Zend + 1, 1)

    # Read yield model:

    picklefile = "../../yieldgrid/pickle/M%03.0fE%04.1f_fort.10.pickle"%(mass, en)
    df_yield = pd.read_pickle(picklefile)

    if vary=="Mcut":
        params = np.sort(df_yield["Mcut"].unique())
        best_indx = np.where(params==mcut)
    elif vary=="Mmix":
        params = np.sort(df_yield["Mout"].unique())
        best_indx = np.where(params==mmix)
    elif vary=="lfej":
        params = np.sort(df_yield["lfej"].unique())
        best_indx = np.where(params==lfej)
    else:
        print("No parameters called ", vary)
        sys.exit
          
    print("Variation in " + vary +": ", params)
    
    if len(best_indx)!=1:
        print("The best-fit parameter is not found in ", params)
        sys.exit()
    
    indx_min = 0 if best_indx[-1] - 2 < 0 else int(best_indx[-1] -2) 
    indx_max = len(params) if best_indx[-1] + 2 > len(params)-1 else int(best_indx[-1] + 2)
    
    model_logAs = ()
    modellab = ""
    for indx in range(indx_min, indx_max):

        print("Plotting for " + vary + "=%.2f"%(params[indx]))
        if vary=="Mcut":
            mcut = params[indx]
            lab = r"$M_{cut}$"
            unit = r"$M_{\odot}$"
        elif vary=="Mmix":
            mmix = params[indx]
            lab = r"$M_{mix}$"
            unit = ""
        elif vary=="lfej":
            lfej = params[indx]
            lab = r"$\log (f_{ej})$"
            unit = ""

        model_logA, M_H = yields.calc_yield(Znums_model, mass, en, \
                                        mmix, lfej, mcut, fehmodel, df_yield) 
        print(model_logA)
        if indx==indx_min:
            model_logAs = model_logA
            modellab = lab+r"$=%.1f $"%(params[indx]) + unit
        else:
            model_logAs = np.vstack((model_logAs, model_logA))
            modellab = modellab + ","+lab+r"$=%.1f $"%(params[indx]) + unit



    #


    outfigname = "../figs/" + starname + "_" + vary + ".png"

    #tarname,Mass,E51,Mmix,lfej,Mcut,[Fe/H],func,chi2,logA_c,logA_n,logA_o,logA_f,logA_ne,logA_na,logA_mg,logA_al,logA_si,logA_p,logA_s,logA_cl,logA_ar,logA_k,logA_ca,logA_sc,logA_ti,logA_v,logA_cr,logA_mn,logA_fe,logA_co,logA_ni,logA_cu,logA_zn,M_H
    plot_abupattern_xfe(Znums, obs, sig_p, sig_m, flags, fehobs, feherr, Znums_model, model_logAs, fehmodel, outfigname, modellab,melemlows=[21,22])





if __name__ == "__main__":

    #plot_models(before = True, MF = False, model = "Umeda", isotopes = "Basic")
    
    #make_starlist()

    #f = open("starlist.txt", "r")
    #starlist = f.readlines()
    #f.close()
    #input_catalog1 = "../input/EMP_mono_July21_2022/EMP_monomulti_NSNe5_crossmatch_Dec27_Ishigaki18.csv"
    #input_catalog2 = "../input/EMP_mono_July21_2022/EMP_monomulti_NSNe5_crossmatch_Dec27_Saga.csv"
    #pmulti_threshold = 99.0
    #plot_bestfit_mass_hist(starlist, input_catalog1, input_catalog2, pmulti_threshold, pval=False)

    #abundance_datadir = "../data/LAMOST_Subaru"
    #outfigname = "../figs/feh_LAMOST_Subaru.png"
    #plot_fehdist(abundance_datadir, outfigname)


    #datafile = "../data/TestDec19_2022/CS22957-027_abundance.csv"
    #paramfile = "../out/CS22957-027_MethodCalcAllLnlike_Grid_Mass015_Energy01.0.txt"
    datafile = "../data/Hartwig23/CS29498-043_abundance.csv"
    paramfile = "../out/CS29498-043/CS29498-043_MethodCalcAllLnlike_Grid_Mass011_Energy00.5.txt"
    vary="Mmix"
    plot_abupattern_xfe_multimodels(datafile, paramfile, vary)

    #abundance_datadir = "../data/LAMOST_Subaru"
    #starcatalogs = ["../input/LAMOST_Subaru/SAGA_LAMOSTSubaru_Dec19_2022.csv"]
    #outfigname = "../figs/xfe_feh_LAMOST_Subaru.png"
    #star_id = "Object"
    
    #abundance_datadir = "../data/Hartwig23"
    #input_catalog1 = "../input/EMP_mono_July21_2022/EMP_monomulti_NSNe5_crossmatch_Dec27_Ishigaki18.csv"
    #input_catalog2 = "../input/EMP_mono_July21_2022/EMP_monomulti_NSNe5_crossmatch_Dec27_Saga.csv"
    #starcatalogs = [input_catalog1, input_catalog2]
    #outfigname = "../figs/xfe_feh.png"
    #star_id = "Name"
    #symbol = "Ref"
    #plot_feh_xfe(abundance_datadir, starcatalogs, star_id, outfigname)

    #plot_all_lnlike()
   
    #lnlikefiles = glob.glob("../out/*/LnLike_*_MethodCalcAllLnlike_Grid_Mass025_Energy10.0.txt")
    
    #lnlikefiles = glob.glob("../out/*/LnLike_*_MethodCalcAllLnlike_Grid_Mass015_Energy01.0.txt")
    #plot_total_lnlike(lnlikefiles)

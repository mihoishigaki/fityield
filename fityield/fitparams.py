import re 
import numpy as np
import scipy.optimize as op
#from multiprocessing import Pool
import multiprocessing as mp
import pandas as pd

import sys, os
import gc

import time
from random import gauss


fityield_dir=os.environ['HOME']+'/HMP/fityield' 

sys.path.insert(0, os.path.abspath(fityield_dir))

from fityield import yields, GetUncertainty





def lnlike(theta, x, y, yerr, flg, bnds, mass, en, df_yield, znum_model_uplims=[21,22]):

    
    mmix,lfej,feh = theta
  

    if mmix < (bnds[0])[0] or mmix > (bnds[0])[1] or \
       lfej < (bnds[1])[0] or lfej > (bnds[1])[1] or \
       feh < (bnds[2])[0] or feh > (bnds[2])[1]:
        return -np.inf
        
    ymodel, M_H = yields.calc_yield(x, mass, en, mmix, lfej, feh, df_yield)
 
   
    inv_sigma2 = 1.0/yerr**2


    sum=0.0

    for i,ym in enumerate(ymodel):

        theory_lowlim_or_obs_uplim = False

        # Check if the element's model value is treated as a lower limit
        for zmu in znum_model_uplims:
            if x[i] == zmu and y[i] >= ym:
                theory_lowlim_or_obs_uplim = True
                #print("Theory for Znum=",x[i]," is treated as a lower limit")

        # Check if the observed value is an upper limit
        if (flg[i] == 1 or flg[i] == 9) and y[i] >= ym:
            theory_lowlim_or_obs_uplim=True
            #print("Observation for Znum=",x[i]," is treated as an upper limit")
            
        if theory_lowlim_or_obs_uplim == False:
            sum = sum - 0.5 * ((y[i] - ym)**2 * inv_sigma2[i] - np.log(inv_sigma2[i]))
        else:
            continue
    
    return sum





def lnlike2(x, y, yerr, flg, feh, mass, en, dfrow):

    #start = time.perf_counter()
    #start_cpu = time.process_time()

    mmix = dfrow['Mout']
    lfej = dfrow['lfej'] 
    mcut = dfrow['Mcut'] 


    #if mmix < (bnds[0])[0] or mmix > (bnds[0])[1] or \
    #   lfej < (bnds[1])[0] or lfej > (bnds[1])[1] or \
    #   feh < (bnds[2])[0] or feh > (bnds[2])[1]:
    #    return -np.inf

    
    
    
    elemid, massnum, yield_in_solarmass = yields.get_yield_in_mass_from_dfrow(dfrow)

    # Get mass of Fe
    Z_Fe = 26
    M_Fe = yields.get_element_mass(Z_Fe, elemid, massnum, yield_in_solarmass)
    M_H = M_Fe/10**(feh-4.5)/56


    ymodel = \
                yields.get_logA_from_yield_in_mass(x, M_H, elemid, massnum, yield_in_solarmass)




    
    #ymodel, M_H = yields.calc_yield(x, mass, en, mmix, lfej, feh, df_yield)


    inv_sigma2 = 1.0/yerr**2



    lnlike0 =  -0.5 * ((y - ymodel)**2 * inv_sigma2 - np.log(inv_sigma2))


    filt = ((x == 21) & (y >= ymodel)) | ((x == 22) & (y >= ymodel)) | ( ((flg == 1) | (flg == 9) ) & (y >= ymodel)) 

    lnlike = np.where(filt, 0, lnlike0)

    sum = np.sum(lnlike)

    result = [mmix, lfej, mcut, feh, sum]

    #for i,ym in enumerate(ymodel):

    #    theory_lowlim_or_obs_uplim = False

    #    # Check if the element's model value is treated as a lower limit
    #    for zmu in znum_model_uplims:
    #        if x[i] == zmu and y[i] >= ym:
    #            theory_lowlim_or_obs_uplim = True
    #            #print("Theory for Znum=",x[i]," is treated as a lower limit")

    #    # Check if the observed value is an upper limit
    #    if (flg[i] == 1 or flg[i] == 9) and y[i] >= ym:
    #        theory_lowlim_or_obs_uplim=True
    #        #print("Observation for Znum=",x[i]," is treated as an upper limit")

    #    if theory_lowlim_or_obs_uplim == False:
    #        sum = sum - 0.5 * ((y[i] - ym)**2 * inv_sigma2[i] - np.log(inv_sigma2[i]))
    #    else:
    #        continue

    
    #end = time.perf_counter()
    #end_cpu = time.process_time()

    #elapsed_time = end-start
    #elapsed_time_cpu = end_cpu - start_cpu
    
    #print("Elapsed:%.1f sec"%(elapsed_time), " CPU:%.1f sec"%(elapsed_time_cpu))

    return (result)




def lnlike3(x, y, yerr_p, yerr_m, flg, feh, mass, en, yddata):

    #start = time.perf_counter()
    #start_cpu = time.process_time()
    

    # yddata should be:  'Mcut', 'Mout', 'lfej', 'h1mass', 'h2mass', 'femass', yields....
    mcut = yddata[0]
    mmix = yddata[1]
    lfej = yddata[2] 
    h1mass = yddata[3]
    h2mass = yddata[4]
    femass = yddata[5]
    yd = yddata[6]

    M_H = femass/10**(feh-4.5)/56     # Total mass (progenitor star + ISM) of H atoms
    N_H = M_H/1.0



    #if mmix < (bnds[0])[0] or mmix > (bnds[0])[1] or \
    #   lfej < (bnds[1])[0] or lfej > (bnds[1])[1] or \
    #   feh < (bnds[2])[0] or feh > (bnds[2])[1]:
    #    return -np.inf

  
    ymodel = np.log10(yd/N_H)    


    #ymodel = \
    #            yields.get_logA_from_yield_in_mass(x, M_H, elemid, massnum, yield_in_solarmass)

    #ymodel = 


    
    #ymodel, M_H = yields.calc_yield(x, mass, en, mmix, lfej, feh, df_yield)


    inv_sig2_p = 1.0/yerr_p**2
    inv_sig2_m = 1.0/yerr_m**2

    diff = y - ymodel

    inv_sig2 = np.where(diff > 0, inv_sig2_p, inv_sig2_m)
 
    lnlike0 =  -0.5 * (diff**2 * inv_sig2 - np.log(inv_sig2_p))
    chi20 = diff**2 * inv_sig2

    filt = ((x == 21) & (y >= ymodel)) | ((x == 22) & (y >= ymodel)) | ( ((flg == 1) | (flg == 9) ) & (y >= ymodel)) 

    lnlike = np.where(filt, 0, lnlike0)
    chi2 = np.where(filt, 0, chi20)
    
    dof = np.count_nonzero(lnlike!=0) - 6
    
    #end = time.perf_counter()
    #end_cpu = time.process_time()

    #elapsed_time = end-start
    #elapsed_time_cpu = end_cpu - start_cpu
    
    #print("Elapsed:%.4f sec"%(elapsed_time), " CPU:%.4f sec"%(elapsed_time_cpu))

    return [mmix, lfej, mcut, feh, np.sum(lnlike), np.sum(chi2), dof]


def chi2(theta, x, y, yerr, flg, bnds, mass, en, yieldset, znum_model_uplims=[21,22]):


    mmix,lfej,feh = theta


    if mmix < (bnds[0])[0] or mmix > (bnds[0])[1] or \
       lfej < (bnds[1])[0] or lfej > (bnds[1])[1] or \
       feh < (bnds[2])[0] or feh > (bnds[2])[1]:
        return np.inf

    ymodel, M_H = yields.calc_yield(x, mass, en, mmix, lfej, feh, yieldset)


    inv_sigma2 = 1.0/yerr**2


    sum=0.0



    for i,ym in enumerate(ymodel):

        theory_lowlim_or_obs_uplim = False

        # Check if the element's model value is treated as a lower limit
        for zmu in znum_model_uplims:
            if x[i] == zmu and y[i] >= ym:
                theory_lowlim_or_obs_uplim = True
                #print("Theory for Znum=",x[i]," is treated as a lower limit")

        # Check if the observed value is an upper limit
        if (flg[i] == 1 or flg[i] == 9) and y[i] >= ym:
            theory_lowlim_or_obs_uplim=True
            #print("Observation for Znum=",x[i]," is treated as an upper limit")

        if theory_lowlim_or_obs_uplim == False:
            sum = sum - 0.5 * ((y[i] - ym)**2 * inv_sigma2[i] - np.log(inv_sigma2[i]))
        else:
            continue

    return sum, M_H







def fit_ml(x, y, yerr, flg, theta0, mass, en, yieldset, method):


    mmix, lfej, feh = theta0


    params = get_inputparams([ "bnd_mmix_min", "bnd_mmix_max", "mmix_nsteps", \
                             "bnd_lfej_min", "bnd_lfej_max", "lfej_nsteps", \
                              "bnd_w_feh", "feh_nsteps", "yieldset" ])

    mmix_min = params[0]
    mmix_max = params[1]
    lfej_min = params[3]
    lfej_max = params[4]
    feh_min = feh-0.5
    feh_max = feh+0.5
    
    # Parameters are alpha,Zcc, ZIa, f_Ia
    bnds = ((mmix_min, mmix_max), (lfej_min, lfej_max), (feh_min, feh_max))
        
    
    nll = lambda *args: -lnlike(*args)
    result = op.minimize(nll, theta0, args=(x, y, yerr, flg, bnds, mass, en, yieldset), method = method, bounds = bnds)

    theta = result["x"]
    success = result["success"]
    func = result["fun"]
    jac = result["jac"]
    
    return theta, success, func, jac



def calc_all_lnlike_old(paramfile, x, y, yerr, flg, fehobs, mass, en, interp=True):

    params = get_inputparams(paramfile, [ "bnd_mmix_min", "bnd_mmix_max", "mmix_nsteps", \
                             "bnd_lfej_min", "bnd_lfej_max", "lfej_nsteps", \
                              "bnd_w_feh", "feh_nsteps", "yieldset", "ncpu" ])

    # For continuous values for the fitting parameters
    if interp == True:

        mmix_step = (params[1] - params[0]) / params[2]
        lfej_step = (params[4] - params[3]) / params[5]


    else:
        mmix_step = 0.1
        lfej_step = 0.1


    feh_step = 2.0 * params[6] / params[7]
    fehs = np.arange(fehobs - params[6], fehobs + params[6] + feh_step, feh_step)



    #bnds = ((params[0], params[1] + mmix_step), (params[3], params[4] + lfej_step), \
    #        (fehobs - params[6], fehobs + params[6] + feh_step))


    # Read yield model:

    picklefile = "../../yieldgrid/pickle/M%03.0fE%04.1f_fort.10.pickle"%(mass, en)

    df_yield = pd.read_pickle(picklefile)


    #results = ()
    #for index, row in df_yield.iterrows():
    #    for feh in fehs:
    #        theta = [row['Mout'], row['lfej'], feh]
    #        lnlike_val = lnlike2(theta, x, y, yerr, flg, feh, bnds, mass, en, row)
    #       result = np.c_[theta, [lnlike_val]]
    #        if len(results)==0:
    #            results = result
    #        else:
    #            results = np.vstack((results, result))
    
    results = ()
    for feh in fehs:
        #theta = [row['Mout'], row['lfej'], feh]
        #results0 = []

        results0 = []
        ncpu = int(params[9])
        with mp.Pool(ncpu) as pool:

            #results0 = pool.imap(lnlike2, [(i, x, y, yerr, flg, feh, mass, en, row) \
            #for i, row in df_yield.iterrows()])

            results0 = pool.starmap_async(lnlike2, [(x, y, yerr, flg, feh, mass, en, row) \
                for i, row in df_yield.iterrows()]).get()
        
        pool.close()
        print(results0[0])
        
        if len(results) ==0:
            results = results0
        else:
            results = np.vstack((results, results0))
        print("Finished calculating for [Fe/H]=%.1f"%(feh))
    
    #print(len(results))
    sorted_results = np.array(sorted(results, key = lambda x: (x[0], x[1], x[2], x[3])))


    #x1 = np.arange(params[0], params[1] + mmix_step, mmix_step)
    #x2 = np.arange(params[3], params[4] + lfej_step, lfej_step)

    #feh_step = 2.0 * params[6] / params[7]
    #x3 = np.arange(feh - params[6], feh + params[6] + feh_step, feh_step)

    #yieldset = params[8]
    
    #X1, X2, X3 = np.meshgrid(x1,x2,x3)
    #X = np.c_[np.ravel(X1), np.ravel(X2), np.ravel(X3)]
    #lnlike_val = np.zeros_like(X[:,0])

    
    #thetas = np.c_[X[:,0],X[:,1],X[:,2]]
    

    print("Number of parameter sets = ", len(results))

    # Read yield file 

    #picklefile = "../../yieldgrid/pickle/M%03.0fE%04.1f_fort.10.pickle"%(mass, en)

    #df_yield = pd.read_pickle(picklefile)

    
    #lfej = -3.
    #filt = (df_yield['Mout'] == mmix) & (df_yield['lfej'] == lfej) 



    #bnds = ((params[0], params[1] + mmix_step), (params[3], params[4] + lfej_step), (feh - params[6], feh + params[6] + feh_step))

    #pool = mp.Pool(mp.cpu_count())
    
    #lnlike_val =[]
    #lnlike_val = pool.starmap_async(lnlike2, [(i, theta, x, y, yerr, flg, bnds, mass, en, df_yield) \
    #        for i, theta in enumerate(thetas)]).get()
    
    #pool.close()


    #del df_yield
    #gc.collect()
    
    return(sorted_results)



def calc_all_lnlike(x, y, yerr_p, yerr_m, flg, fehs, mass, en, ncpu):


    # Read yield model:

    picklefile = "../../yieldgrid/pickle/Natoms_M%03.0fE%04.1f_fort.10.pickle"%(mass, en)
    df = pd.read_pickle(picklefile)
    d = df.to_numpy()
    
   
    Z = np.array(x, dtype = int)
    indx = 8 + Z



    
    results = ()
    for feh in fehs:

        results0 = []
        with mp.Pool(int(ncpu)) as pool:

            #results0 = pool.imap(lnlike2, [(i, x, y, yerr, flg, feh, mass, en, row) \
            #for i, row in df_yield.iterrows()])

            results0 = pool.starmap_async(lnlike3, [(x, y, yerr_p, yerr_m, flg, feh, mass, en, row) \
                for row in zip(d[:,2], d[:,4], d[:, 5], d[:, 6], d[:, 7], d[:, 8], d[:,indx]) ]).get()
        
        pool.close()
        
        if len(results) ==0:
            results = results0
        else:
            results = np.vstack((results, results0))
        print("Finished calculating for [Fe/H]=%.1f"%(feh))
    
    #print(len(results))
    sorted_results = np.array(sorted(results, key = lambda x: (x[0], x[1], x[2], x[3])))



    print("Number of parameter sets = ", len(results))

    # Read yield file 

    #picklefile = "../../yieldgrid/pickle/M%03.0fE%04.1f_fort.10.pickle"%(mass, en)

    #df_yield = pd.read_pickle(picklefile)

    
    #lfej = -3.
    #filt = (df_yield['Mout'] == mmix) & (df_yield['lfej'] == lfej) 



    #bnds = ((params[0], params[1] + mmix_step), (params[3], params[4] + lfej_step), (feh - params[6], feh + params[6] + feh_step))

    #pool = mp.Pool(mp.cpu_count())
    
    #lnlike_val =[]
    #lnlike_val = pool.starmap_async(lnlike2, [(i, theta, x, y, yerr, flg, bnds, mass, en, df_yield) \
    #        for i, theta in enumerate(thetas)]).get()
    
    #pool.close()


    #del df_yield
    #gc.collect()
    
    return(sorted_results)

def get_inputparams(paramfile, paramnames):

    # Get input parameters from a input.param file.
    # * "paramnames" should be a list of parameter names as string
    
    f=open(paramfile)
    lines=f.readlines()
    f.close()


    vals = []
    
    for i,paramname in enumerate(paramnames):

        val = ""
        for line0 in lines:
            line = (line0.split("\t"))[0]
            if line.startswith(paramname):
                val=(line.split("="))[1]
                if re.search('\"',val):
                    val = val.replace('"','')
                    val = val.strip()
                
                elif paramname == "yieldset":
                    val = np.int(val)
                else:
                    val = np.float(val)
                break
            
        vals.append(val)
  
    return(vals)

def random_split_normal(mu, lower_sig, upper_sig):

    z = gauss(0, 1)
 
    return mu + z * (lower_sig if z < 0 else upper_sig)

if __name__ == "__main__":

    mu = 5.0
    lower_sig = 1.0
    upper_sig = 2.0

    random_array = []
    for _ in range(1000):
        random_array.append(random_split_normal(mu, lower_sig, upper_sig))
    import matplotlib.pyplot as plt

    #plt.hist(random_array)
    #plt.savefig("test_split_gauss.png")  


    #mass = 11.0
    #en = 0.5
    #picklefile = "../../yieldgrid/pickle/M%03.0fE%04.1f_fort.10.pickle"%(mass, en)

    #df_yield = pd.read_pickle(picklefile)
    #print(df_yield.columns)
    #arr = df_yield.to_numpy()
    #print(arr[0])
   
 

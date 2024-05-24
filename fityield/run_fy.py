import os, time
import sys
import glob
import numpy as np
import pandas as pd

import multiprocessing as mp

# Location of the codes. Edit according to the environment. 
fityield_dir=os.environ['HOME']+'/HMP/fityield' 

sys.path.insert(0, os.path.abspath(fityield_dir))

from fityield import abundance, yields, fitparams, makeplots, GetUncertainty


def run_fy_write_results_old(paramfile, starname, DataID, Znums, obs, sigma, flags, \
                         mass, en, theta0, Zstart, Zend, yieldset, method):
    
    print("Fitting yields of M = %.0f and E51 = %.1f ..."%(mass,en))

    
    if method == "SLSQP":

        print("Start fitting for M = %.0f Msun, E_51 = %.1f"%(mass, en))
        print("Initial guess for Mmix, logfej, [Fe/H] = %.2f, %.2f, %.1f"%\
              (theta0[0], theta0[1], theta0[2]))

        theta_best, success, func, jac = \
            fitparams.fit_ml(Znums, obs, sigma, flags, theta0, mass, en, \
                             yieldset, method)


    elif method == "CalcAllLnlike" or "CalcAllLnlike_Grid":

        feh = theta0[2]

        if method == "CalcAllLnlike":
            interp = True
        elif method == "CalcAllLnlike_Grid":
            interp = False
            
        
        results = fitparams.calc_all_lnlike(paramfile, Znums, obs, sigma, flags, feh, mass, en, interp)



        outpath="../out/"
        all_lnlike_file = outpath + \
        "LnLike_%s_%s_Yieldset%i_Method%s_Mass%03.0f_Energy%04.1f.txt"%\
        (starname, DataID, yieldset, method, mass, en)
        
        np.savetxt(all_lnlike_file,results,fmt='%.2f',delimiter=',')

        #makeplots.plot_lnlike(all_lnlike_file,feh)


        index_for_maxlnlike = [i for i, x in enumerate(results[:,4]) if x == max(results[:,4])]
        #index_for_maxlnlike = np.argmax(results[:,3])

        mmix_fit = results[index_for_maxlnlike, 0]
        lfej_fit = results[index_for_maxlnlike, 1]
        mcut_fit = results[index_for_maxlnlike, 2]
        feh_fit = results[index_for_maxlnlike, 3]
        func = results[index_for_maxlnlike, 4]

        #theta_best = [mmix_fit, lfej_fit, feh_fit]

        

    print("Writing results ...")
    

    # Read yield model:

    picklefile = "../../yieldgrid/pickle/Natoms_M%03.0fE%04.1f_fort.10.pickle"%(mass, en)

    df_yields = pd.read_pickle(picklefile)


    if len(mmix_fit)==1:

        theta_best = [mmix_fit[0], lfej_fit[0], mcut_fit[0], feh_fit[0]]


        Znums_model, model_logA = write_results(theta_best, func[0], mass, en, starname, \
                DataID, Zstart, Zend, yieldset, method, df_yields)

        print("Plotting ...")  
        outfigname = outpath + \
            "%s_%s_Yieldset%i_Method%s_Mass%03.0f_Energy%04.1f_Mout%03.1f_lfej%04.1f_Mcut%04.2f_feh%05.2f.png"%\
            (starname, DataID, yieldset, method, mass, en, mmix_fit, lfej_fit, mcut_fit, feh_fit)
        makeplots.plot_abupattern(Znums, obs, sigma, flags, Znums_model, \
                              model_logA, outfigname)
        #if model_multi==True:
        #    thetas = ()
        #    for tt in [mcut_fit[0]*0.8, mcut_fit[0], mcut_fit[0]*1.2]:
        #        thetas = np.hstack((thetas, [mmix_fit[0], lfej_fit[0], tt, feh_fit[0]]))


    else:
        for j, mmix in enumerate(mmix_fit):
            theta_best = [mmix, lfej_fit[j], mcut_fit[j], feh_fit[j]]

            Znums_model, model_logA = write_results(theta_best, func[j], mass, en, starname, \
                    DataID, Zstart, Zend, yieldset, method, df_yields)
    
            print("Plotting ...")
            outfigname = outpath + \
                "%s_%s_Yieldset%i_Method%s_Mass%03.0f_Energy%04.1f_Mout%03.1f_lfej%04.1f_Mcut%04.2f_feh%05.2f.png"%\
                 (starname, DataID, yieldset, method, mass, en, mmix, lfej_fit[j], mcut_fit[j], feh_fit[j])
            makeplots.plot_abupattern(Znums, obs, sigma, flags, Znums_model, \
                                     model_logA, outfigname) 

    return()



def run_fy_write_results(paramfile, datadir, withBS = False, plot = "Best"):

    start_time = time.time()    
    
    #starname, DataID, Znums, obs, sigma, flags, \
    #                     mass, en, theta0, Zstart, Zend, yieldset, method):


    DataID, Zstart, Zend, bnd_w_feh, feh_nsteps, method, ncpu = \
        fitparams.get_inputparams(paramfile, ["DataID", "Zstart", "Zend", \
                                               "bnd_w_feh", "feh_nsteps", "method", "ncpu"])

    #nputparams(paramfile, [ "bnd_mmix_min", "bnd_mmix_max", "mmix_nsteps", \
    #                         "bnd_lfej_min", "bnd_lfej_max", "lfej_nsteps", \
    #                          "bnd_w_feh", "feh_nsteps", "yieldset", "ncpu" ])



    print("Fitting yields to : ", DataID)




    # Read the data

    print("Reading observed abundances...")

    if DataID == "Nordlander19":

        datafile = '../data/Nordlander19/Table1_ion_averaged.txt'

    elif DataID == "Nordlander19_3DNLTE":

        datafile = '../data/Nordlander19/Table1_updated_3DNLTE.txt'

    else:
        datafile = datadir  + DataID + "_abundance.csv"  
    

    if os.path.isfile(datafile):

        if "Nordlander" in DataID:
            starname, species, df_obs = \
                abundance.read_nordlander19(Zstart ,Zend, datafile)
            Znums = abundance.get_atomic_number(species)
            obs,sigma_ref,flags = abundance.get_abundance_Znum(Znums, species, df_obs)

        else: 
            starname, species, Znums, obs, sigma_ref, flags = \
                abundance.read_monoenriched(Zstart ,Zend, datafile)
    else:

        print(datafile, " does not exist. Stop here.")
        sys.exit()
    
    print("Abundance data for ", starname, " read.")
    

    filt = (Znums!=21) & (Znums!=22)
    print("Number of measurement (except for Sc or Ti): %i"%(len(obs[filt])))

    if len(obs[filt])<7:
      
        print("Number of measurement not sufficient. Stop here.")  
        return()


    # Asymetrical uncertaintie
    if "Hartwig" in datadir:
        feherr = 0.1   # Assumed uncertainty in [Fe/H]
        sigma_p, sigma_m = abundance.get_Uncertainty_Znum(Znums, feherr)
    else:
        feherr = sigma_ref[Znums==26]
        sigma_p = sigma_ref
        sigma_m = sigma_ref
    



    # Observed [Fe/H] and the [Fe/H] range for the model
    fehobs = (obs[Znums==26])[0] - (-4.5)
    feh_step = 2.0 * bnd_w_feh / feh_nsteps
    fehs = np.arange(fehobs - bnd_w_feh, fehobs + bnd_w_feh + feh_step, feh_step)


    # Output directory
    outpath = "../out/" + starname + "/"
    if not os.path.isdir(outpath):
        os.makedirs(outpath)





    # Roop over 10 (M, E51) sets of models
    
    for mass in [11.0, 13.0, 15.0, 25.0, 40.0, 100.0]:

        if mass == 11.0:
            ens = [0.5]
        elif mass == 13.0:
            ens = [0.5, 1.0]
        elif mass == 15.0:
            ens = [1.0]
        elif mass == 25.0:
            ens = [1.0, 10.0]
        elif mass==40.0: 
            ens = [1.0, 30.0]
        elif mass==100.0:
            ens = [1.0, 60.0]
                
        for en in ens:

            #if mass!=15.0 or en!=1.0:
            #    continue

            print("Fitting yields of M = %.0f and E51 = %.1f ..."%(mass,en))

    
            if method == "SLSQP":

                print("Start fitting for M = %.0f Msun, E_51 = %.1f"%(mass, en))
                print("Initial guess for Mmix, logfej, [Fe/H] = %.2f, %.2f, %.1f"%\
                     (theta0[0], theta0[1], theta0[2]))
                theta_best, success, func, jac = \
                        fitparams.fit_ml(Znums, obs, sigma, flags, theta0, mass, en, \
                        yieldset, method)

            elif method == "CalcAllLnlike_Grid":

                #print(Znums, obs, sigma_p, sigma_m, fehs)
                #sys.exit()        
                results = fitparams.calc_all_lnlike(Znums, obs, sigma_p, sigma_m, flags, fehs, mass, en, ncpu)


            all_lnlike_file = outpath + \
                "LnLike_%s_Method%s_Mass%03.0f_Energy%04.1f.txt"%\
                    (starname, method, mass, en)
        
            np.savetxt(all_lnlike_file,results,fmt='%.2f',delimiter=',')

            #makeplots.plot_lnlike(all_lnlike_file,feh)

            # results are stored in the order of mmix, lfej, mcut, feh, np.sum(lnlike), np.sum(chi2)

            index_for_maxlnlike = [i for i, x in enumerate(results[:,4]) if x == max(results[:,4])]
            #index_for_maxlnlike = np.argmax(results[:,3])

            mmix_fit = results[index_for_maxlnlike, 0]
            lfej_fit = results[index_for_maxlnlike, 1]
            mcut_fit = results[index_for_maxlnlike, 2]
            feh_fit = results[index_for_maxlnlike, 3]
            func = results[index_for_maxlnlike, 4]
            chi2 = results[index_for_maxlnlike, 5]
            #theta_best = [mmix_fit, lfej_fit, feh_fit]
            dof = results[index_for_maxlnlike, 6]
  
        
        

            print("Writing results ...")
    

            # Read yield model:

            picklefile = "../../yieldgrid/pickle/M%03.0fE%04.1f_fort.10.pickle"%(mass, en)
            #picklefile = "../../yieldgrid/pickle/test3.pickle"
            df_yields = pd.read_pickle(picklefile)

            theta_best = [mmix_fit, lfej_fit, mcut_fit, feh_fit]
            Znums_model, model_logA = write_results(outpath, theta_best, func, chi2, dof, mass, en, starname, \
                      DataID, Zstart, Zend, method, df_yields)

            if len(mmix_fit)==1:

                print("Plotting abundances ...")  
                outfigname = outpath + \
                     "%s_Method%s_Mass%03.0f_Energy%04.1f_Mout%03.1f_lfej%04.1f_Mcut%04.2f_feh%05.2f.png"%\
                     (starname, method, mass, en, mmix_fit, lfej_fit, mcut_fit, feh_fit)
                makeplots.plot_abupattern(Znums, obs, sigma_p, sigma_m, flags, Znums_model, \
                              model_logA, outfigname)

                outfigname = outpath + \
                     "XFe_%s_Method%s_Mass%03.0f_Energy%04.1f_Mout%03.1f_lfej%04.1f_Mcut%04.2f_feh%05.2f.png"%\
                     (starname, method, mass, en, mmix_fit, lfej_fit, mcut_fit, feh_fit)

                makeplots.plot_abupattern_xfe(Znums, obs, sigma_p, sigma_m, flags, fehobs, feherr, \
                   Znums_model, model_logA, feh_fit, outfigname, melemlows=[21,22])
                
                #if change_models==True:
                #    for mcut in [mcut_fit, ] 




            elif plot=="All":   # To plot abundances for all parameter sets
                for j, mmix in enumerate(results[:,0]):
                    theta_best = [mmix, results[j, 1], results[j, 2], results[j, 3]]

                    print("Plotting ...")
                    outfigname = outpath + \
                        "%s_Method%s_Mass%03.0f_Energy%04.1f_Mout%03.1f_lfej%04.1f_Mcut%04.2f_feh%05.2f.png"%\
                        (starname, method, mass, en, mmix, theta_best[1], theta_best[2], theta_best[3])
                    makeplots.plot_abupattern(Znums, obs, sigma_p, sigma_m, flags, Znums_model, \
                                     model_logA, outfigname)


            else:


                for j, mmix in enumerate(mmix_fit):
                    theta_best = [mmix, lfej_fit[j], mcut_fit[j], feh_fit[j]]

                    #Znums_model, model_logA = write_results(theta_best, func[j], chi2[j], mass, en, starname, \
                    #       DataID, Zstart, Zend, method, df_yields)
    
                    print("Plotting ...")
                    outfigname = outpath + \
                        "%s_Method%s_Mass%03.0f_Energy%04.1f_Mout%03.1f_lfej%04.1f_Mcut%04.2f_feh%05.2f.png"%\
                        (starname, method, mass, en, mmix, lfej_fit[j], mcut_fit[j], feh_fit[j])

                    makeplots.plot_abupattern(Znums, obs, sigma_p, sigma_m, flags, Znums_model, \
                                     model_logA, outfigname) 

                    outfigname = outpath + \
                        "XFe_%s_Method%s_Mass%03.0f_Energy%04.1f_Mout%03.1f_lfej%04.1f_Mcut%04.2f_feh%05.2f.png"%\
                        (starname, method, mass, en, mmix, lfej_fit[j], mcut_fit[j], feh_fit[j])

                    makeplots.plot_abupattern_xfe(Znums, obs, sigma_p, sigma_m, flags, fehobs, feherr, \
                          Znums_model, model_logA, feh_fit[j], outfigname, melemlows=[21,22])
    
    print("Process finished with %.1f s."%(time.time() - start_time))
    return()



def write_results(outpath, theta_best, func, chi2, dof, mass, en, \
                  starname, DataID, Zstart, Zend, method, df_yields):

    print("Writing results to files ...")
    print("The number of bestfit parameter sets: %i"%(len(func)))



    outfilename = outpath + \
        "%s_Method%s_Mass%03.0f_Energy%04.1f.txt"%\
       (starname, method, mass, en)


    df = pd.DataFrame({'Starname':[starname]*len(func)})
    df['Mass'] = [mass] * len(func)
    df['E51'] = [en] * len(func)
    df['Mmix'] = theta_best[0]
    df['lfej'] = theta_best[1]
    df['Mcut'] = theta_best[2]
    df['[Fe/H]'] = theta_best[3]
    df['func'] = func
    df['chi2'] = chi2
    df['dof'] = dof
    df=df.round({"Mass":0, "E51":1, "Mmix":1, "lfej":1, "Mcut":2, "[Fe/H]":1, "func":3, "chi2": 4, "dof": 1})

    Znums_model = np.arange(Zstart, Zend + 1, 1)


    model_logAs = ()
    M_Hs = ()
    for j, ff in enumerate(func):   
 
        model_logA, M_H = yields.calc_yield(Znums_model, mass, en, \
                                        theta_best[0][j], theta_best[1][j], \
                                        theta_best[2][j], theta_best[3][j], df_yields)
        if j==0:
            model_logAs = model_logA
        else:
            model_logAs = np.vstack((model_logAs, model_logA))
        M_Hs = np.hstack((M_Hs, M_H))
        
        
    for i,zz in enumerate(Znums_model):

        elem = abundance.get_elem_name([zz])
        column_name, column_name_err, column_name_flag = \
                abundance.get_column_name(elem[0])
        if len(M_Hs)==1:
            df[column_name] = model_logAs[i]
        else:
            df[column_name] = model_logAs[:, i]
        df = df.round({column_name:2})

  
    df['M_H'] = M_Hs
    
    df.to_csv(outfilename, index=False)

    return(Znums_model,model_logA)
    



def main_old(paramfile):
    

    # MAIN

    # Input a data ID from the terminal


    DataID, Zstart, Zend, yieldset, mmix0, lfej0, feh0, method = \
        fitparams.get_inputparams(paramfile, ["DataID", "Zstart", "Zend", \
                                               "yieldset", "mmix_ini",\
                                   "lfej_ini", "feh_ini", "method"])


    print("Fitting yields to : ", DataID)




    # Read the data

    print("Reading observed abundances...")

    if DataID == "Nordlander19":

        datafile = '../data/Nordlander19/Table1_ion_averaged.txt'

    elif DataID == "Nordlander19_3DNLTE":

        datafile = '../data/Nordlander19/Table1_updated_3DNLTE.txt'

    elif 'monoenriched' in DataID:
        datafile = '../data/monoenriched/' + (DataID.split("_"))[0] + "_abundance.csv"  
    
    else:

        print("DataID ", DataID, " is not supported. Sorry.")
        sys.exit() 


    if os.path.isfile(datafile):

        if "Nordlander" in DataID:
            starname, species, df_obs = \
                abundance.read_nordlander19(Zstart ,Zend, datafile)

        elif "monoenriched" in DataID: 
            starname, species, df_obs = \
                abundance.read_monoenriched(Zstart ,Zend, datafile)

    else:

        print(datafile, " does not exist. Stop here.")
        sys.exit()
    
    print("Abundance data for ", starname, ":")
    print(df_obs)
    
    Znums = abundance.get_atomic_number(species)

    obs,sigma,flags = abundance.get_abundance_Znum(Znums, species, df_obs)

    #print("Atomic numbers of the data: ", Znums)
    #print("Abundances in log A: ", obs)
    #print("Abundance errors: ",sigma)
    #print("Abundance flags (see abundance.py): ",flags)


    # Set an initial guess
    
    if feh0 == -99.99:
        feh0 = (obs[Znums==26])[0] - (-4.5)



    mcut0 = 0.0
    theta0 = [mmix0, lfej0, mcut0, feh0]



    if yieldset == 1:

        mass = 25.0
        en = 1.0

        
        run_fy_write_results(paramfile, starname, DataID, Znums, obs, sigma, flags, mass, en, theta0, \
                             Zstart, Zend, yieldset, method)
    

    elif yieldset == 2:

        for mass in [11.0, 13.0, 15.0, 25.0, 40.0, 100.0]:

            if mass == 11.0:
                ens = [0.5]
            elif mass == 13.0:
                ens = [0.5, 1.0]
            elif mass == 15.0:
                ens = [1.0]
            elif mass == 25.0:
                ens = [1.0, 10.0]
            elif mass==40.0: 
                ens = [1.0, 30.0]
            elif mass==100.0:
                ens = [1.0, 60.0]
                
            for en in ens:
                

                #picklefile = "../../yieldgrid/pickle/M%03.0fE%04.1f_fort.10.pickle"%(mass, en)

                #df_yield = pd.read_pickle(picklefile)
                
                run_fy_write_results(paramfile, starname, DataID, Znums, obs, sigma, flags, mass, en, theta0, \
                                     Zstart, Zend, yieldset, method)




    print("Fitting finished successfully.")

    return


if __name__ == "__main__":
    


    #main_old(*sys.argv[1:2]) 
    #main(*sys.argv[1:2])
    #paramfile = "../config/input.params.SMSSJ010839.58-285701.5"
    run_fy_write_results(*sys.argv[1:3])

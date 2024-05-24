#
#     This file is part of FitYield
#     
#
#
#

import os
import sys
import numpy as np
import pandas as pd


fityield_dir=os.environ['HOME']+'/HMP/fityield'

sys.path.insert(0, os.path.abspath(fityield_dir))

from fityield import GetUncertainty


def read_nordlander19(Znum_start,Znum_end, datafile):

    starname='SMSS1605-1443'

    #datafile = datapath + "/Nordlander19/Table1_updated_3DNLTE.txt"
    
    print("Reading abundance data from ", datafile)
    
    df0=pd.read_csv(datafile,sep='\t',dtype={'A(X)':'float','sigtot':'float','flag':'object'})
    df=pd.DataFrame({'Starname':[starname]})

    species=()
    
    for i,elem in enumerate(df0['Species']):

        znum=get_atomic_number([elem])

        
        if znum<Znum_start:
            continue
        if znum>Znum_end:
            break
        
        #elem=elem.strip()

        #elem_name=(elem.split('_'))[0]
        #ion_mol=(elem.split('_'))[1]

        species=np.hstack((species,elem))
        
        column_name,column_name_err,column_name_flag=get_column_name(elem) 
        df[column_name]=df0['A(X)'][i]-12 # Abundances are given by log_epsilon
        df[column_name_err]=df0['sigtot'][i]
        df[column_name_flag]=get_int_flag(str(df0['flag'][i]))

    abundance_data=df

    return(starname,species,abundance_data)





def read_monoenriched(Znum_start,Znum_end, datafile):

    starname=(((datafile.split("/"))[-1]).split("_"))[0]

    
    print("Reading abundance data from ", datafile)
    
    df0=pd.read_csv(datafile,dtype={'A(X)':'float','sigtot':'float','flag':'int'})
    df=pd.DataFrame({'Starname':[starname]})

    species = ()
    Znums = ()
    obs = ()
    sigma = ()
    flags = ()    

    for i,elem in enumerate(df0['Species']):

        znum=get_atomic_number([elem])

        if znum==167:  # Tentatively ignore C+N
            continue   
        if znum<Znum_start:
            continue
        if znum>Znum_end:
            break
        

        if np.isnan(df0['A(X)'][i])==True:
            continue

        Znums = np.hstack((Znums, znum))
        species=np.hstack((species,elem))
        
        A_X = df0['A(X)'][i]
        sig = df0['sigtot'][i]
        flag = df0['flag'][i] 

        obs = np.hstack((obs, A_X))
        sigma = np.hstack((sigma, sig))
        flags = np.hstack((flags, flag))


    return(starname,species, Znums, obs, sigma, flags)





def get_column_name(elem):

    """
    Get the column name for a given element and ion
    """
    
    column_name='logA_'+elem.lower()
    column_name_err=column_name+'_e'
    column_name_flag=column_name+'_fl'    

    
    return(column_name,column_name_err,column_name_flag)





def get_int_flag(binary_str):

    """
    Meaning of the flags
    
    1st bit (+1)  :  Upper limit
    2nd bit (+2)  :  Lower limit
    3rd bit (+4)  :  I and III difference more than 2 sigma
    4th bit (+8)  :  Based on molecular lines
    5th bit (+16) :  3D and/or NLTE values differ by more than 5 sigma
    6th bit (+32) :  Assumed value 

    """
    
    return(int(binary_str,2))

#result=read_nordlander19()


def get_atomic_number(elems):



    # Read atomic number file
    df=pd.read_csv("../data/atomic_numbers.txt",dtype={'Elem':str,'Znum':int},keep_default_na=False)


    znums=()
    
    znums = np.zeros(len(elems))
    for i, elem in enumerate(elems):

        if elem=="CN":
            Znum = 167 # For C+N combined
        else:
            elem = elem.replace(" II", "")
            elem = elem.replace(" I", "")
	        
            # Express the input element name as upper-case character
            elem_upper=elem.upper()

   
            filt = df["Elem"] == elem_upper
  
            Znum=df[filt]["Znum"]


        
        if np.size(Znum)!=1:

            print("Atomic number for ",elem," is not found\n")
            sys.exit()

            
        znums[i] = Znum
        
    return(znums)


def get_elem_name(Znums):


    # Read atomic number file
    df=pd.read_csv("../data/atomic_numbers.txt",dtype={'Elem':str,'Znum':int},keep_default_na=False)


    elemnames=()

    for zz in Znums:

   
        filt = df["Znum"] == zz
  
        elemname=(df[filt]["Elem"]).values


        
        if np.size(elemname) != 1:

            print("Element name for an atomic number = ",zz," is not found\n")
            sys.exit()

        str1 = (elemname[0])[0]
        str2 = ((elemname[0])[1:]).lower()
        elemname_formatted = str1+str2
     
        elemnames = np.hstack((elemnames,elemname_formatted))
        
    return(elemnames)


def get_abundance_Znum(Znums,species,df):

    obs=np.zeros_like(Znums,dtype=np.float64)
    sigma=np.zeros_like(Znums,dtype=np.float64)
    flags=np.zeros_like(Znums,dtype=np.int)

    for i,znum in enumerate(Znums):
    
        c_abu,c_err,c_flag=get_column_name(species[i])
     
        obs[i]=(df[c_abu].values)[0]
        sigma[i]=(df[c_err].values)[0]
        flags[i]=(df[c_flag].values)[0]
    
    return(obs,sigma,flags)








def make_abundance_datatable_monoenriched_old(pmulti_bound, outdir):

    #catalog = "/home/miho/Tilman_Machine_Lerning/EMP_starlist/uniqueEMPstars_XFe_mono-enriched.csv"

    catalog = "../input/EMP_mono_July21_2022/EMP_monomulti_NSNe5_crossmatch.csv"

    df = pd.read_csv(catalog)

    #starname='SMSS1605-1443'

    #datafile = datapath + "/Nordlander19/Table1_updated_3DNLTE.txt"

    #print("Reading abundance data from ", datafile)

    #df0=pd.read_csv(datafile,sep='\t',dtype={'A(X)':'float','sigtot':'float','flag':'object'})
    #df=pd.DataFrame({'Starname':[starname]})

    #Species	A(X)	sigtot	flag 


    feh_error = 0.20


    elems = ["C", "O", "Na", "Mg", "Al", "Si", "Ca", "Cr", "Mn", "Fe", "Co", "Ni", "Zn"]
    xfeerr = [0.20, 0.20, 0.50, 0.30, 0.50, 0.20, 0.20, 0.40, 0.30, 0.20, 0.25, 0.10, 0.25]
    solarabund = [-3.57, -3.31, -5.76, -4.40, -5.55, -4.49, -5.66, -6.36, -6.57, -4.50, -7.01, -5.78, -7.44]

    #outdir = "../data/monoenriched/"



    ct = 0

    for i, row in df.iterrows():

        starname = row["Name"]


        N_SN = row["MyPred"]
        
        p_multi = row["xMean"]
        std = row["xStd"]   
        #print(starname, N_SN)

        #if N_SN>1.0:
        #    continue

        if p_multi + std >= pmulti_bound:
            continue

        ct = ct + 1

       
        feh = float(row["FeH"])
    
        species = ()
        xhs = ()
        sigma = ()
        flag = ()



        for j, elem in enumerate(elems):


            species = np.hstack((species, elem))
            if "[" + elem + "/Fe]" in row.index:
                xfe = row["[" + elem + "/Fe]"] 
                if xfe == -99.0:
                    xhs = np.hstack((xhs, -99.99))
                else:
                    xhs = np.hstack((xhs, xfe + feh))
            elif "[Fe/" + elem + "]" in row.index:
                fex = row["[Fe/" + elem + "]"]
                if fex == -99.0:
                    xhs = np.hstack((xhs, -99.99))
                else:
                    xhs = np.hstack((xhs, -1.*fex + feh))
            elif elem == "Fe":
                xhs = np.hstack((xhs, feh))
            else:
                xhs = np.hstack((xhs, -99.99))
            sigma = np.hstack((sigma, xfeerr[j]))   # This should be replaced by [X/H] errors
            flag = np.hstack((flag, "000000"))


        abundance = ()

        for j, xh in enumerate(xhs):
            if xh == -99.99:
                abundance = np.hstack((abundance, -99.99))
            else:
                abundance = np.hstack((abundance, np.round(xh + solarabund[j] + 12, 2)))


        data = {"Species":species, "A(X)":abundance, "sigtot":sigma, "flag":flag} 
        dfout = pd.DataFrame(data)
       
        # Count the number of elements:
        filt = dfout["A(X)"]!=-99.99
        nmeasurement = len(dfout[filt])
        print(starname, N_SN, nmeasurement)

        dfout.to_csv(outdir + "/" + starname + "_abundance.csv", index=False)

    print("Number of mono-enriched stars: %i"%(ct))





def make_abundance_datatable_monoenriched(catalogs, elems, outdir, pmulti_bound):

    #catalog = "/home/miho/Tilman_Machine_Lerning/EMP_starlist/uniqueEMPstars_XFe_mono-enriched.csv"

    #catalog = "../input/EMP_mono_July21_2022/EMP_monomulti_NSNe5_crossmatch_Oct14.csv"


    for catalog in catalogs:  

        df = pd.read_csv(catalog)

        ct = 0

        for i, row in df.iterrows():

            if pmulti_bound == 0:   # For the LAMOST_Subaru sample
        
                starname = row["Object"]

            else:
                starname = row["Name"]


                N_SN = row["MyPred"]
        
                p_multi = row["xMean"]
                std = row["xStd"]   
                #print(starname, N_SN)

                #if N_SN>1.0:
                #    continue

                #if p_multi + std >= pmulti_bound:
                if p_multi >=pmulti_bound:
                    continue
   
 
            species = ()
            xhs = ()
            abundances = ()
            sigma = ()
            flag = ()

            znums = get_atomic_number(elems)
            
            for j, elem in enumerate(elems):

            
                species = np.hstack((species, elem))

                # Get Solar abundance for this element
                
                solar = get_solar_abundance(znums[j])

              

                xh = np.nan
                sig = np.nan
                fl = np.nan 
                cdetected = 0
                ndetected = 0

                # Read table data
                if elem == "Li":

                    if "A(Li I)" in row.index and "A(Li)" in row.index:
                        if np.isnan(row["A(Li I)"])==False:
                            xh = row["A(Li I)"]
                            sig = row["d(Li I)"]
                            fl = row["f(Li I)"]
                        elif np.isnan(row["A(Li)"])==False:
                            xh = row["A(Li)"]
                            sig = row["d(Li)"]
                            fl = row["f(Li)"]
                    elif "A(Li)" in row.index:
                        xh = row["A(Li)"]
                        sig = row["d(Li)"]
                        fl = row["f(Li)"]


                elif elem == "C":

                    if "[C/H]" in row.index:

                        if "[CH/H]" in row.index:
                            if np.isnan(row["[C/H]"])==False:
                                xh = row["[C/H]"]
                                sig = row["d(C)"]
                                fl = row["f(C)"]
                            elif np.isnan(row["[CH/H]"])==False:
                                xh = row["[CH/H]"]
                                sig = row["d(CH)"]
                                fl = row["f(CH)"]
                            elif np.isnan(row["[C I/H]"])==False:
                                xh = row["[C I/H]"]
                                sig = row["d(C I)"]
                                fl = row["f(C I)"]
                            elif np.isnan(row["[C2/H]"])==False:
                                xh = row["[C2/H]"]
                                sig = row["d(C2)"]
                                fl = row["f(C2)"]
                        else:
                            xh = row["[C/H]"]
                            sig = row["d(C)"]
                            fl = row["f(C)"]

                    if np.isnan(xh) == False:
                        cdetected = 1
                        ch = xh 
                        chsig = sig
                        chfl = fl

                elif elem == "N":
	
                    if "[N/H]" in row.index:

                        if "[NH/H]" in row.index:

                            if np.isnan(row["[N/H]"])==False:
                                xh = row["[N/H]"]
                                sig = row["d(N)"]
                                fl = row["f(N)"]
                            elif np.isnan(row["[NH/H]"])==False:
                                xh = row["[NH/H]"]
                                sig = row["d(NH)"]
                                fl = row["f(NH)"]
                        else:
                            xh = row["[N/H]"]
                            sig = row["d(N)"]
                            fl = row["f(N)"]            

                    if np.isnan(xh) == False:
                        ndetected = 1
                        nh = xh 
                        nhsig = sig
                        nhfl = fl

                elif elem == "CN":
                    if cdetected == 1 and ndetected == 1:
                        if chfl == np.nan:
                            if nhfl == 0:
                                xh = np.log10(10**(ch - 3.57) + 10**(nh - 4.17)) - solar
                            else: 
                                xh = ch
                                sig = 0. 
                                fl = "<"
                        else:
                            if nhfl == np.nan:
                                xh = nh
                                sig = 0.
                                fl = "<"
                    elif cdetected == 1:
                        if chfl == np.nan:
                            xh = ch
                            sig = 0.
                            fl = ">"   # Lower limit
                        else:
                            xh = ch 
                            sig = 0.
                            fl = chfl    # Upper limit
                    elif ndetected == 1:   
                        if nhfl == np.nan:
                            xh = nh
                            sig = 0.
                            fl = ">"
                        else:
                            xh = nh 
                            sig = 0.
                            fl = nhfl
 
                elif elem == "O": 

                    if "[O/H]" in row.index and "[O I/H]" in row.index:
                        if np.isnan(row["[O/H]"])==False:
                            xh = row["[O/H]"]
                            sig = row["d(O)"]
                            fl = row["f(O)"]
                        elif np.isnan(row["[O I/H]"])==False:
                            xh = row["[O I/H]"]
                            sig = row["d(O I)"]
                            fl = row["f(O I)"]
                        elif np.isnan(row["[OH/H]"])==False:
                            xh = row["[OH/H]"]
                            sig = row["d(OH)"]
                            fl = row["f(OH)"]

                    elif "[O/H]" in row.index:
                        xh = row["[O/H]"]
                        sig = row["d(O)"]
                        fl = row["f(O)"]

                else:
                    if "[" + elem + " I/H]" in row.index and "[" + elem + " II/H]" in row.index and "[" + elem + "/H]" in row.index: 

                        if np.isnan(row["[" + elem + " I/H]"])==False and np.isnan(row["[" + elem + " II/H]"])==False:
                            if row["f(" + elem + " I)"]!="<" and row["f(" + elem + " II)"]!="<":
                                xh = 0.5 * (row["[" + elem + " I/H]"] + row["[" + elem + " II/H]"])
                                sig = np.sqrt(np.sum((row["[" + elem + " I/H]"] - xh)**2 + (row["[" + elem + " II/H]"] - xh)**2))
                                fl = np.nan
                            elif row["f(" + elem + " I)"]!="<":
                                xh = row["[" + elem + " I/H]"]
                                sig = row["d(" + elem + " I)"] 
                                fl = row["f(" + elem + " I)"]
                            elif row["f(" + elem + " II)"]!="<":
                                xh = row["[" + elem + " II/H]"]
                                sig = row["d(" + elem + " II)"]
                                fl = row["f(" + elem + " II)"]
                        elif np.isnan(row["[" + elem + " I/H]"])==False:
                            xh = row["[" + elem + " I/H]"]
                            sig = row["d(" + elem + " I)"]
                            fl = row["f(" + elem + " I)"] 
                        elif np.isnan(row["[" + elem + " II/H]"])==False:
                            xh = row["[" + elem + " II/H]"]
                            sig = row["d(" + elem + " II)"]
                            fl = row["f(" + elem + " II)"]
                        elif np.isnan(row["[" + elem + "/H]"])==False:
                            xh = row["[" + elem + "/H]"]
                            sig = row["d(" + elem + ")"]
                            fl = row["f(" + elem + ")"]
                    elif "[" + elem + " I/H]" in row.index and "[" + elem + "/H]" in row.index:
                        if np.isnan(row["[" + elem + " I/H]"])==False:
                            xh = row["[" + elem + " I/H]"]
                            sig = row["d(" + elem + " I)"]
                            fl = row["f(" + elem + " I)"]
                        else:
                            xh = row["[" + elem + "/H]"]
                            sig = row["d(" + elem + ")"]
                            fl = row["f(" + elem + ")"]

                    elif "[" + elem + "/H]" in row.index:
                        xh = row["[" + elem + "/H]"]
                        sig = row["d(" + elem + ")"]
                        fl = row["f(" + elem + ")"]
                    elif  "[" + elem + " I/H]" in row.index:
                        xh = row["[" + elem + " I/H]"]
                        sig = row["d(" + elem + " I)"]
                        fl = row["f(" + elem + " I)"]
                    elif "[" + elem + " II/H]" in row.index:
                        xh = row["[" + elem + " II/H]"]
                        sig = row["d(" + elem + " II)"]
                        fl = row["f(" + elem + " II)"]
                    elif "[" + elem + "/H]" in row.index:
                        xh = row["[" + elem + "/H]"]
                        sig = row["d(" + elem + ")"]
                        fl = row["f(" + elem + ")"]

                xhs = np.hstack((xhs, xh))
                if np.isnan(xh) == True:
                    abundances = np.hstack((abundances, np.nan))
                else:
                    abundances = np.hstack((abundances, np.round(xh + solar, 2)))
                sigma = np.hstack((sigma, sig))
                flag = np.hstack((flag, fl))
                #print(elem, xh, sig, fl)
                #flag = np.hstack((flag, "000000"))


            fls = ()
            for j, fl in enumerate(flag):
                if fl=="<":
                    fls = np.hstack((fls, int(1)))
                elif fl==">":
                    fls = np.hstack((fls, int(-1)))
                else:
                    fls = np.hstack((fls, int(0)))

            data = {"Species":species, "A(X)":abundances, "sigtot":sigma, "flag":fls} 
            dfout = pd.DataFrame(data)
       
            # Count the number of elements:
            filt = (dfout["A(X)"].notna()) & (dfout["flag"]==0) & (dfout["Species"]!="Ti") & (dfout["Species"]!="Sc")  & (dfout["Species"]!="Li")
            nmeasurement = len(dfout[filt])
            #print(starname, N_SN, nmeasurement)
        
            if nmeasurement<=7:
                continue
            else:
                ct = ct + 1
                dfout.to_csv(outdir + "/" + starname + "_abundance.csv", index=False)

        print("Number of mono-enriched stars: %i"%(ct))



def get_Uncertainty_Znum(Znums, feherr):


    elemnames = get_elem_name(Znums)

    sigma_p = np.zeros(len(Znums))
    sigma_m = np.zeros(len(Znums))
    for i, znum in enumerate(Znums):

        if znum==26:
            sigma_p[i] = feherr
            sigma_m[i] = feherr
        else:
            ratio = "[" + elemnames[i] + "/Fe]"
            print(ratio) 
            # Get errors in [X/Fe]
            sig_p, sig_m = GetUncertainty.get_sigma_obs(ratio)

            # Get errors in [X/H] or logA
            sigma_p[i] = np.sqrt(sig_p**2 - feherr**2) 
            sigma_m[i] = np.sqrt(sig_m**2 - feherr**2)
    return(sigma_p, sigma_m)


def get_solar_abundance(z):

    solarfile='../utility/solarabund/ipcc.dat'

    zsuns,abusuns=np.loadtxt(solarfile,usecols=(1,5),unpack=True,skiprows=1)

    if z==10 or z==18:
        return np.nan
    elif z==167:
        indx_C = list(zsuns).index(6)
        indx_N = list(zsuns).index(7)
        abusun = np.log10(10**abusuns[indx_C] + 10**abusuns[indx_N])
    else:

        indx = list(zsuns).index(z)
        abusun = abusuns[indx]

    return(abusun)



def logA2XFe(znum, logA, feh):

    solarFe = get_solar_abundance(26)

    # Convirt logA with solar abundance
    xfes = np.zeros_like(logA)
    for i, xx in enumerate(znum):
        solarabu = get_solar_abundance(xx)
        if solarabu!=np.nan:
            xfes[i] = logA[i] - solarabu - feh
        else:
            xfes[i] = np.nan
    return(xfes)


 


if __name__ == "__main__":

     catalogs = ["../input/EMP_mono_July21_2022/EMP_monomulti_NSNe5_crossmatch_Dec27_Saga.csv", \
	"../input/EMP_mono_July21_2022/EMP_monomulti_NSNe5_crossmatch_Dec27_Ishigaki18.csv"]
     pmulti_bound = 99.
     elems = ["Li", "C", "N", "CN", "O", "Na", "Mg", "Al", "Si", "S", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]
     outdir = "../data/Hartwig23"
     make_abundance_datatable_monoenriched(catalogs, elems, outdir, pmulti_bound)


     #catalog ="../input/LAMOST_Subaru/SAGA_LAMOSTSubaru_Dec19_2022.csv" 
     
     #elems = ["Li", "C", "Na", "Mg", "Si", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Zn"]
     #outdir = "../data/LAMOST_Subaru"
     #pmulti_bound = 0.0
     #make_abundance_datatable_monoenriched(catalog, elems, outdir, pmulti_bound)

#    ratio="[Fe/Na]"
#    result = GetUncertainty.get_sigma_obs(ratio)
#    print(result)

   

    #feherr = 0.1
    #sigma_p, sigma_m = get_Uncertainty_Znum([24], feherr)
    #znum = [6, 8, 12]
    #logA = [ -3.4, -5.2, -7.5]
    #feh = -3.0
    #xfes = logA2XFe(znum, logA, feh)
    #print(xfes)




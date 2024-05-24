#
#     This file is part of FitYield
#     
#
#     This file contains programs to read the yield model 
#

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os,sys,re,glob

from multiprocessing import Pool


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fityield import abundance,yields


def get_mco(mass):

    m, mcos = np.loadtxt('../../models/interpolate/m_mco_Mcuts.dat', usecols=(0, 1), unpack = True)


    mco = np.nan
    for i, mm in enumerate(m):
        if mm==mass:
            mco = mcos[i] 

    return(mco)


def get_yield_in_mass_file(mass,en,mmix,lfej,yieldset):

    if yieldset==1:
        
        yield_path="/home/miho/HMP/yieldgrid/modeloutputs/"
        
    elif yieldset==2:
        
        yield_path="/home/miho/HMP/yieldgrid/modeloutputs_Mcuts/"

    if mass==11.:
        mcut=1.41
    elif mass==13.:
        mcut=1.47
    elif mass==15.:
        mcut=1.41
    elif mass==25.:
        mcut=1.69
    elif mass==40.:
        mcut=2.42
    elif mass==100.:
        mcut=3.63

    mco = get_mco(mass)
    yield_in_mass_file = yield_path + "M%03iE%04.1f_Mcut%4.2fMCO%4.2fMout%4.2flfej%4.1f_fort.10.dat"%(mass,en,mcut,mco,mmix,lfej)
        

    return(yield_in_mass_file)


def get_logA_from_yield_in_mass(znums, M_H, elemid, massnum, yield_in_solarmass):


    # Number of atoms for H (neglecting the fraction of duterium in M_H)
    N_H = yield_in_solarmass[0]/massnum[0] +\
        yield_in_solarmass[1]/massnum[1] + M_H/1.0


    N_X = np.zeros(np.size(znums), dtype = np.float64)


    for ii, znum in enumerate(znums):

        yield_in_Natoms = 0.0

        for jj, ei in enumerate(elemid):

            znum_ei = abundance.get_atomic_number([ei])
            
            if znum_ei != znum:
                continue

            yield_in_Natoms = yield_in_Natoms + yield_in_solarmass[jj]/massnum[jj]

        N_X[ii] = yield_in_Natoms

    yields_in_logA = np.log10(N_X/N_H)


    return(yields_in_logA)




def get_Natoms_from_yield_in_mass(znums, elemid, massnum, yield_in_solarmass):


    # Number of atoms for H (neglecting the fraction of duterium in M_H)
    #N_H = yield_in_solarmass[0]/massnum[0] +\
    #    yield_in_solarmass[1]/massnum[1] + M_H/1.0


    N_atoms = np.zeros(np.size(znums), dtype = float)

    for ii, znum in enumerate(znums):
        
 
        yield_in_Natoms = 0.0

        for jj, ei in enumerate(elemid):

            znum_ei = abundance.get_atomic_number([ei])
            
            if znum_ei != znum:
                continue

            yield_in_Natoms = yield_in_Natoms + yield_in_solarmass[jj]/massnum[jj]

        N_atoms[ii] = yield_in_Natoms
 
    return(N_atoms)






def get_logA_from_yield_in_mass_file(znums,M_H,yield_in_mass_file):


    elemid,massnum,yield_in_solarmass = read_yield_in_mass(yield_in_mass_file)

    # Number of atoms for H (neglecting the fraction of duterium in M_H)
    N_H=yield_in_solarmass[0]/massnum[0]+\
        yield_in_solarmass[1]/massnum[1]+M_H/1.0
    

    N_X=np.zeros(np.size(znums),dtype=np.float64)

    
    for ii,znum in enumerate(znums):

        yield_in_Natoms=0.0

        for jj,ei in enumerate(elemid):

            znum_ei=abundance.get_atomic_number([ei])

            if znum_ei!=znum:
                continue

            yield_in_Natoms=yield_in_Natoms+yield_in_solarmass[jj]/massnum[jj]

        N_X[ii]=yield_in_Natoms
        
    yields_in_logA=np.log10(N_X/N_H)    
            

    return(yields_in_logA)





def get_yield_in_mass_from_df(df0, mmix, lfej, mcut):

    filt = (df0['Mout'] == np.round(mmix, 1)) & (df0['lfej'] == np.round(lfej, 1)) & (df0['Mcut'] == np.round(mcut, 2))

    df = df0[filt]


    if len(list(df.index))==0:

        print("following param not found!", mmix, lfej)

        sys.exit()


    rindx = (list(df.index))[0]
    cindxs = list(df.columns)

    elemid=()
    massnum=()
    yield_in_solarmass=()

    for i in range(6, len(cindxs)):
        cindx = cindxs[i]
        elemid = np.hstack((elemid, re.sub('[0-9]', '', cindx)))
        massnum = np.hstack((massnum, re.sub('[a-z]', '', cindx)))
        yield_in_solarmass = np.hstack((yield_in_solarmass, df.at[rindx, cindx]))
 

    massnum[0] = 1   # Mass number for p
    massnum[1] = 2    # Mass number for d
    elemid[0] = "h"
    elemid[1] = "h"

    massnum = np.float64(massnum)


    return(elemid,massnum, yield_in_solarmass)




def get_yield_in_mass_from_dfrow(dfrow):

    # dfrow should be a series, extracted by iterrows() method
    # This is very slow!!     

    #rindx = (list(dfrow.index))[0]
    cindxs = list(dfrow.index)

    elemid=()
    massnum=()
    yield_in_solarmass=()

    for i in range(6, len(cindxs)):
        cindx = cindxs[i]
        elemid = np.hstack((elemid, re.sub('[0-9]', '', cindx)))
        massnum = np.hstack((massnum, re.sub('[a-z]', '', cindx)))

        yield_in_solarmass = np.hstack((yield_in_solarmass, dfrow[cindx]))
 

    massnum[0] = 1   # Mass number for p
    massnum[1] = 2    # Mass number for d
    elemid[0] = "h"
    elemid[1] = "h"


    massnum = np.float64(massnum)
    return(elemid,massnum, yield_in_solarmass)





def read_yield_in_mass(yield_in_mass_file):

    
    f=open(yield_in_mass_file)
    lines=f.readlines()
    f.close()

    elemid=()
    massnum=()
    yield_in_solarmass=()
    
    for i,line in enumerate(lines):

        data=(line.strip()).split()
    
        if i==16:
            ncols=4
        else:
            ncols=5
            
        for j in range(0,ncols):
            elemid=np.hstack((elemid,re.sub('[0-9]','',(data[j*2]))))
            massnum=np.hstack((massnum,re.sub('[a-z]','',(data[j*2]))))
            yield_in_solarmass=np.hstack((yield_in_solarmass,np.float64(data[j*2+1])))

        massnum[0]=1   # Mass number for p
        massnum[1]=2   # Mass number for d
        elemid[0]="h"
        elemid[1]="h"


    massnum_float=np.float64(massnum)

    yield_in_solarmass_float=np.float64(yield_in_solarmass)
    
    return(elemid,massnum_float,yield_in_solarmass_float)


def get_element_mass(Znum, elemid, massnum, yield_in_solarmass):

    yieldmass=0.0

    for i,em in enumerate(elemid):
        znum_em=abundance.get_atomic_number([em])
        if znum_em==Znum:
            yieldmass=yieldmass+yield_in_solarmass[i]
        else:
            continue

    return(yieldmass)


def get_element_mass_file(Znum,yield_in_mass_file):
    
    elemid,massnum,yield_in_solarmass=read_yield_in_mass(yield_in_mass_file)

    yieldmass=0.0
    
    for i,em in enumerate(elemid):
        znum_em=abundance.get_atomic_number([em])
        if znum_em==Znum:
            yieldmass=yieldmass+yield_in_solarmass[i]
        else:
            continue

    return(yieldmass)




def calc_yield(znums, mass, en, mmix0, lfej0, mcut0, feh, df_yield):
   

    #elemid, massnum, yield_in_solarmass = get_yield_in_mass_from_df(df_yield, mmix0, lfej0)



        
    elemid, massnum, yield_in_solarmass = get_yield_in_mass_from_df(df_yield, mmix0, lfej0, mcut0)

    # Get mass of Fe
    Z_Fe = 26
    M_Fe = get_element_mass(Z_Fe, elemid, massnum, yield_in_solarmass)
    M_H = M_Fe/10**(feh-4.5)/56
    
        
    yields_in_logA = \
                get_logA_from_yield_in_mass(znums, M_H, elemid, massnum, yield_in_solarmass)
        
    
        
    return(yields_in_logA, M_H)




def calc_yield_interp(znums,mass,en,mmix0,lfej0,feh,df_yield):
   

    #elemid, massnum, yield_in_solarmass = get_yield_in_mass_from_df(df_yield, mmix0, lfej0)



    # The steps of the mmix and lfej parameters. Must be 0.1 at the moment. 
    mmix_step=0.1
    lfej_step=0.1

    # Determine the grid points that bracket the specified values
    mmix1=mmix0-(mmix0%mmix_step)
    if mmix0 == 2.0:
        mmix2 = 2.0
    else:
        mmix2=mmix1+mmix_step
        
    lfej1=lfej0-(lfej0%lfej_step)
    if lfej0 == 0.0:
        lfej2 = 0.0
    else:
        lfej2=lfej1+lfej_step


    M_Fes = np.zeros(4)

    yields_in_logA = np.zeros((4,np.size(znums)))
    
    for i in range(0,4):

        if i==0:
            mmix=mmix1
            lfej=lfej1
        elif i==1:
            mmix=mmix1
            lfej=lfej2
        elif i==2:
            mmix=mmix2
            lfej=lfej1
        elif i==3:
            mmix=mmix2
            lfej=lfej2
            

        if lfej==0.0:
            lfej=-0.0
            
            
        #yield_in_mass_file=get_yield_in_mass_file(mass,en,mmix,lfej,yieldset)

        elemid, massnum, yield_in_solarmass = get_yield_in_mass_from_df(df_yield, mmix, lfej)

        # Get mass of Fe
        Z_Fe=26
        M_Fe=get_element_mass(Z_Fe, elemid, massnum, yield_in_solarmass)
        M_H=M_Fe/10**(feh-4.5)/56
    
        M_Fes[i]=M_Fe
        
        
        yields_in_logA[i,:]=get_logA_from_yield_in_mass(znums,M_H, elemid, massnum, yield_in_solarmass)


    yields_in_logA_mmix1 = interp_2yields(yields_in_logA[0,:],yields_in_logA[1,:],lfej1,lfej2,lfej0)
    yields_in_logA_mmix2 = interp_2yields(yields_in_logA[2,:],yields_in_logA[3,:],lfej1,lfej2,lfej0)

    yields_in_logA_intp = interp_2yields(yields_in_logA_mmix1,yields_in_logA_mmix2,mmix1,mmix2,mmix0)


    M_Fe_mmix1 = np.interp(lfej0,[lfej1,lfej2],[M_Fes[0],M_Fes[1]])
    M_Fe_mmix2 = np.interp(lfej0,[lfej1,lfej2],[M_Fes[2],M_Fes[3]])
    M_Fe_intp = np.interp(mmix0,[mmix1,mmix2],[M_Fe_mmix1,M_Fe_mmix2])

    M_H_intp =  M_Fe_intp/10**(feh-4.5)/56
    
    # Check if the interpolation works
    #plot_yields_intp(znums,yields_in_logA,yields_in_logA_intp)
        
        
    return(yields_in_logA_intp,M_H_intp)





def interp_2yields(y1,y2,x1,x2,x):

    y=np.zeros_like(y1)

    for i,yy in enumerate(y1):

        y[i]=np.interp(x,[x1,x2],[yy,y2[i]])

    return(y)


def plot_yields_intp(znum,y,yintp):

    fig,ax=plt.subplots(1,1)
    for i in range(0,4):
        
        ax.plot(znum,y[i,:],linestyle=":",color="blue",linewidth=1,alpha=0.5)
    ax.plot(znum,yintp,linestyle="-",color="red",linewidth=1,alpha=0.5)
        
    ax.set_xlim(5,32)
    ax.set_ylim(-9,-8.5)
    ax.set_xlabel("Z")
    ax.set_ylabel("logA")
    
    plt.show()

    return


def write_pickle(mass, en):

    path = "../../yieldgrid/modeloutputs_Mcuts"
    print(path + "/M%03.0fE%04.1f_*"%(mass,en))
    
    filelist = glob.glob(path + "/M%03.0fE%04.1f_*_fort.10.dat"%(mass,en))
    outpickle = "../../yieldgrid/pickle/M%03.0fE%04.1f_fort.10.pickle"%(mass, en)
    #filelist = [path + "/M040E30.0_Mcut2.42MCO13.89Mout0.50lfej-3.6_fort.10.dat", \
    #        #    path + "/M040E30.0_Mcut2.42MCO13.89Mout0.50lfej-5.0_fort.10.dat"]

    niso = 84
    nfiles = len(filelist)

    mass = ()
    en = ()
    Mcut = ()
    MCO = ()
    Mout = ()
    lfej = ()
    elems = ['' for i in range(niso)]
    yds = [[0.0 for i in range(niso)] for j in range(nfiles)]
  
  

    for j, file in enumerate(filelist):

        # Get information on the model parameters
        filename = (file.split("/"))[-1]
        mass = np.hstack((mass, float(filename[1:4])))
        en = np.hstack((en, float(filename[5:9])))
        Mcut = np.hstack((Mcut, float(((filename.split("Mcut"))[1])[:4])))
        MCO = np.hstack((MCO, float(((filename.split("MCO"))[1])[:4])))
        Mout = np.hstack((Mout, float(((filename.split("Mout"))[1])[:4])))
        lfej = np.hstack((lfej, float(((filename.split("lfej"))[1])[:4])))
        
        # Read the yield file
        elem, massnum, yd = read_yield_in_mass(file)
        if j == 0:
            elems=[elem[i] + str(int(massnum[i])) for i in range(niso)]
        yds[j] = [yd[i] for i in range(niso)]
    

    # Define a dictionary 

    data = {"mass":mass, "energy":en, "Mcut":Mcut, "MCO":MCO, "Mout":Mout, "lfej":lfej}
    
    for i in range(niso):
        data[elems[i]] = [yds[j][i] for j in range(nfiles)]

    df = pd.DataFrame(data)
    df.to_pickle(outpickle)

    return



def write_pickle_elem(mass, en):

    path = "../../yieldgrid/modeloutputs_Mcuts"
    print(path + "/M%03.0fE%04.1f_*"%(mass,en))
    
    filelist = glob.glob(path + "/M%03.0fE%04.1f_*_fort.10.dat"%(mass,en))
    outpickle = "../../yieldgrid/pickle/Natoms_M%03.0fE%04.1f_fort.10.pickle"%(mass, en)

    #filelist = [path + "/M013E01.0_Mcut1.47MCO2.39Mout1.00lfej-4.1_fort.10.dat", \
    #            path + "/M013E01.0_Mcut1.40MCO2.39Mout1.00lfej-4.1_fort.10.dat", \
    #            path + "/M013E01.0_Mcut1.55MCO2.39Mout1.00lfej-4.1_fort.10.dat", \
    #            path + "/M013E01.0_Mcut1.32MCO2.39Mout1.00lfej-4.1_fort.10.dat", \
    #            path + "/M013E01.0_Mcut1.62MCO2.39Mout1.00lfej-4.1_fort.10.dat"]
    #outpickle = "../../yieldgrid/pickle/test.pickle"

    #filelist = [path + "/M025E01.0_Mcut1.52MCO6.29Mout0.10lfej-0.6_fort.10.dat", \
    #            path + "/M025E01.0_Mcut1.60MCO6.29Mout0.10lfej-0.6_fort.10.dat", \
    #            path + "/M025E01.0_Mcut1.69MCO6.29Mout0.10lfej-0.6_fort.10.dat", \
    #            path + "/M025E01.0_Mcut1.77MCO6.29Mout0.10lfej-0.6_fort.10.dat", \
    #            path + "/M025E01.0_Mcut1.86MCO6.29Mout0.10lfej-0.6_fort.10.dat"]
    #outpickle = "../../yieldgrid/pickle/test2.pickle"


    #filelist = [path + "/M015E01.0_Mcut1.41MCO3.02Mout0.10lfej-0.7_fort.10.dat", \
    #            path + "/M015E01.0_Mcut1.34MCO3.02Mout0.10lfej-0.7_fort.10.dat", \
    #            path + "/M015E01.0_Mcut1.27MCO3.02Mout0.10lfej-0.7_fort.10.dat", \
    #            path + "/M015E01.0_Mcut1.48MCO3.02Mout0.10lfej-0.7_fort.10.dat", \
    #            path + "/M015E01.0_Mcut1.55MCO3.02Mout0.10lfej-0.7_fort.10.dat"]
    #outpickle = "../../yieldgrid/pickle/test3.pickle"


    #niso = 84

    nfiles = len(filelist)
    nelem = 30

    mass = ()
    en = ()
    Mcut = ()
    MCO = ()
    Mout = ()
    lfej = ()
    h1mass = ()
    h2mass = ()
    femass = ()

    elems = ['' for i in range(nelem)]
    yds = [[0.0 for i in range(nelem)] for j in range(nfiles)]

    #elems = ['' for i in range(niso)]
    #yds = [[0.0 for i in range(niso)] for j in range(nfiles)]
  
    znums = np.arange(1, nelem + 1, 1)
    elemnames = abundance.get_elem_name(znums)

    for j, file in enumerate(filelist):

        # Get information on the model parameters
        filename = (file.split("/"))[-1]
        mass = np.hstack((mass, float(filename[1:4])))
        en = np.hstack((en, float(filename[5:9])))
        Mcut = np.hstack((Mcut, float(((filename.split("Mcut"))[1])[:4])))
        MCO = np.hstack((MCO, float(((filename.split("MCO"))[1])[:4])))
        Mout = np.hstack((Mout, float(((filename.split("Mout"))[1])[:4])))
        lfej = np.hstack((lfej, float(((filename.split("lfej"))[1])[:4])))
        
        # Read the yield file
        elem, massnum, yd = read_yield_in_mass(file)
        h1mass = np.hstack((h1mass, yd[0]))
        h2mass = np.hstack((h2mass, yd[1]))
     
        # Get mass of Fe
        Z_Fe = 26
        femass = np.hstack((femass, get_element_mass(Z_Fe, elem, massnum, yd)))

        N_atoms = get_Natoms_from_yield_in_mass(znums, elem, massnum, yd)
        #if j == 0:
        #    elems=[elem[i] + str(int(massnum[ for i in range(nelem)] 
        #if j == 0:
        #    elems=[elem[i] + str(int(massnum[i])) for i in range(niso)]
        yds[j] = [N_atoms[i] for i in range(nelem)]
    

    # Define a dictionary 

    data = {"mass":mass, "energy":en, "Mcut":Mcut, "MCO":MCO, "Mout":Mout, "lfej":lfej, "h1mass":h1mass, "h2mass":h2mass, "femass":femass}
    
    for i in range(nelem):
        data[elemnames[i]] = [yds[j][i] for j in range(nfiles)]

    df = pd.DataFrame(data)
    df.to_pickle(outpickle)

    return


def write_pickle_all():

    path = "../../yieldgrid/modeloutputs_Mcuts"

    mass = [11., 13., 13., 15., 25., 25., 40., 40., 100., 100.]
    en = [0.5, 0.5, 1.0, 1.0, 1.0, 10.0, 1.0, 30.0, 1.0, 60.0]

    args = list(zip(mass, en))


    with Pool(10) as pool:
        pool.starmap(write_pickle_elem, args)   

    #    
    #        #print(path + "/M%03.0fE%04.1f_*"%(mass,en))
    #        filelist = glob.glob(path + "/M%03.0fE%04.1f_*_fort.10.dat"%(mass,en))
    #        #print(filelist)
    #        outpickle = "../../yieldgrid/pickle/M%03.0fE%04.1f_fort.10.pickle"%(mass, en)
    #        #filelist = [path + "/M040E30.0_Mcut2.42MCO13.89Mout0.50lfej-3.6_fort.10.dat", \
    #        #    path + "/M040E30.0_Mcut2.42MCO13.89Mout0.50lfej-5.0_fort.10.dat"]

    #        write_pickle(filelist, outpickle)
            
    return

if __name__ == "__main__":
    mass = 11.0
    en = 0.5
    #write_pickle_elem(mass, en)
    #write_pickle_all()

    write_pickle(mass, en)


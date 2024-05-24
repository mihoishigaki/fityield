import numpy as np
import random

#Chiaki Kobayashi, Feb 23 2021:
#For observations, for these elements, the main error source is the NLTE effect,
#and the errors are independent. I just updated the numbers for [X/Fe]
#and you can do err_[X/Y] = sqrt( err_[X/Fe]**2 + err_[Y/Fe]**2).
#For models, a few factors are taken into account;
#namely the elements that are synthesized in the same region,
#e.g., Cr&Mn and Co&Zn are not independent, so the error cannot be the sum of squares.

#dict for theoretical uncertainties
th_dict = {
'[C/O]':   [0.1,0.1],
'[C/Na]':  [0.1,0.5],
'[O/Na]':  [0.1,0.5],
'[C/Mg]':  [0.1,0.1],
'[O/Mg]':  [0.1,0.1],
'[Na/Mg]': [0.5,0.1],
'[C/Al]':  [0.1,0.5],
'[O/Al]':  [0.1,0.5],
'[Na/Al]': [0.2,0.2],
'[Mg/Al]': [0.1,0.5],
'[C/Si]':  [0.1,0.2],
'[O/Si]':  [0.1,0.2],
'[Na/Si]': [0.5,0.2],
'[Mg/Si]': [0.1,0.2],
'[Al/Si]': [0.5,0.2],
'[C/Ca]':  [0.1,0.2],
'[O/Ca]':  [0.1,0.2],
'[Na/Ca]': [0.5,0.2],
'[Mg/Ca]': [0.1,0.2],
'[Al/Ca]': [0.5,0.2],
'[Si/Ca]': [0.2,0.2],
'[C/Cr]': [0.1,0.15],
'[O/Cr]': [0.1,0.15],
'[Na/Cr]': [0.5,0.15],
'[Mg/Cr]': [0.1,0.15],
'[Al/Cr]': [0.5,0.15],
'[Si/Cr]': [0.2,0.15],
'[Ca/Cr]': [0.2,0.15],
'[C/Mn]': [0.1,0.2],
'[O/Mn]': [0.1,0.2],
'[Na/Mn]': [0.5,0.2],
'[Mg/Mn]': [0.1,0.2],
'[Al/Mn]': [0.5,0.2],
'[Si/Mn]': [0.2,0.2],
'[Ca/Mn]': [0.2,0.2],
'[Cr/Mn]': [0.1,0.1],
'[C/Fe]': [0.1,0.1],
'[O/Fe]': [0.1,0.1],
'[Na/Fe]': [0.5,0.1],
'[Mg/Fe]': [0.1,0.1],
'[Al/Fe]': [0.5,0.1],
'[Si/Fe]': [0.2,0.1],
'[Ca/Fe]': [0.2,0.1],
'[Cr/Fe]': [0.15,0.1],
'[Mn/Fe]': [0.2,0.1],
'[C/Co]': [0.1,0.3],
'[O/Co]': [0.1,0.3],
'[Na/Co]': [0.5,0.3],
'[Mg/Co]': [0.1,0.3],
'[Al/Co]': [0.5,0.3],
'[Si/Co]': [0.2,0.3],
'[Ca/Co]': [0.1,0.3],
'[Cr/Co]': [0.15,0.3],
'[Mn/Co]': [0.2,0.3],
'[Fe/Co]': [0.05,0.25],
'[C/Ni]': [0.1,0.15],
'[O/Ni]': [0.1,0.15],
'[Na/Ni]': [0.5,0.15],
'[Mg/Ni]': [0.1,0.15],
'[Al/Ni]': [0.5,0.15],
'[Si/Ni]': [0.2,0.15],
'[Ca/Ni]': [0.2,0.15],
'[Cr/Ni]': [0.15,0.15],
'[Mn/Ni]': [0.2,0.15],
'[Fe/Ni]': [0.05,0.1],
'[Co/Ni]': [0.25,0.1],
'[C/Zn]': [0.1,0.3],
'[O/Zn]': [0.1,0.3],
'[Na/Zn]': [0.5,0.3],
'[Mg/Zn]': [0.1,0.3],
'[Al/Zn]': [0.5,0.3],
'[Si/Zn]': [0.1,0.3],
'[Ca/Zn]': [0.2,0.3],
'[Cr/Zn]': [0.15,0.3],
'[Mn/Zn]': [0.2,0.3],
'[Fe/Zn]': [0.05,0.25],
'[Co/Zn]': [0.2,0.2],
'[Ni/Zn]': [0.1,0.25]
}

obs_dict = {
'[C/O]': [0.28,0.28],
'[C/Na]': [0.28,0.56],
'[O/Na]':  [0.28,0.54],
'[C/Mg]':  [0.36,0.28],
'[O/Mg]':  [0.36,0.28],
'[Na/Mg]': [0.58,0.28],
'[C/Al]':  [0.54,0.28],
'[O/Al]':  [0.54,0.28],
'[Na/Al]': [0.71,0.28],
'[Mg/Al]': [0.54,0.36],
'[C/Si]':  [0.28,0.28],
'[O/Si]':  [0.28,0.28],
'[Na/Si]': [0.54,0.28],
'[Mg/Si]': [0.28,0.36],
'[Al/Si]': [0.28,0.54],
'[C/Ca]':  [0.28,0.28],
'[O/Ca]':  [0.28,0.28],
'[Na/Ca]': [0.54,0.28],
'[Mg/Ca]': [0.28,0.36],
'[Al/Ca]': [0.28,0.54],
'[Si/Ca]': [0.28,0.28],
'[C/Cr]': [0.45,0.28],
'[O/Cr]': [0.45,0.28],
'[Na/Cr]': [0.64,0.28],
'[Mg/Cr]': [0.45,0.36],
'[Al/Cr]': [0.45,0.54],
'[Si/Cr]': [0.45,0.28],
'[Ca/Cr]': [0.45,0.28],
'[C/Mn]': [0.36,0.28],
'[O/Mn]': [0.36,0.28],
'[Na/Mn]': [0.58,0.28],
'[Mg/Mn]': [0.36,0.36],
'[Al/Mn]': [0.36,0.54],
'[Si/Mn]': [0.36,0.28],
'[Ca/Mn]': [0.36,0.28],
'[Cr/Mn]': [0.36,0.45],
'[C/Fe]': [0.2,0.2],
'[O/Fe]': [0.2,0.2],
'[Na/Fe]': [0.5,0.2],
'[Mg/Fe]': [0.2,0.3],
'[Al/Fe]': [0.2,0.5],
'[Si/Fe]': [0.2,0.2],
'[Ca/Fe]': [0.2,0.2],
'[Cr/Fe]': [0.2,0.4],
'[Mn/Fe]': [0.2,0.3],
'[C/Co]': [0.28,0.28],
'[O/Co]': [0.28,0.28],
'[Na/Co]': [0.54,0.28],
'[Mg/Co]': [0.28,0.36],
'[Al/Co]': [0.28,0.54],
'[Si/Co]': [0.28,0.28],
'[Ca/Co]': [0.28,0.28],
'[Cr/Co]': [0.28,0.45],
'[Mn/Co]': [0.28,0.36],
'[Fe/Co]': [0.2,0.2],
'[C/Ni]': [0.28,0.28],
'[O/Ni]': [0.28,0.28],
'[Na/Ni]': [0.54,0.28],
'[Mg/Ni]': [0.28,0.36],
'[Al/Ni]': [0.28,0.54],
'[Si/Ni]': [0.28,0.28],
'[Ca/Ni]': [0.28,0.28],
'[Cr/Ni]': [0.28,0.45],
'[Mn/Ni]': [0.28,0.36],
'[Fe/Ni]': [0.2,0.2],
'[Co/Ni]': [0.28,0.28],
'[C/Zn]': [0.28,0.28],
'[O/Zn]': [0.28,0.28],
'[Na/Zn]': [0.54,0.28],
'[Mg/Zn]': [0.28,0.36],
'[Al/Zn]': [0.28,0.54],
'[Si/Zn]': [0.28,0.28],
'[Ca/Zn]': [0.28,0.28],
'[Cr/Zn]': [0.28,0.45],
'[Mn/Zn]': [0.28,0.45],
'[Fe/Zn]': [0.2,0.2],
'[Co/Zn]': [0.28,0.28],
'[Ni/Zn]': [0.28,0.28]
}

#Get element-specific uncertainty for theoretical yields
def get_sigma_th(ratio):

        elems = [(elemnames.strip("[")).strip("]")  for elemnames in ratio.split("/")]

        default_value = [-9.99,-9.99]#if abundance is not in dict
        
        vals1 = th_dict.get("["+elems[0] + "/" + elems[1] + "]", default_value)
        vals2 = th_dict.get("["+elems[1] + "/" + elems[0] + "]", default_value)
 
        if vals1[0]!=-9.99:
            return(vals1)
        elif vals2[0]!=-9.99:
            return([vals2[1],vals2[0]])

#Get element-specific uncertainty for observations
def get_sigma_obs(ratio):
        
        elems = [(elemnames.strip("[")).strip("]")  for elemnames in ratio.split("/")]
        default_value = [-9.99,-9.99]#if abundance is not in dict

        vals1 = obs_dict.get("["+elems[0] + "/" + elems[1] + "]", default_value)
        vals2 = obs_dict.get("["+elems[1] + "/" + elems[0] + "]", default_value)

        #print(vals1)
        #print(vals2)

        if vals1[0]!=-9.99:
            return(vals1)
        elif vals2[0]!=-9.99:
            return([vals2[1],vals2[0]])
        else:
            return([0.2, 0.2])   # ! To be updated!


#sample element-specific uncertainty for theoretical yields
def get_scatter_th(ratio):
    sigma = get_sigma_th(ratio)
    i = random.choice([0,1])
    sigma = sigma[i]#positive or negative?

    if(i==0):#positive
        return np.abs(np.random.normal(loc=0.0, scale=sigma, size=None))
    else:
        return -1.0*np.abs(np.random.normal(loc=0.0, scale=sigma, size=None))


#sample element-specific uncertainty for observations
def get_scatter_obs(ratio):
    sigma = get_sigma_obs(ratio)
    i = random.choice([0,1])
    sigma = sigma[i]#positive or negative?

    if(i==0):#positive
        return np.abs(np.random.normal(loc=0.0, scale=sigma, size=None))
    else:
        return -1.0*np.abs(np.random.normal(loc=0.0, scale=sigma, size=None))

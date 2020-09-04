#%%
"""
Test Eddington ratio distribution shapes
"""
import numpy as np
import matplotlib.pyplot as plt
#%%
# Edd ratio values range
edd_bin = np.arange(-4, 0, 0.001)
edd=10 ** edd_bin

def schechter(edd,arg1,arg2):
   prob = ((edd / (10. ** arg1)) ** arg2)
   return prob* np.exp(-(edd / (10. ** arg1)))

def gaussian(edd,arg1,arg2):
   return np.exp(-(edd - arg2) ** 2. / (2.*arg1 ** 2.))
#%%
# Shankar + Schechter z=1
arg1=-0.8
arg2= 1.6

plt.plot(edd,schechter(edd,arg1,arg2));
#%%
# Schechter test
arg1=2.21748
arg2= 0.969

schechter_max=arg2*10**arg1
print(f'Position of the maximum: {schechter_max}')

x=np.arange(2,2.5,0.001)
x=10**x


plt.plot(x,schechter(x,arg1,arg2))
plt.axvline(schechter_max);
#plt.yscale('log');
#%%
# shankar + gaussian
arg1= 0.05
arg2= 0.25

plt.plot(edd,gaussian(edd,arg1,arg2));
#%%
# shankar + gaussian
arg1= 0.2
arg2= 0.6

plt.plot(edd,gaussian(edd,arg1,arg2));

# %%
# plots k corrs

# lusso modif 
def lusso(x, a1, a2, a3, b):
   return a1*x + a2*x**2 + a3*x**3 + b

incr=0.01
#Fits from table2, type2, Spectro+photo,488
#range from fig 9 Lbol=[9.8-12.2]
pars_t2=[0.23, 0.05, 0.001, 1.256]
Lbol2=np.arange(start=9.8,stop=12.2, step=incr)
bol_corr2=lusso(Lbol2-12,*pars_t2)
Lbol2_erg = Lbol2 + 33.585 #from Lsun to erg/s

#Fits from table2, type1, Spectro+photo,373
# range from fig 9 Lbol=[10.8-13.2]
pars_t1 = [0.288, 0.111, -0.007, 1.308]
L_max = 13.55 # where bol_corr1=100
Lbol1 = np.arange(start=10.8,stop=L_max, step=incr)
bol_corr1 = lusso(Lbol1-12,*pars_t1)
Lbol1_erg = Lbol1 + 33.585 #from Lsun to erg/s

#Combine equations, eq 1 from Lusso 2012
start = np.min(Lbol1)
ending = np.max(Lbol2)
LX = np.arange(start=start,stop=ending, step=incr)
bol_corr12 = lusso(LX-12,*pars_t1)
bol_corr22 = lusso(LX-12,*pars_t2)
bol_tot = bol_corr12*(LX-start)/(ending-start)+bol_corr22*(ending-LX)/(ending-start)
Lbol_tot = LX + 33.585 #from Lsun to erg/s

# Low luminosity (She et al. 2017)
lx_final = np.arange(start=20.,stop=np.min(Lbol2_erg), step=incr)
low_lum = np.log10(16)*np.ones(len(lx_final))

# Compose to make one array:
# low lum + AGN2
a = tuple([Lbol2_erg < np.min(Lbol1_erg)])
lx_final = np.concatenate((lx_final,Lbol2_erg[a]))
corr_final = np.concatenate((low_lum,bol_corr2[a]))
# + intermidiate area
lx_final = np.concatenate((lx_final,Lbol_tot))
corr_final = np.concatenate((corr_final,bol_tot))
# + AGN1 up to 100
a = tuple([Lbol1_erg > np.max(Lbol2_erg)])
lx_final = np.concatenate((lx_final,Lbol1_erg[a]))
corr_final = np.concatenate((corr_final,bol_corr1[a]))
# + flat area
lx_high = np.arange(start=max(Lbol1_erg),stop=48.6,step=incr)
corr_high = 2*np.ones(len(lx_high))
lbol_final = np.concatenate((lx_final,lx_high))
corr_final = np.concatenate((corr_final,corr_high))

# duras
# table1, (Klbol),general
pars = [10.96, 11.93, 17.79]
k_corr = pars[0]*(1+((lbol_final - 33.485)/pars[1])**pars[2])

# %%

plt.plot(Lbol2_erg,10**bol_corr2) #type2
plt.plot(Lbol1_erg,10**bol_corr1) #type1
plt.plot(Lbol_tot,10**bol_tot) #combination of previous
plt.plot(lbol_final,10**corr_final) #combination + low and high luminosity
plt.yscale('log')
plt.xlim(34,48.6)

plt.plot(lbol_final,k_corr);

# %%
l_bol=np.load('l_bol.npy')
print(l_bol[:10])
# %%
# duras
k_corr = pars[0]*(1+((l_bol - 33.485)/pars[1])**pars[2])
plt.scatter(l_bol,k_corr)
plt.yscale('log');

lum=l_bol-k_corr
lum[:10]
# %%
print(l_bol[:10])
print(k_corr[:10])
np.nanmax(lum)
# %%

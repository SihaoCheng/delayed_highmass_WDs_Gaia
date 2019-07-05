import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, votable
from astropy.table import Table, vstack, hstack
import os, sys
from scipy.interpolate import interp1d
import importlib
import emcee
from multiprocessing import Pool
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic # Low-level frames
import astropy.units as u
import WD_HR # a function package 
import WD_models # a module w


test_number = '0'
t_gap_eff = 0.743
Q_IS_MERGER= True # Does the extra-delayed population also have the WD-WD merger delay?
DELAY_INDEX = 0.7 # the absolute value of the power index of the delay time distribution for mergers. 
DELAY_CUT = 0.5 # 

# Set parameters
##------------------------------------------------------------------------------------------------------------------------
Nv = 13
NQ = 7
delay_lim = 15

# Define star formation functions
##------------------------------------------------------------------------------------------------------------------------
# t0: age from born, t1: merger from born, tc: cooling age
# t0 = t1 + tc
end_of_SF = 11
age_T = 7
stromberg_k = 80
n = 400; n_tc = 4000
t0 = np.linspace(0,end_of_SF,n)

##--------------------------------------------------------------------------------------------------
def select_WD(WD_table, bprp_excess=1.3, bprp_error=0.05, ev=2, G=19.5, d_ed=10, distance=500,astrometric_excess=1.5):
    selected = (WD_table['phot_bp_rp_excess_factor']<bprp_excess)*\
        (WD_table['parallax']/WD_table['parallax_error']>d_ed)*\
        ((np.sqrt((WD_table['phot_bp_mean_flux_error']/WD_table['phot_bp_mean_flux'])**2+\
        (WD_table['phot_rp_mean_flux_error']/WD_table['phot_rp_mean_flux'])**2)\
                  <bprp_error ))*\
           (WD_table['phot_g_mean_mag']<G) * (WD_table['ev']<ev) * (1/WD_table['parallax']*1000<distance)\
            *(WD_table['astrometric_excess_noise']<astrometric_excess)
    selected_table = WD_table[selected].copy()
    for spec_type, model in [['DA_thick','CO'],['DA_thick','ONe'],['DB','CO'],['DB','ONe'],['DA_thin','CO'],['DA_thin','ONe']]:
        age_planet, mass_planet = para_of_WD(selected_table, spec_type, model) 
        selected_table['age_'+spec_type+'_'+model] = age_planet
        selected_table['mass_'+spec_type+'_'+model] = mass_planet
    print(selected.sum())
    return selected, selected_table

def para_of_WD(WD_table, spec_type, model):
    WD_model = eval('WD_models.'+spec_type+'_'+model)
    age_planet = 10**WD_model['grid_logage_func'](WD_table['bp_rp'],WD_table['G'])/10**9
    mass_planet = WD_model['grid_mass_func'](WD_table['bp_rp'],WD_table['G'])
    return age_planet, mass_planet


##------------------------------------------------------------------------------------------------------------------------
def SFR_merger(t0, para):
        return (t0>=0.0)*(t0<end_of_SF)*1
    
SFR = SFR_merger

def p_merger_delay(t1,dt,d_index):   
    if d_index<0 and d_index>-10:
        return (t1>=dt)*(np.abs(t1)+0.01)**(d_index)
        #return (t1>=1.5)*(np.abs(t1)+0.01)**(d_index) + (t1>=dt)*(t1<1.5)*(np.abs(1.5)+0.01)**(d_index)
    if d_index>0 and d_index<10:
        return (t1>=dt)*(np.abs(t1)+0.01)**(-d_index)
        #return (t1>=1)*(np.abs(t1)+0.1)**(-d_index) + (t1>=dt)*(t1<1)*(np.abs(1)+0.1)**(-d_index)
    if d_index>10 or d_index<-10:
        new_time = np.array([ 0.27,   0.81,   1.35,   1.89,   2.43,   2.97,   3.51,   4.05,   4.59,   5.13,
                   5.67,   6.21,   6.75,   7.29,   7.83,   8.37,   8.91,   9.45,   9.99,  10.53,
                  11.07,  11.61,  12.15,  12.69,  13.23,])
        new_rate = np.array([  0,   1.01587302e-13,   7.76334776e-14,   6.08946609e-14,
                       6.92640693e-14,   3.11688312e-14,   2.22222222e-14,   1.24098124e-14,
                       9.81240981e-15,   4.61760462e-15,   8.65800866e-16,   1.73160173e-15,
                       8.65800866e-16,   8.65800866e-16,   1.15440115e-15,   1.44300144e-15,
                       1.73160173e-15,   1.44300144e-15,   2.30880231e-15,   2.02020202e-15,
                       8.65800866e-16,   1.73160173e-15,   1.15440115e-15,   2.02020202e-15,
                       2.88600289e-15,])*1e13 #8.36940837e-15
        return interp1d(new_time, new_rate, kind='nearest', bounds_error=False, fill_value=0)(t1)

def pdf_tct0_array(tc, t0, para, SFR, delay_cut, delay_index):
    array = SFR(t0, para) * p_merger_delay(t0 - tc, delay_cut, delay_index)
    return array, array.mean()

def pdf_tct0(tc, t0, para, normalization_factor_Q, SFR, delay_cut, delay_index):
    return SFR(t0, para) * p_merger_delay(t0 - tc, delay_cut, delay_index) / normalization_factor_Q

def prep_get_v(table):
    c = SkyCoord(ra=table['ra']*u.deg, dec=table['dec']*u.deg, \
         pm_ra_cosdec=table['pmra']*u.mas/u.yr, pm_dec=table['pmdec']*u.mas/u.yr)
    pml = c.transform_to(Galactic).pm_l_cosb.value
    pmb = c.transform_to(Galactic).pm_b.value
    factor = 1/1000 * (1/np.array(table['parallax'])*1000)*1.5*10**8/3600/24/365.24
    return pml, pmb, factor

def get_v_delayed_3D(age,l,b,pml,pmb,factor,v_drift,U,V,W):
    A = 1*15.3
    B = 1*-11.9
    C = 1*-3.2
    K = 1*-3.3
    d = factor / (1.5*10**8/3600/24/365.24)
    vL = pml*factor - ( A*np.cos(2*l/180*np.pi)+B-C*np.sin(2*l/180*np.pi) )*np.cos(b/180*np.pi)*d + \
            (V+v_drift)*np.cos(l/180*np.pi)-U*np.sin(l/180*np.pi)
    vB = pmb*factor + ( (A*np.sin(2*l/180*np.pi)+C*np.cos(2*l/180*np.pi)+K)*np.sin(2*b/180*np.pi)/2 )*d +\
            W*np.cos(b/180*np.pi) - ((V+v_drift)*np.sin(l/180*np.pi)+U*np.cos(l/180*np.pi))*np.sin(b/180*np.pi)
    return vL, vB


# Define MCMC functions
##------------------------------------------------------------------------------------------------------------------------
def m3(a,b):
    result = np.array([[(a[0,:]*b[:,0]).sum(),(a[0,:]*b[:,1]).sum(),(a[0,:]*b[:,2]).sum()],\
                     [(a[1,:]*b[:,0]).sum(),(a[1,:]*b[:,1]).sum(),(a[1,:]*b[:,2]).sum()],\
                     [(a[2,:]*b[:,0]).sum(),(a[2,:]*b[:,1]).sum(),(a[2,:]*b[:,2]).sum()],\
                        [0,0,0]])
    return result

def LBR_XYZ(l,b,L,B,sx,sy,sz):
    Sigma = np.array([[sx**2, 0, 0],[0, sy**2, 0],[0, 0, sz**2]])
    A = np.array([[-np.sin(l/180*np.pi), -np.sin(b/180*np.pi)*np.cos(l/180*np.pi), \
                   np.cos(b/180*np.pi)*np.cos(l/180*np.pi)],\
                  [np.cos(l/180*np.pi), -np.sin(b/180*np.pi)*np.sin(l/180*np.pi), \
                   np.cos(b/180*np.pi)*np.sin(l/180*np.pi)],\
                  [0, np.cos(b/180*np.pi), np.sin(b/180*np.pi)]])

    #print(A.shape, Sigma.shape, sx.shape)
    Sigma_lbr = m3(m3(A.T, Sigma),A)
    Sigma_lb = Sigma_lbr[:2,:2]
    S00 = Sigma_lb[0,0]
    S11 = Sigma_lb[1,1]
    S01 = Sigma_lb[0,1]
    S10 = Sigma_lb[1,0]
    detS = S00*S11 - S01*S10
    return 1/(detS)**0.5/(2*np.pi) * np.exp(-1/2/detS*
                                  np.dot(
                                      np.dot(
                                          np.array([L,B,0]), np.array([[S11, -S01, 0],[-S10, S00, 0], [0,0,0]])),
                                      np.array([L,B,0]).T
                                        )
                                 ), \
            1/Sigma_lb[0,0]**0.5/(2*np.pi)**0.5 * np.exp(-L**2/2/Sigma_lb[0,0]), \
            1/Sigma_lb[1,1]**0.5/(2*np.pi)**0.5 * np.exp(-B**2/2/Sigma_lb[1,1])

def velocity_scatter_3D(age, index, v4, vT, v0):
    age_trans = 10
    return (age<age_T)*((age / 4)**index*v4+v0) + \
            (age>=age_T)*(age<age_trans)*(vT - (age-age_trans)/(age_T-age_trans)*(vT-((age_T / 4)**index*v4+v0))) +\
            (age>=age_trans)*vT

def velocity_density_3D(l, b, vL, vB, age, para_v, is_vL=True):
    #index_UV_t = para_v[0], V_10 = para_v[1], v_T = para_v[2]
    #index_W = para_v[3], W_10 = para_v[4]
    #sy_to_sx = para_v[5]
    #v0 = para_v[6], v0_z = para_v[7], v_T_z = para_v[8]
    #sy_to_sx_T = para_v[9]
    #U = para_v[10]
    #V = para_v[11]
    #W = para_v[12]
    sx = velocity_scatter_3D(age, para_v[0], para_v[1], para_v[2], para_v[6])
    sy = velocity_scatter_3D(age, para_v[0], para_v[1]*para_v[5], para_v[2]*para_v[9],\
                             para_v[6]*para_v[5])
    sz = velocity_scatter_3D(age, para_v[3], para_v[4], para_v[8], para_v[7])
    p_vLvB, p_vL, p_vB = LBR_XYZ(l,b,vL,vB,sx,sy,sz)
    return p_vLvB

def velocity_density_3D_check(l, b, vL, vB, age, para_v, is_vL):
    sx = velocity_scatter_3D(age, para_v[0], para_v[1], para_v[2], para_v[6])
    sy = velocity_scatter_3D(age, para_v[0], para_v[1]*para_v[5], para_v[2]*para_v[9],\
                             para_v[6]*para_v[5])
    sz = velocity_scatter_3D(age, para_v[3], para_v[4], para_v[8], para_v[7])
    p_vLvB, p_vL, p_vB = LBR_XYZ(l,b,vL,vB,sx,sy,sz)
    if is_vL == True:
        return p_vL
    else:
        return p_vB 

def ln_likelihood_pheno(para,
                        mass, age, pml, pmb, factor, l, b,
                        mass_Q, age_Q, pml_Q, pmb_Q, factor_Q, l_Q, b_Q,
                       is_check, velocity_density_3D_check_func=None, is_vL=True,
                       not_fit_UVW=False,not_fit_index=False,fixv=False):
    para_v = para[0:Nv].copy()
    para_Q = para[Nv:Nv+NQ].copy()
    delay_index = DELAY_INDEX#para_Q[3]#
    delay_cut = DELAY_CUT
    #stromberg_k = para_Q[4]
    if not_fit_UVW==True:
        U = 11#para_v[10]
        V = 7.5#para_v[11]
        W = 7#para_v[12]
        U_Q = U
        V_Q = V
        W_Q = W
    else:
        U = para_v[10]
        V = para_v[11]
        W = para_v[12]
        U_Q = 11#para_v[10]#para[Nv+3]
        V_Q = 7.5#para_v[11]#para_Q[4]
        W_Q = 7#para_v[12]#para_Q[6]
    if not_fit_index==True:
        para_v[0] = 0.38
        para_v[3] = 0.5
    if fixv==True:
        para_v[1] = 35
        para_v[2] = 70
        para_v[4] = 17
        para_v[5] = 0.60
        para_v[6] = 0
        para_v[7] = 0
        para_v[8] = 40
        para_v[9] = 0.60
    if test_number == 'noQ':
        para_Q[0] = 0
    if test_number == 'nomerger':
        para_Q[5] = 0
    if test_number == 'noQmerger':
        para_Q[0] = 0
        para_Q[5] = 0
    #if METHOD == 'Run_MCMC':
    para_Q[3] = -end_of_SF
    
    early = np.array(1.22 - (age - 0.6) * 0.2> mass) #for ONe: 1.22; CO: 1.25
    late = np.array(1.22 - (age - 0.6) * 0.2 < mass)    
    
    ## tct0_array is the pdf of tc, t0, range: tc: 0~age_lim+delay, t0: 0~end_of_SFR
    ## normalization_factor_Q is the mean of tct0_array
    tct0_array, normalization_factor_Q = pdf_tct0_array(np.linspace(0.1,end_of_SF,n_tc).reshape(-1,1),
                                               np.linspace(0,end_of_SF,n).reshape(1,-1),
                                               para_Q[3:],
                                               SFR, delay_cut, delay_index)
    ## tc_weight is the norm factor of pdf of t0 given a tc. tct0_array / tc_weight
    tc_weight = tct0_array.mean(1)/normalization_factor_Q
    tc_weight = interp1d(np.linspace(0.1,end_of_SF,n_tc),
                         tc_weight,
                         fill_value=0.0,
                         bounds_error=False)
    
    ## For the ordinary merger population:
    tct0_array_merger = tct0_array
    normalization_factor_Q_merger = normalization_factor_Q
    tc_weight_merger = tc_weight

    
    ## v_delay for different possible t0
    v_drift_delayed = velocity_scatter_3D(t0, para_v[0], para_v[1], para_v[2], para_v[6])**2/stromberg_k

    vL_delayed, vB_delayed = get_v_delayed_3D(t0.reshape(1,-1), 
                              l.reshape(-1,1),
                              b.reshape(-1,1),
                              pml.reshape(-1,1),
                              pmb.reshape(-1,1), 
                              factor.reshape(-1,1),
                              v_drift_delayed.reshape(1,-1),
                             U,V,W)    
    vL_Q_delayed, vB_Q_delayed = get_v_delayed_3D(t0.reshape(1,-1), 
                                l_Q.reshape(-1,1), 
                                b_Q.reshape(-1,1),
                                pml_Q.reshape(-1,1), 
                                pmb_Q.reshape(-1,1), 
                                factor_Q.reshape(-1,1),
                                v_drift_delayed.reshape(1,-1),
                               U,V,W) 
    vL_delayed_Q, vB_delayed_Q = get_v_delayed_3D(t0.reshape(1,-1), 
                              l.reshape(-1,1),
                              b.reshape(-1,1),
                              pml.reshape(-1,1),
                              pmb.reshape(-1,1), 
                              factor.reshape(-1,1),
                              v_drift_delayed.reshape(1,-1),
                             U_Q,V_Q,W_Q)    
    vL_Q_delayed_Q, vB_Q_delayed_Q = get_v_delayed_3D(t0.reshape(1,-1), 
                                l_Q.reshape(-1,1), 
                                b_Q.reshape(-1,1),
                                pml_Q.reshape(-1,1), 
                                pmb_Q.reshape(-1,1), 
                                factor_Q.reshape(-1,1),
                                v_drift_delayed.reshape(1,-1),
                               U_Q,V_Q,W_Q) 
    v_drift = velocity_scatter_3D(age, para_v[0], para_v[1], para_v[2], para_v[6])**2/stromberg_k
    v_drift_Q = velocity_scatter_3D(age_Q, para_v[0], para_v[1], para_v[2], para_v[6])**2/stromberg_k
    vL, vB = np.array(get_v_delayed_3D(age, l, b, pml, pmb, factor, v_drift, U,V,W))
    vL_Q, vB_Q = np.array(get_v_delayed_3D(age_Q, l_Q, b_Q, pml_Q, pmb_Q, factor_Q, v_drift_Q, U,V,W))
    if is_check == True:
        velocity_density_3D_func = velocity_density_3D_check_func
        background_pdf_factor = 1    
    else:
        velocity_density_3D_func = velocity_density_3D
        background_pdf_factor = 400
    ## pdf of v,tc =
        ## [pdf(v) for ordinary * uniform(tc)] * ordinary fraction +
        ## [pdf(v) for merger * merger fraction(t0)
    ## pdf_tct0 is (age_lim+delay) * end_of_SF * pdf(tc,t0)
    
    ## early
    if Q_IS_MERGER == True:
        n_Q_delay = 20
        density_array_early = np.log(
            velocity_density_3D_func(l[early],b[early],
                                vL[early], vB[early], age[early], para_v, is_vL,
                                    )\
               * (1-para_Q[0]-para_Q[2]-para_Q[5]) +\
            ( velocity_density_3D_func(l[early].reshape(-1,1),b[early].reshape(-1,1),
                                 vL_delayed[early], vB_delayed[early], t0.reshape(1,-1), para_v, is_vL,
                                      )\
               * pdf_tct0(age[early].reshape(-1,1),
                          t0.reshape(1,-1),
                          para_Q[3:],
                          normalization_factor_Q,
                          SFR, delay_cut, delay_index
                          ) * para_Q[0]\
             
             +velocity_density_3D_func(l[early].reshape(-1,1),b[early].reshape(-1,1),
                                 vL_delayed[early], vB_delayed[early], t0.reshape(1,-1), para_v, is_vL,
                                     )\
               * pdf_tct0(age[early].reshape(-1,1),
                          t0.reshape(1,-1),
                          para_Q[3:],
                          normalization_factor_Q_merger,
                          SFR_merger, delay_cut, delay_index
                          ) * para_Q[5]\
            ).mean(1) +\
            1/400/background_pdf_factor*1\
                * para_Q[2]
                                    ) - np.log( (1-para_Q[0]-para_Q[5]) +\
                                               para_Q[0]*tc_weight(age[early]) +\
                                               para_Q[5]*tc_weight_merger(age[early])
                                              )
        
        ## late
        density_array_late = np.log(
            velocity_density_3D_func(l[late],b[late],
                                vL[late], vB[late], age[late], para_v, is_vL,
                                    ) \
               * (1-para_Q[0]-para_Q[2]-para_Q[5]) +\
            (velocity_density_3D_func(l[late].reshape(-1,1),b[late].reshape(-1,1),
                                 vL_delayed[late], vB_delayed[late], t0.reshape(1,-1), para_v, is_vL,
                                     )\
               * pdf_tct0(age[late].reshape(-1,1) + para_Q[1],
                          t0.reshape(1,-1),
                          para_Q[3:],
                          normalization_factor_Q,
                          SFR, delay_cut, delay_index
                         ) * para_Q[0] +\
             velocity_density_3D_func(l[late].reshape(-1,1),b[late].reshape(-1,1),
                                 vL_delayed[late], vB_delayed[late], t0.reshape(1,-1), para_v, is_vL,
                                     )\
               * pdf_tct0(age[late].reshape(-1,1),
                          t0.reshape(1,-1),
                          para_Q[3:],
                          normalization_factor_Q_merger,
                          SFR_merger, delay_cut, delay_index
                         ) * para_Q[5]\
            ).mean(1) +\
            1/400/background_pdf_factor\
                * para_Q[2]
                                   ) - np.log( (1-para_Q[0]-para_Q[5]) +\
                                              para_Q[0]*tc_weight(age[late]+para_Q[1]) +\
                                              para_Q[5]*tc_weight_merger(age[late])
                                             )
        # 0: index of WD (tc) ,  1: t0,  2: delay on branch 
        ## Q
        a1, a2 = vL_Q_delayed.shape
        density_array_Q = np.log(
            velocity_density_3D_func(l_Q, b_Q,
                                vL_Q, vB_Q, age_Q, para_v, is_vL)\
                * (1-para_Q[0]-para_Q[2]-para_Q[5]) +\
            (velocity_density_3D_func(l_Q.reshape(-1,1), b_Q.reshape(-1,1),
                                 vL_Q_delayed, vB_Q_delayed, t0.reshape(1,-1), para_v, is_vL).reshape(a1,a2,1)\
               * pdf_tct0(age_Q.reshape(-1,1,1) + np.linspace(0,1,n_Q_delay).reshape(1,1,-1)*para_Q[1],
                          t0.reshape(1,-1,1),
                          para_Q[3:],
                          normalization_factor_Q,
                          SFR, delay_cut, delay_index
                         )\
            ).mean((1,2)) * (para_Q[1]+t_gap_eff) / t_gap_eff * para_Q[0] +\
            (velocity_density_3D_func(l_Q.reshape(-1,1),b_Q.reshape(-1,1),
                                 vL_Q_delayed, vB_Q_delayed, t0.reshape(1,-1), para_v, is_vL,
                                     )\
               * pdf_tct0(age_Q.reshape(-1,1),
                          t0.reshape(1,-1),
                          para_Q[3:],
                          normalization_factor_Q_merger,
                          SFR_merger, delay_cut, delay_index
                         )\
            ).mean(1) * para_Q[5] +\
            1/400/background_pdf_factor*para_Q[2]  
                                ) - np.log( (1-para_Q[0]-para_Q[5])+\
                                           para_Q[0] * tc_weight(age_Q.reshape(-1,1) +\
                                                                np.linspace(0,1,n_Q_delay).reshape(1,-1)*para_Q[1]
                                                                ).mean(1) * (para_Q[1] + t_gap_eff) / t_gap_eff +\
                                           para_Q[5]*tc_weight_merger(age_Q)
                                          )
        
    if Q_IS_MERGER == False:
        ## early
        fQ = para_Q[0]
        density_array_early = np.log(
            velocity_density_3D_func(l[early],b[early],
                                vL[early], vB[early], age[early], para_v, is_vL,
                                    )\
               * (1-para_Q[2]-para_Q[5]-para_Q[0]+fQ) +\
            (velocity_density_3D_func(l[early].reshape(-1,1),b[early].reshape(-1,1),
                                 vL_delayed[early], vB_delayed[early], t0.reshape(1,-1), para_v, is_vL,
                                     )\
               * pdf_tct0(age[early].reshape(-1,1),
                          t0.reshape(1,-1),
                          para_Q[3:],
                          normalization_factor_Q_merger,
                          SFR_merger, delay_cut, delay_index
                          ) * para_Q[5]\
            ).mean(1) +\
            1/400/background_pdf_factor\
                * para_Q[2]
                                    ) - np.log( (1-para_Q[5]-para_Q[0]+fQ) +\
                                               para_Q[5]*tc_weight_merger(age[early])
                                              )
        
        ## late
        v_drift_late = velocity_scatter_3D(age+para[1], para_v[0], para_v[1], para_v[2], para_v[6])**2/stromberg_k
        vL_late, vB_late = np.array(get_v_delayed_3D(age+para[1], l, b, pml, pmb, factor, v_drift_late, U,V,W))
        vL_late = vL_late[late]
        vB_late = vB_late[late]
        too_late = age[late]+para_Q[1] > end_of_SF
        fQ_late = fQ*(~too_late)
        density_array_late = np.log(
            velocity_density_3D_func(l[late],b[late],
                                vL[late], vB[late], age[late], para_v, is_vL,
                                    ) \
               * (1-para_Q[0]-para_Q[2]-para_Q[5]) +\
            velocity_density_3D_func(l[late],b[late],
                                vL_late, vB_late, age[late]+para_Q[1], para_v, is_vL,
                                    ) \
               * fQ_late +\
            (velocity_density_3D_func(l[late].reshape(-1,1),b[late].reshape(-1,1),
                                 vL_delayed[late], vB_delayed[late], t0.reshape(1,-1), para_v, is_vL,
                                     )\
               * pdf_tct0(age[late].reshape(-1,1),
                          t0.reshape(1,-1),
                          para_Q[3:],
                          normalization_factor_Q_merger,
                          SFR_merger, delay_cut, delay_index
                         ) * para_Q[5]\
            ).mean(1) + \
            1/400/background_pdf_factor\
                * para_Q[2]
                                   ) - np.log( (1-para_Q[5]-para_Q[0]) +\
                                               fQ_late +\
                                              para_Q[5]*tc_weight_merger(age[late])
                                             )

        # 0: index of WD (tc) ,  1: t0,  2: delay on branch 
        ## Q
        n_Q_delay = 200
        v_drift_Q_Qdelayed = velocity_scatter_3D(age_Q.reshape(-1,1)+np.linspace(0,1,n_Q_delay).reshape(1,-1)*para_Q[1],
                                                 para_v[0], para_v[1], para_v[2], para_v[6])**2/stromberg_k
        vL_Q_Qdelayed, vB_Q_Qdelayed = get_v_delayed_3D(0, 
                                l_Q.reshape(-1,1), 
                                b_Q.reshape(-1,1),
                                pml_Q.reshape(-1,1), 
                                pmb_Q.reshape(-1,1), 
                                factor_Q.reshape(-1,1),
                                v_drift_Q_Qdelayed,
                               U,V,W) 
        too_late = age_Q.reshape(-1,1)+np.linspace(0,1,n_Q_delay).reshape(1,-1)*para_Q[1] > end_of_SF
        density_array_Q = np.log(
            velocity_density_3D_func(l_Q, b_Q,
                                vL_Q, vB_Q, age_Q, para_v, is_vL)\
                * (1-para_Q[0]-para_Q[2]-para_Q[5]) +\
            (velocity_density_3D_func(l_Q.reshape(-1,1), b_Q.reshape(-1,1),
                                 vL_Q_Qdelayed, vB_Q_Qdelayed, 
                                      age_Q.reshape(-1,1)+np.linspace(0,1,n_Q_delay).reshape(1,-1)*para_Q[1],
                                      para_v, is_vL) * ~(too_late)
            ).mean(1) * (para_Q[1]+t_gap_eff) / t_gap_eff * fQ +\
            (velocity_density_3D_func(l_Q.reshape(-1,1),b_Q.reshape(-1,1),
                                 vL_Q_delayed, vB_Q_delayed, t0.reshape(1,-1), para_v, is_vL,
                                     )\
               * pdf_tct0(age_Q.reshape(-1,1),
                          t0.reshape(1,-1),
                          para_Q[3:],
                          normalization_factor_Q_merger,
                          SFR_merger, delay_cut, delay_index
                         )\
            ).mean(1) * para_Q[5] +\
            1/400/background_pdf_factor*para_Q[2] 
                                ) - np.log( (1-para_Q[0]-para_Q[5]) +\
                                           fQ * (~too_late).mean(1) * (para_Q[1] + t_gap_eff) / t_gap_eff +\
                                           para_Q[5]*tc_weight_merger(age_Q)
                                          )
        
    #density_array_Q = 0
    density = density_array_early[~np.isnan(density_array_early)].sum() +\
              density_array_late[~np.isnan(density_array_late)].sum() + \
              density_array_Q[~np.isnan(density_array_Q)].sum()

    if np.isnan(density) or density==0:
        return -np.inf, density_array_early, density_array_late, density_array_Q
    else:
        return density, density_array_early, density_array_late, density_array_Q

def ln_prob(para, 
            mass, age, pml, pmb, factor, l, b,
            mass_Q, age_Q, pml_Q, pmb_Q, factor_Q, l_Q, b_Q,
            not_fit_UVW,not_fit_index,fixv):
    # "power index", "v4", "v_T",
    #"index_z","v4_z",
    #"sy/sx",
    #"v0","v0_z","v_T_z",
    #"sy/sx_T"
    #UVW
    #"fraction", "delay", "background", "SFR_step", "no use", "merger fraction"
    para_v = para[0:Nv].copy()
    para_Q = para[Nv:Nv+NQ].copy()
    if (para_v[0]<0)+(para_v[0]>1)+ (para_v[1]<0)+(para_v[1]>50)+ (para_v[2]<40)+(para_v[2]>90)+\
        (para_v[3]<0)+(para_v[3]>1)+ (para_v[4]<0)+(para_v[4]>50)+\
        (para_v[5]<0.0)+(para_v[5]>1)+\
        (para_v[6]<0)+(para_v[6]>20)+ (para_v[7]<0)+(para_v[7]>20)+ (para_v[8]<0)+(para_v[8]>80)+\
        (para_v[9]<0.0)+(para_v[9]>1)+\
        (para_v[10]<-5)+(para_v[10]>20)+\
        (para_v[11]<-5)+(para_v[11]>20)+\
        (para_v[12]<-5)+(para_v[12]>20)+\
        (para_Q[0]<0)+(para_Q[0]>0.35)+\
        (para_Q[1]<0)+(para_Q[1]>delay_lim)+\
        (para_Q[2]<0)+(para_Q[2]>0.05)+\
        (para_Q[3]<-2)+(para_Q[3]>-0.1)+\
        (para_Q[4]<70)+(para_Q[4]>90)+\
        (para_Q[5]<0)+(para_Q[5]>0.50)+\
        (para_Q[6]<-5)+(para_Q[6]>20):
        return -np.inf
    else:
        density,_,_,_ = ln_likelihood_pheno(para,
                                   mass, age, pml, pmb, factor, l, b,
                                   mass_Q, age_Q, pml_Q, pmb_Q, factor_Q, l_Q, b_Q,
                                           False,
                                           not_fit_UVW=not_fit_UVW,not_fit_index=not_fit_index,fixv=fixv)
        return density
    
    
def ln_likelihood_pheno_merger_rate(para,
                        mass, age, pml, pmb, factor, l, b,
                        mass_Q, age_Q, pml_Q, pmb_Q, factor_Q, l_Q, b_Q,
                       is_check, velocity_density_3D_check_func=None, is_vL=True,
                       not_fit_UVW=False,not_fit_index=False,fixv=False):
    para_v = para[0:Nv].copy()
    para_Q = para[Nv:Nv+NQ].copy()
    delay_index = DELAY_INDEX#para_Q[3]#
    delay_cut = DELAY_CUT
    #stromberg_k = para_Q[4]
    if not_fit_UVW==True:
        U = 11#para_v[10]
        V = 7.5#para_v[11]
        W = 7#para_v[12]
        U_Q = U
        V_Q = V
        W_Q = W
    else:
        U = para_v[10]
        V = para_v[11]
        W = para_v[12]
        U_Q = 11#para_v[10]#para[Nv+3]
        V_Q = 7.5#para_v[11]#para_Q[4]
        W_Q = 7#para_v[12]#para_Q[6]
    if not_fit_index==True:
        para_v[0] = 0.38
        para_v[3] = 0.5
    if fixv==True:
        para_v[1] = 35
        para_v[2] = 70
        para_v[4] = 17
        para_v[5] = 0.60
        para_v[6] = 0
        para_v[7] = 0
        para_v[8] = 40
        para_v[9] = 0.60
    if test_number == 'noQ':
        para_Q[0] = 0
    if test_number == 'nomerger':
        para_Q[5] = 0
    if test_number == 'noQmerger':
        para_Q[0] = 0
        para_Q[5] = 0
    #if METHOD == 'Run_MCMC':
    para_Q[3] = -end_of_SF
    
    early = np.array(1.22 - (age - 0.6) * 0.2> mass) #for ONe: 1.22; CO: 1.25
    late = np.array(1.22 - (age - 0.6) * 0.2 < mass)    
    
    ## tct0_array is the pdf of tc, t0, range: tc: 0~age_lim+delay, t0: 0~end_of_SFR
    ## normalization_factor_Q is the mean of tct0_array
    tct0_array, normalization_factor_Q = pdf_tct0_array(np.linspace(0.1,end_of_SF,n_tc).reshape(-1,1),
                                               np.linspace(0,end_of_SF,n).reshape(1,-1),
                                               para_Q[3:],
                                               SFR, delay_cut, delay_index)
    ## tc_weight is the norm factor of pdf of t0 given a tc. tct0_array / tc_weight
    tc_weight = tct0_array.mean(1)/normalization_factor_Q
    tc_weight = interp1d(np.linspace(0.1,end_of_SF,n_tc),
                         tc_weight,
                         fill_value=0.0,
                         bounds_error=False)
    
    ## For the ordinary merger population:
    tct0_array_merger = tct0_array
    normalization_factor_Q_merger = normalization_factor_Q
    tc_weight_merger = tc_weight

    
    ## v_delay for different possible t0
    v_drift_delayed = velocity_scatter_3D(t0, para_v[0], para_v[1], para_v[2], para_v[6])**2/stromberg_k

    vL_delayed, vB_delayed = get_v_delayed_3D(t0.reshape(1,-1), 
                              l.reshape(-1,1),
                              b.reshape(-1,1),
                              pml.reshape(-1,1),
                              pmb.reshape(-1,1), 
                              factor.reshape(-1,1),
                              v_drift_delayed.reshape(1,-1),
                             U,V,W)    
    vL_Q_delayed, vB_Q_delayed = get_v_delayed_3D(t0.reshape(1,-1), 
                                l_Q.reshape(-1,1), 
                                b_Q.reshape(-1,1),
                                pml_Q.reshape(-1,1), 
                                pmb_Q.reshape(-1,1), 
                                factor_Q.reshape(-1,1),
                                v_drift_delayed.reshape(1,-1),
                               U,V,W) 
    vL_delayed_Q, vB_delayed_Q = get_v_delayed_3D(t0.reshape(1,-1), 
                              l.reshape(-1,1),
                              b.reshape(-1,1),
                              pml.reshape(-1,1),
                              pmb.reshape(-1,1), 
                              factor.reshape(-1,1),
                              v_drift_delayed.reshape(1,-1),
                             U_Q,V_Q,W_Q)    
    vL_Q_delayed_Q, vB_Q_delayed_Q = get_v_delayed_3D(t0.reshape(1,-1), 
                                l_Q.reshape(-1,1), 
                                b_Q.reshape(-1,1),
                                pml_Q.reshape(-1,1), 
                                pmb_Q.reshape(-1,1), 
                                factor_Q.reshape(-1,1),
                                v_drift_delayed.reshape(1,-1),
                               U_Q,V_Q,W_Q) 
    v_drift = velocity_scatter_3D(age, para_v[0], para_v[1], para_v[2], para_v[6])**2/stromberg_k
    v_drift_Q = velocity_scatter_3D(age_Q, para_v[0], para_v[1], para_v[2], para_v[6])**2/stromberg_k
    vL, vB = np.array(get_v_delayed_3D(age, l, b, pml, pmb, factor, v_drift, U,V,W))
    vL_Q, vB_Q = np.array(get_v_delayed_3D(age_Q, l_Q, b_Q, pml_Q, pmb_Q, factor_Q, v_drift_Q, U,V,W))
    if is_check == True:
        velocity_density_3D_func = velocity_density_3D_check_func
        background_pdf_factor = 1    
    else:
        velocity_density_3D_func = velocity_density_3D
        background_pdf_factor = 400
    ## pdf of v,tc =
        ## [pdf(v) for ordinary * uniform(tc)] * ordinary fraction +
        ## [pdf(v) for merger * merger fraction(t0)
    ## pdf_tct0 is (age_lim+delay) * end_of_SF * pdf(tc,t0)
    
    if Q_IS_MERGER == False or Q_IS_MERGER == True:
        ## early
        fQ = 0
        density_array_early = np.log(
            velocity_density_3D_func(l[early],b[early],
                                vL[early], vB[early], age[early], para_v, is_vL,
                                    )\
               * (1-para_Q[2]-para_Q[5]-para_Q[0]+fQ) +\
            (velocity_density_3D_func(l[early].reshape(-1,1),b[early].reshape(-1,1),
                                 vL_delayed[early], vB_delayed[early], t0.reshape(1,-1), para_v, is_vL,
                                     )\
               * pdf_tct0(age[early].reshape(-1,1),
                          t0.reshape(1,-1),
                          para_Q[3:],
                          normalization_factor_Q_merger,
                          SFR_merger, delay_cut, delay_index
                          ) * para_Q[5]\
            ).mean(1) +\
            1/400/background_pdf_factor\
                * para_Q[2]
                                    ) - np.log( (1-para_Q[5]-para_Q[0]+fQ) +\
                                               para_Q[5]*tc_weight_merger(age[early])
                                              )
    #density_array_Q = 0
    density = density_array_early[~np.isnan(density_array_early)].sum()

    if np.isnan(density) or density==0:
        return -np.inf, density_array_early
    else:
        return density, density_array_early

def ln_prob_merger_rate(para, 
            mass, age, pml, pmb, factor, l, b,
            mass_Q, age_Q, pml_Q, pmb_Q, factor_Q, l_Q, b_Q,
            not_fit_UVW,not_fit_index,fixv):
    # "power index", "v4", "v_T",
    #"index_z","v4_z",
    #"sy/sx",
    #"v0","v0_z","v_T_z",
    #"sy/sx_T"
    #UVW
    #"fraction", "delay", "background", "SFR_step", "no use", "merger fraction"
    para_v = para[0:Nv].copy()
    para_Q = para[Nv:Nv+NQ].copy()
    if (para_v[0]<0)+(para_v[0]>1)+ (para_v[1]<0)+(para_v[1]>50)+ (para_v[2]<40)+(para_v[2]>90)+\
        (para_v[3]<0)+(para_v[3]>1)+ (para_v[4]<0)+(para_v[4]>50)+\
        (para_v[5]<0.0)+(para_v[5]>1)+\
        (para_v[6]<0)+(para_v[6]>20)+ (para_v[7]<0)+(para_v[7]>20)+ (para_v[8]<0)+(para_v[8]>80)+\
        (para_v[9]<0.0)+(para_v[9]>1)+\
        (para_v[10]<-5)+(para_v[10]>20)+\
        (para_v[11]<-5)+(para_v[11]>20)+\
        (para_v[12]<-5)+(para_v[12]>20)+\
        (para_Q[0]<0)+(para_Q[0]>0.35)+\
        (para_Q[1]<0)+(para_Q[1]>delay_lim)+\
        (para_Q[2]<0)+(para_Q[2]>0.05)+\
        (para_Q[3]<-2)+(para_Q[3]>-0.1)+\
        (para_Q[4]<70)+(para_Q[4]>90)+\
        (para_Q[5]<0)+(para_Q[5]>0.80)+\
        (para_Q[6]<-5)+(para_Q[6]>20):
        return -np.inf
    else:
        density,_ = ln_likelihood_pheno_merger_rate(para,
                                   mass, age, pml, pmb, factor, l, b,
                                   mass_Q, age_Q, pml_Q, pmb_Q, factor_Q, l_Q, b_Q,
                                           False,
                                           not_fit_UVW=not_fit_UVW,not_fit_index=not_fit_index,fixv=fixv)
        return density

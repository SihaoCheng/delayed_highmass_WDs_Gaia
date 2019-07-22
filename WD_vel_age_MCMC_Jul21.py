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
import WD_HR
import WD_models
import WD_MCMC_func


test_number = sys.argv[2]
WD_MCMC_func.test_number = test_number

if sys.argv[3]=='M':
    METHOD = 'Run_MCMC'
    WD_MCMC_func.t_gap_eff = 0.743
if sys.argv[3]=='S':
    METHOD = 'Simulate'
    WD_MCMC_func.t_gap_eff = 0.505

if sys.argv[4]=='T':
    NOT_FIT_UVW= True
if sys.argv[4]=='F':
    NOT_FIT_UVW = False

if sys.argv[5]=='T':
    NOT_FIT_INDEX= True
if sys.argv[5]=='F':
    NOT_FIT_INDEX = False

if sys.argv[6]=='T':
    FIXV= True
if sys.argv[6]=='F':
    FIXV = False
    
if sys.argv[7]=='T':
    WD_MCMC_func.Q_IS_MERGER= True
if sys.argv[7]=='F':
    WD_MCMC_func.Q_IS_MERGER= False
    
WD_MCMC_func.DELAY_INDEX = -float(sys.argv[8])    
WD_MCMC_func.DELAY_CUT = float(sys.argv[9])

if len(sys.argv)>10:
    DELAY = int(sys.argv[10])

##--------------------------------------------------------------------------------------------------
agents = 30
chunksize = 1
number = agents

from WD_MCMC_func import Nv, NQ, end_of_SF, age_T, DELAY_INDEX, DELAY_CUT, Q_IS_MERGER, stromberg_k
burning = 200
then_run = 400
gap = 5
ndim, nwalkers = Nv+NQ, 50


# Load WD table
##------------------------------------------------------------------------------------------------------------------------
#WD_warwick_smaller = np.load('/datascope/menard/group/scheng/Gaia/WD_warwick_smaller.npy')[0]['WD_warwick_smaller']
SELECTION_PARA = [1.4,0.10,2,22,8,300]
WD_warwick_smaller = np.load('/datascope/menard/group/scheng/Gaia/WD_warwick_smaller.npy')[0]['WD_warwick_smaller']
_, WD_warwick_smaller = WD_MCMC_func.select_WD(WD_warwick_smaller,SELECTION_PARA[0],SELECTION_PARA[1],SELECTION_PARA[2],
                                               SELECTION_PARA[3],SELECTION_PARA[4],SELECTION_PARA[5])

if WD_MCMC_func.Q_IS_MERGER==False:
    WD_MCMC_func.n = 400
    WD_MCMC_func.n_tc = 8000


# Select the WDs Suitable for MCMC
##------------------------------------------------------------------------------------------------------------------------
mass_min = 1.08#1.07#1.09
mass_max = 1.23#1.22#1.27
distance1 = 0
distance2 = int(sys.argv[1])
spec_type = 'H'
model = 'o'
WD_model = WD_models.load_model('f', 'f', model, spec_type)
age_lim = 3.5
WD_warwick_smaller['mass'] = WD_warwick_smaller['mass_' + spec_type + '_' + model]
WD_warwick_smaller['age'] = WD_warwick_smaller['age_' + spec_type + '_' + model]
Q_branch = np.array((WD_warwick_smaller['mass']>mass_min)*(WD_warwick_smaller['mass']<mass_max)*\
         (1/WD_warwick_smaller['parallax']*1000>distance1)*(1/WD_warwick_smaller['parallax']*1000<distance2)*\
        (WD_HR.func_select(WD_warwick_smaller['bp_rp'],WD_warwick_smaller['G'],13.20,1.2,0.20,-0.40,0.10)) )
WD = WD_warwick_smaller[np.array((WD_warwick_smaller['mass']>mass_min)*(WD_warwick_smaller['mass']<mass_max)*\
        (1/WD_warwick_smaller['parallax']*1000>distance1)*(1/WD_warwick_smaller['parallax']*1000<distance2)\
         * ~Q_branch )]
WD_Q = WD_warwick_smaller[Q_branch]

print('length of WD: ',len(WD), 'length of WD_Q: ',len(WD_Q))


# prepare to get v
pml,pmb,factor = WD_MCMC_func.prep_get_v(WD)
pml_Q, pmb_Q, factor_Q = WD_MCMC_func.prep_get_v(WD_Q)
v_drift = (((WD['age']+0.1)/10.1)**0.2*40)**2/stromberg_k
v_drift_Q = (((WD_Q['age']+0.1)/10.1)**0.2*40)**2/stromberg_k
vL, vB = np.array(WD_MCMC_func.get_v_delayed_3D(WD['age'], WD['l'], WD['b'], pml, pmb, factor, v_drift,11,7.5,7))
vL_Q, vB_Q = np.array(WD_MCMC_func.get_v_delayed_3D(WD_Q['age'], WD_Q['l'], WD_Q['b'], pml_Q, pmb_Q, factor_Q, v_drift_Q,\
                                       11,7.5,7))

selection = np.array(((WD['age'])<age_lim)*((WD['age'])>0.1)*(np.abs(vL<200))*(np.abs(vB<200)))
mass = np.array(WD['mass'][selection])
age = np.array(WD['age'][selection])
l = np.array(WD['l'][selection])
b = np.array(WD['b'][selection])
vL = vL[selection]
vB = vB[selection]
pml = pml[selection]
pmb = pmb[selection]
factor = factor[selection]

selection_Q = np.array(((WD_Q['age'])<age_lim)*((WD_Q['age'])>0.1)*(np.abs(vL_Q<200))*(np.abs(vB_Q<200)))
mass_Q = np.array(WD_Q['mass'][selection_Q])
age_Q = np.array(WD_Q['age'][selection_Q])
l_Q = np.array(WD_Q['l'][selection_Q])
b_Q = np.array(WD_Q['b'][selection_Q])
vL_Q = vL_Q[selection_Q]
vB_Q = vB_Q[selection_Q]
pml_Q = pml_Q[selection_Q]
pmb_Q = pmb_Q[selection_Q]
factor_Q = factor_Q[selection_Q]

##--------------------------------------------------------------------------------------------------
def parallel(i):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, WD_MCMC_func.ln_prob, 
                                    args=[mass, age, pml, pmb, factor, l, b,
                                          mass_Q, age_Q, pml_Q, pmb_Q, factor_Q, l_Q, b_Q,
                                          NOT_FIT_UVW, NOT_FIT_INDEX, FIXV])
    a_random_number = np.random.randint(0,100000)
    np.random.seed(i+a_random_number)
    # "power index", "v10", "v_T",
    #"index_z","v10_z",
    #"sy/sx",
    #"v0","v0_z","v_T_z",
    #"sy/sx_T"
    #UVW
    #"fraction", "delay", "background"
    #
    p0 = [np.concatenate((\
            np.random.rand(1)*0.1*2+0.3-0.1, np.random.rand(1)*10*2+30-10, np.random.rand(1)*10*2+65-10,
            np.random.rand(1)*0.15*2+0.5-0.15, np.random.rand(1)*10*2+15-10,
            np.random.rand(1)*0.1*2+0.67-0.1,
            np.random.rand(1)*4*2+5-4, np.random.rand(1)*4*2+5-4, np.random.rand(1)*10*2+40-10,
            np.random.rand(1)*0.1*2+0.63-0.1,
            np.array([7,5,5])+np.random.rand(3)*np.array([5,5,5]),
            np.array([0.02,8,0])+np.random.rand(3)*np.array([0.15,4,0.01]),
            np.array([-1,77,0.15,5])+np.random.rand(4)*np.array([0.3,5,0.15,5]) )) for j in range(nwalkers)]
    pos, _, _ = sampler.run_mcmc(p0, burning)
    sampler.reset()
    sampler.run_mcmc(pos, then_run)
    a = sampler.flatchain[::gap,:]
    return a



# Run MCMC
##--------------------------------------------------------------------------------------------------
if METHOD == 'Run_MCMC':
    with Pool(processes=agents) as pool:
        result = pool.map(parallel, np.arange(number), chunksize)  
    sampling_per_agent = nwalkers*(then_run//gap)
    
    para_v = np.empty((number*sampling_per_agent,Nv))
    para_Q = np.empty((number*sampling_per_agent,NQ))
    for i in range(number):
        para_v[(i*sampling_per_agent):((i+1)*sampling_per_agent)] = result[i][:,:Nv]
        para_Q[(i*sampling_per_agent):((i+1)*sampling_per_agent)] = result[i][:,Nv:Nv+NQ]
    para_v = para_v.reshape(agents,nwalkers,then_run//gap,Nv).transpose((2,1,0,3))\
                                            .reshape(number*nwalkers*(then_run//gap),Nv)
    para_Q = para_Q.reshape(agents,nwalkers,then_run//gap,NQ).transpose((2,1,0,3))\
                                            .reshape(number*nwalkers*(then_run//gap),NQ)
    #---------------------------------------------------------------------------------------------------------------------------
    para_input = np.median(np.concatenate((para_v,para_Q),axis=1)[-50000:,:],axis=0)
    
    x_list = ['np.arange(0,15,0.2)','np.arange(0,0.45,0.01)','np.arange(0,0.45,0.01)']
    PDF_test_name = ['delay_test','mfraction','Qfraction']
    changed_para = [Nv+1, Nv+5 , Nv+0]
    for PDF_test_index in range(3):
        pdf_sim = np.empty_like(eval(x_list[PDF_test_index]))
        pdf_Q_sim = np.empty_like(eval(x_list[PDF_test_index]))
        pdf_e_sim = np.empty_like(eval(x_list[PDF_test_index]))
        pdf_l_sim = np.empty_like(eval(x_list[PDF_test_index]))
    
        for i,x in enumerate(eval(x_list[PDF_test_index])):
            para = para_input.copy()
            para[changed_para[PDF_test_index]] = x
            pdf_sim[i], temp1, temp2, temp = WD_MCMC_func.ln_likelihood_pheno(para, mass, age, pml, pmb, factor, l, b,
                                           mass_Q, age_Q, pml_Q, pmb_Q, factor_Q, l_Q, b_Q,
                                                   False,
                                                   not_fit_UVW=NOT_FIT_UVW,not_fit_index=NOT_FIT_INDEX,fixv=FIXV)
            pdf_e_sim[i] = temp1[~np.isnan(temp1)].sum()
            pdf_l_sim[i] = temp2[~np.isnan(temp2)].sum()
            pdf_Q_sim[i] = temp[~np.isnan(temp)].sum()
        exec( PDF_test_name[PDF_test_index]+'= [pdf_sim,pdf_e_sim,pdf_l_sim,pdf_Q_sim ]')

    #---------------------------------------------------------------------------------------------------------------------------
    if Q_IS_MERGER == True:
        suffix = '.npy'
    if Q_IS_MERGER == False:
        suffix = 'Qisnotmerger.npy'
    
    np.save('/datascope/menard/group/scheng/Gaia/WD_vel_age_MCMC_Feb12/MCMC_power_'+sys.argv[8]+'_'+sys.argv[9]+'_'+\
            str(mass_min)+'_'+str(distance2)+'_'+str(age_lim)+'_'+\
            spec_type+'_'+model+'_'+str(end_of_SF)+'_T'+\
            str(age_T)+'_'+sys.argv[4]+sys.argv[5]+sys.argv[6]+test_number+suffix,
            np.array([{'para_Q':para_Q, 'para_v':para_v,
                       'delay_test':delay_test, 'mfraction':mfraction, 'Qfraction':Qfraction,
                       'data_length':[selection.sum(),selection_Q.sum()],
                     'para_input':para_input,
                      'selection_para':SELECTION_PARA,
                      'delay_index_cut':[DELAY_INDEX, DELAY_CUT]}]) )
    print('all finished')


    

Q_FRACTION = 0.07
M_FRACTION = 0.15
DELAY = 10

if METHOD == 'Simulate':
    WD_MCMC_func.t_gap_eff = 0.705#

##--------------------------------------------------------------------------------------------------
agents = 1
chunksize = 1
number = agents

# Simulate and then run MCMC
##--------------------------------------------------------------------------------------------------  
if METHOD == 'Simulate':
    para = np.array([0.39, 27, 64,
                 0.6, 12,
                 0.65,
                 7.6 ,4.0 ,35 ,0.58,
                 10,7.5,6.5,
                 Q_FRACTION, DELAY ,0.000 , 10, 7.5, M_FRACTION, 6.5]) # 3Gyr
    
    para_input = para.copy()
    
    def simulation_m(nwalkers,then_run,para,m_or_Q='Q'):
        burning = 600
        then_run = 1
        ndim = 2
        
        def ln_pdf_tct0_m(tct0, para):
            tc = tct0[0]
            t0 = tct0[1]
            
            para_v = para[0:Nv]
            para_Q = para[Nv:]
                
            density = WD_MCMC_func.pdf_tct0(tc,t0,para_Q[3:],1,WD_MCMC_func.SFR_merger)#*((tc<=2)*1+(tc>2)*0.3)
            
            if np.isnan(density) or density==0:
                return -np.inf
            else:
                if t0<0.1 or t0>=end_of_SF or tc<0.1 or tc>=age_lim:
                    return -np.inf
                else:
                    return np.log(density)
                
        def ln_pdf_tct0_Q(tct0, para):
            tc = tct0[0]
            t0 = tct0[1]
            
            para_v = para[0:Nv]
            para_Q = para[Nv:]
            density = WD_MCMC_func.pdf_tct0(tc,t0,para_Q[3:],1,WD_MCMC_func.SFR)#*((tc<=2)*1+(tc>2)*0.3)
            
            if np.isnan(density) or density==0:
                return -np.inf
            else:
                if t0<0.1 or t0>=end_of_SF or tc<0.1 or tc>=age_lim+para_Q[1]:
                    return -np.inf
                else:
                    return np.log(density)
                
                
        ln_pdf_tct0 = ln_pdf_tct0_Q
        if m_or_Q=='m':
            ln_pdf_tct0 = ln_pdf_tct0_m        
        
        if Q_IS_MERGER==False and m_or_Q=='Q':
            a = np.random.rand(nwalkers*then_run).reshape(-1,1)*(end_of_SF-0.1)+0.1
            a = np.concatenate((a,a),axis=1)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_pdf_tct0, 
                                        args=[para])
            p0 = [ np.array([1.5,10.5])+np.random.rand(2)*np.array([1,1]) for j in range(nwalkers)] #SFR2
            pos, _, _ = sampler.run_mcmc(p0, burning)
            sampler.reset()
            sampler.run_mcmc(pos, then_run)
            sampler.reset()
            sampler.run_mcmc(pos, then_run)
            a = sampler.flatchain
        return a
    
    #-----------------------------------------------------------------------------------------------------------
    
    
    N_total = selection.sum()+selection_Q.sum()
    
    tct0_array, normalization_factor_Q = \
            WD_MCMC_func.pdf_tct0_array(np.linspace(0.1,end_of_SF,WD_MCMC_func.n_tc).reshape(-1,1),
                                               np.linspace(0,end_of_SF,WD_MCMC_func.n).reshape(1,-1),
                                               para[Nv+3:],
                                               WD_MCMC_func.SFR)
    ## tc_weight is the norm factor of pdf of t0 given a tc. tct0_array / tc_weight
    tc_weight = tct0_array.mean(1)/normalization_factor_Q
    Q_fraction_select = ( np.linspace(0.1,end_of_SF,WD_MCMC_func.n_tc)>0.1 )*\
                           ( np.linspace(0.1,end_of_SF,WD_MCMC_func.n_tc)<=age_lim+para[Nv+1] )
    m_fraction_select = ( np.linspace(0.1,end_of_SF,WD_MCMC_func.n_tc)>0.1 )*\
                           ( np.linspace(0.1,end_of_SF,WD_MCMC_func.n_tc)<=age_lim )
    
    length_Q = (age_lim+para[Nv+1]>end_of_SF)*(end_of_SF-0.1)+\
                    (age_lim+para[Nv+1]<=end_of_SF)*(age_lim+para[Nv+1]-0.1)
    all_frac_sim = (tc_weight[Q_fraction_select].mean()*para[Nv:][0]*length_Q/(age_lim-0.1) +\
                    tc_weight[m_fraction_select].mean()*para[Nv:][5] +\
                        1-para[Nv:][0]-para[Nv:][5] )
    Q_frac_sim = tc_weight[Q_fraction_select].mean()*para[Nv:][0]*length_Q/(age_lim-0.1)\
                            / all_frac_sim
    m_frac_sim = tc_weight[m_fraction_select].mean()*para[Nv:][5] / all_frac_sim
    
    N_Q = int(N_total*Q_frac_sim/2)*2
    N_m = int(N_total*m_frac_sim/2)*2
    N_o = N_total - N_Q - N_m
    
    #-----------------------------------------------------------------------------------------------------------
    
    def select_age(mass, tc, delay=0):
        early = mass<(1.22 - (tc - 0.6 + WD_MCMC_func.t_gap_eff/2) * 0.2)
        late = mass>(1.22 - (tc - delay - 0.6 - WD_MCMC_func.t_gap_eff/2) * 0.2)
        branch = ~(early + late)
        return early, late, branch
    
    N_o_young = int(N_o * (2-0.1)*1/((2-0.1)*1 + (age_lim-2)*0.3))
    tc_o = np.concatenate(( np.random.rand(N_o_young)*(2-0.1)+0.1, np.random.rand(N_o-N_o_young)*(age_lim-2)+2 ))
    tc_o = np.random.rand(N_o)*(age_lim-0.1)+0.1
    t0_o = tc_o.copy()
    mass_o = np.random.rand(N_o)*(mass_max-mass_min)+mass_min
    
    if N_Q > 5:
        tct0_Q = simulation_m(N_Q,1,para,'Q')
        tc_Q = tct0_Q[:,0]
        t0_Q = tct0_Q[:,1]
        #print(len(mass_Q),len(tc_Q), N_Q, len(tct0_Q))
        mass_Q = np.random.rand(N_Q)*(mass_max-mass_min)+mass_min
        Q_early, Q_late, Q_branch = select_age(mass_Q, tc_Q, para[Nv:][1])
        tc_Q[Q_branch] = (1.22 - mass_Q[Q_branch])/0.2 + 0.6 - WD_MCMC_func.t_gap_eff/2 + \
                                np.random.rand(Q_branch.sum())*WD_MCMC_func.t_gap_eff
        tc_Q[Q_late] -= para[Nv:][1] 
    else:
        N_Q = 0
        tc_Q = np.zeros(0)
        t0_Q = np.zeros(0)
        mass_Q = np.zeros(0)
    
    if N_m > 5:
        tct0_m = simulation_m(N_m,1,para,'m')
        tc_m = tct0_m[:,0]
        t0_m = tct0_m[:,1]
        mass_m = np.random.rand(N_m)*(mass_max-mass_min)+mass_min
    else:
        N_m = 0
        tc_m = np.zeros(0)
        t0_m = np.zeros(0)
        mass_m = np.zeros(0)
    #para_input[Nv+0] = N_Q/(N_o+N_Q+N_m)
    #para_input[Nv+5] = N_m/(N_o+N_Q+N_m)
    
    early_sim, late_sim, branch_sim = select_age(np.concatenate((mass_o, mass_Q, mass_m)),
                                                 np.concatenate((tc_o, tc_Q, tc_m)) )
    mass_sim = np.concatenate((mass_o, mass_Q, mass_m))[~branch_sim]
    mass_sim_Q = np.concatenate((mass_o, mass_Q, mass_m))[branch_sim]
    age_sim = np.concatenate((tc_o, tc_Q, tc_m))[~branch_sim]
    age_sim_Q = np.concatenate((tc_o, tc_Q, tc_m))[branch_sim]
    t0_sim = np.concatenate((t0_o, t0_Q, t0_m))[~branch_sim]
    t0_sim_Q = np.concatenate((t0_o, t0_Q, t0_m))[branch_sim]
    
    #----------------------------------------------------------------------------------------------------------
    
    shuffle = np.arange(len(l)+len(l_Q))
    np.random.shuffle(shuffle)
    
    selection_sim = np.zeros(N_total,dtype=bool)
    selection_sim[:(~branch_sim).sum()] = True
    selection_sim_Q = np.zeros(N_total,dtype=bool)
    selection_sim_Q[(~branch_sim).sum():] = True
    
    l_sim = np.concatenate((l,l_Q))[shuffle][selection_sim]
    b_sim = np.concatenate((b,b_Q))[shuffle][selection_sim]
    factor_sim = np.concatenate((factor,factor_Q))[shuffle][selection_sim]
    
    l_sim_Q = np.concatenate((l,l_Q))[shuffle][selection_sim_Q]
    b_sim_Q = np.concatenate((b,b_Q))[shuffle][selection_sim_Q]
    factor_sim_Q = np.concatenate((factor,factor_Q))[shuffle][selection_sim_Q]
    
    #----------------------------------------------------------------------------------------------------------
    
    para_v = para_input[0:Nv].copy()
    para_Q = para_input[Nv:].copy()
    
    U_sim,V_sim,W_sim = para_v[10],para_v[11],para_v[12]
    
    index_sim, v10_sim, vT_sim, sy_sx, v0_sim, sy_sx_T = \
                            para_v[0], para_v[1], para_v[2], para_v[5], para_v[6], para_v[9]
    index_z_sim, v10_z_sim, vT_z_sim, v0_z_sim = para_v[3], para_v[4], para_v[8], para_v[7]
    
    def sigma_v(age, index_sim, v10_sim, vT_sim, v0_sim):
        sx_sim = WD_MCMC_func.velocity_scatter_3D(age, index_sim, v10_sim, vT_sim, v0_sim)
        sy_sim = WD_MCMC_func.velocity_scatter_3D(age, index_sim, v10_sim*sy_sx, vT_sim*sy_sx_T,\
                                 v0_sim*sy_sx)
        sz_sim = WD_MCMC_func.velocity_scatter_3D(age, index_z_sim, v10_z_sim, vT_z_sim, v0_z_sim)
        return sx_sim, sy_sim, sz_sim
    
    def mv3(a,b):
        result = np.array([(a[0,:]*b).sum(),\
                         (a[1,:]*b).sum(),\
                         (a[2,:]*b).sum(),\
                            0])
        return result
    
    def LB_sampling(l,b,sx,sy,sz,v_drift,U,V,W):
        Sigma = np.array([[sx**2, 0, 0],[0, sy**2, 0],[0, 0, sz**2]])
        A = np.array([[-np.sin(l/180*np.pi), -np.sin(b/180*np.pi)*np.cos(l/180*np.pi), \
                   np.cos(b/180*np.pi)*np.cos(l/180*np.pi)],\
                  [np.cos(l/180*np.pi), -np.sin(b/180*np.pi)*np.sin(l/180*np.pi), \
                   np.cos(b/180*np.pi)*np.sin(l/180*np.pi)],\
                  [0, np.cos(b/180*np.pi), np.sin(b/180*np.pi)],
                 [0,0,0]])
        x,y,z = np.random.multivariate_normal((0,0,0),np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]]),len(l)).T
        x = x*sx - U
        y = y*sy - v_drift - V
        z = z*sz - W
        return mv3(A.T,np.array([x,y,z,0]).T)
    
    def generate_pm(t0_sim, l_sim, b_sim, factor_sim):
        v_drift_sim = WD_MCMC_func.velocity_scatter_3D(t0_sim, index_sim, v10_sim, vT_sim, v0_sim)**2/80
        sx_sim, sy_sim, sz_sim = sigma_v(t0_sim, index_sim, v10_sim, vT_sim, v0_sim)
        vL_sim,vB_sim,_,_ = LB_sampling(l_sim,b_sim,sx_sim,sy_sim,sz_sim,v_drift_sim,U_sim,V_sim,W_sim)
        pml_sim, pmb_sim = vL_sim/factor_sim, vB_sim/factor_sim
        return pml_sim, pmb_sim
    
    #----------------------------------------------------------------------------------------------------------
    
    pml_sim_Q, pmb_sim_Q = generate_pm(t0_sim_Q, l_sim_Q, b_sim_Q, factor_sim_Q)
    pml_sim, pmb_sim = generate_pm(t0_sim, l_sim, b_sim, factor_sim)
    selection = (np.abs(pml_sim) * factor_sim < 200) * (np.abs(pml_sim) * factor_sim < 200) * ((age_sim) > 0.1)
    selection_Q = (np.abs(pml_sim_Q) * factor_sim_Q < 200) * (np.abs(pml_sim_Q) * factor_sim_Q < 200) *\
    ((age_sim_Q) > 0.1)
    
    mass, age, pml, pmb, factor, l, b, mass_Q, age_Q, pml_Q, pmb_Q, factor_Q, l_Q, b_Q = \
            mass_sim[selection], age_sim[selection], pml_sim[selection], pmb_sim[selection],\
            factor_sim[selection], l_sim[selection], b_sim[selection], \
            mass_sim_Q[selection_Q], age_sim_Q[selection_Q], pml_sim_Q[selection_Q], pmb_sim_Q[selection_Q], \
            factor_sim_Q[selection_Q], l_sim_Q[selection_Q], b_sim_Q[selection_Q] 
    
    v_drift = (((age+0.1)/10.1)**0.2*40)**2/80
    v_drift_Q = (((age_Q+0.1)/10.1)**0.2*40)**2/80
    vL, vB = np.array(WD_MCMC_func.get_v_delayed_3D(age, l, b, pml, pmb, factor, v_drift,
                                                U_sim,V_sim,W_sim))
    vL_Q, vB_Q = np.array(WD_MCMC_func.get_v_delayed_3D(age_Q, l_Q, b_Q, pml_Q, pmb_Q, factor_Q, 
                                                    v_drift_Q,U_sim,V_sim,W_sim))
    
    
    
    
    
    with Pool(processes=agents) as pool:
        result = pool.map(parallel, np.arange(number), chunksize)  
    number = agents
    para_v = np.empty((number*nwalkers*(then_run//gap),Nv))
    para_Q = np.empty((number*nwalkers*(then_run//gap),NQ))
    for i in range(number):
        para_v[(i*nwalkers*(then_run//gap)):((i+1)*nwalkers*(then_run//gap))] = result[i][:,:Nv]
        para_Q[(i*nwalkers*(then_run//gap)):((i+1)*nwalkers*(then_run//gap))] = result[i][:,Nv:]
    para_v = para_v.reshape(agents,nwalkers,then_run//gap,Nv).transpose((2,1,0,3))\
                                            .reshape(number*nwalkers*(then_run//gap),Nv)
    para_Q = para_Q.reshape(agents,nwalkers,then_run//gap,NQ).transpose((2,1,0,3))\
                                            .reshape(number*nwalkers*(then_run//gap),NQ)

    #---------------------------------------------------------------------------------------------------------------------------
    
    def PDF_simulation(parameter_index,parameter_min,parameter_max,parameter_number=20,parameter_name='default'):
        x_list = np.linspace(parameter_min,parameter_max,parameter_number)
        pdf_sim = np.empty_like(x_list)
        pdf_Q_sim = np.empty_like(x_list)
        pdf_e_sim = np.empty_like(x_list)
        pdf_l_sim = np.empty_like(x_list)
        
        for i,x in enumerate(x_list):
            para = para_input.copy()
            para[parameter_index] = x
            pdf_sim[i], temp1, temp2, temp = WD_MCMC_func.ln_likelihood_pheno(para, mass, age, pml, pmb, factor, l, b,
                                           mass_Q, age_Q, pml_Q, pmb_Q, factor_Q, l_Q, b_Q,
                                                   False,
                                                   not_fit_UVW=NOT_FIT_UVW,not_fit_index=NOT_FIT_INDEX,fixv=FIXV)
            pdf_e_sim[i] = temp1[~np.isnan(temp1)].sum()
            pdf_l_sim[i] = temp2[~np.isnan(temp2)].sum()
            pdf_Q_sim[i] = temp[~np.isnan(temp)].sum()
            return [pdf_sim,pdf_e_sim,pdf_l_sim,pdf_Q_sim ]   

    delay_test = PDF_simulation()[pdf_sim,pdf_e_sim,pdf_l_sim,pdf_Q_sim ]   
   
    #---------------------------------------------------------------------------------------------------------------------------
    suffix = '.npy'
    np.save('/datascope/menard/group/scheng/Gaia/WD_vel_age_MCMC_Dec12/simulation_'+\
            str(mass_min)+'_'+str(distance2)+'_'+str(age_lim)+'_'+\
            spec_type+'_'+model+'_SFR'+str(SFR_TYPE)+'_'+str(end_of_SF)+'_T'+str(age_T)+'_delay'+\
            str(DELAY)+'_'+sys.argv[4]+sys.argv[5]+sys.argv[6]+test_number+suffix, \
            np.array([{'para_Q':para_Q, 'para_v':para_v, 'para_input':para_input, 'SFR_TYPE':SFR_TYPE,\
                      'delay_test':delay_test, 'SFR_step':SFR_step, 'Qfraction':Qfraction, \
                       'simulated_data':[mass, age, pml, pmb, factor, l, b,
                                       mass_Q, age_Q, pml_Q, pmb_Q, factor_Q, l_Q, b_Q],
                      'data_length':[selection.sum(),selection_Q.sum()],
                      'selection_para':SELECTION_PARA}]) )

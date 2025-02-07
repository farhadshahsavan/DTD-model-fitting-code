# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 14:34:42 2022

@author: Farhad
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 10:26:59 2021

@author: Farhad
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from scipy.integrate import quadrature, quad
from datetime import datetime
import math


def pre(Dres, t0):
    return 0.2*(np.pi**2 * Dres**2)*(t0**2)
def relaxation (x, T):
    return np.exp(-(x/T))
def exp_term_DQ (x_dq, Dres, t0, shift_factor):
    return np.exp(-2* pre (Dres, t0)* (np.exp(-x_dq*shift_factor/t0) +x_dq*shift_factor/t0 -1))
def sinh_term_DQ (x_dq, Dres, t0, shift_factor):
    return np.sinh(pre (Dres, t0)* (np.exp(-2*x_dq*shift_factor/t0)-2*np.exp(-x_dq*shift_factor/t0)+1))
def exp_term_MQ(x_mq, Dres, t0, shift_factor):
    return np.exp(-pre (Dres, t0)* (4* np.exp(-x_mq*shift_factor/t0)- np.exp(-2*x_mq*shift_factor/t0)+ 2* x_mq*shift_factor/t0- 3 ))
def inside_exp_term_dq (x_dq, Dres, t0):
    return -2 * pre (Dres, t0)* (np.exp(-x_dq/t0) +x_dq/t0 -1)
def inside_sinh_term_dq (x_dq, Dres, t0):
    return pre (Dres, t0)* (np.exp(-2* x_dq/t0) -2*np.exp(-x_dq/t0)+ 1)
def inside_exp_term_mq (x_mq, Dres, t0):
    return -pre (Dres, t0)* (4* np.exp(-x_mq/t0)- np.exp(-2*x_mq/t0)+ 2* x_mq/t0- 3 )
def residue (dat, fit_dat): 
    return (dat - fit_dat)
def R_square (dat, fit_dat):
    return  1-(np.sum(residue(dat, fit_dat)**2) / np.sum(((dat-np.mean(dat))**2)))
def chi_squared (expec_val,comboY):
    return np.sum((comboY-expec_val)**2/abs(expec_val))

def data_preparation() :    
    p = 1
    for file in os.listdir(loc):
        if file.endswith(".txt"):
            inputaddress = os.path.join(loc, file)
            namefile = os.path.join(file)
            namefile = namefile[:-4]
            temp[p] = int(namefile) #each file should named the associated teperature
            shift_factor[p] = 10**(c1*temp[p]/(c2 + temp[p])) #rheological shift factor based on WLF
            # READING THE TEXT FILE AND COPY IT TO DATA
            data1 = np.loadtxt(inputaddress)
            num_rows, num_cols = data1.shape
            maxim = np.where(data1[:,2] == np.amax(data1[:,2]))
            boundry[p] = int(maxim[0]) + 0
            length_MQ[p] = num_rows
            data[p,0:num_rows,0] = np.round(data1[:,0],4)
            data[p,0:num_rows,1] = (data1[:,1] + data1[:,2])/(data1[0,1] + data1[0,2])
            data[p,0:num_rows,2] = data1[:,2]/(data1[0,1] + data1[0,2])
            
            p=p+1
    return p-1
def print_result(fittedParameters,R2_DQ,  R2_MQ, chi_2, outputadress):
    f = open(outputadress + "\Awexp-RESULT-prediction.txt","a")
    print("#------------------------------------------------------------------------------", file = f)
    print("#fitting at", start_time, file= f)
    print("#alpha = ", alpha,"    ","#NMR_alpha = ", NMR_alpha,"    ", "#p2 = ", p**2,"    ", "#root =", root,"    ", "#n_segments =", n_tau, file= f)
    print("Dres= ", round(fittedParameters[0],3), '\n', "Me= ", round(fittedParameters[1],3), '\n', "tau_e= ", round(fittedParameters[2],6), '\n', file= f)
    #print("fi_bb ", round(fittedParameters[2]/(fittedParameters[2]+ fittedParameters[3]*fittedParameters[1]),3), '\n', "fi_a ", round(fittedParameters[3]*fittedParameters[1]/(fittedParameters[2]+ fittedParameters[3]*fittedParameters[1]),3), '\n', file= f)
    for i in range (1, n+1):
        print("f1_", str(i), "=", round(f1[i], 2),'\n', file=f)
        print("T1_", str(i), "=", round(fittedParameters[2+i],1), "      ", "T2_", str(i), "=", round(fittedParameters[2+n+i],1),'\n', file=f)

        for i in range (1, n+1):
        print ("R2_DQ_" +str(i)+ "_"+ str(temp[i]) +"C=", round(R2_DQ[i],5), '\t', "R2_MQ_" +str(i) + "_"+ str(temp[i]) +"C=", round(R2_MQ[i],5), '\t', "chi-squared_" +str(i) + "_"+ str(temp[i]) +"C=", round(chi_2[i],6), file=f)
    f.close()
def plot_result ( yy, outputadress):
    pointer = 0
    for i in range(1,n+1):
        boundryliney = [0, 1]
        boundrylinex = [data[i,boundry[i],0], data[i,boundry[i],0]]
        plt.plot(data[i,0:length_MQ[i],0],data[i,0:length_MQ[i],1], 'k.', data[i,1:length_MQ[i],0],data[i,1:length_MQ[i],2],'r.', data[i,1:length_MQ[i],0], yy[pointer:pointer+length_MQ[i]-1],'c-.', data[i,1:length_MQ[i],0],yy[pointer+length_MQ[i]-1:pointer+ 2*(length_MQ[i]-1)],'c-.',boundrylinex, boundryliney, 'y:')
        pointer = pointer + 2*(length_MQ[i]-1)
        plt.rcParams["legend.fontsize"]= 10
        plt.gca().legend(('MQ','DQ', 'simultaneous fitting-dilution theory'))
        plt.ylim(0.001, 1)
        plt.ylabel('Intensity', fontsize= 8)
        plt.xlabel('DQ-time evolution(ms)', fontsize= 8)
        plt.yscale('log')
        # ADJUST SAVE FIGURE FILE
        plt.savefig(outputadress +"/" + str(temp[i]) + "C-AW-arm-retraction.png", dpi=600)
        #plt.show()
        plt.clf()

def integration_opt_fi1 (x, b):
    global upper_lim_1 , cache_1, doll
    if upper_lim_1 > b :
        cache_1 =0
        upper_lim_1 = 0
    extr = quad(x, upper_lim_1, b)[0]
    num_integral = cache_1 + extr
    cache_1 = num_integral
    upper_lim_1 = b
    
    return cache_1
def integration_opt_fi2 (x, b):
    global upper_lim_2 , cache_2, doll
    if upper_lim_2 > b :
        cache_2 =0
        upper_lim_2 = 0
    num_integral = cache_2 + quad(x, upper_lim_2, b)[0]
    cache_2 = num_integral
    upper_lim_2 = b
    return cache_2

def combine_data (data_c, s):
    if s == 0 :
        return np.array([]), np.array([]) 
    else:
        return np.concatenate((combine_data (data_c, s-1)[0] ,data_c[s,1:boundry[s],0], data_c[s,1:length_MQ[s],0])), np.concatenate((combine_data (data_c, s-1)[1] ,data_c[s,1:boundry[s],2], data_c[s,1:length_MQ[s],1]))

def exp_fun_arm (t,m,tau):   
    if m == 1 :
        return np.exp(-t/tau[0])
    else:
        return exp_fun_arm (t,m-1, tau) + np.exp(-t/tau[m-1])
def U_bb_xb(x_b):
    return (1-((1-x_b)**(alpha+1) * (1+(1+alpha)*x_b)))
def diff_1_Ub (x_b, fi_bb, s_b):
    c = 15 *(-((1 + alpha)* (1 - x_b)**(1 + alpha)) + (1 + alpha)* (1 - x_b)**alpha *(1 + (1 + alpha)* x_b)) * fi_bb**alpha * s_b / (8* (1 + alpha)* (2 + alpha))
    return c
def diff_2_Ub(x_b, fi_bb, s_b):
    c = 15 *(2 *(1 + alpha)**2 *(1 - x_b)**alpha - alpha* (1 + alpha)* (1 - x_b)**(-1 + alpha) *(1 + (1 + alpha)* x_b))* fi_bb**alpha * s_b/ (8* (1 + alpha)* (2 + alpha))
    return c
def rett_bb(x_b, tau_a_long, p, q, s_b, fi_bb):
    
    tau_bb_early = 375/8192 *(np.pi/(p**2)) * q * s_b**3 * x_b**4 *tau_a_long * fi_bb**(3*alpha)
    U_b = (15*s_b*(fi_bb**alpha)/(8*(1+alpha)*(2+alpha))) * U_bb_xb(x_b)
    tau_bb_late = 25 * s_b**2 * fi_bb**(2*alpha) * q * tau_a_long * np.exp(U_b)* ((2*np.pi/(diff_2_Ub(0, fi_bb, s_b)))**0.5)/(8*p**2 * diff_1_Ub (x_b, fi_bb, s_b))
    return (tau_bb_early * np.exp(U_b))/(1+(tau_bb_early * np.exp(U_b)/tau_bb_late))
def frac_unrelaxed_segment(t, tau_rep, x_c, fi_a, fi_bb):
    unrelaxed_fraction = fi_a * relaxed_segments_counter (t, tau_arm[1:])/n_tau + fi_bb*(relaxed_segments_counter (t, tau_s[1:rep_treshold])/n_tau + (1-x_c))#* rep_tube_survival(t, tau_rep))
    return unrelaxed_fraction
def frac_unrelaxed_segment_final(t, tau_rep, x_c, fi_a, fi_bb, rep_treshold):
    unrelaxed_fraction = fi_a * relaxed_segments_counter (t, tau_a[1:])/n_tau + fi_bb*(relaxed_segments_counter (t, tau_clf_bb[1:rep_treshold])/n_tau + (1-x_c))#* rep_tube_survival(t, tau_rep))
    return unrelaxed_fraction
def relaxed_segments_counter (t, tau):
    count = np.exp(-t/tau)
    return sum(count)
def rep_tube_survival(t, tau_rep):
    tube_survival = 0
    for i in range(count_rep):
        pp= 2*i+1
        tube_survival = tube_survival + (8/(np.pi**2 * (pp)**2))* relaxation (t, tau_rep/(pp**2))
    return tube_survival
        
def integration(x_mq, Dres, tau_d, tau_rep, x_c, fi_a, fi_bb):
    integrand_fi_1 = lambda t: frac_unrelaxed_segment(t, tau_rep, x_c, fi_a, fi_bb)**NMR_alpha * 0.2* (np.pi**2)* (Dres**2 * np.exp(-t/tau_d)) 
    integrand_fi_2 = lambda t: frac_unrelaxed_segment(t, tau_rep, x_c, fi_a, fi_bb)**NMR_alpha * t* 0.2* (np.pi**2)* (Dres**2 * np.exp(-t/tau_d))
    integrand_fi1_fi2_2 = lambda t: frac_unrelaxed_segment(t, tau_rep, x_c, fi_a, fi_bb)**NMR_alpha * (2*x_mq-t)* 0.2* (np.pi**2)* (Dres**2 * np.exp(-t/tau_d))
    f_1 = integration_opt_fi1 (integrand_fi_1, x_mq)
    f_2 = integration_opt_fi2 (integrand_fi_2, x_mq)
    
    return 2*(x_mq*f_1 - f_2), (f_2 + quad(integrand_fi1_fi2_2, x_mq, 2*x_mq)[0])
def integral_fit (comboX, *par):
    #Par[i], 0=Dres,1=Me, 2=Te, 3=q, 4=T1,5=T2, 6=T2 7=T3 ,8=T4, 9=T5, 10= T6, 9=T7, 10 = T2_1, 11= T2_2
    #x_dq = comboX[:boundry] # first data
    #x_mq = time[1:length_MQ[1]] # second data
    s_a = np.zeros(n_tau+1)
    x_bb = np.zeros(n_tau+1)
    for i in range (1, n_tau+1):
        s_a[i] = i/n_tau
        x_bb[i] = i/n_tau
    fit_sum = np.array([])
    Me = par[1]
    tau_e = par[2]
    Z_a = M_a/Me
    Z_bb = M_bb/Me
    fi_b = M_bb/(M_bb + q*M_a)
    fi_a = 1-fi_b
    fi_bb = fi_b
    tail_bb = 0
    U_a = (15/4)* Z_a* ((1-((1-fi_a*s_a)**(alpha+1))*(1+(1+alpha)*fi_a*s_a)))/(fi_a**2 * (alpha+1) * (alpha+2))
    tau_early = (225/256)*(np.pi**3) * tau_e* (Z_a* s_a)**4
    tau_late = (np.pi**5 *( 2/15))**0.5 * tau_e * Z_a**1.5* np.exp(U_a)/(s_a * (1-(fi_a*s_a))**alpha)
    tau_a = (tau_early * np.exp(U_a))/(1+(tau_early*np.exp(U_a)/tau_late))
    tau_a_longest = tau_a[n_tau]
    tau_clf_bb = rett_bb(x_bb, tau_a_longest, p , q, Z_bb, fi_bb)
    tau_reptation = (25/(8*np.pi**2 * p**2)) * (1-x_bb)**2 * Z_bb **2 * fi_bb**(2*alpha) * tau_a_longest * q

    for i in range (1,n+1):
        global upper_lim_1 , cache_1, upper_lim_2 , cache_2
        cache_1 =0
        upper_lim_1 = 0
        cache_2 =0
        upper_lim_2 = 0
        x_mq = data[i,1:length_MQ[i],0]
        fit_DQ = np.zeros(len(x_mq))
        fit_MQ = np.zeros(len(x_mq))
        fit_DQ_arm = np.zeros(len(x_mq))
        fit_MQ_arm = np.zeros(len(x_mq))
        fit_DQ_bb = np.zeros(len(x_mq))
        fit_MQ_bb = np.zeros(len(x_mq))
        fit_DQ_rep = np.zeros(len(x_mq))
        fit_MQ_rep = np.zeros(len(x_mq))
        border_bb = 0
        global tau_arm, tau_s, rep_treshold
        tau_arm = tau_a/shift_factor[i]
        tau_s = tau_clf_bb/ shift_factor[i]
        tau_rept = tau_reptation/ shift_factor[i]
        j=1
        ss = True
        while tau_arm[j] * tail_factor < x_mq[0] and ss:
            j+=1
            if j>n_tau:
                ss= False
                j-=1
        border_arm = j-1
        
        tt = 1
        while tau_s[tt] < tau_rept[tt]:
            if tau_s[tt] * tail_factor < x_mq[0]:
                border_bb = tt
            tt+=1
        x_c = (tt-1)/n_tau
        rep_treshold = tt
        tail = fi_a* ((border_arm/n_tau)) + (tail_bb)+ fi_bb* (border_bb/n_tau)
        tau_rep = tau_rept[tt]
        for j in range (1,n_tau+1):
            if tau_arm[j] * tail_factor > x_mq[0]:
                fi1_2 , fi1_fi2 = v_integration(x_mq, par[0], tau_arm[j], tau_rep, x_c, fi_a, fi_bb)
                fit_DQ_arm = fit_DQ_arm + np.sinh(fi1_fi2) * np.exp(-fi1_2)/n_tau
                fit_MQ_arm = fit_MQ_arm + (np.exp(fi1_fi2) * np.exp(-fi1_2))/n_tau

            if j < rep_treshold:
                if tau_s[j] * tail_factor > x_mq[0]:
                    fi1_2 , fi1_fi2 = v_integration(x_mq, par[0], tau_s[j], tau_rep, x_c, fi_a, fi_bb)
                    fit_DQ_bb = fit_DQ_bb + np.sinh(fi1_fi2) * np.exp(-fi1_2)/n_tau
                    fit_MQ_bb = fit_MQ_bb + (np.exp(fi1_fi2) * np.exp(-fi1_2))/n_tau

        fi1_2_sum = np.zeros(len(x_mq))
        fi1_fi2_sum = np.zeros(len(x_mq))
        for tt in range(count_rep):
            pp = 2*tt +1
            fi1_2 , fi1_fi2 = v_integration(x_mq, par[0], tau_rep/(pp**2), tau_rep, x_c, fi_a, fi_bb)
            fi1_2_sum = fi1_2_sum + (8/(np.pi**2 * (pp)**2))* fi1_2
            fi1_fi2_sum = fi1_fi2_sum + (8/(np.pi**2 * (pp)**2))* fi1_fi2
            
        fit_MQ_rep = (np.exp(fi1_fi2_sum) * np.exp(-fi1_2_sum ))
        fit_DQ_rep = np.sinh(fi1_fi2_sum) * np.exp(-fi1_2_sum )
        #tail = fi_a* ((border_arm/n_tau)) + (tail_bb)+ fi_bb* (border_bb/n_tau)
        f1[i] = 1-tail
        fit_DQ = (fit_DQ_arm * fi_a + fi_bb* (fit_DQ_bb + (1-x_c)* fit_DQ_rep))* relaxation(x_mq, par[2+i]) #* f1[i]
        fit_MQ = (fit_MQ_arm * fi_a + fi_bb* (fit_MQ_bb + (1-x_c)* fit_MQ_rep)) * relaxation(x_mq, par[2+i])  + (1-f1[i])* relaxation(x_mq, par[2+n+i])

        fit_sum = np.concatenate((fit_sum, fit_DQ[0:boundry[i]-1], fit_MQ))                                
    return fit_sum**(1/root)
def integral_fit_compelete (x_m, *par):
    #Par[i], 0=Dres,1=kappa,2=tau_d_0 3=f1,4=T1, 5=T2 6=T3 ,7=T4, 8=T5, 9= T6, 10=T7, 11 = T2_1, 12= T2_2
    #x_dq = comboX[:boundry] # first data
    #x_mq = comboX[boundry[0]-1:length_MQ[0]-1] # second data
    s_a = np.zeros(n_tau+1)
    x_bb = np.zeros(n_tau+1)
    for i in range (1, n_tau+1):
        s_a[i] = i/n_tau
        x_bb[i] = i/n_tau
    fit_sum = np.array([])
    Me = par[1]
    tau_e = par[2]
    #q = par[3]
    Z_a = M_a/Me
    Z_bb = M_bb/Me
    fi_b = M_bb/(M_bb + q*M_a)
    fi_a = 1-fi_b
    fi_bb = fi_b
    tail_bb = 0
    global tau_a, tau_clf_bb
    U_a = (15/4)* Z_a* ((1-((1-fi_a*s_a)**(alpha+1))*(1+(1+alpha)*fi_a*s_a)))/(fi_a**2 * (alpha+1) * (alpha+2))
    tau_early = (225/256)*(np.pi**3) * tau_e* (Z_a* s_a)**4
    tau_late = (np.pi**5 *( 2/15))**0.5 * tau_e * Z_a**1.5* np.exp(U_a)/(s_a * (1-(fi_a*s_a))**alpha)
    tau_a = (tau_early * np.exp(U_a))/(1+(tau_early*np.exp(U_a)/tau_late))
    tau_a_longest = tau_a[n_tau]
    tau_clf_bb = rett_bb(x_bb, tau_a_longest, p , q, Z_bb, fi_bb)
    tau_reptation = (25/(8*np.pi**2 * p**2)) * (1-x_bb)**2 * Z_bb **2 * fi_bb**(2*alpha) * tau_a_longest * q

    for i in range (1,n+1):
        global upper_lim_1 , cache_1, upper_lim_2 , cache_2
        cache_1 =0
        upper_lim_1 = 0
        cache_2 =0
        upper_lim_2 = 0
        x_mq = data[i,1:length_MQ[i],0]
        fit_DQ = np.zeros(len(x_mq))
        fit_MQ = np.zeros(len(x_mq))
        fit_DQ_arm = np.zeros(len(x_mq))
        fit_MQ_arm = np.zeros(len(x_mq))
        fit_DQ_bb = np.zeros(len(x_mq))
        fit_MQ_bb = np.zeros(len(x_mq))
        fit_DQ_rep = np.zeros(len(x_mq))
        fit_MQ_rep = np.zeros(len(x_mq))
        border_bb = 0

        global tau_arm, tau_s, rep_treshold
        tau_arm = tau_a/shift_factor[i]
        tau_s = tau_clf_bb/ shift_factor[i]
        tau_rept = tau_reptation/ shift_factor[i]
        j=1
        ss = True
        while tau_arm[j] * tail_factor < x_mq[0] and ss:
            j+=1
            if j>n_tau:
                ss= False
                j-=1
        border_arm = j-1
        
        tt = 1
        while tau_s[tt] < tau_rept[tt]:
            if tau_s[tt] * tail_factor < x_mq[0]:
                border_bb = tt
            tt+=1
        x_c = (tt-1)/n_tau
        rep_treshold = tt
        tail = fi_a* ((border_arm/n_tau)) + (tail_bb)+ fi_bb* (border_bb/n_tau)
        tau_rep = tau_rept[tt]
        for j in range (1,n_tau+1):
            if tau_arm[j] *tail_factor > x_mq[0]:
                fi1_2 , fi1_fi2 = v_integration(x_mq, par[0], tau_arm[j], tau_rep, x_c, fi_a, fi_bb)
                fit_DQ_arm = fit_DQ_arm + np.sinh(fi1_fi2) * np.exp(-fi1_2)/n_tau
                fit_MQ_arm = fit_MQ_arm + (np.exp(fi1_fi2) * np.exp(-fi1_2))/n_tau

            if j < rep_treshold:
                if tau_s[j] * tail_factor > x_mq[0]:
                    fi1_2 , fi1_fi2 = v_integration(x_mq, par[0], tau_s[j], tau_rep, x_c, fi_a, fi_bb)
                    fit_DQ_bb = fit_DQ_bb + np.sinh(fi1_fi2) * np.exp(-fi1_2)/n_tau
                    fit_MQ_bb = fit_MQ_bb + (np.exp(fi1_fi2) * np.exp(-fi1_2))/n_tau

        fi1_2_sum = np.zeros(len(x_mq))
        fi1_fi2_sum = np.zeros(len(x_mq))
        for tt in range(count_rep):
            pp = 2*tt +1
            fi1_2 , fi1_fi2 = v_integration(x_mq, par[0], tau_rep/(pp**2), tau_rep, x_c, fi_a, fi_bb)
            fi1_2_sum = fi1_2_sum + (8/(np.pi**2 * (pp)**2))* fi1_2
            fi1_fi2_sum = fi1_fi2_sum + (8/(np.pi**2 * (pp)**2))* fi1_fi2
            
        fit_MQ_rep = (np.exp(fi1_fi2_sum) * np.exp(-fi1_2_sum ))
        fit_DQ_rep = np.sinh(fi1_fi2_sum) * np.exp(-fi1_2_sum )
        f1[i] = 1-tail
        fit_DQ = (fit_DQ_arm * fi_a + fi_bb* (fit_DQ_bb + (1-x_c)* fit_DQ_rep))* relaxation(x_mq, par[2+i]) #* f1[i]
        fit_MQ = (fit_MQ_arm * fi_a + fi_bb* (fit_MQ_bb + (1-x_c)* fit_MQ_rep)) * relaxation(x_mq, par[2+i])  + (1-f1[i])* relaxation(x_mq, par[2+n+i])
        fit_sum = np.concatenate((fit_sum, fit_DQ, fit_MQ))
        fit_DQ_arm_report = (fit_DQ_arm * fi_a )* relaxation(x_mq, par[2+i])
        fit_DQ_bb_report = fi_bb* (fit_DQ_bb + (1-x_c)* fit_DQ_rep)* relaxation(x_mq, par[2+i]) 
        f = open(outputadress + "\\" + str(temp[i])+ "C-fitdata.txt","w")
        g = open(outputadress + "\\" + str(temp[i])+ "C-fit-arm and backbone contribution(first am-secon bb).txt","w")
        h = open(outputadress + "\\relaxation spectrum at 0C-arm-CLFbb-reptation.txt","w")
        for j in range (1 ,length_MQ[i]):
            print(x_mq[j-1], '     ',data[i,j,1], '     ' , data[i,j,2], '     ' , fit_MQ[j-1], '     ' , fit_DQ[j-1], '     ' , data[i,boundry[i],0], file=f)
            print(x_mq[j-1], '     ' , fit_DQ_arm_report [j-1] , '     ',fit_DQ_bb_report [j-1], file=g)
        f.close()
        g.close()
    tau_a = tau_a / 1000
    tau_clf_bb = tau_clf_bb /1000
    tau_reptation = tau_reptation /1000
    for j in range (1,n_tau+1):
        print(j, '     ' , tau_a [j] , '     ',tau_clf_bb[j], '     ',tau_reptation[j], file=h)
    h.close()

    time_correlation = np.logspace (-8, 4, 60)
    fi_unrelaxed = v_frac_unrelaxed_segment_final(time_correlation , tau_reptation[rep_treshold], x_c, fi_a, fi_bb, rep_treshold)
    corr_address_arm = outputadress + "\\correlation functions_arm" 
    corr_address_bb = outputadress + "\\correlation functions_bb"
    corr_fun_bb_rept = np.zeros (len(time_correlation))
    if not os.path.exists(corr_address_arm):
        os.makedirs(corr_address_arm)
    if not os.path.exists(corr_address_bb):
        os.makedirs(corr_address_bb)
    for j in range(1, n_tau+1):
        corr_fun_arm = fi_unrelaxed**NMR_alpha * np.exp(-time_correlation/tau_a[j])
        f = open(corr_address_arm + "\\arm_Seg_No_" + str(j)+ ".txt","w")
        for cc in range (len(time_correlation)):
            print(time_correlation[cc], '     ',corr_fun_arm[cc], file=f)
        if j <= rep_treshold:            
            if j < rep_treshold:
                g = open(corr_address_bb + "\\clf_bb_Seg_No_" + str(j)+ ".txt","w")
                corr_fun_bb_clf = fi_unrelaxed**NMR_alpha * np.exp(-time_correlation/tau_clf_bb[j])
                for t in range(len(time_correlation)):
                    print(time_correlation[t], '     ',corr_fun_bb_clf[t], file=g)

            else:
                g = open(corr_address_bb + "\\rept_bb_Seg_No_" + str(j)+ ".txt","w")
                for tt in range(count_rep):
                    pp = 2*tt+1
                    corr_fun_bb_rept = corr_fun_bb_rept + fi_unrelaxed**NMR_alpha * (8/(np.pi**2 * (pp)**2))* np.exp(-time_correlation/(tau_reptation[rep_treshold]/(pp)**2))
                for t in range(len(time_correlation)):
                    print(time_correlation[t], '     ',corr_fun_bb_rept[t], file=g)
            g.close()
        f.close()
    init_range = x_mq[0]* shift_factor[1]/1000
    upper_range = x_mq[len(x_mq)-1] * shift_factor [n]/1000
    f = open(outputadress + "\\NMR-covered range.txt","w")
    print("initial range=", init_range, file=f)
    print("end range=", upper_range, file=f)
    f.close()

    return fit_sum
def fitting_scipy(comboX, comboY,initialParameters, outputadress):
    #initialParameters = np.array([0.1, 2, 0.2])
    bnd= [[0.00001, 100, 0.000001, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],[2, 10000, 2000, "inf", "inf", "inf", "inf", "inf", "inf", "inf", "inf", "inf", "inf", "inf", "inf", "inf", "inf" ]]
    # curve fit the combined data to the combined function
    fittedParameters, pcov = curve_fit(integral_fit, comboX, comboY, initialParameters, bounds= bnd)
    standard_dev = np.sqrt(np.diag(pcov)) # compute one standard deviation errors on the parameters
    return fittedParameters

def AWEXP_fit (location):
    global outputadress
    outputadress = location + "\hierarchicalmodel-timedependent-Sb-determine-Me-taue-numerical-fulldata"
    if not os.path.exists(outputadress):
        os.makedirs(outputadress)
    # ASSIGN DIRECTORY FOR SAVING THE FIGURE
    combo = combine_data (data,n)
    comboX = combo[0]
    comboY = combo[1]
    initial_par= np.array([0.27, 4700, 0.11, 4 , 5 ,7 , 9, 9 , 10, 11, 8, 10, 13, 14, 15, 16, 17])

    results = fitting_scipy(comboX, abs(comboY) **(1/root), initial_par, outputadress)
    y_r =(integral_fit(comboX, *results) )**root
    R2_DQ = np.zeros(n+1)
    R2_MQ = np.zeros(n+1)
    chi_2 = np.zeros(n+1)
    pointer = 0
    for i in range (1,n+1):
        R2_DQ[i] = R_square ((comboY[pointer:pointer + boundry[i]-1]), y_r[pointer: pointer + boundry[i]-1])
        R2_MQ[i] = R_square ((comboY[pointer + boundry[i]-1:pointer + boundry[i]-1 + length_MQ[i]-1]), y_r[pointer + boundry[i]-1:pointer + boundry[i]-1 + length_MQ[i]-1])
        chi_2[i] = chi_squared (y_r[pointer: pointer + boundry[i]-1 + length_MQ[i]-1],comboY[pointer: pointer + boundry[i]-1 + length_MQ[i]-1])
        pointer = pointer + boundry[i]-1 + length_MQ[i]-1
    
    print_result(results,R2_DQ,  R2_MQ, chi_2, outputadress)
    #comboX_compelete = time
    y_r_compelete =(integral_fit_compelete(data[1,1:length_MQ[1], 0], *results) )
    plot_result (y_r_compelete, outputadress)
    # PRINT THE RESULTS AND SAVING IT TO A FILE IN OUTADDRESS. 'a' is reffered to appending the file. it means it add new results to previous ones. if you want to delete previous you shod use 'w'

start_time = datetime.now()
M_0 = 68.1
M_a = 13800
M_bb = 68000

alpha = 1 #0.17
alpha_new = 0 #5/3
NMR_alpha= 1
p = (40)**-0.5

q = 8.5
#f: number of arms attached to a branch point
f = 1
count_rep =7
tail_factor = 3
v_frac_unrelaxed_segment_final = np.vectorize(frac_unrelaxed_segment_final)
v_integration = np.vectorize(integration)
v_integral_fit = np.vectorize(integral_fit)
v_integration_opt_fi1 = np.vectorize(integration_opt_fi1)
v_integration_opt_fi2 = np.vectorize(integration_opt_fi2)

n_tau = 50  #number of points for modelling distribution of terminal relaxation time
root = 1
c1 = 6.14015 #WLF constant
c2 = 114.791 #WLF constant
temp = np.zeros(10)
files = 10 # number of text file in the folder
shift_factor = np.zeros(10)
data_real = np.zeros(shape = (files,100,3))
data = np.zeros(shape = (files,100,3))
boundry =  np.zeros(10, dtype= int)
length_MQ =  np.zeros(10, dtype=int)
firstDres = 0.2
firstf1 = 0.5
firstT1 = 4
firstT2 = 25
#firstT3 = 90
fittingrange = 33
root_c= np.array([2, 1])
NMR_alpha_check = np.array([2, 1])
p_friction = np.array([40**-0.5, 12**-0.5])
alpha_doon = np.array([1, 4/3])
loc = input("please give me the address of the folder which contains your data")
n = data_preparation()
#print (shift_factor)
f1 = np.zeros(n+1)
time = data[1,0:length_MQ[1],0]
t0 = time[1]-0.00001
t0 = t0* shift_factor[1]
for root_check in root_c:
    for NMR_exp in NMR_alpha_check:
        for cc in range (2):
            start_time = datetime.now()
            root = root_check
            NMR_alpha= NMR_exp
            alpha = alpha_doon[cc]
            p = p_friction[cc]
            p2_loc = round(p,2)
            print(root)
            print(NMR_alpha)
            NMR_alpha_loc = round(NMR_alpha, 2)
            alpha_loc = round(alpha, 2)
            location = loc +"\\"+ "test" +"root-" + str(root)+ "-alpha-NMR-" + str(NMR_alpha_loc)+ "-alpha-" + str(alpha_loc)+ "-P-" + str(p2_loc) 
            AWEXP_fit (location)
            end_time = datetime.now()
            print('Duration: {}'.format(end_time - start_time))
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
       
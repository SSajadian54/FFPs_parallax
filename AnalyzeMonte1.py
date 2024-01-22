import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import warnings
warnings.filterwarnings("ignore")
rcParams["font.size"] = 13
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
threA=15000.0
from scipy.ndimage.filters import gaussian_filter
import pandas as pd 
import seaborn as sns
################################################################################

def Asymmet(icon, ndw):
    f1=open("./files/MONT/datA{0:d}.dat".format(icon),"r")
    nd= sum(1 for line in f1)  
    dat=np.zeros((nd,9)) 
    dat=np.loadtxt("./files/MONT/datA{0:d}.dat".format(icon))
    if(nd!=ndw): 
        print("Big error data_number: ", nd, ndw, icon)
        input("Enter a number ")
    tt0=int(np.argmax(dat[:,3]))#magnification
    asym=0.0; 
    conn=0.000000000674531
    for j in range(nd): 
        n1=int(tt0-j-1);
        n2=int(tt0+j+1);  
        if(n2<nd and n1>=0 and dat[n1,3]>1.05 and dat[n2,3]>1.05):##
            errA =float(abs(dat[n1,4])+abs(dat[n2,4]))*0.5
            asym+=float(dat[n2,3]-dat[n1,3])**2.0/errA**2.0
            conn+=1.0
    return( np.sqrt(asym/conn) )
    
################################################################################

nam0=[r"$\rm{counter}$", r"$\xi-\phi_{0}(\rm{deg})$", r"$\log_{10}[\pi_{\rm{rel}}(\rm{mas})]$", #3
     r"$\rm{Structure}_{\rm l}$", r"$\log_{10}[M_{\rm l}(M_{\oplus})]$", r"$D_{\rm l}(\rm{kpc})$", r"$v_{\rm{l}}(\rm{km}/s)$",#7
     r"$\rm{Structure}_{\star}$", r"$\rm{CL}$", r"$M_{\star}(M_{\odot})$", r"$D_{\rm{s}}(\rm{kpc})$", 
     r"$\log_{10}[T_{\star}(\rm{K})]$", r"$R_{\star}(R_{\odot})$", r"$\log_{10}[L/L_{\odot}]$",#14 
     r"$\rm{Type}_{\star}$", r"$\rm{color}(\rm{mag})$", r"$v_{\rm{s}}(\rm{km}/s)$",
     r"$M_{I}(\rm{mag})$", r"$M_{W149}(\rm{mag})$", r"$m_{I}(\rm{mag})$", r"$m_{W149}(\rm{mag})$", #21
     r"$m_{\rm{base}, I}(\rm{mag})$", r"$m_{\rm{base}, W149}(\rm{mag})$", r"$f_{\rm{b}, I}$", r"$f_{\rm{b}, W149}$", 
     r"$N_{\rm{b}, I}$", r"$N_{\rm{b}, W149}$", r"$A_{I}(\rm{mag})$", r"$A_{W149}(\rm{mag})$",#29
     r"$\log_{10}[t_{\rm E}(\rm{days})]$",r"$R_{\rm E}(\rm{au})$", r"$t_{0}(\rm{days})$", 
     r"$\mu_{\rm{rel}}(\rm{mas}/\rm{yrs})$", r"$v_{\rm{rel}}(\rm{km}/s)$", r"$u_{0}$", r"$\tau(\times1.0e6)$", 
     r"$\log_{10}[\rho_{\star}]$", r"$\log_{10}[\theta_{\rm E}(\rm{mas})]$", #38
     r"$\rm{Flag}_{file}$", r"$\rm{Flag_{data}}$", r"$\log_{10}[\Delta\chi^{2}_{\rm{lensing}}]$", r"$\rm{No.}_{\rm{data}}$", 
     r"$\rm{los}$", r"$\mu_{\star,~n1}(\rm{mas}/\rm{yrs})$", r"$\mu_{\star,~n2}(\rm{mas}/ \rm{yrs})$",##45 
     r"$\xi(\rm{deg})$", r"$\mu_{\rm{l},~n1}(\rm{mas}/ \rm{yrs})$", r"$\mu_{\rm{l},~n2}(\rm{mas}/ \rm{yrs})$", 
     r"$\log_{10}[\pi_{\rm E}]$", r"$Amp_{\rm{P}}$", r"$\delta_{\rm{P}}$",r"$Amp_{\rm{A}}$", 
     r"$\delta_{\rm{A}}$", r"$\chi^{2}_{\rm{real}}$", r"$\chi^{2}_{\rm{no}-\rm{parallax}}$", r"$\chi^{2}_{\rm{base}}$", 
     r"$\phi_{0}\rm{(deg})$"]##57 

nam1= [r"$\log_{10}[\delta t_{\rm E}]$", r"$\log_{10}[\delta \rho_{\star}]$", r"$\log_{10}[\delta u_{0}]$",r"$\log_{10}[\delta t_{0}]$",r"$\log_{10}[\delta f_{\rm b}]$",r"$\log_{10}[\mathcal{A}]$"]


nam2= [r"$\log_{10}[M_{\rm l}(M_{\oplus})]$", r"$D_{\rm l}(\rm{kpc})$", r"$D_{\rm s}(\rm{kpc})$", 
      r"$\log_{10}[t_{\rm E}(days)]$",r"$m_{\rm{base}}(\rm{mag})$", r"$f_{\rm b}$", r"$u_{0}$", 
      r"$\xi(\rm{deg})$", r"$\phi_{0}(\rm{deg})$", r"$\log_{10}[\mu_{\rm{rel}}(\rm{mas}/\rm{yrs})]$", 
      r"$\log_{10}[\pi_{\rm E}]$", r"$\log_{10}[\pi_{\rm{rel}}(\rm{mas})]$",r"$\log_{10}[\theta_{\rm E}(\rm{mas})]$",  
      r"$\log_{10}[\rho_{\star}]$"]
      
nam3=[r"$\log_{10}[M_{\rm{l}}(M_{\oplus})]$", r"$D_{\rm{l}}(\rm{kpc})$", r"$\log_{10}[t_{\rm{E}}(\rm{days})$",  
      r"$\log_{10}[\pi_{\rm E}]$", r"$|\xi-\phi_{0}|(\rm{deg})$", r"$\log_{10}[\pi_{\rm{rel}}(\rm{mas})]$", 
      r"$m_{\rm{base}}(\rm{mag})$", r"$u_{0}$", r"$\log_{10}[\theta_{\rm{E}}]$"]
             
nam4=[r"$\log_{10}[\delta t_{\rm E}]$", r"$\log_{10}[\delta \rho_{\star}]$", r"$\log_{10}[\delta u_{0}]$",
      r"$\log_{10}[\delta t_{0}]$",r"$\log_{10}[\delta f_{\rm b}]$", r"$\log_{10}[M_{\rm l}]$", 
      r"$D_{\rm l}$", r"$\log_{10}[t_{\rm E}]$",r"$m_{\rm{base}}$",  
      r"$\log_{10}[\pi_{\rm E}]$", r"$\log_{10}[\pi_{\rm{rel}}]$",
      r"$\log_{10}[\theta_{\rm E}]$",r"$\log_{10}[\rho_{\star}]$"]             
             
################################################################################
'''
f1=open("./BestFFPst.txt","r")
nb=sum(1 for line in f1)
bes=np.zeros(( nb,19 ))
bes=np.loadtxt("./BestFFPst.txt")
u, ind = np.unique(bes[:,0],return_index=True)
print ("number of best array:  ",  nb,  len(ind) )
fd=open("./BestFFPstC.txt","w")
fd.close()
for i in range( len(ind) ):
    fd=open("./BestFFPstC.txt","a")  
    np.savetxt(fd,bes[int(ind[i]),:].reshape(-1,19),fmt='%d  %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f  %.5f %.5f %.5f %.5f %.5f %.5f %.5f  %.5f %.5f  %d')
    fd.close()
'''
f1=open("./Besttot.txt","r")
nb=sum(1 for line in f1)    
#nb=len(ind)    
bes=np.zeros((nb,19 ))
bes=np.loadtxt("./Besttot.txt")

################################################################################

f0=open("./files/MONT/FFP_Roman1t.dat","r")
nr0=sum(1 for line in f0)
par0=np.zeros(( nr0 , 57 ))
par=np.zeros(( nr0-95 , 57 ))
par0=np.loadtxt("./files/MONT/FFP_Roman1t.dat") 
nr=0
for i in range(nr0):
    if(par0[i,29]<10.0):
        par[nr,:]=par0[i,:]
        nr+=1 
        
print("number of net events with tE<10 days:  ",  nr)
print ("Long_tE events:     ",    nr*100.0/nr0,  nr, nr0)         


par[:,4]=np.log10(par[:,4])  # Log Ml
par[:,11]=np.log10(par[:,11])# Log Tstar
par[:,29]=np.log10(par[:,29])# Log tE
par[:,36]=np.log10(par[:,36])# Log rho* 
par[:,37]=np.log10(par[:,37])# Log tetE
par[:,40]=np.log10(par[:,40])# Log Dchi2     DeltaCh2_lensing_baseline 
par[:,48]=np.log10(par[:,48])# Log piE
par[:,45]=par[:,45]*180.0/np.pi # xi indegree

for i in range(nr): 
    if(par[i,45]>180.0):  par[i,45]=par[i,45]-180.0
    if(par[i,56]>180.0):  par[i,56]=par[i,56]-180.0
    par[i,1]=  np.abs(par[i,45]-par[i,56]) # xi-phi0
    if(par[i,1]>180.0):  par[i,1]=par[i,1]-180.0

par[:,2]=np.log10(1.0/par[:,5]-1.0/par[:,10]) # Log pirel

par2=np.zeros((nr,57))
par3=np.zeros((nr,57))
k2=0;  k3=0
for i in range(nr):
    if(abs(par[i,53]-par[i,54])>100.0):  
        par2[k2,:]=par[i,:]
        k2+=1  
    else:  
        par3[k3,:]=par[i,:]
        k3+=1                
print("number of events affected by Parallax:  ", k2,   float(k2*100.0/nr)  , nr)        

#input("Enter a number ")
################################################################################

for i in range(57):
    plt.clf()
    plt.cla()
    fig= plt.figure(figsize=(8,6))
    ax= plt.gca()              
    plt.hist(par[:,i],30,histtype='bar',ec='navy',facecolor='blue',alpha=0.25,rwidth=1.5,label=r"$\rm{Simulated}~\rm{events}$")
    plt.hist(par2[:k2,i],30,histtype='bar',ec='darkgreen',facecolor='green',alpha=0.7,rwidth=1.5,label=r"$\rm{Parallax}-\rm{affected}~\rm{events}$")
    plt.hist(par3[:k3,i],30,histtype='step',color='k',alpha=1.0, lw=1.9,label=r"$\rm{Not}-\rm{affected}~\rm{events}$")
    y_vals =ax.get_yticks()
    ax.set_yticklabels(['{:.2f}'.format(1.0*x*(1.0/nr)) for x in y_vals]) 
    y_vals = ax.get_yticks()
    plt.ylim([np.min(y_vals), np.max(y_vals)*0.95])
    ax.set_ylabel(r"$\rm{Normalized}~\rm{Distribution}$",fontsize=23,labelpad=0.1)
    ax.set_xlabel(str(nam0[i]),fontsize=25,labelpad=0.1)
    plt.xticks(fontsize=21, rotation=0)
    plt.yticks(fontsize=21, rotation=0)
    #plt.grid("True")
    #plt.grid(linestyle='dashed')
    if(i==4):
        plt.legend()
        plt.legend(loc='best',fancybox=True, shadow=True)
        plt.legend(prop={"size":17.0})
    
    fig=plt.gcf()
    fig.tight_layout(pad=0.4)
    fig.savefig("./lightc/Histo/histo{0:d}.jpg".format(i),dpi=200)
    
    
    if (i==4 or i==29 or i==36 or i==2 or i==37): 
        print (i, round(np.log10(np.mean(np.power(10.0,par[:,i]))),2),  round(np.log10(np.mean(np.power(10.0,par2[:k2,i]))),2),  
            round(np.log10(np.mean(np.power(10.0,par3[:k3,i]))),2) )
    else:  
        print (i, round(np.mean(par[:,i]),2),   round(np.mean(par2[:k2,i]),2),    round(np.mean(par3[:k3,i]),2) )
    
print ("****  All histos are plotted *****************************" )   
#input("Enter a number ")
plt.clf()
plt.cla()
fig= plt.figure(figsize=(8,6))
plt.plot(par2[:k2,29], par2[:k2,36], "ro")
fig=plt.gcf()
fig.savefig("test.jpg",dpi=200)
#input("Enter a number ")
################################################################################
erri=[]
fd=open("./Diffile.txt","w")
fd.close()


dif=np.zeros((nr,26)) 
co=0
count=np.zeros((6,3))
thre= np.array([0.1, 0.3, 0.5])
No1=0.0000001
No2=0.00000001
Nasym=np.zeros(nr)
ave=np.zeros((5000,19))
for ia in range(nr): 
    i=ia
    icon, ksipi, Lpirel= int(par[i,0]),par[i,1], par[i,2]
    strucl, LMl, Dl,vl= par[i,3],par[i,4], par[i,5], par[i,6]
    strucs, cl, mass, Ds, LTstar,Rstar, logl= par[i,7], par[i,8], par[i,9], par[i,10], par[i,11], par[i,12], par[i,13]
    types, col, vs, MI, MW149, mapI, mapW= par[i,14], par[i,15],par[i,16], par[i,17], par[i,18], par[i,19], par[i,20]
    magbI, mbs, blendI, fb, NbI, Nbw, ExI,ExW= par[i,21], par[i,22],par[i,23],par[i,24],par[i,25],par[i,26], par[i,27], par[i,28]
    LtE, RE, t0,mul,Vt,u0,opd,Lros,LtetE=par[i,29],par[i,30],par[i,31],par[i,32],par[i,33],par[i,34],par[i,35],par[i,36],par[i,37]
    flagf,flagD,Ldchi1, ndw,li,mus1, mus2=par[i,38], par[i,39],par[i,40], par[i,41], par[i,42], par[i,43], par[i,44]
    xi, mul1, mul2, LpiE,ampM,errM,ampA,errA=par[i,45], par[i,46],par[i,47], par[i,48],par[i,49],par[i,50],par[i,51], par[i,52]
    chi1, chi2, chi3, PhiP =par[i,53], par[i,54], par[i,55], par[i,56]
    Dchi0=abs(chi1-chi2)
    Lmul=np.log10(mul)
    Ml=np.power(10.0,LMl)
    tE=np.power(10.0,LtE)
    pirel=np.power(10.0,Lpirel)
    tetE=np.power(10.0,LtetE)
    piE=np.power(10.0,LpiE)
    ros=np.power(10.0,Lros)
    if(flagf>0 and icon>=0):        asym=Asymmet(icon, ndw)
    if(Dchi0>100.0):                No1+=1.0
    if(Dchi0>100.0 and asym<threA): No2+=1.0
    #if(Dchi0>100.0 and asym>threA): 
    Nasym[i]= asym
        #print (icon, Dchi0,    asym,    ndw)
        #input("Enter a number ")
       
    
    
    ############################################################################
    if(Dchi0>100.0):
        try:
            bb=bes[:,0].tolist().index(icon)
            if(bb<nb and bb>=0 and co<nr):       
                icon2,u0f,eu01,eu02,t0f,et01,et02,tEf,etE1,etE2,fbf,efb1,efb2,rosf,eros1,eros2,CHI2,chi1,dof=bes[bb,:]
                
                dtE= abs(tE-tEf)/tE;    etE= np.sqrt(etE1**2.0+etE2**2.0)*0.5/tE
                dros=abs(ros-rosf)/ros; eros=np.sqrt(eros1**2.0+eros2**2.0)*0.5/ros
                du0= abs(u0-u0f)/abs(u0);    eu0 =np.sqrt(eu01**2.0+eu02**2.0)*0.5/u0
                dt0= abs(t0-t0f)/t0;    et0 =np.sqrt(et01**2.0+et02**2.0)*0.5/t0
                dfb= abs(fb-fbf)/fb;    efb= np.sqrt(efb1**2.0+efb2**2.0)*0.5/fb
                
                ave[co,:]=np.array([Ml,Dl,Ds,mul,u0,t0,tE,ros,fb,mbs, pirel,tetE,piE, dtE,dros,du0,dt0,dfb,asym])#19
                dif[co,:]=np.array([np.log10(dtE),etE , np.log10(dros),eros , np.log10(du0),eu0 , np.log10(dt0),et0, 
                np.log10(dfb),efb ,np.log10(asym),
                LMl,Dl,Ds,LtE,mbs,fb,u0,ksipi, PhiP, Lmul, LpiE, Lpirel, LtetE, Lros, icon])#26 
                fd=open("./Diffile.txt","a")
                np.savetxt(fd,dif[co,:].reshape(-1,26),fmt="%.7f  %.7f  %.8f %.8f  %.7f %.7f  %.7f %.7f  %.7f %.7f  %.7f  %.7f   %.5f  %.5f  %.7f  %.5f  %.5f  %.5f  %.4f  %.4f  %.5f  %.7f  %.8f %.8f  %.8f   %d")
                fd.close()
                #test=np.array([np.log10(dtE),np.log10(dros),np.log10(du0),np.log10(dt0),np.log10(dfb),
                #LMl,Dl,LtE,mbs,fb,u0,ksipi,LpiE, Lpirel, LtetE, Lros])#16 
                #fg=open("./Corfile.txt","a")
                #np.savetxt(fg,test.reshape(-1,16),fmt="%.7f  %.7f  %.7f  %.7f  %.7f  %.7f  %.7f  %.7f  %.5f  %.5f  %.7f  %.6f  %.7f  %.7f  %.7f  %.8f")
                #fg.close()
                
                if(abs(icon2-icon)>1.0 or (abs(CHI2/dof)>5.0 and dof<1000) or float(abs(chi1-CHI2)/dof)>5.0 or CHI2<0.0 or tEf<0.0 or rosf>99.0): 
                    print("Error: ", icon, icon2, CHI2/dof, chi1/dof, dof, tEf,    rosf,  fbf,   t0f,  u0f)
                    #input("Enter a number")
                co+=1
                for j in range(3):
                    if(dtE>thre[j]):    count[0,j]+=1.0
                    if(dt0>thre[j]):    count[1,j]+=1.0
                    if(du0>thre[j]):    count[2,j]+=1.0
                    if(dfb>thre[j]):    count[3,j]+=1.0
                    if(dros>thre[j]):   count[4,j]+=1.0
                    if(dros>thre[j] and dtE>thre[j]) :  count[5,j]+=1.0  # par[i,4]>0.1 and par[i,4]<1000.0
        except: 
            bb=1
        
for j in range(6):   
    count[j,:]=count[j,:]*100.0/co            
################################################################################
df= pd.read_csv("./Diffile.txt",sep=" ",skipinitialspace=True,header=None,usecols=[0,2,4,6,8,11,12,14,15,21,22,23,24], names=nam4)
corrM = df.corr()
#print(corrM)
fig, ax = plt.subplots(figsize=(10, 8))
corrM.style.background_gradient(cmap='coolwarm').set_precision(2)
ax=sns.heatmap(corrM, annot=True, xticklabels=nam4, yticklabels=nam4, annot_kws={"size":13.5},square=True,linewidth=1.0, cbar_kws={"shrink":.99}, linecolor="k",fmt=".1f", cbar=True, vmax=1, vmin=-1, center=0.0, robust=True)#ax=None,
cbar=ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=17, pad=0.0)
plt.xticks(rotation=35,horizontalalignment='right',fontweight='light', fontsize=15)
plt.yticks(rotation=0, horizontalalignment='right',fontweight='light', fontsize=15)
plt.title(r"$\rm{Correlation}~\rm{Matrix}$", fontsize=17)
fig.tight_layout()
plt.savefig("./lightc/Histo/corr2.jpg", dpi=200)
print("**** Correlation matrix was calculated ******** ")
#input("Enter a number ")
################################################################################
print ("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
print ("The cases(icon) with Asymetry >threA: ",Nasym , np.max(Nasym), np.argmax(Nasym))
print ("Fraction of lensing events with Deltachi2>100:  ", No1*100.0/nr )   
print ("Fraction of these parallax_affected events with Asym<threA:  ",   No2*100.0/No1)
print ("Fraction of these parallax_affected events with Asym>threA:  ",   100.0-No2*100.0/No1)  
print ("fraction with DtE/tE>thre: ", round(count[0,0],2)  , round(count[0,1],2),   round(count[0,2],2) )
print ("fraction with Dros/ros>thre:",round(count[4,0],2)  , round(count[4,1],2),   round(count[4,2],2) )
print ("fraction with Du0/u0>thre: ", round(count[2,0],2)  , round(count[2,1],2),   round(count[2,2],2) )
print ("fraction with Dt0/t0>thre: ", round(count[1,0],2)  , round(count[1,1],2),   round(count[1,2],2) )
print ("fraction with Dfb/fb>thre: ", round(count[3,0],2)  , round(count[3,1],2),   round(count[3,2],2) )

print ("fraction with delta tE && delta rhos: ", round(count[5,0],2)  , round(count[5,1],2),   round(count[5,2],2) )

print ("parameters:  Ml,Dl,Ds,mul, u0,t0,tE,ros,fb,mbs, pirel,tetE,piE, dtE,dros,du0,dt0,dfb,asym")

avn= ["Ml","Dl","Ds","mul", "u0","t0","tE","ros","fb","mbs", "pirel","tetE","piE", "dtE","dros","du0","dt0","dfb","asym"]
for i in range(19): 
    print(i,  str(avn[i]) , np.mean(ave[:co ,i]) , "\pm"  ,  np.std(ave[:co ,i]) )   
print ("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")

################################################################################

x1=np.array([-2.0,0.0,2.0,-1.7,15.0,0.0,0.0,0.0,  0,-3.0,-0.5,-2.3,-4.0, -3.0])
x2=np.array([3.5,10.,14.0, 1.5,23.0,1.0,1.0,360.0,360.0,-1.2,3.,0.5, -0.7, 0.5])
y1=np.array([-3.0,-4.0,-4.0,-6.0, -3.0,0.0])
y2=np.array([1.0,2.0, 1.0, 0.0, 2.0, 2.2])
for i in range(6):
    for j in range(14):   
        plt.clf()
        plt.cla()
        fig=plt.figure(figsize=(8,6))
        plt.scatter(dif[:,11+j],dif[:,i*2], marker='o',c='g', s=10.0,alpha=1.0)
        plt.xlabel(str(nam2[j]), fontsize=20)
        plt.ylabel(str(nam1[i]), fontsize=20)
        #plt.yscale('log')
        plt.xlim([ x1[j] , x2[j] ])
        plt.ylim([ y1[i] , y2[i] ])
        plt.xticks(fontsize=19, rotation=0)
        plt.yticks(fontsize=19, rotation=0)
        plt.grid("True")
        plt.grid(linestyle='dashed')
        fig=plt.gcf()
        #plt.subplots_adjust(hspace=.0)
        fig.tight_layout(pad=0.6)
        fig.savefig("./lightc/Histo/Scatter{0:d}_{1:d}.jpg".format(i,j) , dpi=200)
print("****  Scattering are plotted *************************", i)          



for i in range(6):
    for j in range(6):   
        plt.clf()
        plt.cla()
        fig=plt.figure(figsize=(8,6))
        plt.scatter(dif[:,j*2],dif[:,i*2], marker='o',c='g', s=10.0,alpha=1.0)
        plt.xlabel(str(nam1[j]), fontsize=20)
        plt.ylabel(str(nam1[i]), fontsize=20)
        #plt.yscale('log')
        plt.xlim([ y1[j] , y2[j] ])
        plt.ylim([ y1[i] , y2[i] ])
        plt.xticks(fontsize=19, rotation=0)
        plt.yticks(fontsize=19, rotation=0)
        plt.grid("True")
        plt.grid(linestyle='dashed')
        fig=plt.gcf()
        #plt.subplots_adjust(hspace=.0)
        fig.tight_layout(pad=0.6)
        fig.savefig("./lightc/Histo/ScatterB{0:d}_{1:d}.jpg".format(i,j) , dpi=200)


################################################################################
nv=9
ns=9
xar= np.zeros((nv,ns))
num0=np.zeros((nv,ns))
num1=np.zeros((nv,ns,3))

dela=np.zeros((6,nv,ns))
deln=np.zeros((6,nv,ns))

xar[0,:]=np.arange(ns)*float(5.5/ns)-2.0# log10[ML]---4
xar[1,:]=np.arange(ns)*float(8.5/ns)+0.1#DL   ---5
xar[2,:]=np.arange(ns)*float(2.72/ns)-1.7#logtE---- 29 
xar[3,:]=np.arange(ns)*float(3.56/ns)-0.75#logpiE --- 48
xar[4,:]=np.arange(ns)*float(180.0/ns)+0.0#ksi-pi --- 1
xar[5,:]=np.arange(ns)*float(3.4/ns)-3.0#log(pirel) ---- 2       
xar[6,:]=np.arange(ns)*float(8.5/ns)+15.5#mbase   ----  22             
xar[7,:]=np.arange(ns)*float(0.98/ns)+0.0 #u0  ---- 34      
xar[8,:]=np.arange(ns)*float(3.65/ns)-4.0#log(thetaE)  ---- 36


for i in range(nr):
    xx=np.array([par[i,4], par[i,5], par[i,29], par[i,48], abs(par[i,1]), par[i,2] ,par[i,22], par[i,34], par[i,37]]) 
    Dchi0=abs(par[i,53]- par[i,54])
    icon=int(par[i,0])
    try:     bb=dif[:,25].tolist().index(icon)
    except:  bb=-1     

    for j in range(nv):
        tt=-1
        if(xx[j]<=xar[j,0]):       tt=0
        elif(xx[j]>=xar[j,ns-1]):  tt=ns-1
        else:  
            for k in range(ns-1):   
                if((xx[j]-xar[j,k])*(xx[j]-xar[j,k+1])<0.0 or xx[j]==xar[j,k]):  
                   tt=k
                   break
        if(tt<0): 
            print("Error:  tt:  ", tt, xx[j],  xar[j,:],  j)
            input("Enter a number ")
            
        num0[j,tt]+=1.0
        if(Dchi0>100):                               num1[j,tt,0]+=1.0   
        if(Dchi0>100 and bb>=0 and dif[bb,0]>0.02):  num1[j,tt,1]+=1.0     
        if(Dchi0>100 and bb>=0 and dif[bb,2]>0.02):  num1[j,tt,2]+=1.0 
        if(Dchi0>100 and bb>=0):
            dela[0,j,tt] += np.power(10.0,dif[bb,0])#tE
            dela[1,j,tt] += np.power(10.0,dif[bb,2])#ros
            dela[2,j,tt] += np.power(10.0,dif[bb,4])#u0
            dela[3,j,tt] += np.power(10.0,dif[bb,6])#t0
            dela[4,j,tt] += np.power(10.0,dif[bb,8])#fb
            dela[5,j,tt] += np.power(10.0,dif[bb,10])#Aysm 
            deln[0,j,tt] += 1.0
            deln[1,j,tt] += 1.0
            deln[2,j,tt] += 1.0
            deln[3,j,tt] += 1.0
            deln[4,j,tt] += 1.0
            deln[5,j,tt] += 1.0             
################################################################################
for j in range(nv):  
    for i in range(ns):  
        num1[j,i,0]= num1[j,i,0]*100.0/(num0[j,i]+0.00001) +0.05
        num1[j,i,1]= num1[j,i,1]*100.0/(num0[j,i]+0.00001) +0.05
        num1[j,i,2]= num1[j,i,2]*100.0/(num0[j,i]+0.00001) +0.05
        for k in range(6): 
            dela[k,j,i]=dela[k,j,i]/(deln[k,j,i]+0.01)
        
for i in range(5): 
   for j in range(nv): 
       dela[i,j,:]=np.log10(dela[i,j,:])       
       dela[i,j,:]=gaussian_filter(dela[i,j,:],0.8)        


for i in range(nv):
    plt.clf()
    plt.cla()
    plt.figure(figsize=(8,6))
    
    plt.scatter(xar[i,:],dela[0,i,:],marker="o",facecolors='r',edgecolors='r',s=34)
    plt.scatter(xar[i,:],dela[1,i,:],marker= "o",facecolors='g',edgecolors='g',s=34)
    plt.scatter(xar[i,:],dela[2,i,:],marker= "o",facecolors='b',edgecolors='b',s=34)
    plt.scatter(xar[i,:],dela[3,i,:],marker= "o",facecolors='m',edgecolors='m',s=34)
    plt.scatter(xar[i,:],dela[4,i,:],marker= "o",facecolors='k',edgecolors='k',s=34)
    
    
    plt.plot(xar[i,:],dela[0,i,:],lw=1.8,linestyle='--',color="r",label=r"$\log_{10}[\overline{\delta t_{\rm{E}}}]$")
    plt.plot(xar[i,:],dela[1,i,:],lw=1.8,linestyle='-.',color="g",label=r"$\log_{10}[\overline{\delta \rho_{\star}}]$") 
    plt.plot(xar[i,:],dela[2,i,:],lw=1.8,linestyle=':', color="b",label=r"$\log_{10}[\overline{\delta u_{0}}]$")
    plt.plot(xar[i,:],dela[3,i,:],lw=1.8,linestyle='-',color="m",label=r"$\log_{10}[\overline{\delta t_{0}}]$")   
    plt.plot(xar[i,:],dela[4,i,:],lw=1.8,linestyle='--',color="k",label=r"$\log_{10}[\overline{\delta f_{\rm{b}}}]$")
    
        
    #plt.step(xar[i,:],dela[0,i,:],where='mid',lw=1.8,linestyle='--',color="r",label=r"$\log_{10}[\overline{\delta t_{\rm{E}}}]$")
    #plt.step(xar[i,:],dela[1,i,:],where='mid',lw=1.8,linestyle='-.',color="g",label=r"$\log_{10}[\overline{\delta \rho_{\star}}]$") 
    #plt.step(xar[i,:],dela[2,i,:],where='mid',lw=1.8,linestyle=':', color="b",label=r"$\log_{10}[\overline{\delta u_{0}}]$")
    #plt.step(xar[i,:],dela[3,i,:],where='mid',lw=1.8,linestyle='-',color="m",label=r"$\log_{10}[\overline{\delta t_{0}}]$")   
    #plt.step(xar[i,:],dela[4,i,:],where='mid',lw=1.8,linestyle='--',color="k",label=r"$\log_{10}[\overline{\delta f_{\rm{b}}}]$")
    
    #plt.xlim([ xar[i,0],xar[i,ns-1] ])
    plt.ylabel(r"$\rm{Relative}~\rm{Deviations}$",fontsize=17,labelpad=0.1)
    plt.xlabel(str(nam3[i]),fontsize=17,labelpad=0.1)
    plt.xticks(fontsize=17, rotation=0)
    plt.yticks(fontsize=17, rotation=0)
    plt.grid("True")
    plt.grid(linestyle='dashed')
    if(i==2): 
        plt.legend()
        plt.legend(loc='best',fancybox=True, shadow=True)
        plt.legend(prop={"size":16})
    fig=plt.gcf()
    fig.tight_layout(pad=0.6)
    fig.savefig("./lightc/Histo/Newplots{0:d}.jpg".format(i),dpi=200)
    
    
    
    ############################################################################
    x1=np.array([-2.0,0.0,-1.7,-0.8, 0.0,  -3.0, 15.0, 0.0,-4.0])
    x2=np.array([3.0, 9.5, 1.0, 2.6, 360.0, 0.2, 24.0, 1.0,-0.5])


    plt.clf()
    plt.cla()
    plt.figure(figsize=(8,6))
    
    plt.step(xar[i,:],num1[i,:,0],where='mid',lw=2.1,linestyle='-',color="gray",label=r"$\Delta\chi^{2}>100$")
    plt.step(xar[i,:],num1[i,:,1],where='mid',lw=2.1,linestyle='--',color="r",label=r"$\Delta\chi^{2}>100,~\delta t_{\rm{E}}>0.02$")
    plt.step(xar[i,:],num1[i,:,2],where='mid',lw=2.1,linestyle='-.',color="g",label=r"$\Delta\chi^{2}>100,~\delta \rho_{\star}>0.02$")
    
    plt.scatter(xar[i,:],num1[i,:,0],marker= "o",facecolors='k', edgecolors='gray',  s= 36.0)
    plt.scatter(xar[i,:],num1[i,:,1],marker= "o",facecolors='darkred',  edgecolors='r',   s= 36.0)
    plt.scatter(xar[i,:],num1[i,:,2],marker= "o",facecolors='darkgreen',  edgecolors='g', s= 36.0)
    
    plt.xlim([ xar[i,0], xar[i,ns-1] ])
    plt.ylim([0.04,101.0])
    plt.yscale('log')
    
    plt.ylabel(r"$\rm{Fraction}~\rm{of}~\rm{Parallax}-\rm{affected}~\rm{events}[\%]$",fontsize=20,labelpad=0.1)
    plt.xlabel(str(nam3[i]),fontsize=20,labelpad=0.1)
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    #plt.grid("True")
    #plt.grid(linestyle='dashed')
    if(i==5): 
        plt.legend()
        plt.legend(loc='best',fancybox=True, shadow=True)
        plt.legend(prop={"size":17})
    fig.tight_layout(pad=0.6)    
    fig=plt.gcf()
    fig.savefig("./lightc/Histo/fraction{0:d}.jpg".format(i),dpi=200)
    
    
print ("****  All Fraction plots are plotted *****************************" )   
      
      
      
      
      
      

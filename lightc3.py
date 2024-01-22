import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.figure import Figure
rcParams["font.size"] = 11.5
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
import emcee
import pylab
import VBBinaryLensingLibrary as vb
VBB=vb.VBBinaryLensing()
VBB.Tol=1.0e-3; ### accuracy
VBB.SetLDprofile(VBB.LDlinear);
VBB.LoadESPLTable("./ESPL.tbl"); 
################################################################################
f0=open("./files/MONT/FFP_Roman1c.dat","r")
nr=sum(1 for line in f0)
par=np.zeros(( nr , 57 ))
par=np.loadtxt("./files/MONT/FFP_Roman1c.dat") 

fd=open("./Best3b.txt","w")
fd.close()


m1=0.0
m2=0.0
m3=0.0

sm=1.0e-50
Rsun=6.957*pow(10.0,8.0); 
AU=1.4960*pow(10.0,11.0);
year=float(365.2425); 
Tobs=float(62.0)
cade1=float(15.16/60.0/24.0);
ndim=5
Nwalkers=40
nstep=13000
p0=np.zeros((Nwalkers,ndim))
a=np.zeros((ndim,3));  
b=np.zeros(ndim*3+4); 

################################################################################
def likelihood(p, tim, Magni, erra, prt):
    lp=prior(p, prt)
    if(lp<0.0):
        return(-np.inf)
    return lnlike2(p, tim, Magni , erra)
################################################################################
def prior(p, prt): 
    u01, t01, tE1, fb1, rho1 =p[0],   p[1],   p[2],   p[3],   p[4]
    if(u01>sm and u01<1.0 and t01>=1.0 and t01<Tobs and tE1>cade1 and tE1<100.0 and fb1>sm and fb1<1.0 and rho1>sm and rho1<99.0): 
        return(0); 
    return(-1.0);     
################################################################################
def lnlike2(p, tim, Magni, erra):
    u01, t01, tE1, fb1, rho1=p[0], p[1], p[2], p[3],  p[4]
    VBB.a1=0.0;
    As=np.zeros((len(tim)))
    for i in range(len(tim)):  
        u=np.sqrt((tim[i]-t01)*(tim[i]-t01)/tE1/tE1 + u01*u01); 
        As[i]=VBB.ESPLMag2(u,rho1)
    As=fb1*As+1.0-fb1
    return(-1.0*np.sum((As-Magni)**2.0/(erra*erra) ) )
################################################################################
fflag=0
for ia in range(300): 
    i=ia+300
    icon, lat, lon=int(par[i,0]), par[i,1], par[i,2]
    strucl, Ml, Dl, vl= par[i,3], par[i,4], par[i,5], par[i,6]
    strucs, cl, mass, Ds,Tstar, Rstar, logl= par[i,7], par[i,8], par[i,9], par[i,10], par[i,11], par[i,12], par[i,13]
    types, col, vs, MI, MW149, mapI, mapW=    par[i,14], par[i,15],par[i,16], par[i,17], par[i,18], par[i,19], par[i,20]
    magbI, mbs, blendI, fb, NbI, Nbw, ExI, ExW= par[i,21], par[i,22],par[i,23],par[i,24],par[i,25],par[i,26], par[i,27], par[i,28]
    tE, RE, t0, mul, Vt, u0, opd,ros,tetE=par[i,29],par[i,30],par[i,31],par[i,32],par[i,33],par[i,34],par[i,35],par[i,36],par[i,37]
    flagf,flagD, dchi1, ndw,li, mus1, mus2=par[i,38], par[i,39],par[i,40], par[i,41], par[i,42], par[i,43], par[i,44]
    xi, mul1, mul2, piE=    par[i,45], par[i,46],par[i,47], par[i,48]
    ampM, errM, ampA, errA = par[i,49], par[i,50],par[i,51], par[i,52]
    chi1, chi2, chi3, PhiP = par[i,53], par[i,54], par[i,55], par[i,56]
    xi=   float(xi*180.0/np.pi)
    Dchi0=abs(chi1-chi2)
    pirel=float(1.0/Dl-1.0/Ds)
   
    if(flagf>0 and icon>0): 
        f1=open("./files/MONT/datA{0:d}.dat".format(icon),"r")
        nd= sum(1 for line in f1)  
        dat=np.zeros((nd,9)) 
        dat=np.loadtxt("./files/MONT/datA{0:d}.dat".format(icon))
        if(nd!=ndw): 
            print("Big error data_number: ", nd, ndw, icon)
            input("Enter a number ")
        f2=open("./files/MONT/magA{0:d}.dat".format(icon),"r")
        nm=sum(1 for line in f2)  
        mag=np.zeros((nm,17));   
        mag= np.loadtxt("./files/MONT/magA{0:d}.dat".format(icon))
        ########################################################################
        tt0=int(np.argmax(dat[:,3]))
        asym=0.0; conn=0.000000000674531
        for j in range(nd): 
            n1=int(tt0-j-1);
            n2=int(tt0+j+1);  
            if(n2<nd and n1>=0 and dat[n1,3]>1.05 and dat[n2,3]>1.05):
                errA =float(abs(dat[n1,4])+abs(dat[n2,4]))*0.5
                asym+=float(dat[n2,3]- dat[n1,3])**2.0/errA**2.0
                conn+=1.0
        asym=np.sqrt(asym/conn)    
        tim=  np.zeros((nd)); 
        Magni=np.zeros((nd));   
        erra =np.zeros((nd));
        tim=  dat[:,0];    
        Magni=dat[:,3];   
        erra= dat[:,4]+1.0e-8;   
        ########################################################################
        fflag=0
        if(Dchi0>100.0):
            p0[:,0]=np.abs(np.random.normal(u0,0.6,    Nwalkers))#u0
            p0[:,1]=np.abs(np.random.normal(t0,0.6*tE, Nwalkers))#t0
            p0[:,2]=np.abs(np.random.normal(tE,tE,     Nwalkers))#tE
            p0[:,3]=np.abs(np.random.normal(fb,0.2,    Nwalkers))#fb
            p0[:,4]=np.abs(np.random.normal(ros,0.8*ros,Nwalkers))#ros
            prt=np.array([u0, t0, tE, fb, ros])
            sampler=emcee.EnsembleSampler(Nwalkers, ndim, likelihood ,args=(tim,Magni,erra, prt),threads=1)
            xx=0
            fil=open("./Chain3.dat", "w")
            fil.close()
            for param, like, stat in sampler.sample(p0,iterations=nstep, storechain=False):
                xx+=1
                fil=open("./Chain3.dat","a+")
                ssa=np.concatenate((param.reshape((-1,ndim)),like.reshape((-1,1))), axis=1)
                np.savetxt(fil,ssa,fmt ="%.10f   %.10f   %.10f    %.10f   %.10f   %.10f")
                fil.close()
                if(xx%4000==0): print("Step:     " ,  xx)
            #print("****************** END OF MCMC *************************** ")
            Chain=np.zeros((Nwalkers*nstep,ndim+1))
            Chain=np.loadtxt("./Chain3.dat") 
            samples1 =Chain[:,:ndim].reshape((-1,ndim))
            a=map(lambda v:(v[1],v[2]-v[1],v[1]-v[0]),zip(*np.percentile(samples1, [16, 50, 84],axis=0)))
            tt=int(np.argmin(-1.0*Chain[:,ndim]))
            CHI2=abs(lnlike2(Chain[tt,:ndim] ,tim, Magni, erra))
            ffd= abs(Chain[tt,ndim])
            if( abs(CHI2-ffd)>0.2 ):  
                print("BIG ERROR:  chi2s:  ",  CHI2, ffd)
                print("Best-fitted parameters:  ",  Chain[tt, :])
                print("Initial parameters:  ", prt)
                input("Enter a number " )
            du0=0.5*(abs(a[0][1])+abs(a[0][2]));     
            dt0=0.5*(abs(a[1][1])+abs(a[1][2]));
            dtE=0.5*(abs(a[2][1])+abs(a[2][2]));     
            dfb=0.5*(abs(a[3][1])+abs(a[3][2]));
            dro=0.5*(abs(a[4][1])+abs(a[4][2]));
            u0f, t0f, tEf, fbf, rosf= Chain[tt,:ndim]
            b=np.array([icon, u0f,a[0][1],a[0][2],t0f,a[1][1],a[1][2],tEf,a[2][1],a[2][2],fbf,a[3][1],a[3][2],rosf,a[4][1],a[4][2],CHI2,chi1,int(nd-5)])
            save=open("./Best3b.txt","a")
            np.savetxt(save,b.reshape(-1,ndim*3+4),fmt='%d  %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f  %.5f %.5f %.5f %.5f %.5f %.5f %.5f  %.5f %.5f  %d')
            save.close()
            Dchi=float(chi1-CHI2)
            timt=np.arange(t0f-3.5*tEf, t0f+3.5*tEf, 0.01*tEf)
            u2= np.sqrt(u0f**2.0+(timt-t0f)**2.0/tEf/tEf)
            mfit=np.zeros((len(timt)))
            for i in range(len(timt)):  
                mfit[i]=VBB.ESPLMag2(u2[i],rosf)
            mfit=fbf*mfit+1.0-fbf
            fflag=1
        ########################################################################
        '''
        plt.cla()
        plt.clf()
        fig=plt.figure(figsize=(8,6))
        ax1=fig.add_subplot(111)
        plt.errorbar(dat[:,0],dat[:,3],yerr=dat[:,4], fmt=".",markersize=7., color='m',ecolor='m',elinewidth=1.2, capsize=0, alpha=0.9)
        plt.plot(mag[:,0], mag[:,3], "k-", lw=1.5, label=r"$\rm{Magnification}$")
        plt.plot(mag[:,0], mag[:,4], "b--",lw=1.5, label=r"$\rm{Magnification}+\rm{parallax}$")
        if(fflag>0):  
            plt.plot(timt, mfit , 'g-',  label=r"$\rm{Best}-\rm{Fitted}~\rm{model}$",lw=1.0)
        plt.xlabel(r"$\rm{time(days)}$",    fontsize=18)
        plt.ylabel(r"$\rm{Magnification}$", fontsize=18)
        plt.xticks(fontsize=17, rotation=0)
        plt.yticks(fontsize=17, rotation=0)
        pylab.xlim([ np.min(mag[:,0]),  np.max(mag[:,0]) ])
        #plt.xlim([0.0,5.0])
        if(fflag>0):  
            plt.title(
            r"$t_{\rm{E},r}\rm{(days)}=$"+str(round(tE,2))+r"$,~\rho_{\star,r}=$"+str(round(ros,2))+
            r"$,~M_{\rm l}(M_{\oplus})=$"+str(round(Ml,2))+r"$,~\pi_{\rm{rel}}(\rm{mas})=$"+str(round(pirel,3))+ 
            r"$,~\pi_{\rm{E}}=$"+str(round(piE,3))+"\n"+
            r"$t_{\rm{E},b}(\rm{days})=$"+str(round(tEf,2))+r"$\pm$"+str(round(dtE,2))+
            r"$,~\rho_{\star,b}=$"+str(round(rosf,2))+r"$\pm$"+str(round(dro,2))+ 
            r"$,~\chi^{2}_{\rm{real}}/\rm{dof}=$"+str(round(chi1,1))+"/"+str(int(nd-5))+
            r"$,~\chi^{2}_{\rm{best}}/\rm{dof}=$"+str(round(CHI2,1))+"/"+str(int(nd-5))+
            r"$,~\mathcal{A}=$"+str(round(asym,4)),fontsize=13, color="k")
       
        else:
            plt.title(
            r"$t_{\rm{E},r}\rm{(days)}=$"+str(round(tE,2))+r"$,~\rho_{\star,r}=$"+str(round(ros,2))+
            r"$,~M_{\rm l}(M_{\oplus})=$"+str(round(Ml,2))+r"$,~\theta_{\rm E}(\rm{mas})=$"+str(round(tetE,3))+
            r"$,~\mathcal{A}=$"+str(round(asym,4))+r"$,~\Delta\chi^{2}=$"+str(round(Dchi0,1)),fontsize=13, color="k") 
         
        ax1.grid("True")
        ax1.grid(linestyle='dashed')
        plt.legend()
        ax1.legend(prop={"size":13})
        fig=plt.gcf()
        fig.savefig("./lightc/lcA{0:d}.jpg".format(icon),dpi=200)
        
        ######################################################################## 
             
        plt.cla()
        plt.clf()
        fig=plt.figure(figsize=(8,6))
        ax1=fig.add_subplot(111)
        plt.plot(mag[:,13], mag[:,14], "-",  color='k', lw=1.5, label=r"$\rm{Lens}-\rm{Source}$",alpha=1.0)
        plt.plot(mag[:,15], mag[:,16], "--", color='b', lw=1.5, label=r"$\rm{Lens}-\rm{Source}+\rm{Parallax}$",  alpha=1.0)
        plt.plot(mag[:,15]-mag[:,13],mag[:,16]-mag[:,14],":",color='m',lw=1.5,label=r"$\rm{Parallax}-\rm{induced}~\rm{deviation}$",  alpha=1.0)
        #plt.xlim([np.min(mag[:,13])-5*delx, np.max(mag[:,13])+5*delx])
        #plt.ylim([np.min(mag[:,14])-5*dely, np.max(mag[:,14])+5*dely])
        plt.xlabel(r"$u_{x}$", fontsize=18)
        plt.ylabel(r"$u_{y}$", fontsize=18)
        plt.xticks(fontsize=18, rotation=0)
        plt.yticks(fontsize=18, rotation=0)
        plt.title(
        r"$D_{\rm{s}}(\rm{kpc})=$"+str(round(Ds,1))+r"$,~D_{\rm{l}}\rm{(kpc)}=$"+str(round(Dl,1))+
        r"$,~u_{0}=$"+str(round(u0,2))+r"$,~\ksi(\rm{deg})=$"+str(round(xi,1))+
        r"$,~\phi_{0}(\rm{deg})=$"+str(round(PhiP,1)),fontsize=14, color="k")ksi
        ax1.legend(prop={"size":13})
        ax1.grid("True")
        ax1.grid(linestyle='dashed')
        fig=plt.gcf()
        plt.subplots_adjust(hspace=.0)
        fig.savefig("./lightc/traj{0:d}.jpg".format(icon),dpi=200)
        
        print("***************************************************************")
        print("Counter,  icon:  ",   i,    icon)
        print("Chi2 from real_model, realP, bas: ", chi1, chi2, chi3)
        print("Parameters:Ml, Dl, tE, tetE, piE : ", Ml, Dl, tE, tetE,  piE)
        print("DeltaChi, Asym:    ",  abs(chi1-chi2) ,    asym ) 
        print("The parameters of the best-fitted model:  ",  a)
        print("********* Lightcurve & trajectory was plotted ****, No:  ", icon)
        '''
        ########################################################################

        '''
        plt.cla()
        plt.clf()
        fig=plt.figure(figsize=(8,6))
        ax1=fig.add_subplot(111)
        plt.errorbar(dat[:,5],dat[:,6],yerr=dat[:,7],xerr=dat[:,7],fmt=".",markersize=2.0,color='g',ecolor='#C1FFC1',
        elinewidth=0.01,capsize=0,alpha=0.6)
        plt.plot(mag[:,5], mag[:,6], "-", color='r',lw=1.5, label=r"$\theta_{\star,~\odot}$", alpha=1.0)
        plt.plot(mag[:,7], mag[:,8],"--", color='b',lw=1.5, label=r"$\theta+\rm{Parallax}$", alpha=1.0)
        #plt.xlim([np.min(mag[:,5])-5*delx,  np.max(mag[:,5])+5*delx])
        #plt.ylim([np.min(mag[:,6])-5*dely,  np.max(mag[:,6])+5*dely])
        plt.xlabel(r"$\theta x\rm{(mas)}$", fontsize=17)
        plt.ylabel(r"$\theta y\rm{(mas)}$", fontsize=17)
        plt.xticks(fontsize=17, rotation=0)
        plt.yticks(fontsize=17, rotation=0)
        plt.title(r"$M_{\rm{l}}(M_{\oplus})=$"+str(round(Ml,1))+ r"$,~~D_{\rm{l}}\rm{(kpc)}=$"+str(round(Dl,1))+r"$,~\pi_{\rm{E}}=$"+str(round(piE,3))+r"$,~\pi_{\rm{rel}}=$"+str(round(pirel,3)),fontsize=18, color="k")
        ax1.legend(prop={"size":15.5})
        fig=plt.gcf()
        fig.savefig("./lightc/dfA{0:d}.jpg".format(icon),dpi=200)
        '''
        ######################################################################## 


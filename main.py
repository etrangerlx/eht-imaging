
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.optimize import minimize

lambda_=1
min_spacing=0.5*lambda_
ant_pos=np.arange(0,9,1)*min_spacing
ant_pos.shape=(ant_pos.size,1)
ant_num=ant_pos.size

plt.figure(1)
plt.plot(ant_pos,np.ones([ant_pos.size,1]),marker="^")
plt.title("Antanne Structure")
plt.show()


UV=np.arange(0,0,1)
for p in range(0, len(ant_pos)):
    for q in range(0, len(ant_pos)):
        if p < q:
            tmp=(ant_pos[q] - ant_pos[p]) / lambda_
 #           if tmp not in UV:
            UV=np.append(UV,tmp)


UV_conj=-UV[::-1]
UV=np.append(UV,0)
UV=np.append(UV,UV_conj)
UV_noredun,i1,i2=np.unique(UV,return_index=True,return_inverse=True)

plt.figure(2)
plt.plot(UV_noredun,np.ones([UV_noredun.size,1]),marker="^")
plt.title("UV Structure")
plt.show()

length=np.float16(UV_noredun.size)
#length=np.float16(15)
xi=np.arange(-1,1,2/length)
delta=xi[1]-xi[0]
T=50*np.zeros(xi.shape)
T[(0.2*length).astype(int):(0.4*length).astype(int)]=200
T[(0.6*length).astype(int):(0.8*length).astype(int)]=400

plt.figure(3)
plt.plot(xi,T)
plt.show()


T_vec=T.reshape(T.size,1)
xi_vec=xi.reshape(xi.size,1)
UV_vec=UV.reshape(UV.size,1)
UV_noredun_vec=UV_noredun.reshape(UV_noredun.size,1)

F=np.exp(-1j*2*np.pi*(UV_vec*xi_vec.T))
F_noredun=F[i1,:]

V=np.dot(F,T_vec)*delta
V_noredun=V[i1,:]

T_dft_vec=np.dot(F_noredun.T.conj(),V_noredun)*0.5

plt.figure(5)
plt.clf()
plt.plot(xi,T_dft_vec.real)
plt.show()

ll=0
Rxx=np.zeros([ant_num,ant_num],dtype="complex128")
for p in range(0, len(ant_pos)):
    for q in range(0, len(ant_pos)):
        if p < q:
            Rxx[p,q]=V[ll,0]
            ll=ll+1
Rxx=Rxx+Rxx.T.conj()
for p in range(0, len(ant_pos)):
    Rxx[p,p]=V[ll,0]
#error
gama=np.random.rand(ant_num,ant_num)
gama=np.exp(1j*gama*10)
gama=np.diag(np.diag(gama))



Ryy=np.dot(gama,Rxx).dot(np.linalg.inv(gama))
#2pi mod
kk=np.angle(Ryy)-np.angle(Rxx)-np.angle(np.dot(gama,np.ones([ant_num,ant_num])).dot(np.linalg.inv(gama)))

ll=0
Vm=np.zeros(V.shape,dtype="complex128")
for p in range(0, len(ant_pos)):
    for q in range(0, len(ant_pos)):
        if p < q:
            Vm[ll,0]=Ryy[p,q]
            ll=ll+1
Vm=Vm+Vm[::-1].conj()
Vm[ll]=Ryy[p,p]

Vm_noredun=Vm[i1,:]
T_dft_witherror_vec=np.dot(F_noredun.T.conj(),Vm_noredun)*0.5


plt.figure(6)
plt.clf()
plt.plot(xi,T_dft_witherror_vec.real)
plt.show()


#closure phase
Nclosure=(ant_num-2)*(ant_num-1)/2
A1=np.zeros([Nclosure,T_vec.size],dtype="complex128")
A2=np.zeros([Nclosure,T_vec.size],dtype="complex128")
A3=np.zeros([Nclosure,T_vec.size],dtype="complex128")
Bvis=np.zeros([Nclosure,1],dtype="complex128")
ll=0
for ii in np.arange(1,ant_num):
    for jj in np.arange(ii+1,ant_num):
        t1=(ant_pos[ii]-ant_pos[0])/lambda_
        A1[ll]=delta*np.exp(-1j*2*np.pi*(t1*xi_vec.T))

        t2=(ant_pos[0]-ant_pos[jj])/lambda_
        A2[ll]=delta*np.exp(-1j*2*np.pi*(t2*xi_vec.T))

        t3=(ant_pos[jj]-ant_pos[ii])/lambda_
        A3[ll] = delta*np.exp(-1j * 2 * np.pi * (t3*xi_vec.T))

        Bvis[ll]=Rxx[0,ii]*Rxx[0,jj].conj()*Rxx[ii,jj]
        ll = ll + 1
A1*A2*A3

xinit=np.random.rand(T.size) * 2
print xinit
#xinit=[0.15270033,1.75078458,0.6740086,0.70185501,0.25405819,0.13174369,0.71283464,0.17339643,0.24058373,0.94027499,0.40549614,1.62980561,1.21721765]
#xinit=np.random.normal(np.sum(T)/T.size,10,T.size)
#xinit=np.ones(T.shape,dtype="float128")
xinit=T/T.max()+0.5*np.random.rand(T.size) * 2

xinit=np.log(xinit)

#xinit=T
#xinit.shape=(xinit.size,1)
clnum=(ant_num-1)*(ant_num-2)/2
Bvis.shape=(clnum,)
error1=Bvis-(np.dot(A1,xinit)*np.dot(A2,xinit)*np.dot(A3,xinit))
error2=Bvis-(np.dot(A1,T_vec)*np.dot(A2,T_vec)*np.dot(A3,T_vec))


def chisq_bs(imvec):
    """Bispectrum chi-squared"""

    bisamples = np.dot(A1, imvec) * np.dot(A2, imvec) * np.dot(A3, imvec)
    chisq = np.sum(np.abs((Bvis - bisamples) ** 2)) / (2. * len(Bvis))
    return chisq

def chisqgrad_bs(imvec):
    """The gradient of the bispectrum chi-squared"""
    bisamples = np.dot(A1, imvec) * np.dot(A2, imvec) * np.dot(A3, imvec)
    wdiff = ((Bvis - bisamples).conj())
    pt1 = wdiff * (np.dot(A2, imvec) * np.dot(A3, imvec))
    pt2 = wdiff * np.dot(A1, imvec) * np.dot(A3, imvec)
    pt3 = wdiff * np.dot(A1, imvec) * np.dot(A2, imvec)
    out = -np.real(np.dot(pt1, A1) + np.dot(pt2, A2) + np.dot(pt3, A3)) / len(Bvis)
    return out

def chisq_cphase(imvec):
    """Closure Phases (normalized) chi-squared"""
    clphase = np.angle(Bvis)

    clphase_samples = np.angle(np.dot(A1, imvec) * np.dot(A2, imvec) * np.dot(A3, imvec))
    chisq= (2.0/len(clphase)) * np.sum((1.0 - np.cos(clphase-clphase_samples)))
    return chisq

def chisqgrad_cphase(imvec):
    """The gradient of the closure phase chi-squared"""
    clphase = np.angle(Bvis)

    i1 = np.dot(A1, imvec)
    i2 = np.dot(A2, imvec)
    i3 = np.dot(A3, imvec)
    clphase_samples = np.angle(i1 * i2 * i3)

    pref = np.sin(clphase - clphase_samples)
    pt1  = pref/i1
    pt2  = pref/i2
    pt3  = pref/i3
    out  = -(2.0/len(clphase)) * np.imag(np.dot(pt1, A1) + np.dot(pt2, A2) + np.dot(pt3, A3))
    return out

def sflux(imvec):
    '''Total flux constraint'''
    out = -(np.sum(imvec) - np.sum(T)/T.size) ** 2
    return out


def sfluxgrad(imvec):
    """Total flux constraint gradient"""
    out = -2 * (np.abs(imvec) - np.sum(T)/T.size)
    return out
alpha=10e-0
beta=10e-0

def objfunc(imvec):
    if imvec.any() >=0:
        imvec=10/(1+np.exp(-imvec))
    else:
        imvec=10*np.exp(imvec)/(1+np.exp(imvec))
    imvec = np.exp(imvec)
    datterm = chisq_bs(imvec)
    regterm = sflux(imvec)
    conterm = chisq_cphase(imvec)
    return datterm + beta*conterm#  alpha*regterm+
def objgrad(imvec):
    if imvec.any() >=0:
        imvec=10/(1+np.exp(-imvec))
    else:
        imvec=10*np.exp(imvec)/(1+np.exp(imvec))
    imvec = np.exp(imvec)
    datterm = chisqgrad_bs(imvec)
    regterm = sfluxgrad(imvec)
    conterm = chisqgrad_cphase(imvec)
    grad =datterm + beta*conterm#  alpha*regterm+
    grad*=imvec
    return grad

global nit
nit = 0

def plotcur(im_step):
    global nit

    if im_step.any() >=0:
        im_step=10/(1+np.exp(-im_step))
    else:
        im_step=10*np.exp(im_step)/(1+np.exp(im_step))
    
    chi_bvis = chisq_bs(im_step)
    sflux_im = sflux(im_step)
    im_step = np.exp(im_step)
    plot_i(im_step)
    print("i: %d chi: %0.2f s1: %0.2f" % (nit, chi_bvis, sflux_im))
    nit += 1

def plot_i(imvec):

    plt.figure(2)
    plt.ion()
    plt.pause(5.e-2)
    plt.clf()
    plt.plot(xi,imvec.real)
    plotstr = 'T'
    plt.title(plotstr, fontsize=18)
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'T')

#if __name__ == "__main__":
    #nit = 0  # global variable to track the iteration number in the plotting callback
MAXLS = 1000  # maximum number of line searches in L-BFGS-B
NHIST = 1000  # number of steps to store for hessian approx
MAXIT = 1000  # maximum number of iterations
STOP = 1.e-11 # convergence criterion
optdict = {'maxiter': MAXIT, 'ftol': STOP, 'maxcor': NHIST, 'gtol': STOP, 'maxls': MAXLS}  # minimizer dict params
tstart = time.time()
res = minimize(objfunc, xinit, method='L-BFGS-B',jac=objgrad,options=optdict, callback=plotcur)
tstop = time.time()
print(res.x)
print("chi: %0.2f + s1: %0.2f = %0.2f" % (chisq_bs(res.x), sflux(res.x), chisq_bs(res.x) + sflux(res.x)))
plt.figure(8)
if res.x.any() >= 0:
    res.x = 10 / (1 + np.exp(-res.x))
else:
    res.x = 10 * np.exp(res.x) / (1 + np.exp(res.x))

plt.plot(xi,np.exp(res.x.real))
plt.show()


print(np.exp(res.x.real))

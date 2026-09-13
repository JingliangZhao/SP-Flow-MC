# SVR module

import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import torch, random
from einops import rearrange
import torch.nn.functional as F
from scipy.ndimage import zoom
import torch.nn as nn


# SVR module
def voxel_tv_batch(p0s, env, seg, model, dc, nv=50, iter=3):
    
    # p0s   input 3D positions of the particles, size=[np, 3]
    # env   CTA image
    # seg   Segmented vessel lumen, which is used for estimate the width of the 2.5D slice
    # dc    Template for fast extraction of 2.5D slice
    # nv    Number of initial vectors in SVR
    # iter  Number of SVR executions
    
    lp  = p0s.shape[0]
    vv0 = get_vs(np.array([0,0,1]), 10, nv)[0:nv]       # nv random vectors uniformly sampled within a solid angle domain
    vv0 = vv0.repeat(lp, axis=0)                        # size=[lv*lp], each line corresponds to a 2.5D slice
    la  = nv*lp
    
    # Refine the initial vectors three times
    for kk in range(0, iter):
        
        # 1. Info of the 2.5D slices
        pas = []   # pose information of the 2.5D slices
        for kp in range(0, lp):
            for ko in range(0, nv):
                id = kp*nv + ko                                     # the actual ID in vv0
                vn = vv0[id,  :];   [v1, v2] = cross123_r(vn)       # vn-v1-v2, local coordinate system
                pasi = np.hstack([p0s[kp, :], vn, v1, v2])
                pasi[12] = wd_estimate(seg, pasi, dc)               # estimate a suitable width of the slice
                
                pas.append(pasi)
        
        # 2. Extract slices with Info
        pas = np.array(pas);   ims = np.zeros([la, 3, 33, 33])   # 样本总量 nv*lp
        for i in range(0, la):
            ims[i, :] = get_sample_3(pas[i], env, dc, step=2)
        
        # 3. Predice alpha and beta, and further refine vn to vf
        ims = torch.from_numpy(ims).to('cuda', dtype=torch.float)
        with torch.no_grad():
            lb_pred = model(ims).to('cpu').numpy()
            vvf = lb_to_tv(lb_pred, pas)
            
            vv0 = vvf.copy()
    
    # Clustering
    vv2 = np.zeros([lp, 3])
    for kp in range(0, lp):
        t  = vvf[kp*nv:(kp+1)*nv, :]                    # nv vectors that belong to the same particle
        vv2[kp, :] = mean_shift_dv( t, math.pi/6 )
    
    return vv2   # size=[np,3]


# Extract 2.5D slice with fast template
def get_sample_3(pasi, img, dc, step=2):
    
    # step: inter slice spacing
    
    szx, szy, szz = img.shape
    r = pasi[12].astype(np.uint8).tolist()
    d = 16;   sp = r/d
    
    pp0 = pasi[0:3];      vi  = pasi[3:6]       # vi---z-axis
    v1  = sp*pasi[6:9];   v2 = sp*pasi[9:12]
    
    d1 = dc[0];   d2 = dc[1]                    # Fast template
    
    xx = [];   yy = [];   zz = []
    for k in range(-1, 2):
        ppk = pp0 + step*k*vi
        
        xxk = np.round( ppk[0] + d1*v1[0] + d2*v2[0] )
        yyk = np.round( ppk[1] + d1*v1[1] + d2*v2[1] )
        zzk = np.round( ppk[2] + d1*v1[2] + d2*v2[2] )
        
        xx = np.hstack( [xx, xxk] )
        yy = np.hstack( [yy, yyk] )
        zz = np.hstack( [zz, zzk] )
    
    xx[xx<0]=0;   xx[xx>szx-1]=szx-1
    yy[yy<0]=0;   yy[yy>szy-1]=szy-1
    zz[zz<0]=0;   zz[zz>szz-1]=szz-1
    
    ps = img[np.int16(xx), np.int16(yy), np.int16(zz)]
    
    # Restore to image
    sz = 2*d + 1;   im_sz = sz*sz
    ims = np.zeros([3, sz, sz])
    for k in range(0, 3):
        t = ps[k*im_sz : (k+1)*im_sz]
        ims[k] = t.reshape([sz,sz])
    
    return mat2gray_bk(ims)*2 - 1   # Normalize the grayscale to [-1, 1]


# Grayscale Normalization
def mat2gray_bk(im):
    im = im.astype(float)
    m1=np.min(im);   m2=np.max(im)
    
    if m2-m1 < 1e-6:
        return im
    else:
        det = 1/(m2-m1)
        im2 = (im - m1)*det
        
        return im2


# Function for turning vn to vf
def lb_to_tv(lb_pred, pas):
    
    # lb_pred:      predicted alpha and beta by CRN
    # pas:          slice Info
    
    dv = math.pi/4
    B  = lb_pred.shape[0]   # batch_size
    
    vvs = []
    for i in range(0, B):
        ida = lb_pred[i, :]*2 - 1                       # [0, 1] -> [-1, 1]
        [alf, beta] = cart2pol(ida);   alf = alf*dv
        
        vn = pas[i, 3:6];   v1 = pas[i, 6:9];   v2 = pas[i, 9:12]
        
        vv = vn*np.cos(alf) + np.sin(alf)*( v1*np.cos(beta) + v2*np.sin(beta) )
        vvs.append(vv)
    
    return np.array(vvs)


# Mean-shift clustering for vectors
def mean_shift_dv( data, radius ):
    
    # data:     Input vectors, size=[n,3]
    # radius:   Maximum distance between clusters
    
    gama = 0.000001
    
    # mean-shift
    m = data.shape[0];   index = np.arange(0, m)
    stopthresh = 0.001*radius;   radius2 = radius*radius
    visitflag = np.zeros([m,1])
    count = [];   clustern = 0;   clustercenter = []
    
    ddd=0
    while len(index)>0:
        ddd=ddd+1
        if ddd>12:
            break
        
        fg=0
        
        cn = np.floor( len(index)*np.random.rand() ).astype(np.uint16)
        center=data[index[cn],:]
        this_class=np.zeros([m,1])
        
        ccc=0
        while 1:
            ccc=ccc+1
            if ccc>8:
                break
            
            dis=np.zeros(m)
            for i in range(0,m):
                tt=np.dot(center, data[i,:])
                if tt>1:
                    tt=1
                dis[i]=np.arccos(tt)
            
            innerS = np.where(dis<radius);   innerS=innerS[0]
            visitflag[innerS] = 1
            this_class[innerS] = this_class[innerS]+1
            if innerS.shape[0]==1:
                fg=1;   break
            
            newcenter = np.zeros([1,3]);   sumweight=0
            for i in range(0, innerS.shape[0]):
                tt = dis[innerS[i]]
                w = math.exp(tt**2/radius2)
                sumweight = w + sumweight
                newcenter = newcenter + w*data[innerS[i],:]
            
            newcenter = ( newcenter/np.linalg.norm(newcenter) ).reshape([3,])
            
            if np.arccos(np.dot(newcenter, center))<stopthresh:
                break
            
            center=newcenter.copy()
        
        
        if fg==1:
            index = np.where(visitflag==0);   index=index[0];   continue
        
        mergewith=0
        if clustern>0:
            for i in range(0,clustern):
                betw=np.arccos( np.dot(center, clustercenter[i]) )
                if betw<radius/2:
                    mergewith=i;   break
        
        if mergewith==0:
            clustern=clustern+1
            clustercenter.append(center)
        else:
            tt = clustercenter[mergewith] + center;   tt=tt/np.linalg.norm(tt)
            clustercenter.append(tt)
        
        index = np.where(visitflag==0);   index=index[0]
    
    dis=np.zeros(clustern);   ps = [0 for x in range(0,clustern)]
    for i in range(0,clustern):
        di=np.zeros(m)
        for j in range(0,m):
            ab = np.dot(clustercenter[i], data[j,:])
            ab = np.clip(ab, -1+gama, 1-gama)
            di[j] = np.arccos(ab)
        
        tt=np.where(di<radius);   tt=tt[0];   ps[i]=tt
        dis[i]=tt.shape[0]
    
    ii = np.where(dis==np.max(dis));   ii=ii[0][0]
    clustercenter = clustercenter[ii]
    
    return clustercenter























#

#

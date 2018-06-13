import os
import sys
import numpy as np
import h5py
import random 
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from sklearn.neighbors import KDTree 
from numpy import linalg as la
from sklearn import preprocessing

#list_filename = 'train_files.txt'

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def loadDataFile(filename):
    return load_h5(filename)

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def point_group_first(idx_data,data,PGK,PGN):
    batch_size = data.shape[0]
    data_feature_num = data.shape[-1]
    
    pg_data = np.zeros([batch_size,PGK,PGN,data_feature_num])
    pg_data_idx = np.zeros([batch_size,PGK,3])
    pg_data_all = np.zeros([batch_size,PGK,3])
    for point_idx in range (batch_size):
        rerowcol=idx_data[point_idx]
        real_data = data[point_idx]
        KernelList=random.sample(range(rerowcol.shape[0]),PGK)
        #rerowcol_downsample is downsample point
        rerowcol_downsample=rerowcol[KernelList,:]
        kdt = KDTree(rerowcol, leaf_size=30, metric='euclidean')
        pgi=kdt.query(rerowcol_downsample, k=PGN, return_distance=False)
        pgr=np.zeros([PGK,PGN,data_feature_num])
        for n in range(pgi.shape[0]):
            pgr[n]=pointgroup_svd(real_data[pgi[n,:],:])
        pg_data[point_idx] = pgr
        pg_data_idx[point_idx] = rerowcol_downsample
        pg_data_all[point_idx] = pointgroup_svd(rerowcol_downsample)
        
    return pg_data, pg_data_idx,pg_data_all

def pointgroup_svd (pointgroup):
    scaler = preprocessing.StandardScaler().fit(pointgroup)
    pg_mean = scaler.mean_
    pg_sta = scaler.transform(pointgroup)
    U,sigma,VT=la.svd(pg_sta)

    #print(np.dot(pg_mean, VT[:,0]))
    #first
    if (np.dot(pg_mean,VT[:,0])<0):
        U[:,0]=-U[:,0]
        VT[:,0]=-VT[:,0]

    #second
    if (np.dot(pg_mean,VT[:,1])<0):
        U[:,1]=-U[:,1]
        VT[:,1]=VT[:,1]

    #third
    if (np.dot(pg_mean,VT[:,2])<0):
        U[:,2]=-U[:,2]
        VT[:,2]=-VT[:,2]
    
    pg_s = U[:,:3]
    scaler_s = preprocessing.StandardScaler().fit(pg_s)
    pg_s_sta = scaler_s.transform(pg_s)
    # #disply point cloud
    # point = pointgroup
    # point1 = pg_s
    # print(point1.shape)
    # print(point.shape)
    # fig1=plt.figure(dpi=120)  
    # ax1=fig1.add_subplot(111,projection='3d')  
    # plt.title('point cloud1')
    # for i in range(1024):  
    #     col = random.sample(range(1, 100), 3)
    #     ax1.scatter(point1[i,0],point1[i,1],point1[i,2],color=[float(col[0])/100.0, float(col[1])/100.0, float(col[2])/100.0],marker='.',s=10,linewidth=1,alpha=1,cmap='spectral')  
    # ax1.axis('scaled') 
    # ax1.set_xlabel('X Label')  
    # ax1.set_ylabel('Y Label')  
    # ax1.set_zlabel('Z Label') 

    # fig=plt.figure(dpi=120)  
    # ax=fig.add_subplot(111,projection='3d')  
    # plt.title('point cloud')
    # for i in range(1024):  
    #     col = random.sample(range(1, 100), 3)
    #     ax.scatter(point[i,0],point[i,1],point[i,2],color=[float(col[0])/100.0, float(col[1])/100.0, float(col[2])/100.0],marker='.',s=10,linewidth=1,alpha=1,cmap='spectral')  
    # ax.axis('scaled') 
    # ax.set_xlabel('X Label')  
    # ax.set_ylabel('Y Label')  
    # ax.set_zlabel('Z Label')  
    # plt.show()
    return pg_s_sta
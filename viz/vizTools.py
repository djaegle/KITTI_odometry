#!/usr/bin/env python
from __future__ import division
import numpy as np
from scipy.io import loadmat
import h5py
import os
from os.path import join
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def eul2Rot(euls):
    """
    Converts the XYZ intrinsic Euler angle representation of a rotation to a rotation matrix.
    Each column of euls is [psi,theta,phi].T, where:
    psi: rotation about x
    theta: rotation about y
    phi: rotation about z
    """
    if len(np.shape(euls)) == 1:
        euls_new = np.zeros((euls.shape[0],1),dtype=euls.dtype)
        euls_new[:,0] = euls[:]
        euls = euls_new

    psi = euls[0,:]
    theta = euls[1,:]
    phi = euls[2,:]
    rot_mat = np.zeros((3,3,np.shape(euls)[1]),dtype=np.float)

    rot_mat[0,0,:] = np.cos(theta)*np.cos(phi)
    rot_mat[0,1,:] = np.sin(psi)*np.sin(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi)
    rot_mat[0,2,:] = np.cos(psi)*np.sin(theta)*np.cos(phi) + np.sin(psi)*np.sin(phi)
    rot_mat[1,0,:] = np.cos(theta)*np.sin(phi)
    rot_mat[1,1,:] = np.sin(psi)*np.sin(theta)*np.sin(phi) + np.cos(psi)*np.cos(phi)
    rot_mat[1,2,:] = np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi)
    rot_mat[2,0,:] = -np.sin(theta)
    rot_mat[2,1,:] = np.sin(psi)*np.cos(theta)
    rot_mat[2,2,:] = np.cos(psi)*np.cos(theta)

    return rot_mat

def transformsToTrajectory(transform_sequence,sequence_type):
    """
    Converts a sequence of 3D trajectories (e.g. translation or translations+euler angles)
    into a trajectory of 3D points in a common reference frame. The first point is
    taken as the origin, and all translations and rotations are relative to the previous point.
    """
    n_spatial_dims = 3
    trajectory = np.zeros((transform_sequence.shape[0]+1,n_spatial_dims),dtype=np.float)

    rot_accum = np.eye(3,dtype=np.float)

    for i in xrange(transform_sequence.shape[0]):
        if sequence_type == 'trans':
            trans_i = transform_sequence[i,:3]

            # No rotation
            rot_i = np.eye(3,dtype=np.float)
        elif sequence_type == 'trans+euler':
            trans_i = transform_sequence[i,:3]

            # Populate the rotation matrix from XYZ Euler angles
            rot_i = np.squeeze(eul2Rot(transform_sequence[i,3:6]))
        else:
            raise ValueError('Sequence type %s not defined.' %sequence_type)

        rot_accum = np.dot(rot_accum,rot_i)

        # Use the transformation at index trans_i
        trajectory[i+1,:] = np.dot(rot_accum,trans_i) + trajectory[i,:]

    return trajectory

def filterPathTransforms(estimate_trajectory, gt_trajectory, camera_type, adjacent_ims, start_im, camera=1):
    """
    Returns an array of transformations corresponding to the un-augmented trajectory of the
    camera on the original image sequence.

    Optional argument camera sets which camera will be used (1 or 2)
    """

    valid_inds = np.squeeze(np.logical_and(camera_type == camera,adjacent_ims==1))

    sort_order = np.argsort(start_im[valid_inds])

    path_estimates = estimate_trajectory[valid_inds][sort_order]
    path_gt = gt_trajectory[valid_inds][sort_order]

    return path_estimates, path_gt


def plotTrajectory(estimates,ground_truth,kitti_poses,camera_type,adjacent_ims,start_im):
    """
    Plots 3D paths for a ground_truth and estimated trajectory. Takes either translation-only
    or translation+euler angle trajectories.
    """
    if estimates.shape[1] == 3: # Translation only
        sequence_type = 'trans'
    elif estimates.shape[1] == 6: # Rotation as well
        sequence_type = 'trans+euler'
    else:
        raise ValueError('Unknown sequence type.')

    # Include only the transformations between adjacent images
    estimates, ground_truth = filterPathTransforms(estimates,ground_truth,camera_type,adjacent_ims,start_im,camera=1)

    # Convert a trajectory sequence into a sequence of points
    estimate_trajectory = transformsToTrajectory(estimates,sequence_type)
    gt_trajectory = transformsToTrajectory(ground_truth,sequence_type)

    fig = plt.figure()
    ax = fig.add_subplot(121,projection='3d')
    ax.plot(estimate_trajectory[:,0],estimate_trajectory[:,1],estimate_trajectory[:,2], label='Estimated trajectory')
    ax.plot(kitti_poses[:,0],kitti_poses[:,1],kitti_poses[:,2],label='KITTI ground truth')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()

    ax = fig.add_subplot(122)
    ax.plot(estimate_trajectory[:,0],estimate_trajectory[:,2], label='Estimated trajectory')
    ax.plot(kitti_poses[:,0],kitti_poses[:,2],label='KITTI ground truth')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.legend()

    plt.show()

def getRawErrors(estimates,ground_truth):
    """
    Compute the errors between the estimated and ground truth translation and rotation,
    as well as the distance between each pair of points.
    """

    rot_errors = np.zeros(estimates.shape[0]) # In degrees

    trans_errors = np.sqrt(np.sum((estimates[:,:3]-ground_truth[:,:3])**2,axis=1)) # in m
    distances = np.sqrt(np.sum(ground_truth[:,:3]**2,axis=1)) # in m

    # Convert Euler angles to rotation matrices
    rotmats_estimate = eul2Rot(estimates[:,3:].T)
    rotmats_gt = eul2Rot(ground_truth[:,3:].T)

    for i in xrange(rot_errors.shape[0]):
        coserr = (np.trace(np.dot(eul2Rot(estimates[i,3:]).T,eul2Rot(ground_truth[i,3:]))) - 1) / 2
        # To account for numerical instability, just round to -1 or 1
        coserr = min(coserr,1)
        coserr = max(coserr,-1)

        rot_errors[i] = np.arccos(coserr)


    return distances, trans_errors, rot_errors


def main(file_path):
    estimate_data = loadmat(file_path)
    estimates = estimate_data['estimates']
    ground_truth = estimate_data['groundTruth']
    camera_type = estimate_data['cameraType']
    adjacent_ims = estimate_data['adjacentIms']
    start_im = np.squeeze(estimate_data['startIm'])
    kitti_poses = estimate_data['rawPose'][:,3::4]

    plotTrajectory(estimates,ground_truth,kitti_poses,camera_type,adjacent_ims,start_im)


if __name__ == "__main__":
    data_root = './saved_states'
    mat_file = 'mostRecentEstimates_seq09.mat'
    file_path = join(data_root,mat_file)

    main(file_path)

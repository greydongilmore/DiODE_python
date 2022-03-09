#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import nibabel as nb
import numpy as np
from skimage import measure
from sklearn.linear_model import LinearRegression
import math
from scipy.fft import fft
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import interpn
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


def pixLookup(slice_,y,x,zpad, RGB=3):
	"""
	This helper function looks up a pixel value from a given input image
	img is the input image (RGB or Grayscale)
	yx is the coordinate and repEdge tells the condition for pixel values out
	side of img (Use round up convention)
	"""
	pixVal=np.zeros((1,1,RGB))
	
	if RGB==3:
		ROW,COL,_=slice_.shape
	else:
		ROW,COL=slice_.shape
	
	# If the pixel value is outside of image given
	if np.any([(x<=0), (x>COL), (y<=0), (y>ROW)]):
		if zpad:
			pixVal=0
		else:
			y0=y
			x0=x
			y0[y0<1]=1
			x0[x0<1]=1
			y0[y0>ROW]=ROW
			x0[x0>COL]=COL
			# Bug fix suggested by Sanjuro
			pixVal=slice_[np.floor(y0),np.floor(x0),:]
	else:
		if RGB==3:
			pixVal=slice_[int(np.floor(y)),int(np.floor(x)),:]
		else:
			pixVal=slice_[int(np.floor(y)),int(np.floor(x))]
	
	return pixVal

def ea_diode_interpimage(slice_,yx,zpad=True, RGB=False):
	
	if RGB == 0:
		RGB=np.ndim(slice_)
	
	yx0=np.floor(yx)
	wt=yx-yx0
	wtConj=1-wt
	interTop=wtConj[1]*pixLookup(slice_,yx0[0],yx0[1],zpad,RGB) + wt[1]* pixLookup(slice_,yx0[0],yx0[1],zpad,RGB)
	interBtm=wtConj[1]* pixLookup(slice_,yx[0],yx[1],zpad,RGB) + wt[1]* pixLookup(slice_,yx[0],yx[1],zpad,RGB)
	interVal=wtConj[0]*interTop + wt[0]* interBtm

	return interVal

def ea_sample_slice(vol, tracor, wsize, voxmm, coords, el, interp_order=1):
	"""
	function samples a slice from nifti image based on coordinates and the
	wsize parameter (will use coordinate and sample a square that is 2xwsize
	long in distances). wsize can be given as mm or voxel distance (defined
	by voxmm parameter being either 'mm' or 'vox').
	Define parameter vol as spm volume (see spm_vol), define tracor as either
	'tra', 'cor' or 'sag' for sampling direction. define coords as a set of
	points and el defining the point that is being sampled.
	__________________________________________________________________________________
	Copyright (C) 2015 Charite University Medicine Berlin, Movement Disorders Unit
	Andreas Horn
	"""
	
	interpfactor=2
	voxel_dims = (vol.header["dim"])[1:4]
	
	if voxmm=='mm':
		# calculate distance in millimeters (wsize) back to voxels:
		probe=np.linalg.lstsq(ct_obj.affine, np.c_[[0,0,0,1], [wsize,0,0,1]], rcond=None)[0]
		wsize=abs(np.round(probe[0,0]-probe[0,1],0))
	
	getfullframe=0
	if len(coords)==1: # scalar input, only a height is defined. convert to mm space.
		getfullframe=1
	
	if tracor == 'tra':
		if getfullframe:
			boundbox=[
				np.linspace(0, voxel_dims[0], 500),
				np.linspace(0, voxel_dims[1], 500),
				np.linspace(coords, coords, 500)
			]
		else:
			boundbox=[np.arange(coords[0] - wsize, (coords[0] + wsize) + (1/interpfactor), 1/interpfactor)]
			boundbox.extend([
				np.arange(coords[1] - wsize, (coords[1] + wsize) + (1/interpfactor), 1/interpfactor),
				np.tile(coords[2],(1, len(boundbox[0])))[0]
			])
		
		xi, yi= np.meshgrid(boundbox[0],boundbox[1])
		orig_shape=xi.shape
		zi = np.tile(boundbox[2][0], (xi.size,1))
		for arr in (xi,yi,zi):
			arr.shape=-1
		
		ima = np.empty(xi.shape, dtype=float)
		map_coordinates(vol.get_fdata(), (xi,yi,zi), order=interp_order, output=ima)
		slice_=ima.reshape(orig_shape)
		
		sampleheight=(vol.affine @ np.array([1,1, boundbox[2][0],1]))[2]
	elif tracor == 'cor':
		if getfullframe:
			boundbox=[
				np.linspace(0, voxel_dims[0], 500),
				np.linspace(coords, coords, 500),
				np.linspace(0, voxel_dims[2], 500)
			]
		else:
			boundbox=[np.arange(coords[0] - wsize, (coords[0] + wsize) + (1/interpfactor), 1/interpfactor)]
			boundbox.extend([
				np.tile(coords[2],(1, len(boundbox[0])))[0],
				np.arange(coords[2] - wsize, (coords[2] + wsize) + (1/interpfactor), 1/interpfactor),
			])
		
		xi, zi= np.meshgrid(boundbox[0],boundbox[2])
		yi = boundbox[1][0] * np.ones_like(xi)
		orig_shape=xi.shape
		for arr in (xi,yi,zi):
			arr.shape=-1
		
		ima = np.empty(xi.shape, dtype=float)
		map_coordinates(vol.get_fdata(), (xi,yi,zi), order=interp_order, output=ima)
		slice_=ima.reshape(orig_shape)
		
		sampleheight=(vol.affine @ np.array([1,boundbox[1][0],1,1]))[1]
	elif tracor == 'sag':
		if getfullframe:
			boundbox=[
				np.linspace(coords, coords, 500),
				np.linspace(0, voxel_dims[1], 500),
				np.linspace(0, voxel_dims[2], 500)
			]
		else:
			boundbox=[np.arange(coords[1] - wsize, (coords[1] + wsize) + (1/interpfactor), 1/interpfactor)]
			boundbox.extend([
				np.tile(coords[0],(1, len(boundbox[1])))[0],
				np.arange(coords[2] - wsize, (coords[2] + wsize) + (1/interpfactor), 1/interpfactor),
			])
		
		yi, zi= np.meshgrid(boundbox[1],boundbox[2])
		xi = boundbox[0][0] * np.ones_like(yi)
		orig_shape=yi.shape
		for arr in (xi,yi,zi):
			arr.shape=-1
		
		ima = np.empty(yi.shape, dtype=float)
		map_coordinates(vol.get_fdata(), (xi,yi,zi), order=interp_order, output=ima)
		slice_=ima.reshape(orig_shape)
		
		sampleheight=(vol.affine @ np.array([boundbox[0][0],1,1,1]))[0]

	return slice_,boundbox,sampleheight


def ea_diode_intensityprofile(slice_,center,voxsize,radius):
	radius = radius * 2
	vector = np.c_[0,1] * (radius / voxsize[0])
	vectornew = []
	angle = []
	intensity = []
	for k in range(360):
		theta = (2*np.pi/360) * (k)
		rotmat = np.r_[np.c_[math.cos(theta),math.sin(theta)], np.c_[-math.sin(theta),math.cos(theta)]]
		vectornew.append((vector @ rotmat)[0] + center)
		angle.append(theta)
		intensity.append(ea_diode_interpimage(slice_, vectornew[-1]))
	
	vectornew=np.stack(vectornew)
	return angle, np.array(intensity),vectornew


def ea_diode_intensitypeaksFFT(intensity,noPeaks):
	"""
	this function detects 'noPeaks' number of intensity peaks. peaks are constrained to be at 360�/noPeaks angles to each other.
	Function runs a noPeaks * (360�/noPeaks) array over the intensity-profile and finds the angle at which the sum of all peaks is highest.
	"""
	fftint = fft(intensity)
	fftpart = fftint[noPeaks]
	amplitude = abs(fftpart)
	phase = -math.asin(np.real(fftpart) / amplitude)
	
	if np.imag(fftpart) > 0:
		if np.real(fftpart) > 0:
			phase = -np.pi -phase
		else:
			phase = np.pi -phase

	amplitude = (max(intensity) + abs(min(intensity))) / 2
	level = max(intensity) - amplitude
	
	sprofil=[]
	for k in range(360):
		sprofil.append(amplitude * math.sin(np.deg2rad(noPeaks*k)-phase) + level)
	
	peak=[]
	for k in range(noPeaks):
		peak.append((k)*(360/noPeaks))
	peak=np.array(peak)
	
	sumintensity=[]
	for k in range(int(360/noPeaks)):
		sumintensity.append(np.sum([sprofil[int(x)] for x in peak]))
		peak +=1

	maxpeak = np.argmax(sumintensity)
	
	peak=[]
	for k in range(noPeaks):
		peak.append(round(maxpeak + (k)*(360/noPeaks),0))

	peak=[int(x) for x in peak]
	
	return peak,sprofil


def ea_diode_angle2roll(angle,yaw,pitch):
	roll = (math.sin(angle) * math.cos(pitch)) / ((math.cos(angle) * math.cos(yaw)) - (math.sin(angle) * math.sin(yaw) * math.sin(pitch)))
	roll = math.atan(roll)
	if angle < np.pi and roll < 0 and angle - roll > np.pi/2:
		roll = roll + np.pi

	if angle > np.pi and roll > 0 and angle - roll > np.pi/2:
		roll = roll - np.pi
	
	return roll


def ea_diode_rollpitchyaw(roll,pitch,yaw):
	a = pitch #around x axis
	b = yaw #around y axis
	c = roll #around z axis
	
	Mx = np.r_[np.c_[1, 0, 0], np.c_[0, math.cos(a), math.sin(a)], np.c_[0, -math.sin(a), math.cos(a)]]
	My = np.r_[np.c_[math.cos(b), 0, math.sin(b)],  np.c_[0, 1, 0],  np.c_[-math.sin(b), 0, math.cos(b)]]
	Mz = np.r_[np.c_[math.cos(c), -math.sin(c), 0],  np.c_[math.sin(c), math.cos(c), 0],  np.c_[0, 0, 1]]
	
	M = Mx @ My @ Mz
	
	return M,Mz,My,Mx


def ea_diode_lightmarker(roll,pitch,yaw,marker):
	marker = marker[:3].reshape(-1, 1)
	
	ven = np.c_[0,0.65,-0.75].T
	dor = np.c_[0,0.65,0.75].T
	
	M,_,_,_ = ea_diode_rollpitchyaw(roll-(np.pi/2),pitch,yaw)
	ven90 = M.dot(ven)
	dor90 = M.dot(dor)
	
	M,_,_,_ = ea_diode_rollpitchyaw(roll-(3*(np.pi/2)),pitch,yaw)
	ven270 = M.dot(ven)
	dor270 = M.dot(dor)
	
	# calculate intersecting points between vec60/270/300 and the z-plane through the dir-level artifact
	
	# unitvector from ven0 to dor0
	vec90 = (dor90-ven90) / np.linalg.norm(dor90-ven90)
	# ventral point at 0� from the directional level
	dir_ven90 = marker + ven90
	# dorsal point at 0� from the directional level
	dir_dor90 = marker + dor90
	# factor x of how many unitvectors dir_ven0 is distanced from the marker in the z-dimension
	dir_x90 = (marker[2] - dir_ven90[2]) / vec90[2]
	# intersecting point of the line from ven0 to dor0 withe the marker plane in the z-dimension
	dir_90 = dir_ven90 + (dir_x90 * vec90)
	
	vec270 = (dor270-ven270) / np.linalg.norm(dor270-ven270)
	dir_ven270 = marker + ven270
	dir_dor270 = marker + dor270
	dir_x270 = (marker[2] - dir_ven270[2]) / vec270[2]
	dir_270 = dir_ven270 + (dir_x270 * vec270)
	
	# create vectors corresponding to the dark lines of the artiface
	dir_vec1 = (dir_90 - dir_270) / np.linalg.norm(dir_90 - dir_270)
	
	# calculate the angles of the dark lines with respect to the y-axis
	dir_angles = math.atan2(np.linalg.norm(np.cross(dir_vec1.T,np.r_[0,1,0])[0]),np.dot(dir_vec1.T, np.r_[0,1,0]))
	if dir_vec1[0] < 0:
		dir_angles = -dir_angles
	
	dir_angles = np.c_[dir_angles, dir_angles + np.pi]
	dir_angles[dir_angles>2*np.pi] = dir_angles[dir_angles>2*np.pi] - (2* np.pi)
	dir_angles[dir_angles<0] = dir_angles[dir_angles<0] + (2* np.pi)
	dir_angles = (2 *np.pi) - dir_angles
	dir_angles = sorted(dir_angles)[0]
	
	return dir_angles


def ea_diode_perpendicularplane(normvec,p0,X,Y):
	d = -((normvec[0] * p0[0]) + (normvec[1] * p0[1]) + (normvec[2] * p0[2]))
	Z = (-(normvec[0] * X)-(normvec[1] * Y) -d) / normvec[2]
	return Z

def index_coords(corner_locs, interp_locs):
	index = np.arange(len(corner_locs))
	if np.all(np.diff(corner_locs) < 0):
		corner_locs, index = corner_locs[::-1], index[::-1]
	return np.interp(interp_locs, corner_locs, index)

def interp3(x, y, z, vol, xi, yi, zi, interp_order=1):
	"""Sample a 3D array "v" with pixel corner locations at "x","y","z" at the
	points in "xi", "yi", "zi" using linear interpolation. Additional kwargs
	are passed on to ``scipy.ndimage.map_coordinates``."""
	
	Xslice_copy = x.copy()
	Yslice_copy = y.copy()
	Zslice_copy = z.copy()
	
	Xmm_copy = xi.copy()
	Ymm_copy = yi.copy()
	Zmm_copy = zi.copy()
	
	orig_shape=Xslice_copy.shape
	
	for arr in (Xslice_copy,Yslice_copy,Zslice_copy, Xmm_copy, Ymm_copy, Zmm_copy):
		arr.shape=-1
	
	coords = [index_coords(*item) for item in zip([Xmm_copy, Ymm_copy, Zmm_copy],[Xslice_copy, Yslice_copy, Zslice_copy])]
	
	ima = np.empty(Xslice_copy.shape, dtype=float)
	map_coordinates(vol, coords, order=interp_order, output=ima)
	
	return ima.reshape(orig_shape)


def ea_diode_calculateCOG(data,Xslice,Yslice,Zslice):
	data[np.isnan(data)] = 0
	
	xval = Xslice*data
	yval = Yslice*data
	zval = Zslice*data
	
	COG = np.r_[np.sum(xval) / np.sum(data), np.sum(yval) / np.sum(data),  np.sum(zval) / np.sum(data)]
	return COG


def calculatestreaks(p1,p2,radius):
	a = (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2
	b = 2 * (p1[0] * (p2[0] - p1[0]) + p1[1] * (p2[1] - p1[1]))
	c = p1[0] * p1[0] + p1[1] * p1[1] - radius**2
	
	lambda1 = (-b + np.sqrt(b*b - 4*a*c)) / (2*a)
	lambda2 = (-b - np.sqrt(b*b - 4*a*c)) / (2*a)
	
	#intersection of dark streak with validation circle
	x1 = p1[0] + lambda1 * (p2[0] - p1[0])
	y1 = p1[1] + lambda1 * (p2[1] - p1[1])
	x2 = p1[0] + lambda2 * (p2[0] - p1[0])
	y2 = p1[1] + lambda2 * (p2[1] - p1[1])
	
	ws1 = math.atan2(y1, x1)
	ws2 = math.atan2(y2, x2)
	
	# angle clockwise with respect to +y
	ws1 = -(np.pi/2) + ws1
	ws2 = -(np.pi/2) + ws2
	
	if ws1 < 0:
		ws1 = ws1 + (2*np.pi)
	
	if ws2 < 0:
		ws2 = ws2 + (2*np.pi)
	
	return ws1,ws2


def ea_diode_darkstar(roll,pitch,yaw,dirlevel,radius):
	# create vectors symbolizing the gaps between directional contacts at 60, 180 and 300 degrees
	# and transform them to match lead trajectory and directional level
	
	dirlevel = dirlevel[:3]
	
	ven = np.c_[0,0.65,-0.75].T
	dor = np.c_[0,0.65,0.75 ].T
	
	M,_,_,_ = ea_diode_rollpitchyaw(roll-((2*np.pi)/6),pitch,yaw)
	ven60 = M.dot(ven).T[0]
	dor60 = M.dot(dor).T[0]
	M,_,_,_ = ea_diode_rollpitchyaw(roll-(3*(2*np.pi)/6),pitch,yaw)
	ven180 = M.dot(ven).T[0]
	dor180 = M.dot(dor).T[0]
	M,_,_,_ = ea_diode_rollpitchyaw(roll-(5*(2*np.pi)/6),pitch,yaw)
	ven300 = M.dot(ven).T[0]
	dor300 = M.dot(dor).T[0]
	
	# calculate intersecting points between vec60/180/300 and the z-plane through the dir-level artifact
	
	# unitvector from ven60 to dor60
	vec60 = (dor60-ven60) / np.linalg.norm(dor60-ven60)
	# ventral point at 60° from the directional level
	dir_ven60 = dirlevel + ven60
	# dorsal point at 60° from the directional level
	dir_dor60 = dirlevel + dor60
	# factor x of how many unitvectors dir_ven60 is distanced from the dirlevel in the z-dimension
	dir_x60 = (dirlevel[2] - dir_ven60[2]) / vec60[2]
	# intersecting point of the line from ven60 to dor60 with the dirlevel plane in the z-dimension
	dir_60 = dir_ven60 + (dir_x60 * vec60)
	
	vec180 = (dor180-ven180) / np.linalg.norm(dor180-ven180)
	dir_ven180 = dirlevel + ven180
	dir_dor180 = dirlevel + dor180
	dir_x180 = (dirlevel[2] - dir_ven180[2]) / vec180[2]
	dir_180 = dir_ven180 + (dir_x180 * vec180)
	
	vec300 = (dor300-ven300) / np.linalg.norm(dor300-ven300)
	dir_ven300 = dirlevel + ven300
	dir_dor300 = dirlevel + dor300
	dir_x300 = (dirlevel[2] - dir_ven300[2]) / vec300[2]
	dir_300 = dir_ven300 + (dir_x300 * vec300)
	
	dir_angles_new=[]
	p1 = dir_60[:2] - dirlevel[:2]
	p2 = dir_180[:2] - dirlevel[:2]
	dir_angles_new.extend(calculatestreaks(p1,p2,radius))
	p1 = dir_180[:2] - dirlevel[:2]
	p2 = dir_300[:2] - dirlevel[:2]
	dir_angles_new.extend(calculatestreaks(p1,p2,radius))
	p1 = dir_300[:2] - dirlevel[:2]
	p2 = dir_60[:2] - dirlevel[:2]
	dir_angles_new.extend(calculatestreaks(p1,p2,radius))
	dir_angles = sorted(dir_angles_new)
	
	return dir_angles


def ea_diode_intensitypeaksdirmarker(intensity,angles):
	# this function detects 'noPeaks' number of intensity peaks. peaks are constrained to be at 360°/noPeaks angles to each other.
	# Function runs a noPeaks * (360°/noPeaks) array over the intensity-profile and finds the angle at which the sum of all peaks is highest.
	peak = np.round(np.rad2deg(angles),0)
	peak[peak<1] = peak[peak<1] + 359
	peak[peak>359] = peak[peak>359] - 359
	sumintensity = np.sum(intensity[peak.astype(int).tolist()])
	
	return sumintensity


def ea_diode_auto(ct_obj, head_mm_initial, unitvector_mm, elspec):
	
	# final returned dict with values
	solution={}
	
	voxel_dims = (ct_obj.header["dim"])[1:4]
	voxsize = (ct_obj.header["pixdim"])[1:4]
	
	level1centerRelative = elspec['contact_length'] + elspec['contact_spacing']
	level2centerRelative = (elspec['contact_length'] + elspec['contact_spacing']) * 2
	markercenterRelative = elspec['markerpos'] - (elspec['tip_length']* int(not(elspec['tipiscontact']))) - elspec['contact_length']/2
	
	samplelength = 20
	samplingvector_mm = np.vstack([
		np.linspace(head_mm_initial[0], (head_mm_initial[0] + (samplelength*unitvector_mm[0])), samplelength*(2)),
		np.linspace(head_mm_initial[1], (head_mm_initial[1] + (samplelength*unitvector_mm[1])), samplelength*(2)),
		np.linspace(head_mm_initial[2], (head_mm_initial[2] + (samplelength*unitvector_mm[2])), samplelength*(2)),
		np.ones(samplelength*(2))
	])
	
	samplingvector_vx = np.round(np.linalg.lstsq(ct_obj.affine,samplingvector_mm, rcond=None)[0],0)
	
	newcentervector_vx=[]
	for k in range(samplingvector_vx.shape[1]):
		tmp, bb, sh = ea_sample_slice(ct_obj, 'tra', extractradius, 'vox', samplingvector_vx[:3, k],1)
		tmp_label = tmp > 2000
		
		# find connected components remaining
		tmpcent = measure.regionprops(tmp_label.astype(int), tmp)
		centroids = np.array([prop.centroid for prop in tmpcent])
		if len(tmpcent) == 1:
			tmpcent = [int(x) for x in np.round(centroids, 0)[0]]
			newcentervector_vx.append([bb[0][tmpcent[0]],bb[1][tmpcent[1]], samplingvector_vx[2,k],1])
		elif len(tmpcent) > 1:
			tmpind = np.argmin(np.sum(abs(np.vstack(centroids) - np.c_[(tmp_label.shape[1])/2, (tmp_label.shape[1])/2]),1))
			centroids = tmpcent[tmpind].centroid
			tmpcent = [int(x) for x in np.round(centroids,0)]
			newcentervector_vx.append([bb[0][tmpcent[0]],bb[1][tmpcent[1]], samplingvector_vx[2,k],1])
		elif len(tmpcent) == 0:
			newcentervector_vx.append([np.nan]*3)
		
		print(f"Done sample {k+1} of {samplingvector_vx.shape[1]}: {newcentervector_vx[-1][:3]}")
	
	newcentervector_mm = ct_obj.affine @ np.stack(newcentervector_vx).T
	
	# fit linear model to the centers of mass and recalculate head and unitvector
	new = np.arange(0,samplelength,.5)

	lmx = LinearRegression()
	xmdl = lmx.fit(new.reshape(-1, 1),newcentervector_mm[0,:])

	lmy = LinearRegression()
	ymdl = lmy.fit(new.reshape(-1, 1),newcentervector_mm[1,:])

	lmz = LinearRegression()
	zmdl = lmz.fit(new.reshape(-1, 1),newcentervector_mm[2,:])
	
	head_mm = np.array([xmdl.predict(np.array(0).reshape(1,-1))[0],
			ymdl.predict(np.array(0).reshape(1,-1))[0],
			zmdl.predict(np.array(0).reshape(1,-1))[0],1])

	other_mm = np.array([xmdl.predict(np.array(10).reshape(1,-1))[0],
			ymdl.predict(np.array(10).reshape(1,-1))[0],
			zmdl.predict(np.array(10).reshape(1,-1))[0],1])

	unitvector_mm = (other_mm - head_mm)/np.linalg.norm(other_mm - head_mm)
	
	# calculate locations of markers and directional levels
	tail_mm = head_mm + (6 * unitvector_mm)
	marker_mm = head_mm + (markercenterRelative * unitvector_mm.T)
	dirlevel1_mm = head_mm + (level1centerRelative * unitvector_mm)
	dirlevel2_mm = head_mm + (level2centerRelative * unitvector_mm)
	
	# transform to vx
	marker_vx = np.round(np.linalg.lstsq(ct_obj.affine, marker_mm,rcond=None)[0],4)
	dirlevel1_vx = np.round(np.linalg.lstsq(ct_obj.affine, dirlevel1_mm,rcond=None)[0],4)
	dirlevel2_vx = np.round(np.linalg.lstsq(ct_obj.affine, dirlevel2_mm,rcond=None)[0],4)
	
	# in DiODe v2 only one directional level is used, starting at the
	# middle between both directional levels and being optimized later
	dirlevelnew_mm = np.mean(np.c_[dirlevel1_mm,dirlevel2_mm],1)
	dirlevelnew_vx = np.round(np.linalg.lstsq(ct_obj.affine, dirlevelnew_mm, rcond=None)[0],0)
	
	yaw = math.asin(unitvector_mm[0])
	pitch = math.asin(unitvector_mm[1]/math.cos(yaw))
	solution['polar1'] = np.rad2deg(math.atan2(np.linalg.norm(np.cross(np.r_[0,0,1],unitvector_mm[:3])),np.dot(np.r_[0,0,1],unitvector_mm[:3])))
	solution['polar2'] = -1*(np.rad2deg(math.atan2(unitvector_mm[1],unitvector_mm[0]))+ 90)
	
	fftdiff = []
	checkslices = np.linspace(-2,2,9) # check neighboring slices for marker +/- 1mm in .5mm steps

	for k in checkslices:
		checklocation_mm = marker_mm + (unitvector_mm * k)
		checklocation_vx = np.round(np.linalg.lstsq(ct_obj.affine, checklocation_mm,rcond=None)[0],0)
		slice_,_,_=ea_sample_slice(ct_obj,'tra',extractradius,'vox',(checklocation_vx[:3] + np.c_[0,0,k])[0], 1)
# 		if ct_obj.affine[0,0] < 0:
# 			tmp_label = np.flip(tmp_label,0)
# 		
# 		if ct_obj.affine[1,1] < 0:
# 			tmp_label = np.flip(tmp_label,1)
			
		center = np.c_[(tmp_label.shape[0])/2,(tmp_label.shape[0])/2][0]
		radius = 4
		
		# calculate intensityprofile and its FFT for each slice
		_, intensity,_ = ea_diode_intensityprofile(slice_, center,voxsize,radius)
		peak,tmpfft = ea_diode_intensitypeaksFFT(intensity,2)
		valley,_ = ea_diode_intensitypeaksFFT(-intensity,2)
		fftdiff.append(np.mean([tmpfft[int(x)] for x in peak]) - np.mean([tmpfft[int(x)] for x in valley]))
		
		print(f"Done checking slice {k} for marker")
	
	# select slice with maximum difference in fft and respecify
	# marker accordingly
	tmp_shift = checkslices[np.argmax(fftdiff)]
	marker_mm = marker_mm + (unitvector_mm * tmp_shift)
	marker_vx = np.round(np.linalg.lstsq(ct_obj.affine, marker_mm, rcond=None)[0],0)
	
	# extract marker artifact from slice
	artifact_marker,_,_ = ea_sample_slice(ct_obj,'tra', extractradius,'vox', marker_vx[:3], 1)
	
# 	if ct_obj.affine[0,0] < 0:
# 		artifact_marker = np.flip(artifact_marker,0)
# 	
# 	if ct_obj.affine[1,1]  < 0:
# 		artifact_marker = np.flip(artifact_marker,1)
	
	center_marker = np.c_[(artifact_marker.shape[0])/2, (artifact_marker.shape[0])/2][0]
	
	# extract intensity profile from marker artifact
	radius = 4
	angle_out, intensity, vector = ea_diode_intensityprofile(artifact_marker,center_marker,voxsize,radius)
	
	# detect peaks and valleys for marker artifact
	peak, markerfft = ea_diode_intensitypeaksFFT(intensity, 2)
	valley,_ = ea_diode_intensitypeaksFFT(-1*intensity, 2)
	
	# Detect angles of the white streak of the marker (only for intensityprofile-based ambiguity features)
	valley_roll = ea_diode_angle2roll(angle_out[int(valley[0])],yaw,pitch)
	marker_angles = ea_diode_lightmarker(valley_roll,pitch,yaw,marker_mm)
	
	solution={}
	solution['peaks'] = peak
	solution['center_marker'] = center_marker
	solution['valley'] = valley
	solution['markerfft'] = markerfft
	solution['intensity'] = intensity
	solution['vector'] = vector
	solution['angle'] = angle_out
	solution['artifact_marker']=artifact_marker
	solution['rolls_rad'] = [ea_diode_angle2roll(angle_out[int(peak[0])],yaw,pitch),ea_diode_angle2roll(angle_out[int(peak[1])],yaw,pitch)]
	solution['rolls_deg'] = np.rad2deg(solution['rolls_rad'])
	solution['rolls_streak_deg'] = np.rad2deg(marker_angles)
	
	# ASM
	# compares the maximum intensity between the valleys in 3 radii
	ASMradii = [3,6,9]
	ASMintensity_raw=[]
	for k in ASMradii:
		_, ASMintensity_tmp,_ = ea_diode_intensityprofile(artifact_marker,center_marker,voxsize,k)
		ASMintensity_raw.append(ASMintensity_tmp.T)
	
	
	ASMintensity = np.mean(np.stack(ASMintensity_raw), 0)
	if max(ASMintensity[valley[0]:valley[1]]) > max(ASMintensity[list(range(valley[0]))+list(range(valley[1],len(ASMintensity)))]):
		if peak[0] > valley[0] and peak[0] < valley[0]:
			print('ASM decides for peak 1')
			solution['ASM'] = 1
		else:
			print('ASM decides for peak 2')
			solution['ASM'] = 2
	else:
		if peak[0] > valley[0] and peak[0] < valley[1]:
			print('ASM decides for peak 2')
			solution['ASM'] = 2
		else:
			print('ASM decides for peak 1')
			solution['ASM'] = 1
	
	# Center of Mass method
	# this is where shit gets complicated
	# first, to orthogonal vectors, yvec which is the unitvector
	# pointing in the direction of peak(1) and x_vec, perpendicular
	# to it and unitvector are generated
	
	rolltmp = ea_diode_angle2roll(angle_out[peak[0]],yaw,pitch)
	M,_,_,_ = ea_diode_rollpitchyaw(rolltmp,pitch,yaw)
	yvec_mm = M.dot(np.r_[0,1,0])
	xvec_mm = np.cross(unitvector_mm[:3], yvec_mm)
	
	# create meshgrid for CT
	# coordinates in meshgrid format are created for the full
	# ct.img and a permuted ct is exported as Vnew (needed due to
	# the weird meshgrid format in Matlab)
	mincorner_mm = ct_obj.affine.dot(np.r_[1,1,1,1])
	maxcorner_mm = ct_obj.affine.dot(np.r_[np.array(ct_obj.get_fdata().shape), 1])
	
	Xmm=np.arange(mincorner_mm[0], maxcorner_mm[0], (maxcorner_mm[0]-mincorner_mm[0])/(ct_obj.get_fdata().shape[0]))
	Ymm=np.arange(mincorner_mm[1], maxcorner_mm[1],(maxcorner_mm[1]-mincorner_mm[1])/(ct_obj.get_fdata().shape[1]))
	Zmm=np.arange(mincorner_mm[2], maxcorner_mm[2],(maxcorner_mm[2]-mincorner_mm[2])/(ct_obj.get_fdata().shape[2]))
	
	vol_new = ct_obj.get_fdata().copy()
	
	# slice perpendicular
	# a 5mm slice with .1mm resolution is sampled perpendicular to
	# the lead at the position of the marker center and oriented in
	# the direction of x-vec and y-vec
	
	extract_width = 5
	samplingres = .1
	
	Xslice = (np.arange(-extract_width,extract_width+samplingres,samplingres) * xvec_mm[0]) + \
		(np.arange(-extract_width,extract_width+samplingres,samplingres) * yvec_mm[0]).reshape(-1,1) + marker_mm[0]
	
	Yslice = (np.arange(-extract_width,extract_width+samplingres,samplingres)* xvec_mm[1]) + \
		(np.arange(-extract_width,extract_width+samplingres,samplingres) * yvec_mm[1]).reshape(-1,1) + marker_mm[1]
	
	Zslice = ea_diode_perpendicularplane(unitvector_mm, marker_mm, Xslice, Yslice)
	
	
	myslice = interp3(Xslice, Yslice, Zslice, vol_new, Xmm, Ymm, Zmm)
	
	
	COG_mm = ea_diode_calculateCOG((myslice >= 2000).astype(int), Xslice, Yslice, Zslice)
	COG_dir = (COG_mm - marker_mm[:3])/np.linalg.norm((COG_mm - marker_mm[:3]))
	
	if np.sum(abs(yvec_mm-COG_dir)) < np.sum(abs(-yvec_mm-COG_dir)):
		print('COGtrans decides for peak 1')
		solution['COGtrans'] = 0
	else:
		print('COGtrans decides for peak 2')
		solution['COGtrans'] = 1
	
	# slice parallel
	# a 1.5mm slice with .1mm resolution is sampled vertically
	# through the lead and through the marker center and oriented
	# in the direction of y-vec and unitvector
	
	extract_width = 2.5
	samplingres = .1
	
	Xslice = (np.arange(-extract_width,extract_width+samplingres,samplingres)* unitvector_mm[0]) + \
		(np.arange(-extract_width,extract_width+samplingres,samplingres) * yvec_mm[0]).reshape(-1,1) + marker_mm[0]
		
	Yslice = (np.arange(-extract_width,extract_width+samplingres,samplingres)* unitvector_mm[1]) + \
		(np.arange(-extract_width,extract_width+samplingres,samplingres) * yvec_mm[1]).reshape(-1,1) + marker_mm[1]
	
	Zslice = ea_diode_perpendicularplane(xvec_mm, marker_mm, Xslice, Yslice)
	
	
	myslice = interp3(Xslice, Yslice, Zslice, vol_new, Xmm, Ymm, Zmm)
		
	
	COG_mm = ea_diode_calculateCOG((myslice >= 2000).astype(int),Xslice,Yslice,Zslice)
	COG_dir = (COG_mm - marker_mm[:3])/np.linalg.norm((COG_mm-marker_mm[:3]))
	
	if np.sum(abs(yvec_mm-COG_dir)) < np.sum(abs(-yvec_mm-COG_dir)):
		print('COGsag decides for peak 1')
		solution['COGsag'] = 0
	else:
		print('COGsag decides for peak 2')
		solution['COGsag'] = 1
	
	# Slice parallel for visualization
	# a 10mm slice with .1mm resolution is sampled vertically
	# through the lead and through the marker center and oriented
	# in the direction of y-vec and unitvector for later
	# visualization
	
	extract_width = 10
	samplingres = .1
	
	Xslice = (np.arange(-extract_width,extract_width+samplingres,samplingres)* unitvector_mm[0]) + \
		(np.arange(-extract_width,extract_width+samplingres,samplingres) * yvec_mm[0]).reshape(-1,1) + head_mm[0] + 7.5 * unitvector_mm[0]
		
	Yslice = (np.arange(-extract_width,extract_width+samplingres,samplingres)* unitvector_mm[1]) + \
		(np.arange(-extract_width,extract_width+samplingres,samplingres) * yvec_mm[1]).reshape(-1,1) + head_mm[1] + 7.5 * unitvector_mm[1]
	
	Zslice = ea_diode_perpendicularplane(xvec_mm, marker_mm, Xslice, Yslice)
	
	
	finalslice = interp3(Xslice, Yslice, Zslice, vol_new, Xmm, Ymm, Zmm)
	
	
# 	if np.rad2deg(angle_out[peak[0]]) < 90 or np.rad2deg(angle_out[peak[0]]) > 270:
# 		finalslice = np.flip(finalslice,1)
# 	
	# darkstar method
	checkslices = np.arange(-2,2,.5)
	
	sumintensitynew_final={}
	rollangles_final={}
	
	# solution 1
	sumintensitynew=[]
	rollangles=[]
	myroll = ea_diode_angle2roll(angle_out[peak[0]],yaw,pitch)
	for x in checkslices:
		checklocation_mm = dirlevelnew_mm + (unitvector_mm * x)
		checklocation_vx = np.round(np.linalg.lstsq(ct_obj.affine, checklocation_mm, rcond=None)[0],0)
		artifact_tmp,_,_=ea_sample_slice(ct_obj,'tra',extractradius,'vox',checklocation_vx[:3],1)
# 		if ct_obj.affine[0,0] < 0:
# 			artifact_tmp = np.flip(artifact_tmp,0)
# 	
# 		if ct_obj.affine[1,1]  < 0:
# 			artifact_tmp = np.flip(artifact_tmp,1)
		
		center_tmp = np.c_[(artifact_marker.shape[0])/2, (artifact_marker.shape[0])/2][0]
		radius = 8
		
		angle, intensity_tmp,vector = ea_diode_intensityprofile(artifact_tmp,center_tmp,voxsize,radius)
		# determine angles of the 6-valley artifact ('dark star') artifact in each of the slices for +30:-30 deg
		sumintensitynew_tmp=[]
		rollangles_tmp=[]
		for k in range(61):
			roll_shift = k-31
			rolltemp = myroll + np.deg2rad(roll_shift)
			dirnew_angles = ea_diode_darkstar(rolltemp,pitch,yaw,checklocation_mm,radius)
			sumintensitynew_tmp.append(ea_diode_intensitypeaksdirmarker(intensity_tmp,dirnew_angles))
			rollangles_tmp.append(np.deg2rad(roll_shift))
		sumintensitynew.append(sumintensitynew_tmp)
		rollangles.append(rollangles_tmp)
	
	sumintensitynew_final[0]=np.stack([np.array(x) for x in sumintensitynew])
	rollangles_final[0]=np.stack([np.array(x) for x in rollangles])
	
	darkstarangle=[]
	darkstarslice=[]
	darkstarangle.extend([np.argmin(np.min(sumintensitynew_final[0],0))])
	darkstarslice.extend([np.argmin(np.min(sumintensitynew_final[0],1))])
	
	
	# solution 2
	sumintensitynew=[]
	rollangles=[]
	myroll = ea_diode_angle2roll(angle[peak[1]],yaw,pitch)
	for x in checkslices:
		checklocation_mm = dirlevelnew_mm + (unitvector_mm * x)
		checklocation_vx = np.round(np.linalg.lstsq(ct_obj.affine, checklocation_mm, rcond=None)[0],0)
		artifact_tmp,_,_=ea_sample_slice(ct_obj,'tra',extractradius,'vox',checklocation_vx[:3],1)
# 		if ct_obj.affine[0,0] < 0:
# 			artifact_tmp = np.flip(artifact_tmp,0)
# 	
# 		if ct_obj.affine[1,1]  < 0:
# 			artifact_tmp = np.flip(artifact_tmp,1)
# 	
		center_tmp = np.c_[(artifact_marker.shape[0])/2, (artifact_marker.shape[0])/2][0]
		radius = 8
		
		angle, intensity_tmp,vector = ea_diode_intensityprofile(artifact_tmp,center_tmp,voxsize,radius)
		
		# determine angles of the 6-valley artifact ('dark star') artifact in each of the slices for +30:-30 deg
		sumintensitynew_tmp=[]
		rollangles_tmp=[]
		for k in range(61):
			roll_shift = k-31
			rolltemp = myroll + np.deg2rad(roll_shift)
			dirnew_angles = ea_diode_darkstar(rolltemp,pitch,yaw,checklocation_mm,radius)
			sumintensitynew_tmp.append(ea_diode_intensitypeaksdirmarker(intensity_tmp,dirnew_angles))
			rollangles_tmp.append(np.deg2rad(roll_shift))
		sumintensitynew.append(sumintensitynew_tmp)
		rollangles.append(rollangles_tmp)
	
	sumintensitynew_final[1]=np.stack([np.array(x) for x in sumintensitynew])
	rollangles_final[1]=np.stack([np.array(x) for x in rollangles])
	
	darkstarangle.extend([np.argmin(np.min(sumintensitynew_final[1],0))])
	darkstarslice.extend([np.argmin(np.min(sumintensitynew_final[1],1))])
	
	for k in range(2):
		sumintensitynew_final[k] = sumintensitynew_final[k][darkstarslice[k],:]
		rollangles_final[k] = rollangles_final[k][darkstarslice[k],:]
	
	if min(sumintensitynew[0]) < min(sumintensitynew[1]):
		print('Darkstar decides for peak 1')
		solution['Darkstar'] = 0
	else:
		print('Darkstar decides for peak 2')
		solution['Darkstar'] = 1
	
	finalpeak={}
	peakangle={}
	
	# Take COGtrans solution
	finalpeak = peak[solution['COGtrans']]
	
	peakangle = angle[finalpeak]
	roll = ea_diode_angle2roll(peakangle,yaw,pitch)
	
	realsolution = solution['COGtrans']
	
	dirlevelnew_mm = dirlevelnew_mm + (unitvector_mm * checkslices[darkstarslice[realsolution]])
	dirlevelnew_vx = np.round(np.linalg.lstsq(ct_obj.affine, dirlevelnew_mm, rcond=None)[0],0)
	
	artifact_dirnew,_,_ = ea_sample_slice(ct_obj,'tra',extractradius,'vox',dirlevelnew_vx[:3],1)
	
# 	if ct_obj.affine[0,0] < 0:
# 		artifact_dirnew = np.flip(artifact_dirnew,0)
# 	
# 	if ct_obj.affine[1,1]  < 0:
# 		artifact_dirnew = np.flip(artifact_dirnew,1)
	
	center_dirnew = np.c_[(artifact_dirnew.shape[0])/2, (artifact_dirnew.shape[0])/2][0]
	
	anglenew, intensitynew, vectornew = ea_diode_intensityprofile(artifact_dirnew,center_dirnew,voxsize,radius)
	
	rollnew = roll + rollangles[realsolution][darkstarangle[realsolution]]
	dirnew_angles = ea_diode_darkstar(rollnew,pitch,yaw,dirlevelnew_mm,radius)
	dirnew_valleys = np.round(np.rad2deg(dirnew_angles), 0)
	dirnew_valleys[dirnew_valleys > 360] = dirnew_valleys[dirnew_valleys > 360] - 360
	
	
	solution['realsolution'] = realsolution
	solution['finalslice'] = finalslice
	solution['artifact_dirnew'] = artifact_dirnew
	solution['finalpeak'] = finalpeak
	solution['center_marker'] = center_marker
	solution['center_dirnew'] = center_dirnew
	solution['markerfft'] = markerfft
	solution['intensitynew'] = intensitynew
	solution['anglenew'] = anglenew
	solution['vectornew'] = vectornew
	solution['dirnew_valleys'] = dirnew_valleys
	solution['sumintensitynew_final'] = sumintensitynew_final
	solution['rollangles_final'] = rollangles_final
	solution['darkstarangle'] = darkstarangle
	solution['darkstarslice'] = darkstarslice
	
	return solution


def gen_figure(solution, side):
	
	subtitle_text_options={
		'fontsize': 16, 
		'fontweight': 'bold'
		}
	text_options = {'horizontalalignment': 'center',
					'verticalalignment': 'center',
					'fontsize': 15,
					'fontweight': 'bold'}

	fig = plt.figure(figsize=(16,8))
	ax = fig.add_subplot(231)
	ax.imshow(solution[side]['artifact_marker'].T, cmap='gray',alpha=1, vmin=cscale[0], vmax=cscale[1])
	ax.plot(solution[side]['vector'][:,0], solution[side]['vector'][:,1], '-g')
	#ax.set_xticks([]),ax.set_yticks([])
	ax.invert_yaxis(),ax.invert_xaxis()
	ax.scatter(solution[side]['vector'][solution[side]['peaks'],0], solution[side]['vector'][solution[side]['peaks'],1],s=20, color='g',alpha=1)
	
	ax.scatter(solution[side]['vector'][solution[side]['finalpeak'],0], 
		solution[side]['vector'][solution[side]['finalpeak'],1],
		s=10, color='g',alpha=1)
	
	ax.quiver(solution[side]['center_marker'][0], 
		solution[side]['center_marker'][1], 
		solution[side]['vector'][solution[side]['finalpeak'],0] - solution[side]['center_marker'][0],
		solution[side]['vector'][solution[side]['finalpeak'],1] - solution[side]['center_marker'][1],
		linewidth=1,ec='g', angles='xy', scale=.5,scale_units='xy')
	
	ax.scatter(solution[side]['center_marker'][0],
		solution[side]['center_marker'][1],
		s=100, color='m',alpha=1)
	
	for k in solution[side]['valley']:
		xp=[solution[side]['center_marker'][0],(solution[side]['center_marker'][0] + 3 * (solution[side]['vector'][k,0]-solution[side]['center_marker'][0]))]
		yp=[solution[side]['center_marker'][1],(solution[side]['center_marker'][1] + 3 * (solution[side]['vector'][k,1]-solution[side]['center_marker'][1]))]
		ax.plot(xp, yp, '-r')
	
	xlimit=ax.get_xlim()
	ylimit=ax.get_ylim()
	ax.text(np.mean(xlimit),ylimit[1]-.15* np.mean(ylimit),'A', color='w',**text_options)
	ax.text(np.mean(xlimit),ylimit[0]+.15* np.mean(ylimit),'P', color='w',**text_options)
	ax.text(xlimit[0]+0.1*np.mean(xlimit),np.mean(ylimit),'L', color='b',**text_options)
	ax.text(xlimit[1]-0.1*np.mean(xlimit),np.mean(ylimit),'R',color='b', **text_options)
	ax.set_title('Axial View', **subtitle_text_options)
	
	
	ax = fig.add_subplot(232)
	ax.plot(np.rad2deg(solution[side]['angle']), solution[side]['intensity'])
	ax.plot(np.rad2deg(solution[side]['angle']), solution[side]['markerfft'])
	
	ax.scatter(np.rad2deg(np.array(solution[side]['angle'])[solution[side]['peaks']]),
		solution[side]['intensity'][solution[side]['peaks']],
		s=35, facecolors='none', edgecolors='g',alpha=1)
	
	ax.scatter(np.rad2deg(np.array(solution[side]['angle'])[solution[side]['finalpeak']]), solution[side]['intensity'][solution[side]['finalpeak']],s=35, facecolors='g', edgecolors='g',alpha=1)
	ax.scatter(np.rad2deg(np.array(solution[side]['angle'])[solution[side]['valley']]), solution[side]['intensity'][solution[side]['valley']],s=35, facecolors='none', edgecolors='r',alpha=1)
	ax.set_title('Intensity Profile', **subtitle_text_options)
	
	ax = fig.add_subplot(233)
	ax.imshow(np.rot90(solution[side]['finalslice'],3), cmap='gray',alpha=1, vmin=1500, vmax=3000)
	ax.set_xticks([]),ax.set_yticks([])
	ax.set_title('Sagittal View', **subtitle_text_options)
	
	
	ax = fig.add_subplot(234)
	ax.imshow(np.rot90(solution[side]['artifact_dirnew'],3), cmap='gray',alpha=1, vmin=cscale[0], vmax=cscale[1])
	ax.set_xticks([]),ax.set_yticks([])
	ax.set_title('Directional Level', **subtitle_text_options)
	ax.plot(solution[side]['vectornew'][:,0], solution[side]['vectornew'][:,1], '-g')
	
	ax.scatter(solution[side]['vectornew'][[int(x) for x in solution[side]['dirnew_valleys'].tolist()],0], 
			   solution[side]['vectornew'][[int(x) for x in solution[side]['dirnew_valleys'].tolist()],1],s=20, color='g',alpha=1)
	
	for k in [int(x) for x in solution[side]['dirnew_valleys']]:
		xp=[solution[side]['center_dirnew'][0],(solution[side]['center_dirnew'][0] + 1.5 * (solution[side]['vectornew'][k,0]-solution[side]['center_dirnew'][0]))]
		yp=[solution[side]['center_dirnew'][1],(solution[side]['center_dirnew'][1] + 1.5 * (solution[side]['vectornew'][k,1]-solution[side]['center_dirnew'][1]))]
		ax.plot(xp, yp, '-r')
	
	ax.scatter(solution[side]['center_dirnew'][0],
		solution[side]['center_dirnew'][1],
		s=100, color='m',alpha=1)
	
	xlimit=ax.get_xlim()
	ylimit=ax.get_ylim()
	ax.text(np.mean(xlimit),ylimit[0]-.15* np.mean(ylimit),'A', color='w',**text_options)
	ax.text(np.mean(xlimit),ylimit[1]+.15* np.mean(ylimit),'P', color='w',**text_options)
	ax.text(xlimit[0]+0.1*np.mean(xlimit),np.mean(ylimit),'L', color='b',**text_options)
	ax.text(xlimit[1]-0.1*np.mean(xlimit),np.mean(ylimit),'R',color='b', **text_options)
	ax.set_title('Directional Level', **subtitle_text_options)
	
	
	ax = fig.add_subplot(235)
	ax.plot(np.rad2deg(solution[side]['anglenew']), solution[side]['intensitynew'])
	ax.plot(np.rad2deg(solution[side]['angle']), solution[side]['markerfft'])
	
	ax.scatter(np.rad2deg(np.array(solution[side]['anglenew'])[[int(x) for x in solution[side]['dirnew_valleys'].tolist()]]), 
			   solution[side]['intensitynew'][[int(x) for x in solution[side]['dirnew_valleys'].tolist()]],
			   s=20, color='g',alpha=1)
	ax.set_title('Intensity Profile', **subtitle_text_options)
	
	
	
	ax = fig.add_subplot(236)
	ax.plot(np.rad2deg(solution[side]['rollangles_final'][solution[side]['realsolution']]),
			solution[side]['sumintensitynew_final'][solution[side]['realsolution']])
	ax.plot(np.rad2deg(solution[side]['rollangles_final'][int(not(solution[side]['realsolution']))]),
			solution[side]['sumintensitynew_final'][int(not(solution[side]['realsolution']))],
			color='r',alpha=1)
	
	ax.scatter(np.rad2deg(solution[side]['rollangles_final'][solution[side]['realsolution']][solution[side]['rollangles_final'][solution[side]['realsolution']]==0]), 
			   solution[side]['sumintensitynew_final'][solution[side]['realsolution']][solution[side]['rollangles_final'][solution[side]['realsolution']]==0],
			   s=20, color='g',alpha=1)
	
	ax.scatter(np.rad2deg(solution[side]['rollangles_final'][solution[side]['realsolution']][solution[side]['darkstarangle'][solution[side]['realsolution']]]), 
			   solution[side]['sumintensitynew_final'][solution[side]['realsolution']][solution[side]['darkstarangle'][solution[side]['realsolution']]],
			   s=20, color='r',alpha=1)
	
	xy_arrow=(np.rad2deg(solution[side]['rollangles_final'][solution[side]['realsolution']][solution[side]['darkstarangle'][solution[side]['realsolution']]]),
			solution[side]['sumintensitynew_final'][solution[side]['realsolution']][solution[side]['darkstarangle'][solution[side]['realsolution']]])
	
	text_arrow="{:.0f}".format(np.round(solution[side]['sumintensitynew_final'][solution[side]['realsolution']][solution[side]['darkstarangle'][solution[side]['realsolution']]],0))
	
	ax.annotate(text_arrow, xy=xy_arrow, xycoords='data',xytext=(xy_arrow[0]+20, xy_arrow[1]+20), arrowprops=dict(arrowstyle="->", color='black',linewidth=2), fontsize=14)
	
	ax.set_title('Similarity Index', **subtitle_text_options)
	
	plt.tight_layout(pad=2)



#%%


ctpath=r'/home/greydon/Downloads/sub-P238_ses-post_acq-Electrode_run-01_ct.nii.gz'
head_mm_initial = np.array([2.0, 1.0,48.1])
tail_mm_initial = np.array([8.4, 3.9, 69.5])


ctpath=r'/home/greydon/data/data/DBS/derivatives/trajGuide/derivatives/sub-232/source/sub-P232_ses-post_acq-Electrode_run-01_ct.nii.gz'
side = 'right'
head_mm_initial = np.array([24.943,-9.592, 48.404])
tail_mm_initial = np.array([40.572, -7.330, 86.161])
side = 'left'
head_mm_initial = np.array([3.673,-8.265,47.505])
tail_mm_initial = np.array([-5.356,2.707,72.543])


ctpath=r'/home/greydon/data/data/DBS/derivatives/trajGuide/derivatives/sub-231/source/sub-P231_ses-post_acq-Electrode_run-01_ct.nii.gz'
side = 'right'
head_mm_initial = np.array([15.036,-24.577,51.102])
tail_mm_initial = np.array([21.369,-8.614,72.443])

side = 'left'
head_mm_initial = np.array([-5.482,-25.054,51.622])
tail_mm_initial = np.array([-15.636,-11.433,72.096])


elmodel = 'Boston Scientific Vercise Directed'


with open('/home/greydon/Documents/GitHub/DiODE_python/diode_elspec.json') as elspec_file:
	diode_elspec = json.load(elspec_file)

elspec = diode_elspec[elmodel]

sides = ['right','left','3','4','5','6','7','8']
extractradius = 30
cscale=(-50,100)


ct_obj = nb.load(ctpath)
unitvector_mm = (tail_mm_initial - head_mm_initial)/np.linalg.norm(tail_mm_initial - head_mm_initial)

solution={}

solution[side] = ea_diode_auto(ct_obj, head_mm_initial, unitvector_mm, elspec)


gen_figure(solution, side)


#%%
















import numpy as np
import math
import nibabel as nib
from scipy import interpolate
from scipy.ndimage.interpolation import map_coordinates
import matplotlib.pyplot as plt

def ea_sample_slice(vol, tracor, wsize, voxmm, coords, el, interp_order=2):
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
		probe=np.linalg.lstsq(vol.affine, np.c_[[0,0,0,1], [wsize,0,0,1]], rcond=None)[0]
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
			boundbox=[np.arange(coords[0] - wsize, (coords[0] + wsize), 1/interpfactor)]
			boundbox.extend([
				np.arange(coords[1] - wsize, (coords[1] + wsize), 1/interpfactor),
				np.tile(coords[2],(1, len(boundbox[0])))[0]
			])
		# need to flip x and y here
		yi, xi= np.meshgrid(boundbox[0],boundbox[1])
		zi = np.tile(boundbox[2][0], (xi.shape))
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
		sampleheight=(vol.affine @ np.array([boundbox[0][0],1,1,1]))[0]
	
	orig_shape=xi.shape
	for arr in (xi,yi,zi):
		arr.shape=-1
	
	ima = np.empty(xi.shape, dtype=float)
	map_coordinates(vol.get_fdata(), (yi,xi,zi), order=interp_order, output=ima)
	slice_=ima.reshape(orig_shape)
	
	return slice_,boundbox,sampleheight


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

def darkstar(roll,pitch,yaw,dirlevel,radius):
	# create vectors symbolizing the gaps between directional contacts at 60, 180 and 300 degrees
	# and transform them to match lead trajectory and directional level
	
	dirlevel = dirlevel[:3]
	
	ven = np.c_[0,0.65,-0.75].T
	dor = np.c_[0,0.65,0.75 ].T
	
	M,_,_,_ = rollpitchyaw(roll-((2*np.pi)/6),pitch,yaw)
	ven60 = M.dot(ven).T[0]
	dor60 = M.dot(dor).T[0]
	M,_,_,_ = rollpitchyaw(roll-(3*(2*np.pi)/6),pitch,yaw)
	ven180 = M.dot(ven).T[0]
	dor180 = M.dot(dor).T[0]
	M,_,_,_ = rollpitchyaw(roll-(5*(2*np.pi)/6),pitch,yaw)
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

def ea_diode_intensitypeaksFFT(intensity, noPeaks):
	"""
	this function detects 'noPeaks' number of intensity peaks. peaks are constrained to be at 360�/noPeaks angles to each other.
	Function runs a noPeaks * (360�/noPeaks) array over the intensity-profile and finds the angle at which the sum of all peaks is highest.
	"""
	fftint = np.fft.fft(intensity)
	fftpart = fftint[noPeaks+1]
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
	for k in range(0,360):
		sprofil.append(amplitude * math.sin(np.deg2rad(noPeaks*k)-phase) + level)
	sprofil=np.array(sprofil)
	
	peak=[]
	for k in range(noPeaks):
		peak.append(int((k)*(360/noPeaks)))
	
	sumintensity=[]
	for k in range(int(360/noPeaks)):
		sumintensity.append(np.sum(sprofil[peak]))
		peak =[x+1 for x in peak]
	
	maxpeak = np.argmax(sumintensity)
	
	peak=[]
	for k in range(noPeaks):
		peak.append(round(maxpeak + (k)*(360/noPeaks),0).astype(int))
	
	return peak,sprofil

def peak_FFT(valSlice1,noPeaks=2):
	M = np.fft.fft(valSlice1)
	fftpart = M[noPeaks+1]
	amplitude = (max(valSlice1) + abs(min(valSlice1))) / 2
	phase = -math.sin (np.real(fftpart) / amplitude)
	m = max(abs(M))
	M[np.where(abs(M) < 0.99 * m)] = 0
	denoised_val = np.fft.ifft(M)
	angles = denoised_val.argsort()[-2:]
	print(f"Peak angles (in degrees): {angles[0]}, {angles[1]}")

# 	phase = 0
# 	for i in range(359):
# 		if denoised_val[i] == 0:
# 			phase = i
# 			break
# 		elif denoised_val[i] * denoised_val[i + 1] < 0:
# 			phase = i + 0.5
# 			break
# 	
	level = max(valSlice1) - amplitude
	sprofil=np.zeros(360)
	for theta in range(360):
		sprofil[theta]=amplitude * math.sin(np.deg2rad(noPeaks*theta)-phase) + level
	
	dangles = np.array([int(round(angles[0] + (45 - (phase / 2)))), int(round(angles[1] + (45 - phase / 2)))])
	print(f"** Denoised peak angles (in degrees): {dangles[0]}, {dangles[1]}")
	return dangles,angles,sprofil


def index_coords(corner_locs, interp_locs):
	index = np.arange(len(corner_locs))
	if np.all(np.diff(corner_locs) < 0):
		corner_locs, index = corner_locs[::-1], index[::-1]
	return np.interp(interp_locs, corner_locs, index)

def perpendicularplane(normvec,p0,X,Y):
	d = -((normvec[0] * p0[0]) + (normvec[1] * p0[1]) + (normvec[2] * p0[2]))
	Z = (-(normvec[0] * X)-(normvec[1] * Y) -d) / normvec[2]
	return Z

def calculateCOG(ct_obj,xvec_mm,yvec_mm,marker_mm, unitvector_mm, orientation='trans', interp_order=1):
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
	
	if orientation == 'trans':
		# slice perpendicular
		# a 5mm slice with .1mm resolution is sampled perpendicular to
		# the lead at the position of the marker center and oriented in
		# the direction of x-vec and y-vec
		extract_width = 5
		samplingres = .1
		message_txt = 'COGtrans'
		
		Xslice = (np.arange(-extract_width,extract_width+samplingres,samplingres) * xvec_mm[0]) + \
			(np.arange(-extract_width,extract_width+samplingres,samplingres) * yvec_mm[0]).reshape(-1,1) + marker_mm[0]
		
		Yslice = (np.arange(-extract_width,extract_width+samplingres,samplingres)* xvec_mm[1]) + \
			(np.arange(-extract_width,extract_width+samplingres,samplingres) * yvec_mm[1]).reshape(-1,1) + marker_mm[1]
		
		Zslice = perpendicularplane(unitvector_mm, marker_mm, Xslice, Yslice)
	elif orientation == 'sag':
		# slice parallel
		# a 1.5mm slice with .1mm resolution is sampled vertically
		# through the lead and through the marker center and oriented
		# in the direction of y-vec and unitvector
		extract_width = 1.5
		samplingres = .1
		message_txt = 'COGsag'
		
		Xslice = (np.arange(-extract_width,extract_width+samplingres,samplingres)* xvec_mm [0]) + \
			(np.arange(-extract_width,extract_width+samplingres,samplingres) * yvec_mm[0]).reshape(-1,1) + marker_mm[0]
			
		Yslice = (np.arange(-extract_width,extract_width+samplingres,samplingres)* xvec_mm[1]) + \
			(np.arange(-extract_width,extract_width+samplingres,samplingres) * yvec_mm[1]).reshape(-1,1) + marker_mm[1]
		
		Zslice = ea_diode_perpendicularplane(unitvector_mm, marker_mm, Xslice, Yslice)
	
	orig_shape=Xslice.shape
	for arr in (Xslice,Yslice,Zslice, Xmm, Ymm, Zmm):
		arr.shape=-1
	
	coords = [index_coords(*item) for item in zip([Xmm, Ymm, Zmm],[Xslice,Yslice,Zslice])]
	
	myslice = np.empty(Xslice.shape, dtype=float)
	map_coordinates(vol_new, coords, order=interp_order, output=myslice)
	myslice=myslice.reshape(orig_shape)
	
	slice_mask =(myslice >= 2000).astype(int)
	slice_mask[np.isnan(slice_mask)] = 0
	
	xval = Xslice.reshape(orig_shape).T*slice_mask
	yval = Yslice.reshape(orig_shape).T*slice_mask
	zval = Zslice.reshape(orig_shape).T*slice_mask
	
	COG_mm = np.r_[np.sum(xval) / np.sum(slice_mask), np.sum(yval) / np.sum(slice_mask),  np.sum(zval) / np.sum(slice_mask)]
	
	COG_dir = (COG_mm - marker_mm[:3])/np.linalg.norm(COG_mm - marker_mm[:3])
	
	if np.sum(abs(yvec_mm-COG_dir)) < np.sum(abs(-yvec_mm-COG_dir)):
		print(f'{message_txt} decides for peak 1')
		solution = 0
	else:
		print(f'{message_txt} decides for peak 2')
		solution = 1
	
	return solution

def calculateASM(slice_vol, center, voxsize, valley, peak):
	nDegree = 360
	ASMintensity_raw=[]
	ASMradii = [3,6,9]
	for iradius in ASMradii:
		#f = interpolate.interp2d(np.arange(dims[0]), np.arange(dims[1]), slice1, kind='linear')
		valSlice=np.zeros(360)
		for theta in range(360):
			x = iradius / voxsize[1] * math.sin(math.radians(theta)) + center[1]
			y = iradius / voxsize[0] * math.cos(math.radians(theta)) + center[0]
			valSlice[theta] = slice_vol[int(x), int(y)]
		ASMintensity_raw.append(valSlice)
	
	ASMintensity = np.mean(np.stack(ASMintensity_raw), 0)
	
	if max(ASMintensity[valley[0]:valley[1]]) > max(ASMintensity[list(range(valley[0]))+list(range(valley[1],len(ASMintensity)))]):
		if peak[0] > valley[0] and peak[0] < valley[0]:
			print('ASM decides for peak 1')
			solution = 0
		else:
			print('ASM decides for peak 2')
			solution = 1
	else:
		if peak[0] > valley[0] and peak[0] < valley[1]:
			print('ASM decides for peak 2')
			solution = 1
		else:
			print('ASM decides for peak 1')
			solution = 0
	return solution

def angle2roll(angle,yaw,pitch):
	roll = (math.sin(angle) * math.cos(pitch)) / ((math.cos(angle) * math.cos(yaw)) - (math.sin(angle) * math.sin(yaw) * math.sin(pitch)))
	roll = math.atan(roll)
	if angle < np.pi and roll < 0 and angle - roll > np.pi/2:
		roll = roll + np.pi

	if angle > np.pi and roll > 0 and angle - roll > np.pi/2:
		roll = roll - np.pi
	
	return roll


def rollpitchyaw(roll,pitch,yaw):
	a = pitch #around x axis
	b = yaw #around y axis
	c = roll #around z axis
	
	Mx = np.r_[np.c_[1, 0, 0], np.c_[0, math.cos(a), math.sin(a)], np.c_[0, -math.sin(a), math.cos(a)]]
	My = np.r_[np.c_[math.cos(b), 0, math.sin(b)],  np.c_[0, 1, 0],  np.c_[-math.sin(b), 0, math.cos(b)]]
	Mz = np.r_[np.c_[math.cos(c), -math.sin(c), 0],  np.c_[math.sin(c), math.cos(c), 0],  np.c_[0, 0, 1]]
	
	M = Mx @ My @ Mz
	
	return M,Mz,My,Mx

def rotation_matrix(pitch, roll, yaw):
	"""Creates rotation matrix from Euler angles.

	Parameters
	----------
	P1: ndarray
		Starting point coordinates.

	P2 : ndarray
		Ending point coordinates.
	
	Returns
	-------
	NormVec : ndarray
		The normal vector of the vector.

	"""
	pitch, roll, yaw = np.array([pitch, roll, yaw])
	matrix_pitch = np.array([
		[np.cos(pitch), 0, np.sin(pitch)],
		[0, 1, 0],
		[-np.sin(pitch), 0, np.cos(pitch)]
	])
	matrix_roll = np.array([
		[1, 0, 0],
		[0, np.cos(roll), -np.sin(roll)],
		[0, np.sin(roll), np.cos(roll)]
	])
	matrix_yaw = np.array([
		[np.cos(yaw), -np.sin(yaw), 0],
		[np.sin(yaw), np.cos(yaw), 0],
		[0, 0, 1]
	])
	return np.dot(matrix_pitch, np.dot(matrix_roll, matrix_yaw))

def intensitypeaksdirmarker(intensity,angles):
	# this function detects 'noPeaks' number of intensity peaks. peaks are constrained to be at 360°/noPeaks angles to each other.
	# Function runs a noPeaks * (360°/noPeaks) array over the intensity-profile and finds the angle at which the sum of all peaks is highest.
	peak = np.round(np.rad2deg(angles),0)
	peak[peak<0] = peak[peak<0] + 360
	peak[peak>359] = peak[peak>359] - 360
	sumintensity = np.sum(intensity[peak.astype(int).tolist()])
	
	return sumintensity

def leastSquares(A, y):
	orig_shape=y.shape
	if orig_shape[0]<2:
		y = y[:, np.newaxis]
	sol = np.linalg.lstsq(A,y,rcond=None)[0].T
	return sol


subtitle_text_options={
	'fontsize': 16, 
	'fontweight': 'bold'
	}

text_options = {'horizontalalignment': 'center',
				'verticalalignment': 'center',
				'fontsize': 18,
				'fontweight': 'bold'}

surround_text_options={
	'fontsize': 14, 
	'fontweight': 'bold'
	}

#%%

imgFp=ctpath
marker=np.round(marker_vx[:3],0)
segment=np.round(dirlevel2_vx[:3],0)

def main(args):
	# Input arguments:
	# (1) imgFp: string. the file name of the input Post-CT image
	# (2) marker: list. The position of marker. Example: [243, 297, 107]
	# (3) segment: list. The position of the proximal segments. Example: [248, 299, 100]

	marker_dict = {}
	if args.fcsv is not None:
		marker_dict = read_fcsv(args.fcsv)
	else:
		if all(x is not None for x in (args.lh,args.lt)):
			marker_dict['left']={
				'head': args.lh,
				'tail': args.lt,
			}
		if all(x is not None for x in (args.rh,args.rt)):
			marker_dict['right']={
				'head': args.rh,
				'tail': args.rt,
			}

	for iside in list(marker_dict):
		marker = np.asarray(iside['head']).astype(int)
		segment = np.asarray(iside['tail']).astype(int)
		
		unitvector_mm = (marker - segment)/np.linalg.norm(marker - segment)
		polar_angle = math.degrees(math.acos(np.dot(unitvector_mm, [0, 0, 1])))
		
	# 	assert abs(polar_angle) < 50, f"The angle between the lead and the slice " \
	# 								  f"normal is {polar_angle} degrees.\nNote that angles " \
	# 								  f"> 50 degrees could cause inaccurate orientation estimation."
		img = nib.load(args.input_ct)
		V = img.get_fdata()
		header_info = img.header
		dims = header_info['dim'][1:4]
		voxsize = header_info['pixdim'][1:4]
		assert voxsize[0] == voxsize[1], "The X and Y axis should have the same voxsize"

		radius1 = 4  # radius for analysis at marker (default 3 mm)
		radius2 = 8  # radius for analysis at proximal segments (default 8 mm)
		extractradius = 30
		noPeaks= 2
		
		# Initial orientation estimation at marker position
		xi, yi= np.meshgrid(np.arange(dims[0]),np.arange(dims[1]))
		zi = marker[2] * np.ones_like(xi)
		orig_shape=xi.shape
		for arr in (xi,yi,zi):
			arr.shape=-1
		
		ima = np.empty(xi.shape, dtype=float)
		map_coordinates(img.get_fdata(), (xi,yi,zi), order=2, output=ima)
		slice1=ima.reshape(orig_shape)
		
		print("Extract intensity profile from marker artifact")
		nDegree = 360
		xSlice1 = np.zeros(nDegree)
		ySlice1 = np.zeros(nDegree)
		angleSlice1 = np.zeros(nDegree)
		valSlice1 = np.zeros(nDegree)
		#f = interpolate.interp2d(np.arange(dims[0]), np.arange(dims[1]), slice1, kind='linear')
		for theta in range(360):
			xSlice1[theta] = radius1 / voxsize[0] * math.sin(math.radians(theta)) + marker[1]
			ySlice1[theta] = radius1 / voxsize[0] * math.cos(math.radians(theta)) + marker[0]
			angleSlice1[theta] = theta
			valSlice1[theta] = slice1[int(xSlice1[theta]), int(ySlice1[theta])]
		
		
		dangles,angles,sprofil = peak_FFT(valSlice1)
		dvalleys,valleys,_ = peak_FFT(-valSlice1)
		
		print("Extract intensity profile from segments artifact")

		xi, yi= np.meshgrid(np.arange(dims[0]),np.arange(dims[1]))
		zi = segment[2] * np.ones_like(xi)
		orig_shape=xi.shape
		for arr in (xi,yi,zi):
			arr.shape=-1
		
		ima = np.empty(xi.shape, dtype=float)
		map_coordinates(img.get_fdata(), (xi,yi,zi), order=2, output=ima)
		slice2=ima.reshape(orig_shape)
		valSlice2 = np.zeros(nDegree)
		xSlice2 = np.zeros(nDegree)
		ySlice2 = np.zeros(nDegree)
		for theta in range(360):
			xSlice2[theta] = radius2 / voxsize[1] * math.sin(math.radians(theta)) + segment[1]
			ySlice2[theta] = radius2 / voxsize[0] * math.cos(math.radians(theta)) + segment[0]
			valSlice2[theta] = slice2[int(xSlice2[theta]), int(ySlice2[theta])]
		
		
		corrections = np.arange(-30, 31)
		nDegreeSeg = corrections.shape[0]
		valSlice2Sim = np.zeros(nDegreeSeg)
		init_angle = angles[1] + 90
		init_angles = np.linspace(init_angle + corrections[0], init_angle + corrections[-1], nDegreeSeg)
		intervals = [60, 120, 180, 240, 300, 360]
		f2 = interpolate.interp2d(np.arange(dims[0]), np.arange(dims[1]), slice2, kind='linear')
		for i in range(nDegreeSeg):
			marker_angle = init_angles[i]
			xStar=np.zeros(6)
			yStar=np.zeros(6)
			for j in range(6):
				xStar[j] = radius2 * math.sin(math.radians(marker_angle + intervals[j])) + segment[1]
				yStar[j] = radius2 * math.cos(math.radians(marker_angle + intervals[j])) + segment[0]
				valSlice2Sim[i] += f2(xStar[j], yStar[j] )
		
		valSlice2Sim /= 6
		final_correction = corrections[np.argmin(valSlice2Sim)]
		observed_angles = angles + final_correction - 90
		gamma = math.radians(observed_angles[0])
		
		print(f"Corrected peak angle (in degrees): {final_correction}")
		print(f"Observed roll angles at the axial plane (in degrees): {observed_angles[0]}, {observed_angles[1]}")

		beta = math.asin(unitvector_mm[0])  # yaw
		alpha = math.asin(unitvector_mm[1] / math.cos(beta))  # pitch
		rolltmp=angle2roll(angles[0], beta, alpha)
		polar1=np.rad2deg(math.atan2(np.linalg.norm(np.cross(np.r_[0,0,1],unitvector_mm[:3])),
						   np.dot(np.r_[0,0,1],unitvector_mm[:3])))
		polar2=-np.rad2deg(math.atan2(unitvector_mm[1],unitvector_mm[0]))+ 90
		
		M,_,_,_ = rollpitchyaw(rolltmp,alpha,beta)
		yvec_mm = M.dot(np.r_[0,1,0])
		xvec_mm = np.cross(unitvector_mm[:3], yvec_mm)
		
		marker_mm=np.round(img.affine.dot(np.append(marker.T,1)),0)
		
		asm_solution = calculateASM(slice1, marker, voxsize, valleys,angles)
		
		cog_trans_solution = calculateCOG(img, xvec_mm, yvec_mm, marker_mm, unitvector_mm,'trans',1)
		
		cog_sag_solution = calculateCOG(img, xvec_mm, yvec_mm, marker_mm, unitvector_mm,'sag',1)
		
		realsolution = cog_trans_solution
		
		finalpeak = angles[realsolution]
		
		rollangleSolutions=[]
		sumintensitySolutions = []
		
		sumintensitynew=np.zeros(60)
		rolls_rad={}
		rolls_rad[0] = angle2roll(angles[0],beta,alpha)
		rollangles=np.zeros(60)
		for k in range(60):
			roll_shift = k-30
			rollangles[k] = rolls_rad[0] + np.deg2rad(roll_shift)
			dirnew_angles = darkstar(rollangles[k],alpha,beta,segment,radius2)
			sumintensitynew[k]=intensitypeaksdirmarker(valSlice2,dirnew_angles)
		
		sumintensitySolutions.append(sumintensitynew)
		rollangleSolutions.append(rollangles[np.argmin(sumintensitySolutions[0])])
		
		sumintensitynew=np.zeros(60)
		rolls_rad[1] = angle2roll(angles[1],beta,alpha)
		rollangles=np.zeros(60)
		for k in range(60):
			roll_shift = k-30
			rollangles[k] = rolls_rad[1] + np.deg2rad(roll_shift)
			dirnew_angles = darkstar(rollangles[k],alpha,beta,segment,radius2)
			sumintensitynew[k]=intensitypeaksdirmarker(valSlice2,dirnew_angles)
		
		sumintensitySolutions.append(sumintensitynew)
		rollangleSolutions.append(rollangles[np.argmin(sumintensitySolutions[1])])
		
		if min(sumintensitySolutions[0]) < min(sumintensitySolutions[1]):
			print('Darkstar decides for peak 1')
			darkstar_solution = 0
		else:
			print('Darkstar decides for peak 2')
			darkstar_solution = 1
		
		
		dirnew_angles = darkstar(rollangles[np.argmin(sumintensitynew)],alpha,beta,segment,radius2)
		dirValleys = np.round(np.rad2deg(dirnew_angles), 0)
		dirValleys[dirValleys > 359] = dirValleys[dirValleys > 359] - 360
		
		gamma = math.atan((math.sin(gamma) * math.cos(alpha)) / \
						  (math.cos(gamma) * math.cos(beta) - math.sin(alpha) * math.sin(beta) * math.sin(gamma)))
		
		print(f"Pitch, yaw, roll angles (in degrees): {math.degrees(alpha)}, {math.degrees(beta)}, {math.degrees(gamma)}")

		marker_orientation = [-math.sin(gamma) * math.cos(beta),
							  math.cos(gamma) * math.cos(alpha) + math.sin(alpha) * math.sin(beta) * math.sin(gamma),
							  -math.cos(gamma) * math.sin(alpha) + math.sin(gamma) * math.sin(beta) * math.cos(alpha)]
		
		print("Estimated marker orientation")
		print(f"Isotropic: [{marker_orientation[0]}, {marker_orientation[1]}, {marker_orientation[2]}]")

		marker_orientation /= voxsize
		marker_orientation = marker_orientation / np.linalg.norm(marker_orientation)
		print(f"With current spacing: [{marker_orientation[0]}, {marker_orientation[1]}, {marker_orientation[2]}]")
				
		generate_figure(solution)


if __name__ == "__main__":
	
	# Input arguments
	parser = argparse.ArgumentParser(description="Run DiODe directional lead orientation detection.")
	
	parser.add_argument("-i", "--input", dest="input_ct", help="Path to input CT file containing electrodes)")
	parser.add_argument("-f", "--fcsv", dest="fcsv", default=None, help="Path to input Slicer FCSV File (RAS-oriented)")
	parser.add_argument("-lh", dest="lh", default=None, help="Comma seperated list of RAS coordinates for left head (x,y,z)")
	parser.add_argument("-rh", dest="rh", default=None, help="Comma seperated list of RAS coordinates for right head (x,y,z)")
	parser.add_argument("-lt", dest="lt", default=None, help="Comma seperated list of RAS coordinates for left tail (x,y,z)")
	parser.add_argument("-rt", dest="rt", default=None, help="Comma seperated list of RAS coordinates for right tail (x,y,z)")
	args = parser.parse_args()
	
	main(args)


def generate_figure(solution):
	
	fig = plt.figure(figsize=(12,9))
	ax = fig.add_subplot(231)
	ax.imshow(slice1, cmap='gray',alpha=1, vmin=-50, vmax=150,origin='lower')
	
	ax = fig.add_subplot(232)
	ax.imshow(slice1, cmap='gray',alpha=1, vmin=-50, vmax=150,origin='lower')
	plt.xlim([marker[0]-30,marker[0]+30])
	plt.ylim([marker[1]-30,marker[1]+30])
	ax.plot(ySlice1,xSlice1, ':g')
	ax.scatter(ySlice1[angles.tolist()],xSlice1[angles.tolist()],
		s=80, edgecolors='g',color='none',alpha=1)
	
	ax.scatter(ySlice1[finalpeak],xSlice1[finalpeak],
		s=80, color='g',alpha=1)
	
	for k in valleys:
		xp=[marker[0],(marker[0] + 3 * (ySlice1[k]-marker[0]))]
		yp=[marker[1],(marker[1] + 3 * (xSlice1[k]-marker[1]))]
		ax.plot(xp,yp, '--r')
	
	
	ax.quiver(marker[0],marker[1],ySlice1[finalpeak] - marker[0],xSlice1[finalpeak] - marker[1],
		linewidth=2,ec='g', angles='xy', scale=.5,scale_units='xy')
	
	ax.scatter(marker[0],marker[1],s=100, color='m',alpha=1)
	
	xlimit=ax.get_xlim()
	ylimit=ax.get_ylim()
	ax.text(np.mean(xlimit),ylimit[1]-5,'A', color='b',**text_options)
	ax.text(np.mean(xlimit),ylimit[0]+5,'P', color='b',**text_options)
	ax.text(xlimit[1]-5,np.mean(ylimit),'L', color='b',**text_options)
	ax.text(xlimit[0]+5,np.mean(ylimit),'R',color='b', **text_options)
	ax.set_title('Axial View', **subtitle_text_options)
	
	ax = fig.add_subplot(233)
	ax.plot(np.arange(0, 360), valSlice1)
	ax.plot(np.arange(0, 360), sprofil)
	ax.set_xlim(0,361)
	
	ax.set_ylim(np.min([valSlice1, valSlice2])-50,
			  np.max([valSlice1, valSlice2])+50)
	
	ax.scatter(angleSlice1[dangles.tolist()],valSlice1[dangles.tolist()],
		s=120, edgecolors='g', color='none', alpha=1)
	
	ax.scatter(angleSlice1[angles.tolist()],valSlice1[angles.tolist()],
			s=120, facecolors='g', edgecolors='g',alpha=1)
	
	ax.set_title('Intensity Profile', **subtitle_text_options)
	
	ax = fig.add_subplot(234)
	ax.imshow(slice2, cmap='gray',alpha=1, vmin=-50, vmax=150,origin='lower')
	ax.set_xlim([segment[0]-30,segment[0]+30])
	ax.set_ylim([segment[1]-30,segment[1]+30])
	
	ax.set_xticks([]),ax.set_yticks([])
	ax.set_title('Directional Level', **subtitle_text_options)
	ax.plot(ySlice2,xSlice2, ':r')
	ax.scatter(ySlice2[dirValleys.astype(int).tolist()],xSlice2[dirValleys.astype(int).tolist()],
			   s=80, color='r',alpha=1)
	
	for k in dirValleys:
		xp=[segment[0],(segment[0] + 1.5 * (ySlice2[int(k)]-segment[0]))]
		yp=[segment[1],(segment[1] + 1.5 * (xSlice2[int(k)]-segment[1]))]
		ax.plot(xp, yp, '-r')
	
	xlimit=ax.get_xlim()
	ylimit=ax.get_ylim()
	ax.text(np.mean(xlimit),ylimit[1]-5,'A', color='b',**text_options)
	ax.text(np.mean(xlimit),ylimit[0]+5,'P', color='b',**text_options)
	ax.text(xlimit[1]-5,np.mean(ylimit),'L', color='b',**text_options)
	ax.text(xlimit[0]+5,np.mean(ylimit),'R',color='b', **text_options)
	ax.set_title('Directional Level', **subtitle_text_options)
	
	sol_tran=f"COM-Transversal Solution: {rolls_rad[cog_trans_solution]:.2f}"
	sol_sag=f"COM-Sagittal Solution: {rolls_rad[cog_sag_solution]:.2f}"
	sol_star=f"STARS Solution: {rolls_rad[darkstar_solution]:.2f}"
	sol_asm=f"ASM Solution: {rolls_rad[asm_solution]:.2f}"
	pol_ang=f"Polar Angle: {abs(polar1):.0f}"
	resol=f"CT Resolution: {voxsize[0]:.2f}x{voxsize[1]:.2f}x{voxsize[2]:.2f} mm"
	
	ax.text(-.05, -.3,sol_tran, transform=ax.transAxes, **surround_text_options)
	ax.text(-.05, -.4,sol_sag, transform=ax.transAxes, **surround_text_options)
	ax.text(-.05, -.5,sol_star, transform=ax.transAxes, **surround_text_options)
	ax.text(-.05, -.6,sol_asm, transform=ax.transAxes, **surround_text_options)
	ax.text(-.05, -.7,pol_ang, transform=ax.transAxes, **surround_text_options)
	ax.text(-.05, -.8,resol, transform=ax.transAxes, **surround_text_options)
	
	
	ax = fig.add_subplot(235)
	ax.plot(np.arange(0, 360), valSlice2)
	ax.set_xlim(0,361)
	
	ax.set_ylim(np.min([valSlice1, valSlice2])-50,
			  np.max([valSlice1, valSlice2])+50)
	
	ax = fig.add_subplot(236)
	ax.plot(np.rad2deg(rollangleSolutions[realsolution]),
			solution[side]['sumintensitynew_final'][solution[side]['realsolution']])
	ax.set_yticks([])
	ax.plot(np.rad2deg(solution[side]['rollangles_final'][int(not(solution[side]['realsolution']))]),
			solution[side]['sumintensitynew_final'][int(not(solution[side]['realsolution']))],
			color='r',alpha=1)
	
	sub_str=f"{os.path.basename(imgFp).split('_')[0]}"
	
	plt.suptitle(sub_str, fontsize=20, fontweight='bold')
	
	fig.subplots_adjust(hspace=.3, bottom=0.25)

from keras import backend as K
import numpy as np


def Active_Contour_Loss(y_true, y_pred): 

	#y_pred = K.cast(y_pred, dtype = 'float64')

	"""
	lenth term
	"""

	x = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal and vertical directions 
	y = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1]

	delta_x = x[:,:,1:,:-2]**2
	delta_y = y[:,:,:-2,1:]**2
	delta_u = K.abs(delta_x + delta_y) 

	epsilon = 0.00000001 # where is a parameter to avoid square root is zero in practice.
	w = 1
	lenth = w * K.sum(K.sqrt(delta_u + epsilon)) # equ.(11) in the paper

	"""
	region term
	"""

	C_1 = np.ones((256, 256))
	C_2 = np.zeros((256, 256))

	region_in = K.abs(K.sum( y_pred[:,0,:,:] * ((y_true[:,0,:,:] - C_1)**2) ) ) # equ.(12) in the paper
	region_out = K.abs(K.sum( (1-y_pred[:,0,:,:]) * ((y_true[:,0,:,:] - C_2)**2) )) # equ.(12) in the paper

	lambdaP = 1 # lambda parameter could be various.
	
	loss =  lenth + lambdaP * (region_in + region_out) 

	return loss

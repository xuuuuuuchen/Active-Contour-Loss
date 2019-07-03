def ACWE(y_true, y_pred): 
	print "y_true " +str(y_true.shape)
	print "y_pred " +str(y_pred.shape)
	print "y_pred " +str(y_pred)
	u = y_pred
	#u = K.cast(y_pred, dtype = 'float64')
	print "u " +str(u)
	#(?, ?, ?, ?)
	#(?, 1, 256, 256)
	#x = K.zeros(shape = (y_pred.shape[0],y_pred.shape[1],y_pred.shape[2]-1, y_pred.shape[3]))
	#y = K.zeros(shape = (y_pred.shape[0],y_pred.shape[1],y_pred.shape[2],y_pred.shape[3]-1))
	x = u[:,:,1:,:] - u[:,:,:-1,:]
	y = u[:,:,:,1:] - u[:,:,:,:-1]
	print "x " +str(x)
	print "y " +str(y)
	print "x " +str(x.shape)
	print "y " +str(y.shape)
	#(?, 1, 255, 256)
	#(?, 1, 256, 255)
	#for i in range(x.shape[2]):
		#for j in range(y.shape[3]):
    			#delta_x = x[:,:,0:,:-1]
			#delta_y = y[:,:,:-1,0:]
	#delta_u = K.zeros(shape = (x.shape[0],x.shape[1],x.shape[2],y.shape[3]))
	#delta_u = K.sqrt( K.pow( x[:,:,0:,:-1], 2) + K.pow( y[:,:,:-1,0:] , 2))
	delta_x = x[:,:,1:,:-2]**2
	delta_y = y[:,:,:-2,1:]**2
	delta_u = K.abs(delta_x + delta_y)

	print "delta_x " +str(delta_x)
	print "delta_x " +str(delta_x.shape)

	delta_u = K.sqrt(delta_u + 0.00000001)
	print "delta_u " +str(delta_u)
	print "delta_u " +str(delta_u.shape)

	#delta = 0
	delta = K.sum(delta_u)
	#print "delta " +str(delta)
	#print "delta " +str(delta.shape)

	C_1 = np.ones((256, 256))
    	C_2 = np.zeros((256, 256))

	lamda = 1
	mu = 1

	g = 1 

	lenth = g * delta

	region1 = K.abs(K.sum( y_pred[:,0,:,:] * ((y_true[:,0,:,:] - C_1)**2) ) )    
	
	region2 = K.abs(K.sum( (1-y_pred[:,0,:,:]) * ((y_true[:,0,:,:] - C_2)**2) ))

	E =  mu * lenth + lamda * (region1 + region2)

	return E

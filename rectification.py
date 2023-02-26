
def localization_net():

    filt=[64,128,256, 512]
    for i in range(4):
        model.add(tf.keras.conv2d(filt[i], (3,3), stride=1, padding=1, activation='relu'))
    model.add(tf.keras.maxpooling2d(pool_size=(2,2)))
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dense(1024, activation='tanh')) #?
    #number of fiducial points=20 (coordinates-->2*20=40)
    model.add(tf.keras.layers.Dense(40, activation='tanh'))#?
    """
    convolutional layers, pooling layers and fully-connected layers.
    #2k OUPUT FOR REGRESSION (2*20=40)
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    w = np.zeros((64, 6), dtype='float32')
    weights = [w, b.flatten()]

    loc_input = Input(input_shape)

    loc_conv_1 = Conv2D(16, (5, 5), padding='same', activation='relu')(loc_input)
    loc_conv_2 = Conv2D(32, (5, 5), padding='same', activation='relu')(loc_conv_1)
    loc_fla = Flatten()(loc_conv_2)
    loc_fc_1 = Dense(64, activation='relu')(loc_fla)
    loc_fc_2 = Dense(6, weights=weights)(loc_fc_1)

    output = Model(inputs=loc_input, outputs=loc_fc_2)

    return output
    
    """

    return model
def grid_generator():
    '''
    The grid generator estimates the TPS transformation parameters,
    and generates a sampling grid
    :return:
    '''
    #c_p=[c1;:::; cK] in 2*K in rectified image
    #C =[c1;:::; cK] in 2*K in original image
    for i in range(K):
        for j in range(k):
            d[i,j] = np.sqrt((c_p[0,i]-c_p[0,j])**2+(c_p[1,i]-c_p[1,j])**2)  #k*k
            R[i,j] = (d[i,j]**2)*(np.log(d[i,j]**2)) #k*k
    delta_c_p=np.array([np.ones(k,1)  , np.transpose(c_p) , R           ],
                       [np.zeros(1,1) , np.zeros(1,2)        , np.ones(1,K)],
                       [np.zeros(2,1)    , np.zeros(2,2)        , c_p]) #(K+3)*(K+3)
    T = np.transpose(np.dot(np.linalg.inv(delta_c_p) ,np.concatenate(np.transpose(C),np.zeros(3,2)) ))
    R=[]
    ########
    # for each i th pixel R should be calculated
    # tODO: r_hat calculated from c_p and p_hat
    for l in range(K):
        R.append(r_hat[l])
    p_hat_p[i]=np.transpose([1,x_hat[i], y_hat[i], True)
    p[i]=np.dot(T, p_hat_p[i])
    ################################
    return
def rectify()
    localization()
    grid_generator()
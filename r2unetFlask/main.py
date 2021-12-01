import tensorflow as tf
from tensorflow import keras
import os
tf.test.is_built_with_cuda()
tf.config.list_physical_devices('GPU')
# try:
#     # Disable all GPUS
#     tf.config.set_visible_devices([], 'GPU')
#     visible_devices = tf.config.get_visible_devices()
#     for device in visible_devices:
#         assert device.device_type != 'GPU'
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass

# Building Unet by dividing encoder and decoder into blocks
import numpy as np
from keras.models import Model
from keras.layers import add,Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model,model_from_json 

import random

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

ROWS= 320
COLS= 320
CHANNEL=3
train_set_x,train_set_y=np.array([]),np.array([])
test_set_x,test_set_y=np.array([]),np.array([])

def conv_block(input, num_filters):
    x=input
    for i in range(2):
        r = x 
        x = Conv2D(num_filters, 3,padding="same")(input)
        x = BatchNormalization()(x)   #Not in the original network. 
        x = Activation("relu")(x)
        x = Dropout(0.1)(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)  #Not in the original network
    
    shortcut= Conv2D(num_filters,kernel_size=(1,1),padding='same')(input)
    shortcut= BatchNormalization(axis=3)(shortcut)
    
    res_path = add([shortcut,x])
    x = Activation("relu")(res_path)
    return x

#Encoder block: Conv block followed by maxpooling


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p   

#Decoder block
#skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

#Build Unet using the blocks
def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024) #Bridge

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)  #Binary (can be multiclass)

    model = Model(inputs, outputs, name="R2U-Net")
    return model

def read_img(file_path):
    
    img = cv2.imread(file_path,cv2.IMREAD_COLOR)
    return cv2.resize(img, (ROWS,COLS),interpolation=cv2.INTER_CUBIC)

def read_img1(file_path):
    
    img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (ROWS,COLS),interpolation=cv2.INTER_CUBIC)

def prepare_data(images,images1):
    m = len(images) #18
    X = np.zeros((m,ROWS,COLS,CHANNEL),dtype = np.float32) #(18,572,572)
    Y = np.zeros((m,ROWS,COLS),dtype = np.float32)
    
    for i,image_file in enumerate(images):
        img = read_img(image_file)
        X[i]= img
    for i,image_file in enumerate(images1):
        img = read_img1(image_file)
        Y[i]= img
    return X, Y

def predict():
    import random
    ## m = load_model('C:/Users/Aayush Malde/Desktop/aayush documents/TY KJSCE/5th sem/deeplearning/weights/weights3.best.hdf5')
    # imports

    # opening and store file in a variable

    json_file = open('model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()

    # use Keras model_from_json to make a loaded model

    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model

    loaded_model.load_weights("model_weights.h5")
    # pred_test = m.predict(test_set_x[:int(test_set_x.shape[0]*0.9)],verbose=1)
    pred_test = loaded_model.predict(test_set_x,verbose=1)

    pred_test_t=(pred_test>0.5).astype(np.int32)
    ix = random.randint(1,6)
    print(ix)
    plt.imshow(test_set_x[ix])
    plt.show()
    plt.imshow(np.squeeze(test_set_y[ix]))
    plt.show()
    plt.imshow(np.squeeze(pred_test_t[ix]))
    plt.show()

def trainModel(train_set_x,train_set_y):
    checkpoint_filepath = 'C:/Users/Aayush Malde/Desktop/aayush documents/TY KJSCE/5th sem/deeplearning/weights/weights3.best.hdf5'
    # epoch = 100
    callbacks=[
        # tf.keras.callbacks.EarlyStopping(patience=5,monitor='val_loss'),
        tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
    #     save_weights_only=True,
    #     monitor='val_accuracy',
    #     mode='max',
        save_best_only=True)
    ]

    m = load_model('C:/Users/Aayush Malde/Desktop/aayush documents/TY KJSCE/5th sem/deeplearning/weights/weights3.best.h5')
    # print(train_set_x)
    results = m.fit(train_set_x,train_set_y,batch_size=1,validation_split=0.1,epochs = 1,callbacks=callbacks)
    score = m.evaluate(test_set_x, test_set_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("\n\n")
    model_json = m.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    m.save_weights("model_weights.h5")

    # m.save('C:/Users/Aayush Malde/Desktop/aayush documents/TY KJSCE/5th sem/deeplearning/weights/weights3.best.hdf5')
    # m.save('C:/Users/Aayush Malde/Desktop/aayush documents/TY KJSCE/5th sem/deeplearning/weights/weights3.best.h5')
    accuracy = [0.29185113310813904, 0.3563140332698822, 0.3565717339515686, 0.37407225370407104, 0.38156577944755554, 0.43755581974983215, 0.4721435606479645, 0.49421441555023193, 0.5019574761390686, 0.5028943419456482, 0.6342947483062744, 0.6422460675239563, 0.6435920000076294, 0.6428124904632568, 0.6511560678482056, 0.6713759303092957, 0.681251585483551, 0.6825168132781982, 0.6878955364227295, 0.7106055021286011, 0.7044422030448914, 0.7236279845237732, 0.7228743433952332, 0.7283158898353577, 0.7337722778320312, 0.724034309387207, 0.7375541925430298, 0.747056245803833, 0.7500478029251099, 0.74704909324646, 0.7545947432518005, 0.7587853074073792, 0.7692810893058777, 0.7726286053657532, 0.7762250900268555, 0.7704437375068665, 0.7739480137825012, 0.7680127620697021, 0.7740380764007568, 0.7799782752990723, 0.7825944423675537, 0.7840033769607544, 0.7879058122634888, 0.7939940690994263, 0.7959228754043579, 0.7944276332855225, 0.7879281044006348, 0.7855311036109924, 0.7931413650512695, 0.7970063090324402, 0.8021882176399231, 0.8069888353347778, 0.8041878342628479, 0.8073202967643738, 0.8085182905197144, 0.8095899224281311, 0.8114089369773865, 0.8106580972671509, 0.8098942041397095, 0.8043137192726135, 0.8368863463401794, 0.8419960141181946, 0.8423373103141785, 0.840252161026001, 0.8325721025466919, 0.8276280760765076, 0.8324500322341919, 0.8381960391998291, 0.8411491513252258, 0.839722752571106, 0.837855339050293, 0.8343511819839478, 0.8319259881973267, 0.8211880922317505, 0.8317838907241821, 0.8349213600158691, 0.8353754878044128, 0.8384202122688293, 0.8380739688873291, 0.8399016261100769, 0.8430811762809753, 0.8424587845802307, 0.8441699743270874, 0.8440624475479126, 0.8459857702255249, 0.8468044996261597, 0.8466411828994751, 0.8468017578125, 0.8430854082107544, 0.8387032151222229, 0.8412104249000549, 0.8467632532119751, 0.8487294316291809, 0.849193811416626, 0.8509863018989563, 0.8531472086906433, 0.8551860451698303, 0.8544291853904724, 0.8548106551170349, 0.8556130528450012, 0.8528444766998291, 0.8554800152778625, 0.853308379650116, 0.8525157570838928, 0.8523680567741394, 0.8554025292396545, 0.8562651872634888, 0.8550353050231934, 0.8502956628799438, 0.8488872051239014]
    accuracy += results.history['accuracy']
    # print(accuracy)

    
    #plotting accuracy-epoch graph
    x=[]
    for i in range(1,len(accuracy)+1):
        x.append(i)
    plt.plot(x, accuracy)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    # set chart title
    plt.title("Accuracy growth")
    plt.show()

def buildModel():
    m = build_unet([320,320,3])
    m.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    # m.summary()


def trainTest_set(training_set,testing_set,train_set_output_set,test_set_output_set):
    train_set_x, train_set_y = prepare_data(training_set,train_set_output_set) 
    test_set_x, test_set_y = prepare_data(testing_set,test_set_output_set)

    # train_set_x_flatten = train_set_x.reshape(train_set_x.shape[0], ROWS*COLS*CHANNEL).T  #(7056,1000)
    # test_set_x_flatten = test_set_x.reshape(test_set_x.shape[0], ROWS*COLS*CHANNEL).T

    train_set_x = train_set_x/255
    test_set_x = test_set_x/255

    # m = build_unet([128,128,3])
    # m = build_unet([256,256,3])
    # m = build_unet([512,512,3])
    buildModel()
    return train_set_x,train_set_y,test_set_x,test_set_y

def loadDb():
    train_path = 'C:\\Users\\Aayush Malde\\Desktop\\aayush documents\\TY KJSCE\\5th sem\\SC lab\\SC IA\\CHASE\\Train Images\\'
    test_path ='C:\\Users\\Aayush Malde\\Desktop\\aayush documents\\TY KJSCE\\5th sem\\SC lab\\SC IA\\CHASE\\Test Images\\'
    train_set_output='C:\\Users\\Aayush Malde\\Desktop\\aayush documents\\TY KJSCE\\5th sem\\SC lab\\SC IA\\CHASE\\Desired Output\\'
    test_set_output='C:\\Users\\Aayush Malde\\Desktop\\aayush documents\\TY KJSCE\\5th sem\\SC lab\\SC IA\\CHASE\\Desired Output Test\\'

    training_set = [train_path+i for i in os.listdir(train_path)]
    testing_set = [test_path+i for i in os.listdir(test_path)]
    train_set_output_set = [train_set_output+i for i in os.listdir(train_set_output)]
    test_set_output_set = [test_set_output+i for i in os.listdir(test_set_output)]
    # print(training_set)
    # print(testing_set)
    # print(train_set_output_set)

    train_set_x,train_set_y,test_set_x,test_set_y = trainTest_set(training_set,testing_set,train_set_output_set,test_set_output_set)
    return train_set_x,train_set_y,test_set_x,test_set_y

#--main--
train_set_x,train_set_y,test_set_x,test_set_y=loadDb()
n= int(input("Enter choice: "))
if n==1:
    trainModel(train_set_x,train_set_y)
elif n==2:
    predict()



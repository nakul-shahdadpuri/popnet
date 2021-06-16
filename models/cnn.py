import tensorflow as tf
import tensorflow.keras.layers as tfl

class cnn_model(tf.keras.Model):
	def __init__(self,num_of_classes,input_shape):
		    input_img = tf.keras.Input(shape=input_shape)
		    x = tfl.Conv2D(15,3,padding='SAME',activation="linear",input_shape=input_shape[1:])(input_img)
		    x = tfl.ReLU()(x) 
		    x = tfl.MaxPool2D(8,padding="SAME")(x)
		    x = tfl.Dropout(0.2)(x)
		    x = tfl.BatchNormalization()(x)
		 
		     
		    x = tfl.Conv2D(30,3,padding='SAME',activation="linear",input_shape=input_shape[1:])(x)
		    x = tfl.ReLU()(x)
		    x = tfl.Dropout(0.2)(x)
		    x = tfl.BatchNormalization()(x)
		  
		    x = tfl.Conv2D(45,3,padding='SAME',activation="linear",input_shape=input_shape[1:])(input_img)
		    x = tfl.ReLU()(x) 
		    x = tfl.MaxPool2D(4,padding="SAME")(x)
		    x = tfl.Dropout(0.2)(x)
		    x = tfl.BatchNormalization()(x)

		    x = tfl.Conv2D(60,3,padding='SAME',activation="linear",input_shape=input_shape[1:])(x)
		    x = tfl.ReLU()(x)
		    x = tfl.Dropout(0.2)(x)
		    x = tfl.BatchNormalization()(x)
		    
		    x = tfl.Conv2D(60,3,padding='SAME',activation="linear",input_shape=input_shape[1:])(x)
		    x = tfl.ReLU()(x)
		    x = tfl.MaxPool2D(4,padding="SAME")(x)
		    x = tfl.Dropout(0.2)(x)
		    x = tfl.BatchNormalization()(x)
		    
		    x = tfl.Flatten()(x)
		    
		    x = tfl.Dense(200, activation="relu")(x)
		    x = tfl.Dropout(0.2)(x)
		    x = tfl.BatchNormalization()(x)
		    
		    x = tfl.Dense(100, activation="relu")(x)
		    x = tfl.Dropout(0.2)(x)
		    x = tfl.BatchNormalization()(x)

		    x = tfl.Dense(50, activation="relu")(x)
		    x = tfl.Dropout(0.2)(x)
		    x = tfl.BatchNormalization()(x)
		    
		    outputs = tfl.Dense(num_of_classes,activation="softmax")(x)
		    self = tf.keras.Model(inputs=input_img, outputs=outputs)

	def __str__(self):
		return self.summary()





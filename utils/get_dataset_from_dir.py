from tensorflow.keras.preprocessing import image_dataset_from_directory

def get_dataset_from_dir(batch_size,image_size,directory,split):

	train_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=batch_size,
                                             image_size=image_size,
                                             validation_split=split,
                                             subset='training',
                                             seed=69)	
	validation_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=batch_size,
                                             image_size=image_size,
                                             validation_split=split,
                                             subset='validation',
                                             seed=69)
	return train_dataset,validation_dataset



def get_batch_shape(dataset):
	for image_batch,labels in dataset:
		return image_batch.shape, labels.shape
		
def version():
	return "0.0.1v"

def classes(dataset):
	return dataset.class_names

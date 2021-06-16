def get_batch_shape(dataset):
	for image_batch,labels in dataset:
		return image_batch.shape, labels.shape
		
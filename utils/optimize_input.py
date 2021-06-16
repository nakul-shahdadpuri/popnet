def optimize_input(dataset):
	AUTOTUNE = tf.data.AUTOTUNE
	optimized = dataset.cache().prefetch(buffer_size=AUTOTUNE)
	return optimized
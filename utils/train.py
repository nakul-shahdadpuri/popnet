import tensorflow as tf

def train_model(model,training_dataset, testing_dataset, epochs):
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'])
    model.fit(
        training_dataset,
        validation_data=testing_dataset,
        epochs=epochs)
    return model

def save_model(model,name):
    model.save(name)

def test_model(model,testing_dataset):
    return model.evaluate(testing_dataset)


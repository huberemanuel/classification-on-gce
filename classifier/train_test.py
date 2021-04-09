from classifier.train import load_data, load_model

def test_load_data():
    train, test = load_data()
    image_batch, label_batch = next(iter(train))
    assert image_batch.shape == (32, 32, 32, 3)
    assert label_batch.shape == (32, 10)

def test_load_model():
    model = load_model()
    first_layer = model.layers[0]
    last_layer = model.layers[-1]
    assert first_layer.input_shape[0] == (None, 32, 32, 3)
    assert last_layer.output_shape == (None, 10)
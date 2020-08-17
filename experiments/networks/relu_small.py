"""Create a sequential model."""

# Third party modules
from keras.layers import Activation, Dense, Input
from keras.models import Model


def create_model(nb_classes, input_shape, config=None):
    """Create a model."""
    if config is None:
        config = {"model": {}}

    # Network definition
    input_ = Input(shape=input_shape)
    x = input_
    x = Dense(nb_classes)(x)
    x = Activation("softmax")(x)
    model = Model(inputs=input_, outputs=x)
    return model


if __name__ == "__main__":
    model = create_model(100, (20000,))
    model.summary()

    # Third party modules
    from keras.utils import plot_model

    plot_model(model, to_file="baseline.png", show_layer_names=False, show_shapes=True)

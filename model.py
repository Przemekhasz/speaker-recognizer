from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten, Conv2D, MaxPooling2D, Reshape


class ModelCreator:
    """
        Creates a neural network model for classifying voice data.

        The model uses convolutional and LSTM layers, suitable for feature-rich time-series data like audio.
        Understanding LSTM Networks: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    """

    def create_model(self, input_shape, num_classes):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Reshape((128, 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(30, return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model


model_creator = ModelCreator()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from sklearn import svm
from sklearn.neural_network import MLPClassifier


def configure_svm():
    return svm.SVC(kernel="linear", C=100, gamma=0.01)


def configure_mlp():
    return MLPClassifier(hidden_layer_sizes=(25,18,10,5), activation="tanh", solver="lbfgs")


def build_cnn():
    input_shape = (34,1)
    cnn_model = Sequential([Conv1D(64, 2, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Dropout(0.2),
        Conv1D(32, 2, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(6, activation='softmax')
    ])
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return cnn_model
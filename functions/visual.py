import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix


def plot_accuracy(h):
    plt.plot(h.history['accuracy'])
    plt.plot(h.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_loss(h):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_confusion(m, x_test, y_test):
    y_pred = m.predict(x_test)
    cm = confusion_matrix([np.argmax(a) for a in y_test], [np.argmax(a) for a in y_pred])
    df_cm = pd.DataFrame(cm, index=["Water", "Object"], columns=["Water", "Object"])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)

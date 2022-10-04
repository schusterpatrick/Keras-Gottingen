import pandas as pd
import tensorflow as tf
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math

POSSIBLE_PARAMETERS = ["Pytagoras", "Res0", "Res1", "Res2", "Res3", "Res4", "Res5"]
INPUT_PARAMETERS_WITH_TARGET = POSSIBLE_PARAMETERS[1:]
# print(INPUT_PARAMETERS_WITH_TARGET)
TARGET = INPUT_PARAMETERS_WITH_TARGET[-1]
NUM_VDRAIN_VALUES = 4
STARTING_VDRAIN_NUM = [0, 11, 22, 33, 44, 55]
data = {}
import glob, os

os.chdir("C:/Users\Patrick\PycharmProjects\KerasGöttingen\Bottom left_dataFiles _only")
for file in glob.glob("*.txt"):
    data[file] = pd.read_csv(file, sep="\t", skiprows=3)
# print(data["b1r2_out.txt"])
# dict with tuples. keys are names of contacts. start counting from bottom left
pytagoras_dict = {
    "l1": (0, 3),
    "l2": (0, 2),
    "l3": (0, 1),
    "b1": (1, 0),
    "b2": (2, 0),
    "b3": (3, 0),
    "t1": (1, 4),
    "t2": (2, 4),
    "t3": (3, 4),
    "r1": (4, 1),
    "r2": (4, 2),
    "r3": (4, 3),
}
inputs = {}
for key in data.keys():
    currents = {}
    inputs[key] = {}
    currents[key] = {}
    # noch fitten zu ersten x werten
    currents[key]["I_Drain"] = data[key]["I_Drain"].to_list()
    x = [0, -0.5, -1, -1.5]

    # gate_0 = currents[key]["I_Drain"][:4]
    # print(gate_0)
    # gate_1 = currents[key]["I_Drain"][11:15]
    # gate_2 = currents[key]["I_Drain"][22:26]
    # gate_3 = currents[key]["I_Drain"][33:37]
    # gate_4 = currents[key]["I_Drain"][44:48]
    # gate_5 = currents[key]["I_Drain"][55:59]
    pytagoras_x = pytagoras_dict[key[:2]][0] - pytagoras_dict[key[2:4]][0]
    pytagoras_y = pytagoras_dict[key[:2]][1] - pytagoras_dict[key[2:4]][1]
    for input_value in INPUT_PARAMETERS_WITH_TARGET:
        if input_value == "Pytagoras":
            inputs[key]["Pytagoras"] = math.sqrt(pytagoras_x ** 2 + pytagoras_y ** 2)
        else:
            model_lin_reg = scipy.stats.linregress(x, currents[key]["I_Drain"][
                                                  STARTING_VDRAIN_NUM[int(input_value[-1])]:STARTING_VDRAIN_NUM[int(
                                                      input_value[-1])] + NUM_VDRAIN_VALUES])
            res = model_lin_reg.slope
            inputs[key][input_value] = res
    # model_0 = scipy.stats.linregress(x, gate_0)
    # gate_0_res = model_0.slope
    # model_1 = scipy.stats.linregress(x, gate_1)
    # gate_1_res = model_1.slope
    # model_2 = scipy.stats.linregress(x, gate_2)
    # gate_2_res = model_2.slope
    # model_3 = scipy.stats.linregress(x, gate_3)
    # gate_3_res = model_3.slope
    # model_4 = scipy.stats.linregress(x, gate_4)
    # gate_4_res = model_4.slope
    # model_5 = scipy.stats.linregress(x, gate_5)
    # gate_5_res = model_5.slope
    # inputs[key]["Res0"] = gate_0_res
    # inputs[key]["Res1"] = gate_1_res
    # inputs[key]["Res2"] = gate_2_res
    # inputs[key]["Res3"] = gate_3_res
    # inputs[key]["Res4"] = gate_4_res
    # inputs[key]["Res5"] = gate_5_res

# dataset = tf.data.Dataset.from_tensor_slices(pd.DataFrame.from_dict(inputs).to_dict(orient="list")
sc = StandardScaler()

input_list = pd.DataFrame.from_dict(inputs).transpose()
input_list[INPUT_PARAMETERS_WITH_TARGET] = sc.fit_transform(input_list[INPUT_PARAMETERS_WITH_TARGET])
median = input_list[TARGET].quantile(0.5)
input_list[TARGET] = [0 if x < median else 1 for x in input_list[TARGET]]
print(input_list[TARGET])
print(input_list)
print(inputs)
# model = tf.keras.models.Sequential()
# model.add(tf.keras.Input(shape=(16,)))
# model.add(tf.keras.layers.Dense(32, activation='relu'))
# model.add(tf.keras.layers.Dense(32))
target = input_list.pop(TARGET)
# X_train, X_test, y_train, y_test = train_test_split(input_list, target, test_size=0.25)

tf.convert_to_tensor(input_list)


def get_basic_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(len(input_list.keys()),)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='sigmoid'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


model_ml = get_basic_model()
history = model_ml.fit(input_list, target, validation_split=0.25, epochs=100, batch_size=2)
plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()
plt.savefig(f"Res0-4,T=Res5")
# scores = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print(scores)

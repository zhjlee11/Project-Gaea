import tensorflow as tf
import os
print("---------------------------------------------------------------------------------------------")
print("텐서플로우 버전 : {0}".format(tf.__version__))
expath = os.path.join(os.getcwd(), 'saved_model', 'generator_g')
print("Saved Model 경로 : {0}".format(expath))
print("---------------------------------------------------------------------------------------------")

#model = tf.keras.models.load_model('/saved_model/generator_g')
model = tf.saved_model.load(os.path.join(os.getcwd(), 'saved_model', 'generator_g'))

print(list(model.signatures["serving_default"].structured_outputs))
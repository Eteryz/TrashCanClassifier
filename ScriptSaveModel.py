import tensorflow as tf

loaded_model = tf.keras.models.load_model("model_3000epochs_opt_adam.h5")
tf.saved_model.save(loaded_model, "resnet/")

import tensorflow as tf


@tf.function
def train_step(inputs):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        print()

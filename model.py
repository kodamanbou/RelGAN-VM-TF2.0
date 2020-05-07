import tensorflow as tf
import tensorflow_addons as tfa


class RelGAN(tf.keras.Model):
    def __init__(self, num_domains, batch_size=1):
        super(RelGAN, self).__init__()
        self.generator = Generator(num_domains, batch_size)
        self.discriminator = PatchGanDiscriminator()
        self.adversarial = Adversarial()
        self.interpolate = Interpolate()
        self.matching = Matching(num_domains)

    def call(self, inputs, training=None, mask=None):
        input_A_real = inputs[0]
        input_A2_real = inputs[1]
        input_B_real = inputs[2]
        input_C_real = inputs[3]
        input_A_label = inputs[4]
        input_B_label = inputs[5]
        input_C_label = inputs[6]
        alpha = inputs[7]
        alpha_1 = tf.reshape(alpha, [-1, 1])

        vector_A2B = input_B_label - input_A_label
        vector_C2B = input_B_label - input_C_label
        vector_A2C = input_C_label - input_A_label

        outputs = []

        # Generator.
        generation_B = self.generator([input_A_real, vector_A2B])  # A -> B
        generation_B2 = self.generator([input_A2_real, vector_A2B])  # A2 -> B2
        generation_C = self.generator([input_B_real, -vector_C2B])  # B -> C
        generation_A = self.generator([input_C_real, -vector_A2C])  # C -> A

        cycle_A = self.generator([generation_B, -vector_A2B])

        generation_A_identity = self.generator([input_A_real, vector_A2B - vector_A2B])

        generation_alpha = self.generator([input_A_real, vector_A2B * alpha_1])
        generation_alpha_back = self.generator([generation_alpha, -vector_A2B * alpha_1])

        outputs += [generation_B, generation_B2, generation_C, generation_A, cycle_A, generation_A_identity,
                    generation_alpha_back]

        # One-step discriminator.
        discrimination_B_fake = self.discriminator(generation_B)
        discrimination_B_fake = self.adversarial(discrimination_B_fake)
        discrimination_B_real = self.discriminator(input_B_real)
        discrimination_B_real = self.adversarial(discrimination_B_real)

        discrimination_alpha_fake = self.discriminator(generation_alpha)
        discrimination_alpha_fake = self.adversarial(discrimination_alpha_fake)

        # Two-step discriminator.
        discrimination_A_dot_fake = self.discriminator(cycle_A)
        discrimination_A_dot_fake = self.adversarial(discrimination_A_dot_fake)
        discrimination_A_dot_real = self.discriminator(input_A_real)
        discrimination_A_dot_real = self.adversarial(discrimination_A_dot_real)

        outputs += [discrimination_B_fake, discrimination_B_real, discrimination_alpha_fake, discrimination_A_dot_fake,
                    discrimination_A_dot_real]

        # Conditional adversarial.
        sr = [self.discriminator(input_A_real), self.discriminator(input_B_real), vector_A2B]
        sr = self.matching(sr)
        sf = [self.discriminator(input_A_real), self.discriminator(generation_B), vector_A2B]
        sf = self.matching(sf)
        w1 = [self.discriminator(input_C_real), self.discriminator(input_B_real), vector_A2B]
        w1 = self.matching(w1)
        w2 = [self.discriminator(input_A_real), self.discriminator(input_B_real), vector_C2B]
        w2 = self.matching(w2)
        w3 = [self.discriminator(input_A_real), self.discriminator(input_B_real), vector_A2C]
        w3 = self.matching(w3)
        w4 = [self.discriminator(input_A_real), self.discriminator(input_C_real), vector_A2B]
        w4 = self.matching(w4)

        outputs += [sr, sf, w1, w2, w3, w4]

        # Interpolation.
        interpolate_identity = self.discriminator(generation_A_identity)
        interpolate_identity = self.interpolate(interpolate_identity)
        interpolate_B = self.discriminator(generation_B)
        interpolate_B = self.interpolate(interpolate_B)
        interpolate_alpha = self.discriminator(generation_alpha)
        interpolate_alpha = self.interpolate(interpolate_alpha)

        outputs += [interpolate_identity, interpolate_B, interpolate_alpha]
        return outputs


class Generator(tf.keras.Model):
    def __init__(self, num_domains, batch_size):
        super(Generator, self).__init__()
        self.num_domains = num_domains
        self.batch_size = batch_size

        self.h1 = tf.keras.layers.Conv2D(128, kernel_size=(5, 15), padding='same', name='h1_conv')
        self.h1_gates = tf.keras.layers.Conv2D(128, kernel_size=(5, 15), padding='same', name='h1_conv_gates')
        self.h1_glu = tf.keras.layers.Multiply(name='h1_glu')

        self.d1 = Downsample2DBlock(256, kernel_size=(5, 5), strides=2, name_prefix='downsample2d_block1_')
        self.d2 = Downsample2DBlock(256, kernel_size=(5, 5), strides=2, name_prefix='downsample2d_block2_')

        self.resh1 = tf.keras.layers.Conv1D(256, kernel_size=1, strides=1, padding='same', name='resh1_conv')
        self.resh1_norm = tfa.layers.InstanceNormalization(epsilon=1e-6, name='resh1_norm')

        self.res1 = Residual1DBlock(512, kernel_size=3, name_prefix='res1d_block1_')
        self.res2 = Residual1DBlock(512, kernel_size=3, name_prefix='res1d_block2_')
        self.res3 = Residual1DBlock(512, kernel_size=3, name_prefix='res1d_block3_')
        self.res4 = Residual1DBlock(512, kernel_size=3, name_prefix='res1d_block4_')
        self.res5 = Residual1DBlock(512, kernel_size=3, name_prefix='res1d_block5_')
        self.res6 = Residual1DBlock(512, kernel_size=3, name_prefix='res1d_block6_')

        self.resh2 = tf.keras.layers.Conv1D(2304, kernel_size=1, padding='same', name='resh2_conv')
        self.resh2_norm = tfa.layers.InstanceNormalization(epsilon=1e-6, name='resh2_norm')

        # upsampling
        self.u1 = Upsample2DBlock(filters=1024, kernel_size=5, name_prefix='upsampling2d_block1_')
        self.u2 = Upsample2DBlock(filters=512, kernel_size=5, name_prefix='upsampling2d_block2_')

        self.conv_out = tf.keras.layers.Conv2D(1, kernel_size=(5, 15), padding='same', name='conv_out')

    def __call__(self, inputs, training=None, mask=None):
        inputs, vec = inputs[0], inputs[1]
        inputs = tf.expand_dims(inputs, axis=-1)
        l = tf.reshape(vec, [-1, 1, 1, self.num_domains])
        h = tf.shape(inputs)[1]
        w = tf.shape(inputs)[2]
        k = tf.ones([self.batch_size, h, w, self.num_domains])
        k = k * l
        inputs = tf.concat([inputs, k], axis=3)

        h1 = self.h1(inputs)
        h1_gates = self.h1_gates(inputs)
        h1_glu = self.h1_glu([h1, tf.sigmoid(h1_gates)])

        d1 = self.d1(h1_glu)
        d2 = self.d2(d1)
        d3 = tf.squeeze(tf.reshape(d2, shape=(self.batch_size, 1, -1, 2304)), axis=1)
        resh1 = self.resh1(d3)
        resh1_norm = self.resh1_norm(resh1)

        res1 = self.res1(resh1_norm)
        res2 = self.res2(res1)
        res3 = self.res3(res2)
        res4 = self.res4(res3)
        res5 = self.res5(res4)
        res6 = self.res6(res5)

        resh2 = self.resh2(res6)
        resh2_norm = self.resh2_norm(resh2)
        resh3 = tf.reshape(tf.expand_dims(resh2_norm, axis=1), shape=(self.batch_size, 9, -1, 256))

        u1 = self.u1(resh3)
        u2 = self.u2(u1)
        conv_out = self.conv_out(u2)
        out = tf.squeeze(conv_out, axis=-1)

        return out


class PatchGanDiscriminator(tf.keras.Model):
    def __init__(self):
        super(PatchGanDiscriminator, self).__init__()
        self.h1 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=spectral_norm,
                                         name='h1_conv')
        self.h1_gates = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same',
                                               kernel_regularizer=spectral_norm, name='h1_conv_gates')
        self.h1_glu = tf.keras.layers.Multiply(name='h1_glu')

        self.d1 = Downsample2DBlock(256, kernel_size=(3, 3), strides=2, sn=True, name_prefix='downsample2d_block1_')
        self.d2 = Downsample2DBlock(512, kernel_size=(3, 3), strides=2, sn=True, name_prefix='downsample2d_block2_')
        self.d3 = Downsample2DBlock(1024, kernel_size=(3, 3), strides=2, sn=True, name_prefix='downsample2d_block3_')
        self.d4 = Downsample2DBlock(1024, kernel_size=(1, 5), strides=1, sn=True, name_prefix='downsample2d_block4_')

        self.out = tf.keras.layers.Conv2D(1, kernel_size=(1, 3), strides=1, padding='same',
                                          kernel_regularizer=spectral_norm, name='out_conv')

    def __call__(self, inputs, training=None, mask=None):
        inputs = tf.expand_dims(inputs, axis=-1)  # [N, M, T, 1]
        h1 = self.h1(inputs)
        h1_gates = self.h1_gates(inputs)
        h1_glu = self.h1_glu([h1, tf.sigmoid(h1_gates)])

        d1 = self.d1(h1_glu)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        out = self.out(d4)

        return out


class Adversarial(tf.keras.Model):
    def __init__(self):
        super(Adversarial, self).__init__()
        self.adv = tf.keras.layers.Conv2D(1, [1, 3], strides=[1, 1], padding='same', kernel_regularizer=spectral_norm,
                                          name='discriminator_out_conv_adv')

    def call(self, inputs, training=None, mask=None):
        x = self.adv(inputs)
        return x


class Interpolate(tf.keras.Model):
    def __init__(self):
        super(Interpolate, self).__init__()
        self.i0 = tf.keras.layers.Conv2D(128, [3, 3], strides=[1, 1], padding='same', kernel_regularizer=spectral_norm,
                                         name='discriminator_int1_conv')

    def call(self, inputs, training=None, mask=None):
        x = self.i0(inputs)
        interp = tf.reduce_mean(x, axis=3, keepdims=True)
        return interp


class Matching(tf.keras.Model):
    def __init__(self, num_domains):
        super(Matching, self).__init__()
        self.num_domains = num_domains
        self.m1 = tf.keras.layers.Conv2D(1024, [3, 3], strides=[1, 1], padding='same', kernel_regularizer=spectral_norm,
                                         name='discriminator_mat1_conv')
        self.m1_gates = tf.keras.layers.Conv2D(1024, [3, 3], strides=[1, 1], padding='same',
                                               kernel_regularizer=spectral_norm,
                                               name='discriminator_mat1_conv_gates')
        self.m1_glu = tf.keras.layers.Multiply(name='discriminator_mat1_glu')
        self.mat = tf.keras.layers.Conv2D(1, [1, 3], strides=[1, 1], padding='same', kernel_regularizer=spectral_norm,
                                          name='discriminator_out_conv_mat')

    def call(self, inputs, training=None, mask=None):
        f1, f2, vec = inputs[0], inputs[1], inputs[2]
        l = tf.reshape(vec, [-1, 1, 1, self.num_domains])
        b = tf.shape(f1)[0]
        h = tf.shape(f1)[1]
        w = tf.shape(f1)[2]
        k = tf.ones([b, h, w, self.num_domains])
        k = k * l

        m0 = tf.concat([f1, f2, k], axis=3)
        m1 = self.m1(m0)
        m1_gates = self.m1_gates(m0)
        m1_glu = self.m1_glu([m1, tf.sigmoid(m1_gates)])
        mat = self.mat(m1_glu)
        return mat


class Downsample2DBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, activation=None, sn=False, name_prefix=None):
        super(Downsample2DBlock, self).__init__()

        if sn:
            self.h1 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, activation=activation,
                                             kernel_regularizer=spectral_norm, padding='same',
                                             name=name_prefix + 'h1_conv')
            self.h1_gates = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides,
                                                   activation=activation, kernel_regularizer=spectral_norm,
                                                   padding='same', name=name_prefix + 'h1_gates')
        else:
            self.h1 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, activation=activation,
                                             padding='same', name=name_prefix + 'h1_conv')
            self.h1_gates = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides,
                                                   activation=activation,
                                                   padding='same', name=name_prefix + 'h1_gates')

        self.h1_norm = tfa.layers.InstanceNormalization(epsilon=1e-6, name=name_prefix + 'h1_norm')
        self.h1_norm_gates = tfa.layers.InstanceNormalization(epsilon=1e-6, name=name_prefix + 'h1_norm_gates')
        self.h1_glu = tf.keras.layers.Multiply(name=name_prefix + 'h1_glu')

    def __call__(self, inputs, training=None, mask=None):
        h1 = self.h1(inputs)
        h1 = self.h1_norm(h1)
        gates = self.h1_gates(inputs)
        gates = self.h1_norm_gates(gates)
        h1_glu = self.h1_glu([h1, tf.sigmoid(gates)])
        return h1_glu


class Residual1DBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, name_prefix=None):
        super(Residual1DBlock, self).__init__()
        self.h1 = tf.keras.layers.Conv1D(filters, kernel_size=kernel_size, strides=strides,
                                         padding='same', name=name_prefix + 'h1_conv')
        self.h1_norm = tfa.layers.InstanceNormalization(epsilon=1e-6, name=name_prefix + 'h1_norm')
        self.h1_gates = tf.keras.layers.Conv1D(filters, kernel_size=kernel_size, strides=strides,
                                               padding='same', name=name_prefix + 'h1_gates')
        self.h1_norm_gates = tfa.layers.InstanceNormalization(epsilon=1e-6, name=name_prefix + 'h1_norm_gates')
        self.h1_glu = tf.keras.layers.Multiply(name=name_prefix + 'h1_glu')

        self.h2 = tf.keras.layers.Conv1D(filters // 2, kernel_size=kernel_size, strides=strides,
                                         padding='same', name=name_prefix + 'h2_conv')
        self.h2_norm = tfa.layers.InstanceNormalization(epsilon=1e-6, name=name_prefix + 'h1_norm')

    def __call__(self, inputs, training=None, mask=None):
        h1 = self.h1(inputs)
        h1 = self.h1_norm(h1)
        h1_gates = self.h1_gates(inputs)
        h1_gates = self.h1_norm_gates(h1_gates)
        h1_glu = self.h1_glu([h1, tf.sigmoid(h1_gates)])

        h2 = self.h2(h1_glu)
        h2_norm = self.h2_norm(h2)

        h3 = inputs + h2_norm
        return h3


class Upsample2DBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, name_prefix=None):
        super(Upsample2DBlock, self).__init__()
        self.h1 = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides,
                                         padding='same', name=name_prefix + 'h1_conv')
        self.h1_norm = tfa.layers.InstanceNormalization(epsilon=1e-6, name=name_prefix + 'h1_norm')
        self.h1_gates = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides,
                                               padding='same', name=name_prefix + 'h1_gates')
        self.h1_norm_gates = tfa.layers.InstanceNormalization(epsilon=1e-6, name=name_prefix + 'h1_norm_gates')
        self.h1_glu = tf.keras.layers.Multiply(name=name_prefix + 'h1_glu')

    def __call__(self, inputs, training=None, mask=None):
        h1 = self.h1(inputs)
        h1_shuffle = tf.nn.depth_to_space(h1, block_size=2, name='h1_shuffle')
        h1_norm = self.h1_norm(h1_shuffle)
        h1_gates = self.h1_gates(inputs)
        h1_shuffle_gates = tf.nn.depth_to_space(h1_gates, block_size=2, name='h1_shuffle_gates')
        h1_norm_gates = self.h1_norm_gates(h1_shuffle_gates)
        h1_glu = self.h1_glu([h1_norm, tf.sigmoid(h1_norm_gates)])
        return h1_glu


@tf.keras.utils.register_keras_serializable(package='Custom', name='spectral')
def spectral_norm(weight_matrix):
    w_shape = tf.shape(weight_matrix)
    w = tf.reshape(weight_matrix, [-1, w_shape[-1]])
    u = tf.Variable(initial_value=tf.random.normal([1, w_shape[-1]]), trainable=False, name='u')

    u_hat = u
    v_hat = None
    for i in range(1):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    u.assign(u_hat)

    w_norm = w / sigma
    w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

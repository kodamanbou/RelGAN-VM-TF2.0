import tensorflow as tf
import os
import datetime
import io
import matplotlib.pyplot as plt
from librosa import display
from utils import *
from hyperparams import Hyperparameter as hp
from loss import l1_loss, l2_loss
from model import RelGAN

seed = 65535
np.random.seed(seed)
tf.random.set_seed(seed)


@tf.function
def train_step(inputs):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        outputs = model(inputs)
        generation_B = outputs[0]
        generation_B2 = outputs[1]
        generation_C = outputs[2]
        generation_A = outputs[3]
        cycle_A = outputs[4]
        generation_A_identity = outputs[5]
        generation_alpha_back = outputs[6]
        discrimination_B_fake = outputs[7]
        discrimination_B_real = outputs[8]
        discrimination_alpha_fake = outputs[9]
        discrimination_A_dot_fake = outputs[10]
        discrimination_A_dot_real = outputs[11]
        sr = outputs[12]
        sf = outputs[13]
        w1 = outputs[14]
        w2 = outputs[15]
        w3 = outputs[16]
        w4 = outputs[17]
        interpolate_identity = outputs[18]
        interpolate_B = outputs[19]
        interpolate_alpha = outputs[20]

        # cycle loss.
        cycle_loss = l1_loss(y=inputs[0], y_hat=cycle_A)

        # identity loss.
        identity_loss = l1_loss(y=inputs[0], y_hat=generation_A_identity)

        # backward loss.
        backward_loss = l1_loss(y=inputs[0], y_hat=generation_alpha_back)

        # mode-seeking loss.
        mode_seeking_loss = tf.divide(l1_loss(y=inputs[0], y_hat=inputs[1]),
                                      l1_loss(y=generation_B, y_hat=generation_B2))

        # triangle loss.
        triangle_loss = l1_loss(y=inputs[0], y_hat=generation_A)

        # generator loss.
        generator_loss_A2B = l2_loss(y=tf.ones_like(discrimination_B_fake), y_hat=discrimination_B_fake)

        # two-step generator loss.
        two_step_generator_loss_A = l2_loss(y=tf.ones_like(discrimination_A_dot_fake), y_hat=discrimination_A_dot_fake)

        # discriminator loss.
        discriminator_loss_B_real = l2_loss(y=tf.ones_like(discrimination_B_real), y_hat=discrimination_B_real)
        discriminator_loss_B_fake = l2_loss(y=tf.zeros_like(discrimination_B_fake), y_hat=discrimination_B_fake)
        discriminator_loss_B = (discriminator_loss_B_real + discriminator_loss_B_fake) / 2
        discriminator_loss_alpha = l2_loss(y=tf.zeros_like(discrimination_alpha_fake), y_hat=discrimination_alpha_fake)

        # conditional adversarial loss.
        discriminator_loss_cond_sr = l2_loss(y=tf.ones_like(sr), y_hat=sr)
        discriminator_loss_cond_sf = l2_loss(y=tf.zeros_like(sf), y_hat=sf)
        discriminator_loss_cond_w1 = l2_loss(y=tf.zeros_like(w1), y_hat=w1)
        discriminator_loss_cond_w2 = l2_loss(y=tf.zeros_like(w2), y_hat=w2)
        discriminator_loss_cond_w3 = l2_loss(y=tf.zeros_like(w3), y_hat=w3)
        discriminator_loss_cond_w4 = l2_loss(y=tf.zeros_like(w4), y_hat=w4)
        discriminator_loss_cond = discriminator_loss_cond_sr + discriminator_loss_cond_sf + discriminator_loss_cond_w1 \
                                  + discriminator_loss_cond_w2 + discriminator_loss_cond_w3 + discriminator_loss_cond_w4

        generator_loss_cond_sf = l2_loss(y=tf.ones_like(sf), y_hat=sf)

        # interpolation loss.
        discriminator_loss_interp_AB = l2_loss(y=tf.zeros_like(interpolate_identity),
                                               y_hat=interpolate_identity) if rnd == 0 else l2_loss(
            y=tf.zeros_like(interpolate_B), y_hat=interpolate_B)
        discriminator_loss_interp_alpha = l2_loss(
            y=tf.ones_like(interpolate_alpha) * tf.cast(tf.reshape(inputs[7], [-1, 1, 1, 1]), tf.float32),
            y_hat=interpolate_alpha) if rnd == 0 else l2_loss(
            y=tf.ones_like(interpolate_alpha) * tf.cast(tf.reshape(1. - inputs[7], [-1, 1, 1, 1]), tf.float32),
            y_hat=interpolate_alpha)
        discriminator_loss_interp = discriminator_loss_interp_AB + discriminator_loss_interp_alpha
        generator_loss_interp_alpha = l2_loss(y=tf.zeros_like(interpolate_alpha), y_hat=interpolate_alpha)

        # merge the losses.
        generator_loss = generator_loss_A2B + hp.lambda_backward * backward_loss + mode_seeking_loss + hp.lambda_cycle \
                         * cycle_loss + hp.lambda_identity * identity_loss + hp.lambda_triangle * triangle_loss + \
                         hp.lambda_conditional * generator_loss_cond_sf + hp.lambda_interp * generator_loss_interp_alpha
        discriminator_loss = discriminator_loss_B + hp.lambda_interp * discriminator_loss_interp + \
                             hp.lambda_conditional * discriminator_loss_cond

        # Optimizers
        generator_vars = model.generator.trainable_variables
        discriminator_vars = model.discriminator.trainable_variables + model.adversarial.trainable_variables + \
                             model.interpolate.trainable_variables + model.matching.trainable_variables
        grad_gen = gen_tape.gradient(generator_loss, sources=generator_vars)
        grad_dis = dis_tape.gradient(discriminator_loss, sources=discriminator_vars)
        generator_optimizer.apply_gradients(zip(grad_gen, generator_vars))
        discriminator_optimizer.apply_gradients(zip(grad_dis, discriminator_vars))

        # update summaries.
        gen_loss_summary.update_state(generator_loss)
        dis_loss_summary.update_state(discriminator_loss)
        cycle_loss_summary.update_state(cycle_loss)
        identity_loss_summary.update_state(identity_loss)
        triangle_loss_summary.update_state(triangle_loss)
        backward_loss_summary.update_state(backward_loss)
        interpolation_loss_summary.update_state(generator_loss_interp_alpha)
        conditional_loss_summary.update_state(generator_loss_cond_sf)
        generator_loss_A2B_summary.update_state(generator_loss_A2B)
        mode_seeking_loss_summary.update_state(mode_seeking_loss)
        discriminator_B_loss_summary.update_state(discriminator_loss_B)
        discriminator_loss_cond_summary.update_state(discriminator_loss_cond)
        discriminator_loss_interp_summary.update_state(discriminator_loss_interp)


@tf.function
def test_step(inputs):
    # returns generation_B.
    converted = model(inputs)[0][0]
    return converted


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)

    return image


def plot_spec(y):
    figure = plt.figure(figsize=(12, 8))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    display.specshow(D, y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')
    plt.tight_layout()

    return figure


if __name__ == '__main__':
    print('Loading cached data...')
    coded_sps_norms = []
    coded_sps_means = []
    coded_sps_stds = []
    log_f0s_means = []
    log_f0s_stds = []
    for f in glob.glob('pickles/*'):
        coded_sps_norm, coded_sps_mean, coded_sps_std, log_f0s_mean, log_f0s_std = load_pickle(
            os.path.join(f, f'cache{hp.num_mceps}.p'))
        coded_sps_norms.append(coded_sps_norm)
        coded_sps_means.append(coded_sps_mean)
        coded_sps_stds.append(coded_sps_std)
        log_f0s_means.append(log_f0s_mean)
        log_f0s_stds.append(log_f0s_std)

    # model and optimizers.
    num_domains = len(coded_sps_norms)
    model = RelGAN(num_domains, hp.batch_size)
    gen_lr_fn = tf.optimizers.schedules.PolynomialDecay(hp.generator_lr, hp.num_iterations, 1e-05)
    dis_lr_fn = tf.optimizers.schedules.PolynomialDecay(hp.discriminator_lr, hp.num_iterations, 2e-05)
    generator_optimizer = tf.optimizers.Adam(learning_rate=gen_lr_fn, beta_1=0.5)
    discriminator_optimizer = tf.optimizers.Adam(learning_rate=dis_lr_fn, beta_1=0.5)

    # summaries.
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logdir = os.path.join(hp.logdir, current_time)
    summary_writer = tf.summary.create_file_writer(logdir)
    gen_loss_summary = tf.keras.metrics.Mean()
    dis_loss_summary = tf.keras.metrics.Mean()
    cycle_loss_summary = tf.keras.metrics.Mean()
    identity_loss_summary = tf.keras.metrics.Mean()
    triangle_loss_summary = tf.keras.metrics.Mean()
    backward_loss_summary = tf.keras.metrics.Mean()
    interpolation_loss_summary = tf.keras.metrics.Mean()
    conditional_loss_summary = tf.keras.metrics.Mean()
    generator_loss_A2B_summary = tf.keras.metrics.Mean()
    mode_seeking_loss_summary = tf.keras.metrics.Mean()
    discriminator_B_loss_summary = tf.keras.metrics.Mean()
    discriminator_loss_cond_summary = tf.keras.metrics.Mean()
    discriminator_loss_interp_summary = tf.keras.metrics.Mean()

    iteration = 0
    while iteration < hp.num_iterations:
        if iteration % 10000 == 0:
            hp.lambda_triangle *= 0.9
            hp.lambda_backward *= 0.9

        x, x2, x_atr, y, y_atr, z, z_atr = sample_train_data(dataset_A=coded_sps_norms, nBatch=hp.batch_size)

        x_labels = np.zeros([hp.batch_size, num_domains])
        y_labels = np.zeros([hp.batch_size, num_domains])
        z_labels = np.zeros([hp.batch_size, num_domains])
        for b in range(hp.batch_size):
            x_labels[b] = np.identity(num_domains)[x_atr[b]]
            y_labels[b] = np.identity(num_domains)[y_atr[b]]
            z_labels[b] = np.identity(num_domains)[z_atr[b]]

        rnd = np.random.randint(2)
        alpha = np.random.uniform(0, 0.5, size=hp.batch_size) if rnd == 0 else np.random.uniform(0.5, 1.0,
                                                                                                 size=hp.batch_size)

        inputs = [x, x2, y, z, x_labels, y_labels, z_labels, alpha]
        # training.
        train_step(inputs)

        if iteration % 10 == 0:
            with summary_writer.as_default():
                tf.summary.scalar('Generator loss', gen_loss_summary.result(), step=iteration)
                tf.summary.scalar('Cycle loss', cycle_loss_summary.result(), step=iteration)
                tf.summary.scalar('Identity loss', identity_loss_summary.result(), step=iteration)
                tf.summary.scalar('Triangle loss', triangle_loss_summary.result(), step=iteration)
                tf.summary.scalar('Backward loss', backward_loss_summary.result(), step=iteration)
                tf.summary.scalar('Interpolation loss', interpolation_loss_summary.result(), step=iteration)
                tf.summary.scalar('Conditional loss', conditional_loss_summary.result(), step=iteration)
                tf.summary.scalar('Generator adversarial loss', generator_loss_A2B_summary.result(), step=iteration)
                tf.summary.scalar('Mode seeking loss', mode_seeking_loss_summary.result(), step=iteration)
                tf.summary.scalar('Discriminator loss', dis_loss_summary.result(), step=iteration)
                tf.summary.scalar('Discriminator loss B', discriminator_B_loss_summary.result(), step=iteration)
                tf.summary.scalar('Discriminator loss conditional', discriminator_loss_cond_summary.result(),
                                  step=iteration)
                tf.summary.scalar('Discriminator loss interpolation', discriminator_loss_interp_summary.result(),
                                  step=iteration)

                print(f'Iteration: {iteration} \tGenerator loss: {gen_loss_summary.result()} \tDiscriminator loss: '
                      f'{dis_loss_summary.result()}')

                gen_loss_summary.reset_states()
                cycle_loss_summary.reset_states()
                identity_loss_summary.reset_states()
                triangle_loss_summary.reset_states()
                backward_loss_summary.reset_states()
                interpolation_loss_summary.reset_states()
                conditional_loss_summary.reset_states()
                generator_loss_A2B_summary.reset_states()
                mode_seeking_loss_summary.reset_states()
                dis_loss_summary.reset_states()
                discriminator_B_loss_summary.reset_states()
                discriminator_loss_cond_summary.reset_states()
                discriminator_loss_interp_summary.reset_states()

        if iteration % 2500 == 0:
            model.save_weights(os.path.join(hp.weights_dir, 'weights_{:}'.format(iteration)))

        if iteration % 1000 == 0:
            eval_dirs = os.listdir(hp.eval_dir)
            x, x2, x_atr, y, y_atr, z, z_atr = sample_train_data(coded_sps_norms, nBatch=1)
            x_labels = np.zeros([1, num_domains])
            y_labels = np.zeros([1, num_domains])
            z_labels = np.zeros([1, num_domains])
            x_labels[0] = np.identity(num_domains)[x_atr[0]]
            y_labels[0] = np.identity(num_domains)[y_atr[0]]
            z_labels[0] = np.identity(num_domains)[z_atr[0]]
            x_atr = x_atr[0]
            y_atr = y_atr[0]
            eval_dir = os.path.join(hp.eval_dir, eval_dirs[x_atr])
            print(eval_dir)

            for file in glob.glob(eval_dir + '/*.wav'):
                alpha = np.ones(1)
                wav, _ = librosa.load(file, sr=hp.rate, mono=True)
                wav *= 1. / max(0.01, np.max(np.abs(wav)))
                wav = wav_padding(wav, sr=hp.rate, frame_period=hp.duration, multiple=4)
                f0, timeaxis, sp, ap = world_decompose(wav, fs=hp.rate, frame_period=hp.duration)
                f0s_mean_A = np.exp(log_f0s_means[x_atr])
                f0s_meanB = np.exp(log_f0s_means[y_atr])
                f0s_mean_AB = alpha * f0s_meanB + (1 - alpha) * f0s_mean_A
                log_f0s_mean_AB = np.log(f0s_mean_AB)
                f0s_std_A = np.exp(log_f0s_stds[x_atr])
                f0s_std_B = np.exp(log_f0s_stds[y_atr])
                f0s_std_AB = alpha * f0s_std_B + (1 - alpha) * f0s_std_A
                log_f0s_std_AB = np.log(f0s_std_AB)
                f0_converted = pitch_conversion(f0, log_f0s_means[x_atr], log_f0s_stds[x_atr], log_f0s_mean_AB,
                                                log_f0s_std_AB)
                coded_sp = world_encode_spectral_envelop(sp, fs=hp.rate, dim=hp.num_mceps)
                coded_sp_transposed = coded_sp.T
                coded_sp_norm = (coded_sp_transposed - coded_sps_means[x_atr]) / coded_sps_stds[x_atr]
                coded_sp_norm = np.array([coded_sp_norm])

                inputs = [coded_sp_norm, coded_sp_norm, coded_sp_norm, coded_sp_norm, x_labels, y_labels, z_labels,
                          alpha]

                coded_sp_converted_norm = test_step(inputs).numpy()
                if coded_sp_converted_norm.shape[1] > len(f0):
                    coded_sp_converted_norm = coded_sp_converted_norm[:, :-1]
                coded_sps_mean_AB = alpha * coded_sps_means[y_atr] + (1 - alpha) * coded_sps_means[x_atr]
                coded_sps_std_AB = alpha * coded_sps_stds[y_atr] + (1 - alpha) * coded_sps_stds[x_atr]
                coded_sp_converted = coded_sp_converted_norm * coded_sps_std_AB + coded_sps_mean_AB
                coded_sp_converted = np.ascontiguousarray(coded_sp_converted.T)
                decoded_sp_converted = world_decode_spectral_envelop(coded_sp_converted, fs=hp.rate)
                wav_transformed = world_speech_synthesis(f0_converted, decoded_sp_converted, ap, fs=hp.rate,
                                                         frame_period=hp.duration)
                wav_transformed *= 1. / max(0.01, np.max(np.abs(wav_transformed)))

                with summary_writer.as_default():
                    fig = plot_spec(wav_transformed)
                    img = plot_to_image(fig)
                    tf.summary.image(
                        f'Spec_iteration_{iteration}_{eval_dir.split("/")[-1]}_to_{eval_dirs[y_atr].split("/")[-1]}',
                        img, step=iteration)

                    wav_transformed = np.expand_dims(wav_transformed, axis=-1)
                    wav_transformed = np.expand_dims(wav_transformed, axis=0)
                    tf.summary.audio(f'Generated_{eval_dirs[y_atr].split("/")[-1]}_iteration_{iteration}',
                                     wav_transformed, sample_rate=hp.rate, step=iteration)

        iteration += 1

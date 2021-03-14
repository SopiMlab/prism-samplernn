import tensorflow as tf
import numpy as np
import time
from .sample_mlp import SampleMLP
from .frame_rnn import FrameRNN
from .utils import unsqueeze


class SampleRNN(tf.keras.Model):

    def __init__(self, batch_size, frame_sizes, q_levels, q_type, dim, rnn_type,
                 num_rnn_layers, seq_len, emb_size, skip_conn, rnn_dropout):
        super(SampleRNN, self).__init__()
        self.batch_size = batch_size
        self.big_frame_size = frame_sizes[1]
        self.frame_size = frame_sizes[0]
        self.q_type = q_type
        self.q_levels = q_levels
        self.dim = dim
        self.rnn_type = rnn_type
        self.num_rnn_layers = num_rnn_layers
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.skip_conn = skip_conn

        self.big_frame_rnn = FrameRNN(
            rnn_type = self.rnn_type,
            frame_size = self.big_frame_size,
            num_lower_tier_frames = self.big_frame_size // self.frame_size,
            num_layers = self.num_rnn_layers,
            dim = self.dim,
            q_levels = self.q_levels,
            skip_conn = self.skip_conn,
            dropout=rnn_dropout
        )

        self.frame_rnn = FrameRNN(
            rnn_type = self.rnn_type,
            frame_size = self.frame_size,
            num_lower_tier_frames = self.frame_size,
            num_layers = self.num_rnn_layers,
            dim = self.dim,
            q_levels = self.q_levels,
            skip_conn = self.skip_conn,
            dropout=rnn_dropout
        )

        self.sample_mlp = SampleMLP(
            self.frame_size, self.dim, self.q_levels, self.emb_size
        )

    @tf.function
    def train_step(self, data):
        (x, y) = data
        with tf.GradientTape() as tape:
            raw_output = self(x, training=True)
            prediction = tf.reshape(raw_output, [-1, self.q_levels])
            target = tf.reshape(y, [-1])
            loss = self.compiled_loss(
                target,
                prediction,
                regularization_losses=self.losses)
        grads = tape.gradient(loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(target, prediction)
        return {metric.name: metric.result() for metric in self.metrics}

    @tf.function
    def test_step(self, data):
        (x, y) = data
        raw_output = self(x, training=True)
        prediction = tf.reshape(raw_output, [-1, self.q_levels])
        target = tf.reshape(y, [-1])
        loss = self.compiled_loss(
            target,
            prediction,
            regularization_losses=self.losses)
        self.compiled_metrics.update_state(target, prediction)
        return {metric.name: metric.result() for metric in self.metrics}

    # Sampling function
    def sample(self, samples, temperature):
        samples = tf.nn.log_softmax(samples)
        samples = tf.cast(samples, tf.float64)
        # print(f"sample()")
        # print(f" temperature.shape={temperature.shape}") # (1,1024)
        # print(f" samples.shape={samples.shape}") # (1,256)
        samples = samples / temperature
        sample = tf.random.categorical(samples, 1)
        return tf.cast(sample, tf.int32)

    # Inference
    @tf.function
    def inference_step(self, inputs, temperature):
        num_samps = self.big_frame_size
        samples = inputs
        # print("inference_step()")
        # print(f" big_frame_size={self.big_frame_size}")
        # print(f" frame_size={self.frame_size}")
        # print(f" q_levels={self.q_levels}")
        # print(f" dim={self.dim}")
        # print(f" num_rnn_layers={self.num_rnn_layers}")
        # print(f" seq_len={self.seq_len}")
        # print(f" emb_size={self.emb_size}")
        # print(f" temperature.shape={temperature.shape}") # should be (batch_size, big_frame_size)
        big_frame_outputs = self.big_frame_rnn(tf.cast(inputs, tf.float32))
        for t in range(num_samps, num_samps * 2):
            i = t - num_samps
            # print(f" t={t}") # 64
            # print(f" samples.shape={samples.shape}") # (1,64,1)
            if t % self.frame_size == 0:
                frame_inputs = samples[:, t - self.frame_size : t, :]
                # print(f" frame_inputs.shape={frame_inputs.shape}") # (1,16,1)
                big_frame_output_idx = (t // self.frame_size) % (
                    self.big_frame_size // self.frame_size
                )
                # print(f" big_frame_output_idx={big_frame_output_idx}")
                rnn_conditioning_frames = unsqueeze(big_frame_outputs[:, big_frame_output_idx, :], 1)
                # print(f" rnn_conditioning_frames.shape={rnn_conditioning_frames.shape}") # (1,1,1024)
                frame_outputs = self.frame_rnn(
                    tf.cast(frame_inputs, tf.float32),
                    conditioning_frames=rnn_conditioning_frames)
                # print(f" frame_outputs.shape={frame_outputs.shape}") # (1,16,1024)
            sample_inputs = samples[:, t - self.frame_size : t, :]
            # print(f" sample_inputs.shape={sample_inputs.shape}") # (1,16,1)
            frame_output_idx = t % self.frame_size
            # print(f" frame_output_idx={frame_output_idx}")
            mlp_conditioning_frames = unsqueeze(frame_outputs[:, frame_output_idx, :], 1)
            # print(f" mlp_conditioning_frames.shape={mlp_conditioning_frames.shape}") # (1,1,1024)
            sample_outputs = self.sample_mlp(
                sample_inputs,
                conditioning_frames=mlp_conditioning_frames)
            # print(f" sample_outputs.shape={sample_outputs.shape}") # (1,1,256)
            # Generate
            sample_outputs = tf.reshape(sample_outputs, [-1, self.q_levels])
            # print(f" sample_outputs'.shape={sample_outputs.shape}") # (1,256)
            temp = temperature
            if temp.shape[-1] > 1:
                temp = temp[:, i:i+1]
            generated = self.sample(sample_outputs, temp)
            # print(f" generated.shape={generated.shape}")
            generated = tf.reshape(generated, [self.batch_size, 1, 1])
            # print(f" generated'.shape={generated.shape}")
            samples = tf.concat([samples, generated], axis=1)
            # print(f" samples.shape'={samples.shape}")
        return samples[:, num_samps:]

    def reset_rnn_states(self):
        self.big_frame_rnn.reset_states()
        self.frame_rnn.reset_states()

    def call(self, inputs, training=True, temperature=1.0):
        if training==True:
            # UPPER TIER
            big_frame_outputs = self.big_frame_rnn(
                tf.cast(inputs, tf.float32)[:, : -self.big_frame_size, :]
            )
            # MIDDLE TIER
            frame_outputs = self.frame_rnn(
                tf.cast(inputs, tf.float32)[:, self.big_frame_size-self.frame_size : -self.frame_size, :],
                conditioning_frames=big_frame_outputs,
            )
            # LOWER TIER (SAMPLES)
            sample_output = self.sample_mlp(
                inputs[:, self.big_frame_size - self.frame_size : -1, :],
                conditioning_frames=frame_outputs,
            )
            return sample_output
        else:
            return self.inference_step(inputs, temperature)

from __future__ import print_function
import argparse
from functools import reduce
import math
import os
import pkgutil
import re
import shlex
import sys
import time
import json

import tensorflow as tf
import numpy as np
import librosa

from samplernn import (SampleRNN, write_wav, quantize, dequantize, unsqueeze)


OUTPUT_DUR = 3 # Duration of generated audio in seconds
SAMPLE_RATE = 22050 # Sample rate of generated audio
NUM_SEQS = 1
SAMPLING_TEMPERATURE = 0.75
SEED_OFFSET = 0

def get_arguments():
    def check_positive(value):
        val = int(value)
        if val < 1:
             raise argparse.ArgumentTypeError("%s is not positive" % value)
        return val

    def check_env(value):
        m = re.match(r"^Env\(\[(.*?)\], *\[(.*?)\], *\[(.*?)\]\)$", value)
        if m == None:
            return None
            
        levels, times, curve = map(lambda x: list(map(float, re.split(r", *", x))), m.groups())
        return Env(levels, times, curve)
    
    def check_temperature(value):            
        return check_env(value) or float(value)
    
    parser = argparse.ArgumentParser(description='PRiSM TensorFlow SampleRNN Generator')
    parser.add_argument('--output_dir',                 type=str,            default=".",
                                                        help='Path to the generated .wav file')
    parser.add_argument('--output_path',                type=str,            default=None,
                                                        help='Path to the generated .wav file')
    parser.add_argument('--checkpoint_path',            type=str,            required=True,
                                                        help='Path to a saved checkpoint for the model')
    parser.add_argument('--config_file',                type=str,            default=None,
                                                        help='Path to the JSON config for the model')
    parser.add_argument('--dur',                        type=check_positive, default=OUTPUT_DUR,
                                                        help='Duration of generated audio')
    parser.add_argument('--num_seqs',                   type=check_positive, default=NUM_SEQS,
                                                        help='Number of audio sequences to generate')
    parser.add_argument('--sample_rate',                type=check_positive, default=SAMPLE_RATE,
                                                        help='Sample rate of the generated audio')
    parser.add_argument('--temperature',                type=check_temperature,
                                                        default=SAMPLING_TEMPERATURE, nargs='+',
                                                        help='Sampling temperature')
    parser.add_argument('--seed',                       type=str,            help='Path to audio for seeding')
    parser.add_argument('--seed_offset',                type=int,            default=SEED_OFFSET,
                                                        help='Starting offset of the seed audio')
    return parser.parse_args()


# On generation speed: https://github.com/soroushmehr/sampleRNN_ICLR2017/issues/19
# Speed again: https://ambisynth.blogspot.com/2018/09/wavernn.html
# On seeding (sort of): https://github.com/soroushmehr/sampleRNN_ICLR2017/issues/11
# Very interesting article on sampling temperature (including the idea of varying it
# while sampling): https://www.robinsloan.com/expressive-temperature/

'''
def generate_and_save_samples_OLD(model, path, seed, seed_offset=0, dur=OUTPUT_DUR,
                              sample_rate=SAMPLE_RATE, temperature=SAMPLING_TEMPERATURE):

    # Sampling function
    def sample(samples, temperature=SAMPLING_TEMPERATURE):
        samples = tf.nn.log_softmax(samples)
        samples = tf.cast(samples, tf.float64)
        samples = samples / temperature
        return tf.random.categorical(samples, 1)

    q_type = model.q_type
    q_levels = model.q_levels
    q_zero = q_levels // 2
    num_samps = dur * sample_rate

    # Precompute sample sequences, initialised to q_zero.
    samples = np.full((model.batch_size, model.big_frame_size + num_samps, 1), q_zero, dtype='int32')

    # Set seed if provided.
    if seed is not None:
        seed_audio = load_seed_audio(seed, seed_offset, model.big_frame_size)
        samples[:, :model.big_frame_size, :] = quantize(seed_audio, q_type, q_levels)

    print_progress_every = 250
    start_time = time.time()

    # Run the model tiers. Generates a single sample per step. Each frame-level tier
    # consumes one frame of samples per step.
    for t in range(model.big_frame_size, model.big_frame_size + num_samps):

        # Top tier (runs every big_frame_size steps)
        if t % model.big_frame_size == 0:
            inputs = samples[:, t - model.big_frame_size : t, :].astype('float32')
            big_frame_outputs = model.big_frame_rnn(inputs)

        # Middle tier (runs every frame_size steps)
        if t % model.frame_size == 0:
            inputs = samples[:, t - model.frame_size : t, :].astype('float32')
            big_frame_output_idx = (t // model.frame_size) % (
                model.big_frame_size // model.frame_size
            )
            frame_outputs = model.frame_rnn(
                inputs,
                conditioning_frames=unsqueeze(big_frame_outputs[:, big_frame_output_idx, :], 1))

        # Sample level tier (runs once per step)
        inputs = samples[:, t - model.frame_size : t, :]
        frame_output_idx = t % model.frame_size
        sample_outputs = model.sample_mlp(
            inputs,
            conditioning_frames=unsqueeze(frame_outputs[:, frame_output_idx, :], 1))

        # Generate
        sample_outputs = tf.reshape(sample_outputs, [-1, q_levels])
        generated = sample(sample_outputs, temperature)

        # Monitor progress
        start = t - model.big_frame_size
        if start % print_progress_every == 0:
            end = min(start + print_progress_every, num_samps)
            duration = time.time() - start_time
            template = 'Generating samples {} - {} of {} (time elapsed: {:.3f} seconds)'
            print(template.format(start+1, end, num_samps, duration))

        # Update sequences
        samples[:, t] = np.array(generated).reshape([-1, 1])

    # Save sequences to disk
    path = path.split('.wav')[0]
    for i in range(model.batch_size):
        seq = samples[i].reshape([-1, 1])[model.big_frame_size :].tolist()
        audio = dequantize(seq, q_type, q_levels)
        file_name = '{}_{}'.format(path, str(i)) if model.batch_size > 1 else path
        file_name = '{}.wav'.format(file_name)
        write_wav(file_name, audio, sample_rate)
        print('Generated sample output to {}'.format(file_name))
    print('Done')
'''

def create_inference_model(ckpt_path, num_seqs, config):
    model = SampleRNN(
        batch_size = num_seqs, # Generate sequences in batches
        frame_sizes = config['frame_sizes'],
        seq_len = config['seq_len'],
        q_type = config['q_type'],
        q_levels = config['q_levels'],
        dim = config['dim'],
        rnn_type = config.get('rnn_type'),
        num_rnn_layers = config['num_rnn_layers'],
        emb_size = config['emb_size'],
        skip_conn = config.get('skip_conn'),
        rnn_dropout=config.get('rnn_dropout')

    )
    num_samps = config['seq_len'] + model.big_frame_size
    init_data = np.zeros((model.batch_size, num_samps, 1), dtype='int32')
    model(init_data)
    model.load_weights(ckpt_path).expect_partial()
    return model

def load_seed_audio(path, offset, length):
    (audio, _) = librosa.load(path, sr=None, mono=True)
    assert offset + length <= len(audio), 'Seed offset plus length exceeds audio length'
    chunk = audio[offset : offset + length]
    return chunk.reshape(-1, 1)

NUM_FRAMES_TO_PRINT = 4

def lincurve(x, l0, r0, l1, r1, curve):
    if abs(curve) < 0.001:
        return (x - l0) / (r0 - l0) * (r1 - l1) + l1
    grow = math.exp(curve)
    a = (r1 - l1) / (1.0 - grow)
    b = l1 + a
    scaled = (x - l0) / (r0 - l0)
    return b - a * pow(grow, scaled)

class Env:
    def __init__(self, levels=[0,1,0], times=[1,1], curves=[0]):
        self.levels = levels
        self.times = times
        self.curves = curves
        
    def value(self, t):
        if t < 0.0:
            return self.levels[0]
        i = 0
        t0 = 0.0
        t1 = 0.0
        for i in range(len(self.levels)):
            t1 = t0 + self.times[i] if i < len(self.times) else None
            if t1 == None or t < t1:
                break
            t0 = t1
        v0 = self.levels[i]
        if i == len(self.levels)-1:
            return v0
        v1 = self.levels[i+1]
        k = (t-t0)/(t1-t0)
        k = lincurve(k, 0.0, 1.0, 0.0, 1.0, self.curves[i])
        v = (1.0-k)*v0 + k*v1
        return v
    
    def duration(self):
        return sum(self.times)
    
    def discretize(self, n, dur=None):
        if dur == None:
            dur = self.duration()
        sig = [self.value(i/n*dur) for i in range(n)]
        return sig
    
    def __repr__(self):
        return f"Env({self.levels}, {self.times}, {self.curves})"

def get_temperature(temperature, batch_size, num_samps, dur):
    # get_temperature()
    #  temperature=[0.96, 1.0]
    #  batch_size=2
    #  temp=[[0.95999998]
    #  [1.        ]]
    # -> shape = (batch_size, 1)
    print("get_temperature()")
    print(f" temperature={temperature}")
    print(f" batch_size={batch_size}")
    if not isinstance(temperature, list):
        temperature = [temperature]
    have_envs = any(isinstance(temp, Env) for temp in temperature)
    # TODO: handle mixed temperature types
    if have_envs:
        temperature = [env.discretize(num_samps, dur) for env in temperature]
    if len(temperature) < batch_size:
        last_val = temperature[len(temperature)-1]
        while len(temperature) < batch_size:
            temperature = temperature + [last_val]
    elif len(temperature) > batch_size:
        temperature = temperature[:batch_size]
    temperature = tf.reshape(temperature, (batch_size, -1))
    temp = tf.cast(temperature, tf.float64)
    print(f" temp={temp}")
    assert len(temp.shape) > 0, "temp is empty"
    return temp

def format_dur(dur, subdivs=((1,"s"), (60,"m"), (60,"h"), (24,"d"))):
    lens, units = list(zip(*subdivs))
    nums = reduce(lambda a, b: (a[0]//b, [a[0]%b] + a[1]), lens, (dur, []))
    nums = list(zip(nums[1][-2::-1] + [nums[0]], units))[::-1]
    while len(nums) > 1 and nums[0][0] == 0:
        nums.pop(0)
    return " ".join(f"{num}{unit}" for num, unit in nums)

def generate(output_dir, output_path, ckpt_path, config, num_seqs=NUM_SEQS,
             dur=OUTPUT_DUR, sample_rate=SAMPLE_RATE, temperature=SAMPLING_TEMPERATURE,
             seed=None, seed_offset=None, raw_args=[]):
    model = create_inference_model(ckpt_path, num_seqs, config)
    q_type = model.q_type
    q_levels = model.q_levels
    q_zero = q_levels // 2
    num_samps = dur * sample_rate
    # print("generate()")
    # print(f" num_samps={num_samps}") # 128000
    # print(f" temperature={temperature}")
    temperature = get_temperature(temperature, num_seqs, num_samps, dur)
    # print(f" temperature'.shape={temperature.shape}")
    # Save args to disk
    if output_path != None:
        path = output_path
        name = os.path.splitext(os.path.basename(output_path))[0]
        args_path = os.path.join(os.path.dirname(output_path), f"{name}_args.txt")
    else:
        try:
            os.makedirs(output_dir)
        except FileExistsError:
            pass
        path = os.path.join(output_dir, f"generated.wav")
        args_path = os.path.join(output_dir, "args.txt")
    with open(args_path, "w") as fp:
        args_str = " ".join(shlex.quote(arg) for arg in raw_args)
        fp.write(args_str)
    # Precompute sample sequences, initialised to q_zero.
    samples = []
    init_samples = np.full((model.batch_size, model.big_frame_size, 1), q_zero)
    # Set seed if provided.
    if seed is not None:
        seed_audio = load_seed_audio(seed, seed_offset, model.big_frame_size)
        seed_audio = tf.convert_to_tensor(seed_audio)
        init_samples[:, :model.big_frame_size, :] = quantize(seed_audio, q_type, q_levels)
    init_samples = tf.constant(init_samples, dtype=tf.int32)
    samples.append(init_samples)
    # print(f" len(samples)={len(samples)}")
    # print(f" samples[0].shape={samples[0].shape}") # (1,64,1)
    print_progress_every = NUM_FRAMES_TO_PRINT * model.big_frame_size
    start_time = time.time()
    stats = [0.0] * 10
    for i in range(0, num_samps // model.big_frame_size):
        t = i * model.big_frame_size
        # Generate samples
        temp = temperature
        if temp.shape[-1] > 1:
            start = i * model.big_frame_size
            stop = (i+1) * model.big_frame_size
            temp = temperature[:, start:stop]
        # print(f" temp.shape={temp.shape}")
        gen_start_time = time.time()
        frame_samples = model(samples[i], training=False, temperature=temp)
        # print(f" frame_samples.shape={frame_samples.shape}")
        gen_end_time = time.time()
        samples.append(frame_samples)
        # print(f" len(samples')={len(samples)}")
        del stats[0]
        stats.append(gen_end_time - gen_start_time)
        # Monitor progress
        if t % print_progress_every == 0:
            end = min(t + print_progress_every, num_samps)
            step_dur = time.time() - start_time
            stats_num = min(i+1, len(stats)) * model.big_frame_size
            stats_dur = sum(stats)
            time_rem = 0
            if stats_dur > 0:
                rate = stats_num / stats_dur
                num_rem = num_samps - t
                time_rem = int(round(num_rem / rate))
            remaining = format_dur(time_rem)
            print(f'Generated samples {t+1} - {end} of {num_samps} (time elapsed: {step_dur:.3f} seconds, remaining: {remaining})')
    samples = tf.concat(samples, axis=1)
    samples = samples[:, model.big_frame_size:, :]
    # Save sequences to disk
    path = path.split('.wav')[0]
    for i in range(model.batch_size):
        seq = np.reshape(samples[i], (-1, 1))[model.big_frame_size :].tolist()
        audio = dequantize(seq, q_type, q_levels)
        file_name = '{}_{}'.format(path, str(i)) if model.batch_size > 1 else path
        file_name = '{}.wav'.format(file_name)
        write_wav(file_name, audio, sample_rate)
        print('Generated sample output to {}'.format(file_name))
    print('Done')


def find_checkpoint_path(args):
    if os.path.isdir(args.checkpoint_path):
        max_ckpt = None
        for fn in os.listdir(args.checkpoint_path):
            m = re.match(r'^(model\.ckpt-(\d+))\.index$', fn)
            if m:
                num = int(m.group(2))
                if max_ckpt == None or max_ckpt[1] < num:
                    max_ckpt = (m.group(1), num)
        if max_ckpt == None:
            print('no model.ckpt-#.index files found in checkpoint dir')
            sys.exit(1)
        return os.path.join(args.checkpoint_path, max_ckpt[0])
    else:
        return args.checkpoint_path

def find_config(ckpt_path, config_path):
    config = None
    if config_path == None:
        ckpt_dir = os.path.dirname(ckpt_path)
        is_config = lambda fn: not fn.startswith('.') and fn.endswith('.config.json')
        ckpt_configs = [fn for fn in os.listdir(ckpt_dir) if is_config(fn)]
        num_ckpt_configs = len(ckpt_configs)
        if num_ckpt_configs > 1:
            print(f'error: checkpoint directory contains multiple configs: {ckpt_configs}')
            sys.exit(1)
        elif num_ckpt_configs == 1:
            print(f'config: {ckpt_configs[0]} (from checkpoint directory)')
            config_path = os.path.join(ckpt_dir, ckpt_configs[0])
    else:
        print(f'config: {config_path}')
    if config_path != None:
        with open(config_path, 'r') as config_file:
            return json.load(config_file)
    else:
        print('config: default')
        return json.loads(pkgutil.get_data('samplernn_scripts', 'conf/default.config.json'))

    
def main():
    args = get_arguments()
    checkpoint_path = find_checkpoint_path(args)
    print(f'checkpoint: {checkpoint_path}')
    config = find_config(checkpoint_path, args.config_file)
    generate(args.output_dir, args.output_path, checkpoint_path, config, args.num_seqs, args.dur,
             args.sample_rate, args.temperature, args.seed, args.seed_offset, sys.argv[1:])


if __name__ == '__main__':
    main()

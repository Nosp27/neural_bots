import numpy as np
import tensorflow as tf
from keras import layers, activations, models, preprocessing, utils
import keras.preprocessing.text
import pandas as pd
import re


class MessageBot:
    def __init__(self, model_file: str, *, history_df, df_crop_last=None):
        if history_df is None:
            raise RuntimeError
        elif history_df.__class__ is str:
            history_df = self.read_df_standart(history_df)

        self.max_input_length = None
        self.max_output_length = None

        self.model = None
        self.enc_model = None
        self.dec_model = None

        self.encoder_inputs = None
        self.encoder_states = None
        self.decoder_inputs = None
        self.decoder_embedding = None
        self.decoder_lstm = None
        self.decoder_dense = None

        self.lines = self.lines_work(history_df, df_crop_last)

        self.my_tokenizer = self.create_tokenizer()
        self.opponent_tokenizer = self.create_tokenizer()

        self.tokenized_my_lines = self.get_my_lines(self.lines)

        self.num_my_tokens = self.get_num_my_tokens()

        self.tokenized_opponent_lines = self.get_opponent_lines(self.lines)
        self.num_opponent_tokens = self.get_num_opponent_tokens()

        self.max_input_length = self.get_max_input_length()
        self.max_output_length = self.get_max_output_length()

        self.encoder()
        self.load_fitted_model(saved_model_weights_file=model_file)

    def read_df_standart(self, path):
        return pd.read_table(path, names=['me', 'other', 'rest'])

    def lines_work(self, input_df, df_crop_last=None):
        lines = input_df
        lines = lines.drop(columns=['rest'])
        lines = lines.loc[(pd.notna(lines.other)) & (pd.notna(lines.me))]
        lines = lines.loc[~(lines.other.str.contains('(ред.)', regex=False))]
        lines.reset_index(level=0, inplace=True)

        if df_crop_last is not None:
            lines = lines[df_crop_last:]

        return lines

    def create_tokenizer(self):
        return preprocessing.text.Tokenizer()

    def get_my_lines(self, lines):
        my_lines = list(lines.me)

        tokenizer = self.my_tokenizer
        tokenizer.fit_on_texts(my_lines)
        return tokenizer.texts_to_sequences(my_lines)

    def get_max_input_length(self):
        tokenized_my_lines = self.tokenized_my_lines
        length_list = list()
        for token_seq in tokenized_my_lines:
            length_list.append(len(token_seq))
        return np.array(length_list).max()

    def get_num_my_tokens(self):
        return len(self.my_tokenizer.word_index) + 1

    def get_opponent_lines(self, lines):
        opponent_lines = list()
        for line in lines.other:
            opponent_lines.append('<START> ' + line + ' <END>')

        tokenizer = self.opponent_tokenizer
        tokenizer.fit_on_texts(opponent_lines)
        return tokenizer.texts_to_sequences(opponent_lines)

    def get_max_output_length(self):
        tokenized_opponent_lines = self.tokenized_opponent_lines
        length_list = list()
        for token_seq in tokenized_opponent_lines:
            length_list.append(len(token_seq))
        return np.array(length_list).max()

    def get_num_opponent_tokens(self):
        opponent_word_dict = self.opponent_tokenizer.word_index
        return len(opponent_word_dict) + 1

    def encoder(self):
        self.encoder_inputs = tf.keras.layers.Input(shape=(None,))
        encoder_embedding = tf.keras.layers.Embedding(self.num_my_tokens, 256, mask_zero=True)(self.encoder_inputs)
        encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(128, return_state=True)(encoder_embedding)
        self.encoder_states = [state_h, state_c]

        self.decoder_inputs = tf.keras.layers.Input(shape=(None,))
        self.decoder_embedding = tf.keras.layers.Embedding(self.num_opponent_tokens, 256, mask_zero=True)(
            self.decoder_inputs)
        self.decoder_lstm = tf.keras.layers.LSTM(128, return_state=True, return_sequences=True)
        decoder_outputs, _, _ = self.decoder_lstm(self.decoder_embedding, initial_state=self.encoder_states)
        self.decoder_dense = tf.keras.layers.Dense(self.num_opponent_tokens, activation=tf.keras.activations.softmax)
        output = self.decoder_dense(decoder_outputs)

        self.model = tf.keras.models.Model([self.encoder_inputs, self.decoder_inputs], output)
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')

    def load_fitted_model(self, *, saved_model_weights_file):
        self.model.load_weights(saved_model_weights_file)

    def make_inference_models(self):
        encoder_model = tf.keras.models.Model(self.encoder_inputs, self.encoder_states)

        decoder_state_input_h = tf.keras.layers.Input(shape=(128,))
        decoder_state_input_c = tf.keras.layers.Input(shape=(128,))

        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = self.decoder_lstm(
            self.decoder_embedding, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = self.decoder_dense(decoder_outputs)
        decoder_model = tf.keras.models.Model(
            [self.decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        return encoder_model, decoder_model

    def str_to_tokens(self, sentence: str):
        my_word_dict = self.my_tokenizer.word_index
        words = sentence.lower().split()
        tokens_list = list()
        for word in words:
            tokens_list.append(my_word_dict[word])
        return preprocessing.sequence.pad_sequences([tokens_list], maxlen=self.max_input_length, padding='post')

    def translate(self, raw_text: str):
        try:
            return self.translate_internal(raw_text)
        except KeyError:
            return self.translate_internal(raw_text.replace('ё', 'е'))

    def translate_internal(self, raw_text: str):
        text = re.sub(r'[^а-яА-Яё ]', ' ', raw_text)
        opponent_word_dict = self.opponent_tokenizer.word_index
        if not self.enc_model or not self.dec_model:
            self.enc_model, self.dec_model = self.make_inference_models()
        states_values = self.enc_model.predict(self.str_to_tokens(text))
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = opponent_word_dict['start']
        stop_condition = False
        decoded_translation = ''
        while not stop_condition:
            dec_outputs, h, c = self.dec_model.predict([empty_target_seq] + states_values)
            sampled_word_index = np.argmax(dec_outputs[0, -1, :])
            sampled_word = None
            for word, index in opponent_word_dict.items():
                if sampled_word_index == index:
                    decoded_translation += ' {}'.format(word)
                    sampled_word = word

            if sampled_word == 'end' or len(decoded_translation.split()) > self.max_output_length:
                stop_condition = True

            empty_target_seq = np.zeros((1, 1))
            empty_target_seq[0, 0] = sampled_word_index
            states_values = [h, c]

        return decoded_translation[:-3]

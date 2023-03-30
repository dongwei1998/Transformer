# coding=utf-8
# =============================================
# @Time      : 2023-02-15 16:13
# @Author    : DongWei1998
# @FileName  : service.py
# @Software  : PyCharm
# =============================================
import os
import tensorflow as tf
from utils.token_tool import Tokenizers,standardize
from utils.transformer import Transformer
from utils.parameter import parser_opt


# 推理
class Translator(tf.Module):
    def __init__(self, args):

        self.tokenizers_a = Tokenizers(input_vocab=args.input_vocab,max_length=args.max_seq_length,standardize=standardize)
        self.tokenizers_b  = Tokenizers(input_vocab=args.target_vocab, max_length=args.max_seq_length, standardize=standardize)
        # # 获取词表大小
        args.input_vocab_size = self.tokenizers_a.tokenizers.vocabulary_size()
        args.target_vocab_size = self.tokenizers_b.tokenizers.vocabulary_size()
        # 创建模型
        transformer= Transformer(
            num_layers=args.num_layers,
            d_model=args.embedding_size,
            num_heads=args.num_heads,
            dff=args.feed_input_size,
            input_vocab_size=args.input_vocab_size,
            target_vocab_size=args.target_vocab_size,
            dropout_rate=args.dropout_rate)
        # 加载模型
        self.transformer = tf.keras.models.load_model(args.checkpoint_path, custom_objects={'transformer': transformer})



    def __call__(self, sentence):

        sentence = tf.strings.strip(sentence)
        sentence = tf.strings.join(['[START]', sentence, '[END]'], separator=' ')
        sentence = self.tokenizers_a.text_to_ids(sentence)
        sentence = sentence[tf.newaxis]

        encoder_input = sentence

        # As the output language is English, initialize the output with the
        start_end = self.tokenizers_b.text_to_ids(['[START] [END]'])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        # `tf.TensorArray` is required here (instead of a Python list), so that the
        # dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(args.max_seq_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer((encoder_input, output), training=False)
            # Select the last token from the `seq_len` dimension.
            predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

            predicted_id = tf.argmax(predictions, axis=-1)

            # Concatenate the `predicted_id` to the output which is given to the
            # decoder as its input.
            output_array = output_array.write(i + 1, predicted_id[0])
            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())
        # # The output shape is `(1, tokens)`.

        text = self.tokenizers_b.ids_to_text(output[:, 1:])  # Shape: `()`.
        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop.
        # So, recalculate them outside the loop.
        self.transformer((encoder_input, output[:, :-1]), training=False)
        attention_weights = self.transformer.decoder.last_attn_scores


        return text, output, attention_weights



def print_translation(sentence, tokens, ground_truth):
  print(f'{"Input:":15s}: {sentence}')
  print(f'{"Prediction":15s}: {tokens[0]}')
  print(f'{"Ground truth":15s}: {ground_truth}')



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 指定第一块GPU可用
    # 参数加载
    args = parser_opt()
    # 推理模型加载
    model = Translator(args)
    sentence = 'este é um problema que temos que resolver.'
    ground_truth = 'this is a problem we have to solve .'
    # 模型推理
    while True:
        input_str = input('Question :'.encode('utf-8').decode('utf-8'))
        if input_str == 'exit':
            exit()
        text, tokens, attention_weights = model(sentence)
        # 输出打印
        print_translation(sentence, text, ground_truth)

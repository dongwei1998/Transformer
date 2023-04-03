# coding=utf-8
# =============================================
# @Time      : 2023-02-15 16:13
# @Author    : DongWei1998
# @FileName  : service.py
# @Software  : PyCharm
# =============================================
import os
import tensorflow as tf
from utils.parameter import parser_opt
from flask import jsonify, request, Flask
import jieba
# 自适应学习率
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embedding_size, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.embedding_size = tf.cast(embedding_size, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.embedding_size) * tf.math.minimum(arg1, arg2)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    args = parser_opt()
    # 推理模型加载
    reloaded = tf.saved_model.load('./model_2')
    # 目标转换工具加载
    # target_transformer = reloaded.signatures['serving_default']
    # todo 待添加id_to_text 工具

    app = Flask(__name__)
    app.config['JSON_AS_ASCII'] = False
    @app.route('/api/v1/translator', methods=['POST'])
    def predict():
        try:
            # 参数获取
            infos = request.get_json()
            data_dict = {
                'text': ''
            }
            for k, v in infos.items():
                data_dict[k] = v

            text = data_dict['text'].replace('\n', '').replace('\r', '')
            # 参数检查
            if text is None:
                return jsonify({
                    'code': 500,
                    'msg': '请给定参数text！！！'
                })
            else:
                text = ' '.join(word for word in text.split(' '))
            # 直接调用预测的API
            three_input_text = tf.constant(text)
            result = reloaded(three_input_text)
            # text = 'este é o primeiro livro que eu fiz.'
            # result = reloaded(tf.constant(text))
            idx_to_text_list = result.numpy().tolist()[0]
            return jsonify({
                'code': 200,
                'msg': '成功',
                'source_text ': text,
                'translation_text':idx_to_text_list,
            })
        except Exception as e:
            # args.logger.error("异常信息为:{}".format(e))
            return jsonify({
                'code': 500,
                'msg': '预测数据失败!!!',
                'error':e
            })
    # 启动
    app.run(host='0.0.0.0',port=5557)
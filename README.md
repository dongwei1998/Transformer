# Transformer
使用python和tensorflow2.5实现的transformer变形金刚算法模型，可完成翻译任务。


# 训练数据构建
目前使用的 葡萄牙语->英语
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

# 模型训练
python train.py  可以自定义添加日志 打印的一些配置
python train.py  通过 fit训练

# 模型服务启动
python service.py



# 模型结果预测
python flasktest.py
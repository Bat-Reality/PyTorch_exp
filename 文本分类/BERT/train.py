import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import os
from sklearn.model_selection import train_test_split
import numpy as np
from BERT.utils import read_file, pre_processing, attention_masks, evaluate
from BERT.data_loader import get_loader
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# 各数据集的路径
path_prefix = './data'
data_path = os.path.join(path_prefix, 'simplifyweibo_4_moods.csv')
# 预训练模型的路径
pretrained_prefix = './data/pretrained'
tokenizer_path = os.path.join(pretrained_prefix, 'BertTokenizer')

model_dir = './data'
max_len = 62
batch_size = 16
epochs = 2
lr_rate = 2e-5
epsilon = 1e-8


def main():
    # load data
    data_x, data_y = read_file(data_path)

    # data pre_processing
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir=tokenizer_path)
    data_x = [pre_processing(tokenizer, x, max_len) for x in data_x]
    # data_x = torch.tensor(data_x)   # torch.Size([361744, 128])
    # 构建attention_mask
    attn_masks = attention_masks(data_x)

    # split data
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=5, shuffle=True)
    mask_train, mask_test, _, _ = train_test_split(attn_masks, data_y, test_size=0.3, random_state=5, shuffle=True)

    # preparing the training loader
    train_loader = get_loader(x_train, mask_train, y_train, train=True, batch_size=batch_size)
    print('Training loader prepared.')
    # preparing the validation loader
    val_loader = get_loader(x_test, mask_test, y_test, train=False, batch_size=batch_size)
    print('Validation loader prepared.')

    # load model
    model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=4)
    model = model.to(device)

    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=lr_rate, eps=epsilon)
    # 设计scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_acc = 0.
    # run epochs
    for epoch in range(epochs):
        # train for one epoch
        train(train_loader, model, optimizer, scheduler, epoch)

        # evaluate on validation set
        total_acc = validate(val_loader, model)

        if total_acc > best_acc:
            # 如果 validation 的结果好于之前所有的结果，就把当下的模型保存
            best_acc = total_acc
            # torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_batch*100))
            torch.save(model, "{}/ckpt.model".format(model_dir))
            print('saving model with acc {:.3f}'.format(total_acc))


def train(train_loader, model, optimizer, scheduler, epoch):
    # 將 model 的模式设定为 train，这样 optimizer 就可以更新 model 的参数
    model.train()

    train_len = len(train_loader)
    total_loss, total_acc = [], []

    for i, (inputs, masks, labels) in enumerate(train_loader):
        # 1. 放到GPU上
        inputs = inputs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        # 2. 清空梯度
        optimizer.zero_grad()
        # 3. 计算输出
        outputs = model(inputs, attention_mask=masks, labels=labels, token_type_ids=None)
        # 4. 计算损失
        loss, logits = outputs[0], outputs[1]
        total_loss.append(loss.item())
        # 5. 预测结果
        acc = evaluate(logits, labels)
        total_acc.append(acc)
        # 6. 反向传播
        loss.backward()
        # 7. 更新梯度
        optimizer.step()
        scheduler.step()
        if i % 100 == 0:
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.
                  format(epoch + 1, i + 1, train_len, loss.item(), acc * 100), end='\n')
    print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(np.mean(total_loss), np.mean(total_acc) * 100))


def validate(val_loader, model):
    model.eval()  # 將 model 的模式设定为 eval，固定model的参数

    val_len = len(val_loader) / batch_size

    with torch.no_grad():
        total_acc = []
        for i, (inputs, masks, labels) in enumerate(val_loader):
            # 1. 放到GPU上
            inputs = inputs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            # 2. 计算输出
            outputs = model(inputs, attention_mask=masks, token_type_ids=None)
            # 3. 预测结果
            acc = evaluate(outputs[0], labels)
            total_acc.append(acc)
        print("Valid | Acc: {:.3f} ".format(np.mean(total_acc) * 100))
    print('-----------------------------------------------\n')

    return np.mean(total_acc) * 100


if __name__ == '__main__':
    main()

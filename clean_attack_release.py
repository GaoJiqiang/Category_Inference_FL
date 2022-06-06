import logger
import datetime
from tqdm import tqdm
import random
import sys
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import net
import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
utils function
'''
def load_data(params, n_workers, train_data):
    all_range = list(range(len(train_data)))
    random.shuffle(all_range)
    n_workers_dataloader = []
    workers_cluster_label = {}

    if params['equal_divide']:
        data_len = int(len(all_range) / n_workers)
        for worker in range(n_workers):
            sub_indices = all_range[worker * data_len:(worker + 1) * data_len]
            per_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'],
                                                     sampler=torch.utils.data.sampler.SubsetRandomSampler(sub_indices))
            n_workers_dataloader.append(per_loader)
    else:
        mnist_classes = {}
        for ind, x in enumerate(train_data):
            _, label = x
            if label in mnist_classes:
                mnist_classes[label].append(ind)
            else:
                mnist_classes[label] = [ind]
        # label_num = params['n_class_client']
        # per_label_datalen = int((len(train_data) / params['n_workers']) / label_num)
        for i in range(params['n_workers']):
            # if i!=999:
            #     label_num = random.randint(params_server['n_class_perloader'][0], params_server['n_class_perloader'][1])
            # else:
            #     label_num = 1
            label_num = random.randint(params_server['n_class_perloader'][0], params_server['n_class_perloader'][1])
            per_label_datalen = int((len(train_data) / params['n_workers']) / label_num)
            # per_label_datalen =64
            random_n_label = random.sample(range(10), label_num)  # return list
            sub_indices = []
            for j in random_n_label:
                sub_indices += random.sample(mnist_classes[j], per_label_datalen)
            per_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'],
                                                     sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                         sub_indices))
            n_workers_dataloader.append(per_loader)
            workers_cluster_label[i] = random_n_label

    return workers_cluster_label, n_workers_dataloader
def load_logger_data(params, train_data):
    mnist_classes = {}
    for ind, x in enumerate(train_data):
        _, label = x
        if label in mnist_classes:
            mnist_classes[label].append(ind)
        else:
            mnist_classes[label] = [ind]
    label_num = 10
    per_label_datalen = 6
    sub_indices = []
    for j in range(label_num):
        sub_indices += random.sample(mnist_classes[j], per_label_datalen)
    adv_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'],
                                             sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                 sub_indices))
    label = list(range(10))
    return label, adv_loader
def get_epoch_data(all_dataloader, selected_workers):
    epoch_train_data = []
    for i in selected_workers:
        epoch_train_data.append(all_dataloader[i])
    return epoch_train_data
def global_last_round(global_net, global_weight_update):
    for name, data in global_net.state_dict().items():
        data.sub_(global_weight_update[name])
    return global_net
def get_multi_update(global_weight_update, local_weight_update, params):
    shrink = 1.0 / params['each_epoch_workers']

    if not bool(global_weight_update):
        global_weight_update = local_weight_update
        for key in local_weight_update:
            global_weight_update[key] = shrink * global_weight_update[key]  #
    else:
        for key in local_weight_update:
            global_weight_update[key] += shrink * local_weight_update[key]  #
    return global_weight_update
def update_global_model(global_net, global_weight_update):
    for name, data in global_net.state_dict().items():
        data.add_(global_weight_update[name])
    return global_net
def local_training(global_net, local_data, params):
    local_net = net.simpleNet(28 * 28, 300, 100, 10)

    # local_net = net.Activation_Net(28 * 28, 300, 100, 10)
    # local_net = net.Batch_Net(28 * 28, 300, 100, 10)
    # local_net = net.CifarNet()
    # local_net=net.ResNet18()

    local_net = local_net.to(device)
    for name, param in global_net.state_dict().items():
        local_net.state_dict()[name].copy_(param.clone())

    local_optimizer = optim.SGD(local_net.parameters(), lr=params['learning_rate'])
    local_criterion = nn.CrossEntropyLoss()
    weight_update = {}

    local_net.train()
    for local_epoch in range(params['local_retrain']):
        total_loss = 0
        for batch_id, batch in enumerate(local_data):
            data, label = batch
            if dataset == 'mnist':
                data = data.view(data.size(0), -1)  # 64*28*28-->64*784*1
            data = data.to(device)
            label = label.to(device)
            output = local_net(data)
            loss = local_criterion(output, label)
            local_optimizer.zero_grad()
            loss.backward()
            local_optimizer.step()
            total_loss += loss.data
        # print('local_epoch: {}, loss: {:.4}'.format(local_epoch,total_loss))
    for name, data in local_net.state_dict().items():
        weight_update[name] = data - global_net.state_dict()[name]  # record model updates

    return weight_update
def test(global_net, test_data, global_epoch):
    global_net.eval()
    eval_loss = 0
    eval_acc = 0
    global_criterion = nn.CrossEntropyLoss()
    for data in test_data:
        img, label = data
        if dataset == 'mnist':
            img = img.view(img.size(0), -1)
        img = img.to(device)
        label = label.to(device)
        out = global_net(img)
        loss = global_criterion(out, label)
        eval_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('global_epoch: {}, Test Loss: {:.6f}, Acc: {:.6f}'.format(
        global_epoch,
        eval_loss / (len(test_dataset)),
        eval_acc / (len(test_dataset))
    ))
    return eval_loss / len(test_dataset), eval_acc / len(test_dataset)
'''
server function
'''
def load_server_data(test_data, loader_num):
    #####hby#####The server constructs the auxiliary  data set##################
    aux_data, unknown = torch.utils.data.random_split(test_data, [params_server['propotion'], len(test_data)-params_server['propotion']])  # 500 for auxiliary data, 9500 for test data.
    cover_class = params_server['n_class_perloader'][1] # number of categories contained in auxiliary data

    print('size of auxdata is：', len(aux_data))
    print('auxdata covers class:', cover_class)

    n_workers_dataloader = []
    workers_cluster_label = []
    mnist_classes = {}
    per_loader_len = int(len(train_dataset) / params_fl['n_workers'])
    # Block data by category
    for ind, x in enumerate(aux_data):
        _, label = x
        if label in mnist_classes:
            mnist_classes[label].append(ind)
        else:
            mnist_classes[label] = [ind]
    # label_num = params_fl['n_class_client']
    # per_label_datalen=int(per_loader_len / label_num)
    for i in range(loader_num):
        label_num = random.randint(params_server['n_class_perloader'][0], params_server['n_class_perloader'][1])
        per_label_datalen = int(per_loader_len / label_num)
        # per_label_datalen=64
        random_n_label = random.sample(range(cover_class), label_num)  # return list
        sub_indices = []
        for j in random_n_label:
            if per_label_datalen > len(mnist_classes[j]):
                for i in range(per_label_datalen):
                    sub_indices.append(mnist_classes[j][0])
            else:
                sub_indices += random.sample(mnist_classes[j], per_label_datalen)  # 随机选x张
        per_loader = torch.utils.data.DataLoader(aux_data, batch_size=params_fl['batch_size'],
                                                 sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                     sub_indices))
        n_workers_dataloader.append(per_loader)
        workers_cluster_label.append(random_n_label)
    return n_workers_dataloader, workers_cluster_label
def server_training(global_net, class_data):
    server_net = net.simpleNet(28 * 28, 300, 100, 10)
    server_net = server_net.to(device)
    for name, param in global_net.state_dict().items():
        server_net.state_dict()[name].copy_(param.clone())
    server_optimizer = optim.SGD(server_net.parameters(), lr=params_fl['learning_rate'])
    server_criterion = nn.CrossEntropyLoss()
    weight_update = {}
    server_net.train()
    for server_epoch in range(params_server['server_retrain']):
        total_loss = 0
        for batch_id, batch in enumerate(class_data):
            data, label = batch
            if dataset == 'mnist':
                data = data.view(data.size(0), -1)  # 64*28*28--->64*784*1
            data = data.to(device)
            label = label.to(device)
            output = server_net(data)
            loss = server_criterion(output, label)
            server_optimizer.zero_grad()
            loss.backward()
            server_optimizer.step()
            total_loss += loss.data
        # print('local_epoch: {}, loss: {:.4}'.format(local_epoch,total_loss))
    for name, data in server_net.state_dict().items():
        weight_update[name] = data - global_net.state_dict()[name]  # 记录参数更新
    return weight_update
def server_inference(test_data, global_net):
    # Some data were sampled from the test dataset as auxiliary data
    server_data_loader, record_label = load_server_data(test_data, params_server['loader_num'])
    train_label = []
    train_data = []
    # attacker simulates client training and collects model updates.
    print('inference model is training.')
    for i in tqdm(range(len(server_data_loader))):
        weightUpdate = server_training(global_net, server_data_loader[i])
        train_label.append(record_label[i])
        train_data.append(weightUpdate['layer3.weight'].flatten().cpu().numpy().reshape(-1).tolist())
    '''
    prepare data form for training 
    '''
    train_label = MultiLabelBinarizer().fit_transform(train_label)  # onehot multi-label
    a = np.zeros(shape=(len(train_label), 10 - params_server['n_class_perloader'][1]))
    train_label = np.hstack((train_label, a))  # padding
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.2, random_state=42)
    '''
    multi-label training 
    '''
    # neigh = KNeighborsClassifier(n_neighbors=10)
    # clf = DecisionTreeClassifier(random_state=0)
    clf = RandomForestClassifier(oob_score=True)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    '''
    test inference model
    '''
    precision_list = []
    recall_list = []
    for i in range(len(y_predict)):
        precision = precision_score(y_test[i], y_predict[i])
        recall = recall_score(y_test[i], y_predict[i])
        precision_list.append(precision)
        recall_list.append(recall)
    final_precision = sum(precision_list) / (len(precision_list))
    final_recall = sum(recall_list) / (len(recall_list))
    final_f1 = (2 * final_precision * final_recall) / (final_precision + final_recall)
    print('inference model accuracy score', accuracy_score(y_test, y_predict))
    print('inference model precision score', final_precision)
    print('inference model recall score', final_recall)
    print('inference model f1 score', final_f1)
    # print('dai wai score', clf.oob_score_)
    return clf
'''
metrics function
'''
def find_same(predict_result, truth_result):
    y_predict_result = predict_result[0].tolist()
    true_label = [0] * 10
    for i in truth_result:
        true_label[i] = 1
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    # 比较
    for i in range(10):
        if true_label[i] == 1 and y_predict_result[i] == 1:
            TP += 1
        if true_label[i] == 0 and y_predict_result[i] == 1:
            FP += 1
        if true_label[i] == 1 and y_predict_result[i] == 0:
            FN += 1
        if true_label[i] == 0 and y_predict_result[i] == 0:
            TN += 1
    if TP + FP == 0:
        precision = 0
    else:
        precision = float(TP / (TP + FP))  #
    if TP + FN == 0:
        recall = 0
    else:
        recall = float(TP / (TP + FN))  #
    hamming_loss = (FP + FN) / 10
    return precision, recall, hamming_loss
def statistic(type, p_list, r_list, h_list):
    av_precision = sum(p_list) / len(p_list)
    av_recall = sum(r_list) / len(r_list)
    F1 = (2 * av_precision * av_recall) / (av_recall + av_precision)
    hamming_loss = sum(h_list) / len(h_list)
    print(type)
    print('average_precision:{} average_recall:{} F1:{} average_hamming_loss:{}'.format(av_precision, av_recall, F1,
                                                                                        hamming_loss))

'''
main function
'''
def federated_learning_secure(params, train_data, test_data):  # simulate the training process of SecAgg
    workers_label, n_workers_dataloader = load_data(params, params['n_workers'],
                                                    train_data)  # assign data to each client.
    adv_label, adv_loader = load_logger_data(params, train_data)  # noise-logger data
    global_net = net.simpleNet(28 * 28, 300, 100, 10) # new model
    global_net = global_net.to(device)
    '''
    some metrics lists for recording
    '''
    noise_precision_list = []
    noise_recall_list = []
    noise_hamming_list = []
    denoise_precision_list = []
    denoise_recall_list = []
    denoise_hamming_list = []
    true_precision_list = []
    true_recall_list = []
    true_hamming_list = []
    loss_list = []
    acc_list = []
    '''
    FL training
    '''
    for i in range(params['epoch']):
        global_weight_update = {}
        cunrrent_workers_label = []
        if i < params['warm_up']:  # warm up training. attacker can choose attack timing.
            '''
            A subset of users were randomly selected for this round of training
            '''
            epoch_workers = random.sample(range(params['n_workers']), params['each_epoch_workers'])
            epoch_working_data = get_epoch_data(n_workers_dataloader, epoch_workers)
            for id, worker_data in enumerate(epoch_working_data):
                local_weight_update = local_training(global_net, worker_data, params)  # local update
                global_weight_update = get_multi_update(global_weight_update, local_weight_update,
                                                        params)  # server collect client local update
            global_net = update_global_model(global_net, global_weight_update)  # update global model
        else:
            if i == params['warm_up']:
                clf = server_inference(test_dataset, global_net)  # train inference model
                print('inference model is trained')
            '''
            start  inference:
            1. the first round training, select random a subset of clients for training.
            2. the second  round training, replaces the last client in the subset.
            '''
            if i % 2 == 0:
                adv_weight_update1 = local_training(global_net, adv_loader, params)  # noise-logger
                epoch_workers = random.sample(range(params['n_workers'] - 1), params['each_epoch_workers'])
                record_worker = epoch_workers
                epoch_working_data = get_epoch_data(n_workers_dataloader, epoch_workers)
                for id, worker_data in enumerate(epoch_working_data):
                    local_weight_update = local_training(global_net, worker_data, params)
                    if id == len(epoch_working_data) - 1:
                        inference_true_weight = local_weight_update['layer3.weight']
                    global_weight_update = get_multi_update(global_weight_update, local_weight_update,
                                                            params)
                global_net = update_global_model(global_net, global_weight_update)
                record_weight1 = global_weight_update
            else:
                adv_weight_update2 = local_training(global_net, adv_loader, params)  # noise-logger
                epoch_workers = record_worker
                victim_label = workers_label[epoch_workers[-1]]
                epoch_workers[-1] = int(params['backadv_num'])  # Replace the last client with the collusion client.
                epoch_working_data = get_epoch_data(n_workers_dataloader, epoch_workers)
                for id, worker_data in enumerate(epoch_working_data):
                    local_weight_update = local_training(global_net, worker_data, params)
                    if id == len(epoch_working_data) - 1:
                        record_backadv = local_weight_update['layer3.weight']
                    global_weight_update = get_multi_update(global_weight_update, local_weight_update,
                                                            params)
                # global_net = update_global_model(global_net, global_weight_update)
                global_net = global_last_round(global_net, record_weight1)  # Roll back the model and repeat the attack test
                record_weight2 = global_weight_update

                '''
                inference  target model update and denoise
                '''
                noise_inference_weight = (record_weight1['layer3.weight'] - record_weight2['layer3.weight']) * int(
                    params['each_epoch_workers']) + record_backadv
                approx_noise = (adv_weight_update1['layer3.weight'] - adv_weight_update2['layer3.weight']) * int(
                    params['each_epoch_workers'] - 1)
                denoise_inference_weight = noise_inference_weight - approx_noise
                noise = noise_inference_weight - inference_true_weight

                noise_inference_weight_list = noise_inference_weight.flatten().cpu().numpy().reshape(1, -1).tolist()
                denoise_inference_weight_list = denoise_inference_weight.flatten().cpu().numpy().reshape(1, -1).tolist()
                true_weight_list = inference_true_weight.flatten().cpu().numpy().reshape(1, -1).tolist()
                noise = noise.flatten().cpu().numpy().reshape(1, -1).tolist()

                # inference prediction
                noise_predict_inferWeightResult = clf.predict(noise_inference_weight_list)
                noise_precision, noise_recall, noise_hamming = find_same(noise_predict_inferWeightResult, victim_label)
                denoise_predict_inferWeightResult = clf.predict(denoise_inference_weight_list)
                denoise_precision, denoise_recall, denoise_hamming = find_same(denoise_predict_inferWeightResult,
                                                                               victim_label)
                true_predict_inferWeightResult = clf.predict(true_weight_list)
                true_precision, true_recall, true_hamming = find_same(true_predict_inferWeightResult, victim_label)

                noise_precision_list.append(noise_precision)
                noise_recall_list.append(noise_recall)
                noise_hamming_list.append(noise_hamming)
                denoise_precision_list.append(denoise_precision)
                denoise_recall_list.append(denoise_recall)
                denoise_hamming_list.append(denoise_hamming)
                true_precision_list.append(true_precision)
                true_recall_list.append(true_recall)
                true_hamming_list.append(true_hamming)
                # precision, recall, hamming = random_guess(victim_label)  # random guess vs our work
                '''print(
                    'victim_predict_InferWeight:{} precesion:{} recall:{}'.format(predict_inferWeightResult, precision,
                                                                                  recall))
                print('victim_predict_TrueWeight:{}'.format(clf.predict(inference_true_weight_list)))
                print('victim_true_label:{}'.format(victim_label))'''

        '''print('current_epoch:{},current_workers:{}'.format(i, epoch_workers))
        for x in epoch_workers:
            cunrrent_workers_label.append(workers_label[x])
        print('current_workers_label:{}'.format(cunrrent_workers_label))'''

        # model accuracy
        if (i + 1) % 20 == 0:
            loss, acc = test(global_net, test_data, i)
            loss_list.append(loss)
            acc_list.append(acc)
    print('best acc', np.max(acc_list))
    ####################################################################### attack results
    statistic('noise', noise_precision_list, noise_recall_list, noise_hamming_list)
    statistic('denoise', denoise_precision_list, denoise_recall_list, denoise_hamming_list)
    statistic('true', true_precision_list, true_recall_list, true_hamming_list)
    print('attack times:', len(noise_precision_list))



if __name__ == "__main__":
    FL_setting = True  # choose FL
    dataset = 'mnist'
    if not os.path.exists("./command_output"):
        os.mkdir("./command_output")
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    sys.stdout = logger.Logger(f'./command_output/{current_time}_FL_{FL_setting}.txt')

    # hyper-parameters
    params_fl = {  # FL
        'n_workers': 1000,  # total clients
        'batch_size': 64,
        'epoch': 300,
        'each_epoch_workers': 10,
        'local_retrain': 2,
        'learning_rate': 0.01,  # mnist 0.01 cifar 0.001
        'equal_divide': False, # determine how data is divided
        'warm_up': 40,  # warm up training.
        'n_class_client': 1,  # assign [n_class_client] kinds of data to each client.
        'backadv_num': 999, # collusion client ID
    }
    params_server = {
        'propotion': 500,  # auxiliary data size
        'server_data_num': 5000, # repeated sampling from auxiliary data
        'server_retrain': 2,
        'n_class_perloader': [1, 10], # determine the number of categories in each batch of the auxiliary data
        'loader_num': 5000,  # the number of auxiliary data pieces
        'batch_size': 64,
        'learning_rate': 0.01,
    }
    ###################################################################################################################  mnist
    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    ###################################################################################################################
    federated_learning_secure(params_fl, train_dataset, test_loader)
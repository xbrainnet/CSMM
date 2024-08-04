from subjectAlignedModels.SANN_eye import SANN_eye, weight_init
from source_select import Source_Select

import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from mydata_loader import Mydataset
import numpy as np
import scipy.io as scio
from metric_IV import cal_metrics
from pandas import DataFrame


def evaluate(model, data, label):
    cuda = True
    cudnn.benchmark = True
    alpha = 0
    """ test """
    if cuda:
        model = model.cuda()
        data = data.cuda()
        label = label.cuda()
    n_total = len(data)
    data = data.float()
    label = label.long()
    class_output, _ = model(input_data=data, alpha=alpha)
    pred = torch.max(class_output, 1)[1]
    n_correct = torch.sum(pred == label)
    accu = n_correct / n_total
    return accu, pred, label


def model_forward(model, data):
    cuda = True
    cudnn.benchmark = True
    alpha = 0
    data = torch.from_numpy(data)

    """ test """
    if cuda:
        model = model.cuda()
        data = data.cuda()

    data = data.float()
    shallow_output = model.shallow(data)
    deep_output = model.deep.de_fc1(shallow_output)

    return deep_output


def train_and_evaluate(model, subjectIndex, source_data, source_label, target_data, target_label, domain_size):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # initialize
    lr = 0.002
    batch_size = 64
    batch_size_t = 64
    n_epoch = 2000
    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    for p in model.parameters():
        p.requires_grad = True
    # training
    model = model.to(device)
    best_accu_s = 0.0
    best_accu_t = 0.0
    for epoch in range(n_epoch):
        Source = source_data
        Source_Label = source_label
        for domain_id in range(int(len(source_data) / domain_size)):
            # select domain
            sel_source_id = Source_Select(model, Source, target_data, domain_size)
            Sour_data = Source[sel_source_id * domain_size:(sel_source_id + 1) * domain_size]
            Sour_label = Source_Label[sel_source_id * domain_size:(sel_source_id + 1) * domain_size]
            # make dataset
            source_dataset = Mydataset(Sour_data, Sour_label)
            dataloader_source = torch.utils.data.DataLoader(
                dataset=source_dataset,
                batch_size=batch_size,
                shuffle=True)
            target_dataset = Mydataset(target_data, target_label)
            dataloader_target = torch.utils.data.DataLoader(
                dataset=target_dataset,
                batch_size=batch_size_t,
                shuffle=True)
            len_dataloader = min(len(dataloader_source), len(dataloader_target))
            data_source_iter = iter(dataloader_source)
            data_target_iter = iter(dataloader_target)

            for i in range(len_dataloader):
                p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                s_data, s_label = data_source_iter.next()
                s_data = s_data.float()
                s_label = s_label.long()
                batch_size = len(s_label)
                domain_label_s = torch.zeros(batch_size).long()
                model.zero_grad()
                s_data = s_data.to(device)
                s_label = s_label.to(device)
                class_output, _ = model(input_data=s_data, alpha=alpha)
                err_s_label = loss_class(class_output, s_label)

                # training model using target and source data
                data_target = data_target_iter.next()
                t_data, _ = data_target
                t_data = t_data.float()
                t_data = t_data.to(device)
                batch_size = len(t_data)
                domain_label_t = torch.ones(batch_size).long()
                data_s_t = torch.cat([s_data, t_data], dim=0)
                label_s_t = torch.cat([domain_label_s, domain_label_t], dim=0)
                num_sample = data_s_t.shape[0]
                dims = data_s_t.shape[1]
                index = np.random.choice(np.arange(num_sample), size=num_sample, replace=False)
                data_s_t = data_s_t[index]
                label_s_t = label_s_t[index]
                data_s_t = data_s_t.to(device)
                label_s_t = label_s_t.to(device)
                _, domain_output = model(input_data=data_s_t, alpha=alpha)
                err_domain = loss_domain(domain_output, label_s_t)
                err = err_domain + err_s_label
                err.backward()
                optimizer.step()
                sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_domain: %f' \
                                 % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                                    err_domain.data.cpu().numpy()))
                sys.stdout.flush()
            Source = np.delete(Source, range(sel_source_id * domain_size, (sel_source_id + 1) * domain_size), 0)
            Source_Label = np.delete(Source_Label,
                                     range(sel_source_id * domain_size, (sel_source_id + 1) * domain_size), 0)
        torch.save(model, 'sdann_models_save_IV_eye/1/model_epoch_current.pth')
        my_net = torch.load('sdann_models_save_IV_eye/1/model_epoch_current.pth')
        my_net = my_net.eval()
        accu_s, _, _ = evaluate(my_net, source_data, source_label)
        accu_t, pred_t, label_t = evaluate(my_net, target_data, target_label)
        pred_t = pred_t.cpu().numpy()
        label_t = label_t.cpu().numpy()
        label_t = label_t.astype(np.int32)
        pred_t = pred_t.tolist()
        label_t = label_t.tolist()

        acc, macro_f1, auc, _, _, _, _, _, _, _, _ = cal_metrics(label_t, pred_t)
        print('\nAccuracy of the target dataset: %f macro_f1:%f auc:%f' % (acc, macro_f1, auc))
        if accu_t > best_accu_t:
            best_accu_s = accu_s
            best_accu_t = accu_t
            torch.save(my_net, 'sdann_models_save_IV_eye/1/model_epoch_best{}.pth'.format(subjectIndex))
    print('============ Summary ============= \n')
    print('Accuracy of the %s dataset: %f' % ('11 source domain', best_accu_s))
    print('Accuracy of the %s dataset: %f' % ('target', best_accu_t))
    print('Corresponding model was save in sdann_models_save_IV_eye/1/model_epoch_best{}.pth'.format(subjectIndex))
    return best_accu_s, best_accu_t



if __name__ == '__main__':
    datapath = r'datapath'
    filenameList = ['1_1.mat', '2_1.mat', '3_1.mat',
                    '4_1.mat', '5_1.mat', '6_1.mat', '7_1.mat', '8_1.mat',
                    '9_1.mat', '10_1.mat', '11_1.mat',
                    '12_1.mat', '13_1.mat', '14_1.mat', '15_1.mat'
                    ]
    ACC_list = []
    macro_F1_List = []
    AUC_list = []
    Precison0_List = []
    Precison1_List = []
    Precison2_List = []
    Precison3_List = []
    Recall0_List = []
    Recall1_List = []
    Recall2_List = []
    Recall3_List = []
    for subjectIndex in range(len(filenameList)):
        # load data
        filename = filenameList[subjectIndex]
        DATA = scio.loadmat(datapath + filename)
        X1 = DATA['S_eye']
        Y1 = DATA['S_label'].reshape(-1)
        X2 = DATA['T_eye']
        Labels = DATA['T_label'].reshape(-1)
        domain_size = X2.shape[0]
        randomindex1 = np.random.choice(np.arange(len(X1)), size=len(X1), replace=False)
        randomindex2 = np.random.choice(np.arange(len(X2)), size=len(X2), replace=False)
        X1 = torch.from_numpy(X1)
        X1 = X1[randomindex1]
        X1 = X1.float()

        X2 = X2[randomindex2]
        X2 = torch.from_numpy(X2)
        X2 = X2.float()

        Y1 = torch.from_numpy(Y1)
        Y1 = Y1[randomindex1]
        Y1 = Y1.long()

        Labels = Labels[randomindex2]
        Y2 = torch.from_numpy(Labels)
        Y2 = Y2.long()

        # create SANN
        model = SANN_eye()
        model.apply(weight_init)
        print('Subject id:', subjectIndex)
        _, _ = train_and_evaluate(model, subjectIndex, X1, Y1, X2, Y2, domain_size)
        print('_' * 50)

        model_name = 'sdann_models_save_IV_eye/1/model_epoch_best' + str(subjectIndex) + '.pth'
        my_net = torch.load(model_name)
        my_net = my_net.eval()
        acc_SDANN, pred, label = evaluate(my_net, X2, Y2)
        pred = pred.cpu().numpy()
        label = label.cpu().numpy()
        label = label.astype(np.int32)
        pred = pred.tolist()
        label = label.tolist()

        acc, macro_f1, auc, p0, r0, p1, r1, p2, r2, p3, r3 = cal_metrics(label, pred)
        ACC_list.append(acc)
        macro_F1_List.append(macro_f1)
        AUC_list.append(auc)
        Precison0_List.append(p0)
        Precison1_List.append(p1)
        Precison2_List.append(p2)
        Precison3_List.append(p3)
        Recall0_List.append(r0)
        Recall1_List.append(r1)
        Recall2_List.append(r2)
        Recall3_List.append(r3)
        print('Accuracy of the SDANN method:%f macro-f1:%f auc:%f' % (acc_SDANN, macro_f1, auc))
        print('-' * 50)

    data = {'acc': ACC_list, 'macro_F1': macro_F1_List, 'auc': AUC_list, 'Pre0': Precison0_List, 'Rec0': Recall0_List,
            'Pre1': Precison1_List, 'Rec1': Recall1_List, 'Pre2': Precison2_List, 'Rec2': Recall2_List,
            'Pre3': Precison3_List, 'Rec3': Recall3_List}
    df = DataFrame(data)
    df.to_excel('SEED_IV_result_eye/1/result1.xlsx')

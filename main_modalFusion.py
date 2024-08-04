import random
from subjectAlignedModels.SANN import SANN, weight_init
from subjectAlignedModels.SANN_eye import SANN_eye
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from modalFusionModels.contrastiveModel_IV import EFCL_seed_final
import argparse
import ruamel.yaml as yaml
from optim import create_optimizer
import utils
import sys
from sklearn import preprocessing
import scipy.io as scio
from metric_IV import cal_metrics
from pandas import DataFrame


class Dataset_seed(Dataset):
    def __init__(self, data):
        self.datas = data

    def __getitem__(self, index):
        return self.datas[index]

    def __len__(self):
        return len(self.datas)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def preprocess_dataset(DATA):
    Train_datas = []
    Test_datas = []
    train1 = DATA['S_eeg']
    train2 = DATA['S_eye']
    test1 = DATA['T_eeg']
    test2 = DATA['T_eye']
    train_label = DATA['S_label']
    test_label = DATA['T_label']

    min_max_scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    data1 = np.concatenate((train1, test1), axis=0)
    scaler_eeg = min_max_scaler1.fit(data1)
    train_data_eeg = scaler_eeg.transform(train1)
    test_data_eeg = scaler_eeg.transform(test1)
    min_max_scaler2 = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    data2 = np.concatenate((train2, test2), axis=0)
    scaler_eye = min_max_scaler2.fit(data2)
    train_data_eye = scaler_eye.transform(train2)
    test_data_eye = scaler_eye.transform(test2)

    train_size = train_data_eeg.shape[0]
    test_size = test_data_eeg.shape[0]
    # Session1:
    trail_boundary = [0, 42, 65, 114, 146, 168, 208, 246, 298, 334, 376, 388, 415, 469, 533, 568, 612, 624, 667, 701]
    # Session2:
    # trail_boundary=[0,55,80,114,150,203,230,264,310,344,364,424,436,472,516,565,610,647,691,710]
    # Session3:
    # trail_boundary=[0,42,74,97,142,190,216,280,303,329,345,396,437,476,495,539,553,598,620,659]

    id_sample = 0
    for i in range(train_size):
        example = {}
        example_x = torch.from_numpy(train_data_eeg[i, :])
        example_x = example_x.float()
        example_x = torch.unsqueeze(example_x, 1)

        example_y = torch.from_numpy(train_data_eye[i, :])
        example_y = example_y.float()
        example_y = torch.unsqueeze(example_y, 1)

        example['eeg'] = example_x
        example['eye'] = example_y
        trail_id = 0
        for item_id, item_j in enumerate(trail_boundary):
            if id_sample < item_j:
                trail_id = item_id
                break

        example['idx'] = trail_id
        example['label'] = train_label[0][i]
        id_sample = id_sample + 1
        Train_datas.append(example)

    for i in range(test_size):
        example = {}
        example_x = torch.from_numpy(test_data_eeg[i, :])
        example_x = example_x.float()
        example_y = torch.from_numpy(test_data_eye[i, :])
        example_y = example_y.float()
        example['eeg'] = example_x
        example['eye'] = example_y
        example['idx'] = -1
        example['label'] = test_label[0][i]
        Test_datas.append(example)

    return Train_datas, Test_datas, scaler_eeg, scaler_eye


def Train(model, data_loader, optimizer, epoch, device, config):
    model.train()
    running_correct = 0.0
    for i, data in enumerate(data_loader):
        eeg = data['eeg']
        eye = data['eye']
        idx = data['idx']
        labels = data['label']
        eeg = eeg.to(device, non_blocking=True)
        eye = eye.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss_contrast, loss_match, loss_class, pred = model(eeg, eye, alpha=alpha, idx=idx, labels=labels)
        _, pred_labels = torch.max(pred, 1)
        pred_labels = pred_labels.to(device, non_blocking=True)
        running_correct += torch.sum(pred_labels == labels)
        loss = loss_contrast * 0.9 + loss_match + loss_class
        sys.stdout.flush()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return running_correct


def Test(model, TESTDATA, device, eeg_model, eye_model, scale_eeg, scale_eye):
    eeg_data = TESTDATA['T_eeg']
    eeg_data = eeg_data.to(device)
    eye_data = TESTDATA['T_eye']
    eye_data = eye_data.to(device)
    real_label = TESTDATA['T_label']
    Real_labels = np.squeeze(real_label)
    Real_labels = torch.from_numpy(Real_labels)

    feature = eeg_model.shallow.s_bn1(eeg_model.shallow.s_fc1(eeg_data))
    feature = eeg_model.shallow.s_drop1(eeg_model.shallow.s_relu1(feature))
    feature = eeg_model.shallow.s_fc2(feature)
    T_eeg_trans = feature.cpu().detach().numpy()
    T_eeg_trans = T_eeg_trans.astype(np.float64)
    T_eeg_aligned = scale_eeg.transform(T_eeg_trans)

    feature = eye_model.shallow.s_bn1(eye_model.shallow.s_fc1(eye_data))
    feature = eye_model.shallow.s_drop1(eye_model.shallow.s_relu1(feature))
    feature = eye_model.shallow.s_fc2(feature)
    T_eye_trans = feature.cpu().detach().numpy()
    T_eye_trans = T_eye_trans.astype(np.float64)
    T_eye_aligned = scale_eye.transform(T_eye_trans)
    T_eeg_aligned = torch.from_numpy(T_eeg_aligned).float().to(device)
    T_eye_aligned = torch.from_numpy(T_eye_aligned).float().to(device)

    model.eval()
    running_correct = 0.0
    loss_class, predict, pre_before_softmax = model.model_testforward(T_eeg_aligned, T_eye_aligned, labels=Real_labels)
    _, fusion_pred = torch.max(predict, 1)
    Real_labels = Real_labels.to(device)
    fusion_correct = torch.sum(fusion_pred == Real_labels)

    eeg_output, _ = eeg_model(input_data=eeg_data, alpha=0)
    eeg_pred = torch.max(eeg_output, 1)[1]
    eeg_correct = torch.sum(eeg_pred == Real_labels)

    eye_output = eye_model.shallow(eye_data)
    eye_output = eye_model.deep(eye_output)
    eye_output = eye_model.class_classifier(eye_output)
    _, eye_pred = torch.max(eye_output, 1)
    eye_correct = torch.sum(eye_pred == Real_labels)

    ismm_predict = predict
    ismm_predict_normed = F.normalize(ismm_predict, p=1, dim=1)
    eeg_predict_normed = F.normalize(eeg_output, p=1, dim=1)
    eye_predict_normed = F.normalize(eye_output, p=1, dim=1)
    All_predict = ismm_predict_normed + eeg_predict_normed + eye_predict_normed
    _, All_pred_label = torch.max(All_predict, 1)
    All_correct = torch.sum(All_pred_label == Real_labels)

    return Real_labels, fusion_correct, fusion_pred, eeg_correct, eeg_pred, eye_correct, eye_pred, All_correct, All_pred_label


def main_seed(args, config, file_dir, eeg_model_name, eye_model_name, subjectIndex):
    device = torch.device(args.device)
    #  Align DATA
    DATA = scio.loadmat(file_dir)
    S_eeg = DATA['S_eeg']
    S_eeg = torch.from_numpy(S_eeg)
    S_eeg = S_eeg.float()
    S_eye = DATA['S_eye']
    S_eye = torch.from_numpy(S_eye)
    S_eye = S_eye.float()
    S_Label = DATA['S_label']

    T_eeg = DATA['T_eeg']
    T_eeg = torch.from_numpy(T_eeg)
    T_eeg = T_eeg.float()
    T_eye = DATA['T_eye']
    T_eye = torch.from_numpy(T_eye)
    T_eye = T_eye.float()
    T_Label = DATA['T_label']
    PER_LEN = T_Label.shape[1]
    TESTDATA = {'T_eeg': T_eeg, 'T_eye': T_eye, 'T_label': T_Label}

    # load eeg model
    net_eeg = torch.load(eeg_model_name)
    net_eeg = net_eeg.eval()
    net_eeg.to('cpu')

    feature = net_eeg.shallow.s_bn1(net_eeg.shallow.s_fc1(S_eeg))
    feature = net_eeg.shallow.s_drop1(net_eeg.shallow.s_relu1(feature))
    feature = net_eeg.shallow.s_fc2(feature)
    S_eeg_trans = feature.detach().numpy()
    S_eeg_trans = S_eeg_trans.astype(np.float64)

    feature = net_eeg.shallow.s_bn1(net_eeg.shallow.s_fc1(T_eeg))
    feature = net_eeg.shallow.s_drop1(net_eeg.shallow.s_relu1(feature))
    feature = net_eeg.shallow.s_fc2(feature)
    T_eeg_trans = feature.detach().numpy()
    T_eeg_trans = T_eeg_trans.astype(np.float64)

    # load eye model
    net_eye = torch.load(eye_model_name)
    net_eye = net_eye.eval()
    net_eye.to('cpu')

    feature = net_eye.shallow.s_bn1(net_eye.shallow.s_fc1(S_eye))
    feature = net_eye.shallow.s_drop1(net_eye.shallow.s_relu1(feature))
    feature = net_eye.shallow.s_fc2(feature)
    S_eye_trans = feature.detach().numpy()
    S_eye_trans = S_eye_trans.astype(np.float64)

    feature = net_eye.shallow.s_bn1(net_eye.shallow.s_fc1(T_eye))
    feature = net_eye.shallow.s_drop1(net_eye.shallow.s_relu1(feature))
    feature = net_eye.shallow.s_fc2(feature)
    T_eye_trans = feature.detach().numpy()
    T_eye_trans = T_eye_trans.astype(np.float64)

    Data = {'S_eeg': S_eeg_trans, 'S_eye': S_eye_trans, 'S_label': S_Label, 'T_eeg': T_eeg_trans, 'T_eye': T_eye_trans,
            'T_label': T_Label}

    Train_datas, Test_datas, scaler_eeg, scaler_eye = preprocess_dataset(Data)
    train_dataset = Dataset_seed(Train_datas)
    trainData_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, drop_last=True)

    net_eeg.to('cuda:0')
    net_eye.to('cuda:0')
    ########Model#######
    print("-----------------Creating model--------------------")
    print(subjectIndex)
    model = EFCL_seed_final(config=config)
    model = model.to(device)
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    best_all_correct = 0.0
    best_eeg_correct = 0
    best_eye_correct = 0
    best_fusion_correct = 0
    best_Predict_all = []
    best_label = []
    best_Predict_eeg = []
    best_Predict_eye = []
    best_Predict_fusion = []
    for epoch in range(200):
        running_correct = Train(model, trainData_loader, optimizer, epoch, device, config)
        correct_pre = running_correct
        print("\033[0;32;40m Epoch:{},Train Accuracy is:{:.4f}%\033[0m".format(epoch,
                                                                               100 * correct_pre / len(train_dataset)))
        Real_labels, fusion_correct, fusion_pred, eeg_correct, eeg_pred, eye_correct, eye_pred, All_correct, All_pred_label = Test(
            model, TESTDATA, device, net_eeg, net_eye, scaler_eeg, scaler_eye)
        print("Test all Accuracy is:{:.4f}%".format(100 * All_correct / PER_LEN))
        print("Test fusion only Accuracy is:{:.4f}%".format(100 * fusion_correct / PER_LEN))
        if All_correct > best_all_correct:
            best_all_correct = All_correct
            best_Predict_all = All_pred_label
            best_label = Real_labels
            best_Predict_eeg = eeg_pred
            best_eeg_correct = eeg_correct
            best_Predict_eye = eye_pred
            best_eye_correct = eye_correct
            best_Predict_fusion = fusion_pred
            best_fusion_correct = fusion_correct
            torch.save(model, 'fusion_save_model_IV/1/model_epoch_best{}.pth'.format(subjectIndex))

    best_all_correct = best_all_correct.cpu().numpy()
    best_eeg_correct = best_eeg_correct.cpu().numpy()
    best_eye_correct = best_eye_correct.cpu().numpy()
    best_fusion_correct = best_fusion_correct.cpu().numpy()

    best_Predict_all = best_Predict_all.cpu().numpy()
    best_Predict_eeg = best_Predict_eeg.cpu().numpy()
    best_Predict_eye = best_Predict_eye.cpu().numpy()
    best_Predict_fusion = best_Predict_fusion.cpu().numpy()
    best_label = best_label.cpu().numpy()
    best_label = best_label.astype(int)
    best_label = best_label.tolist()

    print("-------------------Start testing-------------------")
    print("Final ALL Accuracy:{:.4f}".format(100 * best_all_correct / PER_LEN))
    print("only Fusion Accuracy:{:.4f}".format(100 * best_fusion_correct / PER_LEN))
    print("Only EEG Accuracy:{:.4f}".format(100 * best_eeg_correct / PER_LEN))
    print("Only EYE Accuracy:{:.4f}".format(100 * best_eye_correct / PER_LEN))

    return best_label, best_Predict_all, best_Predict_eeg, best_Predict_eye, best_Predict_fusion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/eeg_eye.yaml')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    data_path = r'datapath....'
    filenameList = ['1_1.mat', '2_1.mat', '3_1.mat',
                    '4_1.mat', '5_1.mat', '6_1.mat', '7_1.mat', '8_1.mat',
                    '9_1.mat', '10_1.mat', '11_1.mat',
                    '12_1.mat', '13_1.mat', '14_1.mat', '15_1.mat'
                    ]
    model_eeg_pth = './sdann_models_save_IV/1/model_epoch_best'
    model_eye_pth = './sdann_models_save_IV_eye/1/model_epoch_best'
    ACC_list_all = []
    macro_F1_List_all = []
    AUC_list_all = []
    Precison0_List_all = []
    Precison1_List_all = []
    Precison2_List_all = []
    Precison3_List_all = []
    Recall0_List_all = []
    Recall1_List_all = []
    Recall2_List_all = []
    Recall3_List_all = []

    ACC_list_EEG = []
    macro_F1_List_EEG = []
    AUC_list_EEG = []
    Precison0_List_EEG = []
    Precison1_List_EEG = []
    Precison2_List_EEG = []
    Precison3_List_EEG = []
    Recall0_List_EEG = []
    Recall1_List_EEG = []
    Recall2_List_EEG = []
    Recall3_List_EEG = []

    ACC_list_EYE = []
    macro_F1_List_EYE = []
    AUC_list_EYE = []
    Precison0_List_EYE = []
    Precison1_List_EYE = []
    Precison2_List_EYE = []
    Precison3_List_EYE = []
    Recall0_List_EYE = []
    Recall1_List_EYE = []
    Recall2_List_EYE = []
    Recall3_List_EYE = []

    ACC_list_fusion = []
    macro_F1_List_fusion = []
    AUC_list_fusion = []
    Precison0_List_fusion = []
    Precison1_List_fusion = []
    Precison2_List_fusion = []
    Precison3_List_fusion = []
    Recall0_List_fusion = []
    Recall1_List_fusion = []
    Recall2_List_fusion = []
    Recall3_List_fusion = []

    run_list = []
    for f_id in range(len(filenameList)):
        filename = filenameList[f_id]
        print(filename)
        run_list.append(filename)
        file_dir = data_path + filename
        eeg_model_name = model_eeg_pth + str(f_id) + '.pth'
        eye_model_name = model_eye_pth + str(f_id) + '.pth'

        best_label, best_Predict_all, best_Predict_eeg, best_Predict_eye, best_Predict_fusion = main_seed(args, config,
                                                                                                          file_dir,
                                                                                                          eeg_model_name,
                                                                                                          eye_model_name,
                                                                                                          f_id)

        acc_all, macro_f1_all, auc_all, p0_all, r0_all, p1_all, r1_all, p2_all, r2_all, p3_all, r3_all, = cal_metrics(
            best_label, best_Predict_all)
        ACC_list_all.append(acc_all)
        macro_F1_List_all.append(macro_f1_all)
        AUC_list_all.append(auc_all)
        Precison0_List_all.append(p0_all)
        Precison1_List_all.append(p1_all)
        Precison2_List_all.append(p2_all)
        Precison3_List_all.append(p3_all)
        Recall0_List_all.append(r0_all)
        Recall1_List_all.append(r1_all)
        Recall2_List_all.append(r2_all)
        Recall3_List_all.append(r3_all)

        acc_EEG, macro_f1_EEG, auc_EEG, p0_EEG, r0_EEG, p1_EEG, r1_EEG, p2_EEG, r2_EEG, p3_EEG, r3_EEG = cal_metrics(
            best_label, best_Predict_eeg)
        ACC_list_EEG.append(acc_EEG)
        macro_F1_List_EEG.append(macro_f1_EEG)
        AUC_list_EEG.append(auc_EEG)
        Precison0_List_EEG.append(p0_EEG)
        Precison1_List_EEG.append(p1_EEG)
        Precison2_List_EEG.append(p2_EEG)
        Precison3_List_EEG.append(p3_EEG)
        Recall0_List_EEG.append(r0_EEG)
        Recall1_List_EEG.append(r1_EEG)
        Recall2_List_EEG.append(r2_EEG)
        Recall3_List_EEG.append(r3_EEG)

        acc_EYE, macro_f1_EYE, auc_EYE, p0_EYE, r0_EYE, p1_EYE, r1_EYE, p2_EYE, r2_EYE, p3_EYE, r3_EYE = cal_metrics(
            best_label, best_Predict_eye)
        ACC_list_EYE.append(acc_EYE)
        macro_F1_List_EYE.append(macro_f1_EYE)
        AUC_list_EYE.append(auc_EYE)
        Precison0_List_EYE.append(p0_EYE)
        Precison1_List_EYE.append(p1_EYE)
        Precison2_List_EYE.append(p2_EYE)
        Precison3_List_EYE.append(p3_EYE)
        Recall0_List_EYE.append(r0_EYE)
        Recall1_List_EYE.append(r1_EYE)
        Recall2_List_EYE.append(r2_EYE)
        Recall3_List_EYE.append(r3_EYE)

        acc_fusion, macro_f1_fusion, auc_fusion, p0_fusion, r0_fusion, p1_fusion, r1_fusion, p2_fusion, r2_fusion, \
            p3_fusion, r3_fusion = cal_metrics(best_label, best_Predict_fusion)
        ACC_list_fusion.append(acc_fusion)
        macro_F1_List_fusion.append(macro_f1_fusion)
        AUC_list_fusion.append(auc_fusion)
        Precison0_List_fusion.append(p0_fusion)
        Precison1_List_fusion.append(p1_fusion)
        Precison2_List_fusion.append(p2_fusion)
        Precison3_List_fusion.append(p3_fusion)
        Recall0_List_fusion.append(r0_fusion)
        Recall1_List_fusion.append(r1_fusion)
        Recall2_List_fusion.append(r2_fusion)
        Recall3_List_fusion.append(r3_fusion)

    data_all = {'filename': run_list, 'acc_all': ACC_list_all, 'macro_F1_all': macro_F1_List_all,
                'auc_all': AUC_list_all, 'Pre0_all': Precison0_List_all, 'Rec0_all': Recall0_List_all,
                'Pre1_all': Precison1_List_all, 'Rec1_all': Recall1_List_all, 'Pre2_all': Precison2_List_all,
                'Rec2_all': Recall2_List_all,
                'Pre3_all': Precison3_List_all, 'Rec3_all': Recall3_List_all}
    df_all = DataFrame(data_all)
    df_all.to_excel('result_IV/result_all_session1.xlsx')

    data_eeg = {'filename': run_list, 'acc_EEG': ACC_list_EEG, 'macro_F1_EEG': macro_F1_List_EEG,
                'auc_EEG': AUC_list_EEG, 'Pre0_EEG': Precison0_List_EEG,
                'Rec0_EEG': Recall0_List_EEG, 'Pre1_EEG': Precison1_List_EEG, 'Rec1_EEG': Recall1_List_EEG,
                'Pre2_EEG': Precison2_List_EEG, 'Rec2_EEG': Recall2_List_EEG,
                'Pre3_EEG': Precison3_List_EEG, 'Rec3_EEG': Recall3_List_EEG}
    df_eeg = DataFrame(data_eeg)
    df_eeg.to_excel('result_IV/result_only_EEG_s1.xlsx')

    data_eye = {'filename': run_list, 'acc_EYE': ACC_list_EYE, 'macro_F1_EYE': macro_F1_List_EYE,
                'auc_EYE': AUC_list_EYE, 'Pre0_EYE': Precison0_List_EYE,
                'Rec0_EYE': Recall0_List_EYE,
                'Pre1_EYE': Precison1_List_EYE, 'Rec1_EYE': Recall1_List_EYE, 'Pre2_EYE': Precison2_List_EYE,
                'Rec2_EYE': Recall2_List_EYE, 'Pre3_EYE': Precison3_List_EYE,
                'Rec3_EYE': Recall3_List_EYE}
    df_eye = DataFrame(data_eye)
    df_eye.to_excel('result_IV/result_only_EYE_s1.xlsx')

    data_fusion = {'filename': run_list, 'acc_fusion': ACC_list_fusion, 'macro_F1_fusion': macro_F1_List_fusion,
                   'auc_fusion': AUC_list_fusion, 'Pre0_fusion': Precison0_List_fusion,
                   'Rec0_fusion': Recall0_List_fusion,
                   'Pre1_fusion': Precison1_List_fusion, 'Rec1_fusion': Recall1_List_fusion,
                   'Pre2_fusion': Precison2_List_fusion,
                   'Rec2_fusion': Recall2_List_fusion, 'Pre3_fusion': Precison3_List_fusion,
                   'Rec3_fusion': Recall3_List_fusion}
    df_fusion = DataFrame(data_fusion)
    df_fusion.to_excel('result_IV/result_only_fusion_s1.xlsx')

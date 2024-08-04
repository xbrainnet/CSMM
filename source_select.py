import numpy as np
import torch
import torch.nn.functional as F


def Source_Select(model, Source_data, Target_data, domain_size):
    num_domains = int(len(Source_data) / domain_size)
    min_neg_loss_domain = 0.0
    min_Source_id = -1
    device = torch.device("cuda")
    loss_domain = torch.nn.NLLLoss()
    for i in range(num_domains):
        source_tmp = Source_data[i * domain_size:(i + 1) * domain_size]
        source = source_tmp.float()
        target = Target_data.float()
        size_s = len(source)
        size_t = len(target)
        domain_label_source = torch.zeros(size_s)
        domain_label_target = torch.ones(size_t)

        data_source_target = torch.cat([source, target], dim=0)
        label_source_target = torch.cat([domain_label_source, domain_label_target], dim=0)
        num_s_t = data_source_target.shape[0]
        # dims = data_source_target.shape[1]
        index = np.random.choice(np.arange(num_s_t), size=num_s_t, replace=False)
        data_source_target = data_source_target[index]
        label_source_target = label_source_target[index]
        data_source_target = data_source_target.to(device)
        label_source_target = label_source_target.to(device)
        label_source_target = label_source_target.long()

        _, domain_output = model(input_data=data_source_target, alpha=0)
        err_domain = loss_domain(domain_output, label_source_target)
        err_domain = -err_domain

        if (i == 0):
            min_Source_id = i
            min_neg_loss_domain = err_domain
        else:
            if err_domain < min_neg_loss_domain:
                min_Source_id = i
                min_neg_loss_domain = err_domain

    return min_Source_id

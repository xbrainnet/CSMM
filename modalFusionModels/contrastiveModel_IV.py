import torch
from torch import nn
import torch.nn.functional as F
from modalFusionModels.attentionModule import EEG_SAM, EYE_SAM, CrossAttention, weight_init
from einops import rearrange


class EFCL_seed_final(nn.Module):  # EEG Eye Contrastive Learning
    def __init__(self,
                 config=None,
                 ):
        super().__init__()
        embed_dim = config['embed_dim']
        eeg_width = config['eeg_dim']
        eye_width = config['eye_dim']
        self.EEG_encoder = EEG_SAM()
        self.EEG_encoder.apply(weight_init)
        self.eye_encoder = EYE_SAM()
        self.EEG_encoder.apply(weight_init)

        self.fusionModule = CrossAttention()
        self.fusionModule.apply(weight_init)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.efm_head = nn.Linear(32, 2)
        self.classlinear1 = nn.Linear(32, 4)
        self.classsoftmax = nn.LogSoftmax(dim=1)
        self.activate = nn.ReLU()
        self.normlayer = nn.LayerNorm(32)

        # create momentum models
        self.EEG_encoder_m = EEG_SAM()
        self.eye_encoder_m = EYE_SAM()

        self.model_pairs = [[self.EEG_encoder, self.EEG_encoder_m],
                            [self.eye_encoder, self.eye_encoder_m]]

        self.copy_params()

        # create the queue
        self.register_buffer("eeg_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("eye_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), 0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.eeg_queue = nn.functional.normalize(self.eeg_queue, dim=0)
        self.eye_queue = nn.functional.normalize(self.eye_queue, dim=0)

    def forward(self, eeg, eye, alpha, idx, labels):
        eeg = torch.squeeze(eeg)
        eeg = torch.unsqueeze(eeg, 1)
        eeg_embeds = self.EEG_encoder(eeg)
        eeg_embeds = rearrange(eeg_embeds, 'b c d -> b (c d)')

        eye = torch.squeeze(eye)
        eye = torch.unsqueeze(eye, 1)
        eye_embeds = self.eye_encoder(eye)

        eye_embeds = rearrange(eye_embeds, 'b c d ->b (c d)')

        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)

        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        with torch.no_grad():
            self._momentum_update()
            eeg_embeds_m = self.EEG_encoder_m(eeg)
            eeg_embeds_m = rearrange(eeg_embeds_m, 'b c d -> b (c d)')
            eeg_feat_all = torch.cat([eeg_embeds_m.t(), self.eeg_queue.clone().detach()], dim=1)

            eye_embeds_m = self.eye_encoder_m(eye)
            eye_embeds_m = rearrange(eye_embeds_m, 'b c d ->b (c d)')
            eye_feat_all = torch.cat([eye_embeds_m.t(), self.eye_queue.clone().detach()], dim=1)

        sim_e2f = eeg_embeds @ eye_feat_all / self.temp
        sim_f2e = eye_embeds @ eeg_feat_all / self.temp

        loss_e2f = -torch.sum(F.log_softmax(sim_e2f, dim=1) * sim_targets, dim=1).mean()
        loss_f2e = -torch.sum(F.log_softmax(sim_f2e, dim=1) * sim_targets, dim=1).mean()
        loss_efc = (loss_e2f + loss_f2e) / 2
        self._dequeue_and_enqueue(eeg_embeds_m, eye_embeds_m, idx)

        # forward the positve image-text pair
        concat_embeds = torch.cat([eeg_embeds, eye_embeds], dim=1)
        concat_embeds = torch.unsqueeze(concat_embeds, dim=1)
        output_pos = self.fusionModule(concat_embeds)
        with torch.no_grad():
            bs = eeg.size(0)
            weights_e2f = F.softmax(sim_e2f[:, :bs] + 1e-4, dim=1)
            weights_f2e = F.softmax(sim_f2e[:, :bs] + 1e-4, dim=1)

        # select a negative eye for each eeg
        eye_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_e2f[b], 1).item()
            eye_embeds_neg.append(eye_embeds[neg_idx])
        eye_embeds_neg = torch.stack(eye_embeds_neg, dim=0)

        # select a negative eeg for each eye
        eeg_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_f2e[b], 1).item()
            eeg_embeds_neg.append(eeg_embeds[neg_idx])
        eeg_embeds_neg = torch.stack(eeg_embeds_neg, dim=0)

        eye_embeds_all = torch.cat([eye_embeds, eye_embeds_neg], dim=0)
        eeg_embeds_all = torch.cat([eeg_embeds_neg, eeg_embeds], dim=0)

        concat_embeds_all = torch.cat([eeg_embeds_all, eye_embeds_all], dim=1)
        concat_embeds_all = torch.unsqueeze(concat_embeds_all, dim=1)
        output_neg = self.fusionModule(concat_embeds_all)

        vl_embeddings = torch.cat([output_pos, output_neg], dim=0)
        vl_embeddings2 = rearrange(vl_embeddings, "b n h -> b (n h)")
        vl_output = self.efm_head(vl_embeddings2)

        efm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(eeg.device)
        loss_class = torch.nn.NLLLoss()

        loss_efm = F.cross_entropy(vl_output, efm_labels)

        ## loss_class
        embed_out = output_pos[:, :, :]
        embed_out2 = rearrange(embed_out, "b n h -> b (n h)")
        x = self.classlinear1(embed_out2)
        x = self.classsoftmax(x)

        labels = labels.to(eeg.device)
        labels = labels.to(torch.int64)
        loss_class = loss_class(x, labels)
        return loss_efc, loss_efm, loss_class, x

    def model_testforward(self, eeg, eye, labels):
        eeg = torch.squeeze(eeg)
        eeg = torch.unsqueeze(eeg, 1)
        eeg_embeds = self.EEG_encoder(eeg)

        eye = torch.squeeze(eye)
        eye = torch.unsqueeze(eye, 1)
        eye_embeds = self.eye_encoder(eye)

        concat_embeds = torch.cat([eeg_embeds, eye_embeds], dim=2)
        output_pos = self.fusionModule(concat_embeds)

        loss_nll = torch.nn.NLLLoss()

        ## loss_class
        embed_out = output_pos[:, :, :]
        embed_out2 = rearrange(embed_out, "b n h -> b (n h)")
        x = self.classlinear1(embed_out2)
        x_after_softmax = self.classsoftmax(x)

        labels = labels.to(eeg.device)
        labels = labels.to(torch.int64)
        loss_class = loss_nll(x, labels)
        return loss_class, x_after_softmax, x

    @torch.no_grad()
    def copy_params(self):
        for modal_pairs in self.model_pairs:
            for param, param_m in zip(modal_pairs[0].parameters(), modal_pairs[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, eeg_feat, eye_feat, idx):
        eeg_feats = eeg_feat
        eye_feats = eye_feat
        idxs = idx
        batch_size = eeg_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0
        self.eeg_queue[:, ptr:ptr + batch_size] = eeg_feats.T
        self.eye_queue[:, ptr:ptr + batch_size] = eye_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

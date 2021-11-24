import numpy as np
import torch
import math
import torch.nn.functional as NF

from torch_scatter import scatter_mean


class FeatureBank:
    def __init__(self, obj_n, memory_budget, device, update_rate=0.1, thres_close=0.95, thres_down=0.65):
        super(FeatureBank, self).__init__()

        self.obj_n = obj_n
        self.update_rate = update_rate
        self.thres_close = thres_close
        self.thres_down = thres_down
        self.device = device

        self.info = [None for _ in range(obj_n)]
        self.peak_n = np.zeros(obj_n)
        self.replace_n = np.zeros(obj_n)

        self.class_budget = memory_budget // obj_n
        if obj_n == 2:
            self.class_budget = 0.8 * self.class_budget

    def init_bank(self, keys, valuespos, valuesneg, frame_idx=0):

        self.keys = keys
        # print(self.keys.shape)

        self.valuespos = valuespos
        self.valuesneg = valuesneg

        _, d_keys, bank_n = keys.shape
        # print(self.device)
        self.info = torch.zeros((bank_n, 2), device=self.device)
        self.info[:, 0] = frame_idx
        self.peak_n = max(self.peak_n, self.info.shape[0])

    def update(self, prev_key, prev_value_pos, prev_value_neg, frame_idx):

        batch, d_key, bank_n = self.keys.shape    #bank_n=TWH
        _, d_val_pos, _ = self.valuespos.shape
        _, d_val_neg, _ = self.valuesneg.shape

        normed_keys = NF.normalize(self.keys, dim=1)
        normed_prev_key = NF.normalize(prev_key, dim=1)
        # print(normed_prev_key.shape) #[1, 64, 24*24]
        mag_keys = self.keys.norm(p=2, dim=1)
        # print(mag_keys.shape) #[1, 24*24]
        corr = torch.matmul(normed_keys.transpose(1, 2), normed_prev_key)  # bank_n, prev_n
        # print(corr.shape) #[1, 576, 576]
        related_bank_idx = corr.argmax(dim=1, keepdim=True)  # 1, 1, HW
        # print(related_bank_idx.shape) #[1, 1, 576]
        related_bank_corr = torch.gather(corr, 1, related_bank_idx)  # 1, 1, HW

        # take the average of related-bank-corr, update the thres_down ,then save them
        self.thres_down = torch.mean(related_bank_corr[0][0]) / (math.exp(1)/2)
        # print(self.thres_down)

        # greater than threshold, merge them
        selected_idx = (related_bank_corr[0][0] > self.thres_close).nonzero()
        # print(selected_idx.shape) #[551, 1]

        if selected_idx.shape[0] > 0:
            class_related_bank_idx = related_bank_idx[0, 0, selected_idx[:, 0]]  # selected_HW
            # print(class_related_bank_idx.shape) #[551]

            unique_related_bank_idx, cnt = class_related_bank_idx.unique(dim=0, return_counts=True)  # selected_HW
            # print(unique_related_bank_idx.shape) #[341]
            # Update key
            key_bank_update = torch.zeros((batch, d_key, bank_n), dtype=torch.float, device=self.device)  # d_key, THW
            key_bank_idx = class_related_bank_idx.unsqueeze(0).expand(d_key, -1).unsqueeze(0).expand(batch, d_key, -1)  # d_key, HW
            # print(key_bank_idx.shape) #[B, C, WH]
            # print(normed_prev_key[:, :, selected_idx[:, 0]].shape) #[B, C, WH]
            scatter_mean(normed_prev_key[:, :, selected_idx[:, 0]], key_bank_idx, dim=2, out=key_bank_update)
            # d_key, selected_HW
            if class_related_bank_idx.shape != 0:
                self.keys[:, :, unique_related_bank_idx] = mag_keys[:, unique_related_bank_idx] * \
                    ((1 - self.update_rate) * normed_keys[:, :, unique_related_bank_idx] +
                     self.update_rate * key_bank_update[:, :, unique_related_bank_idx])

            # Update value-pos
            normed_values0 = NF.normalize(self.valuespos, dim=1)
            # print(normed_values0.shape) #B,C,WH
            normed_prev_value0 = NF.normalize(prev_value_pos, dim=1)
            # print(normed_prev_value0) #B,C,WH
            mag_values0 = self.valuespos.norm(p=2, dim=1)
            # print(mag_values0) #batch,WH
            val_bank_update0 = torch.zeros((batch, d_val_pos, bank_n), dtype=torch.float, device=self.device)
            val_bank_idx0 = class_related_bank_idx.unsqueeze(0).expand(d_val_pos, -1).unsqueeze(0).expand(batch, d_val_pos, -1)
            scatter_mean(normed_prev_value0[:, :, selected_idx[:, 0]], val_bank_idx0, dim=2, out=val_bank_update0)

            self.valuespos[:, :, unique_related_bank_idx] = mag_values0[:, unique_related_bank_idx] * \
                ((1 - self.update_rate) * normed_values0[:, :, unique_related_bank_idx] +
                 self.update_rate * val_bank_update0[:, :, unique_related_bank_idx])

            # Update value-neg
            normed_values1 = NF.normalize(self.valuesneg, dim=1)
            normed_prev_value1 = NF.normalize(prev_value_neg, dim=1)
            # print(prev_value_neg)
            mag_values1 = self.valuesneg.norm(p=2, dim=1)
            # print(mag_values1)
            val_bank_update1 = torch.zeros((batch, d_val_pos, bank_n), dtype=torch.float, device=self.device)
            val_bank_idx1 = class_related_bank_idx.unsqueeze(0).expand(d_val_pos, -1).unsqueeze(0).expand(batch, d_val_pos, -1)
            scatter_mean(normed_prev_value1[:, :, selected_idx[:, 0]], val_bank_idx1, dim=2, out=val_bank_update1)

            self.valuesneg[:, :, unique_related_bank_idx] = mag_values1[:, unique_related_bank_idx] * \
                 ((1 - self.update_rate) * normed_values1[:, :, unique_related_bank_idx] +
                 self.update_rate * val_bank_update1[:, :, unique_related_bank_idx])

        # less than the threshold, concat them
        selected_idx = ((self.thres_down <= related_bank_corr[0][0]) & (related_bank_corr[0][0] <= self.thres_close)).nonzero()


        # if self.class_budget < bank_n + selected_idx.shape[0]:
        #     self.remove(selected_idx.shape[0], frame_idx)

        self.keys = torch.cat([self.keys, prev_key[:, :, selected_idx[:, 0]]], dim=2)
        self.valuespos = torch.cat([self.valuespos, prev_value_pos[:, :, selected_idx[:, 0]]], dim=2)
        self.valuesneg = torch.cat([self.valuesneg, prev_value_neg[:, :, selected_idx[:, 0]]], dim=2)
        # print(self.keys.shape)
        # print(self.valuesneg.shape)
        new_info = torch.zeros((selected_idx.shape[0], 2), device=self.device)
        new_info[:, 0] = frame_idx
        # print(frame_idx)
        self.info = torch.cat([self.info, new_info], dim=0)

        self.peak_n = max(self.peak_n, self.info.shape[0])

        self.info[:, 1] = torch.clamp(self.info[:, 1], 0, 1e5)  # Prevent inf
        # print(self.info[:,1])

    def remove(self, request_n, frame_idx):

        old_size = self.keys.shape[2]

        LFU = frame_idx - self.info[:, 0]  # time length
        # print(LFU)
        LFU = self.info[:, 1] / LFU
        # print(LFU)
        thres_dynamic = int(LFU.min()) + 1   #1
        # print(thres_dynamic)
        iter_cnt = 0

        while True:
            selected_idx = LFU > thres_dynamic
            self.keys = self.keys[:, :, selected_idx]
            self.valuespos = self.valuespos[:, :, selected_idx]
            self.valuesneg = self.valuesneg[:, :, selected_idx]
            self.info = self.info[selected_idx]
            LFU = LFU[selected_idx]
            iter_cnt += 1

            balance = (self.class_budget - self.keys.shape[2]) - request_n
            if balance < 0:
                thres_dynamic = int(LFU.min()) + 1
            else:
                break

        new_size = self.keys.shape[2]
        self.replace_n += old_size - new_size

        return balance

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KnnSim(nn.Module):
    # anchor_feature: features from all database, used to select nearest neighbors
    # abchor_label: label for anchor_features
    # graph: K, num_neighbors
    def __init__(self, anchor_feature, anchor_label, graph=50, mode='knn'):
        super(KnnSim, self).__init__()
        self.mode = mode
        self.anchor_feature = anchor_feature
        self.anchor_label = anchor_label.contiguous().view(-1, 1)
        self.graph = graph

    def forward(self, features, labels=None, mask=None, reduction=False):
        # features/contrast_feature: query features, usually a batch
        # labels: label for query features, a batch
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        labels, t_labels = labels
        labels = labels.contiguous().view(-1, 1)
        contrast_feature = features 
        anchor_dot_contrast = torch.matmul(contrast_feature, self.anchor_feature.T) # dot-similarities: [N_contrast, N_achor]

        if self.mode == 'knn':
            anchor_dot_contrast, anchor_dot_contrast_index = torch.topk(anchor_dot_contrast, self.graph, dim=1) # find k nearests
            mask = torch.eq(labels, self.anchor_label.T).float().to(device) # [N_contrast, N_achor]
            mask = torch.gather(mask, 1, anchor_dot_contrast_index) # k近邻的标签是否一致 [N_contrast, K]
            mean_log_prob_pos = (mask).sum(1) / self.graph
        elif self.mode == 'knn_sim':
            anchor_dot_contrast, anchor_dot_contrast_index = torch.topk(anchor_dot_contrast, self.graph, dim=1) 
            mask = torch.eq(labels, self.anchor_label.T).float().to(device) 
            mask = torch.gather(mask, 1, anchor_dot_contrast_index) 
            mean_log_prob_pos = (mask).sum(1) / self.graph

            mean_sim = (anchor_dot_contrast).sum(1) / self.graph
            mean_sim = mean_sim.view(-1)

            mean_log_prob_pos[mean_log_prob_pos != mean_log_prob_pos] = 0
            loss = - mean_log_prob_pos

            if reduction:
                loss = loss.mean() 
            else:
                loss = loss.view(-1)
            return loss, mean_sim

        mean_log_prob_pos[mean_log_prob_pos != mean_log_prob_pos] = 0
        loss = - mean_log_prob_pos # 使得output分数越小，标签置信度越高。 GMM聚类时， 小均值的类代表干净样本

        # print('t_labels:', t_labels.reshape(-1))
        # print('n_labels:', labels.reshape(-1))
        # print('loss:', loss.reshape(-1))

        if reduction:
            loss = loss.mean()
        else:
            loss = loss.view(-1)
        return loss
    

class KnnCtsLoss(nn.Module):
    """
    Knn contrastive loss
    """
    def __init__(self, ):
        super(KnnCtsLoss, self).__init__()

    def forward(self, features, sigma=5, temperature=0.1, margin=10.):  #
        
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        features = features.contiguous().view(features.shape[0], -1)
        features = F.normalize(features, p=2., dim=1)
        sim_matrix = torch.matmul(features, features.T)
        
        sim_pos_k, pos_index_k = torch.topk(sim_matrix, sigma+1, dim=1)
        sim_neg_k, neg_index_k = torch.topk(-sim_matrix, sigma, dim=1)

        sim_pos_k = sim_pos_k[:, 1:] / temperature
        sim_neg_k = -sim_neg_k / temperature
        
        pos_pairs = torch.exp(sim_pos_k) # [N, sigma]
        neg_sum = (torch.sum(torch.exp(sim_neg_k), dim=1)).unsqueeze(1).repeat(1, sigma) # [N, sigma]

        cts_loss = -(torch.sum(torch.log(pos_pairs/neg_sum), dim=1)/sigma).mean()
        cts_loss = torch.max(cts_loss + torch.tensor(margin, device=device), torch.tensor(0.0, device=device))

        return cts_loss


class KnnCtsLoss2(nn.Module):
    """
    Knn contrastive loss with stability and non-negativity constraints
    """
    def __init__(self):
        super(KnnCtsLoss2, self).__init__()

    def forward(self, features, sigma=5, temperature=0.1, margin=10.):
        device = features.device

        features = features.contiguous().view(features.shape[0], -1)
        features = F.normalize(features, p=2, dim=1)

        sim_matrix = torch.matmul(features, features.T)
        sim_pos_k, pos_index_k = torch.topk(sim_matrix, sigma + 1, dim=1)

        # Construct negative mask
        batch_size = sim_matrix.size(0)
        pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
        pos_mask.scatter_(1, pos_index_k, True)
        neg_mask = ~pos_mask

        # Extract negative similarities
        sim_neg_k = sim_matrix.masked_select(neg_mask).view(batch_size, -1)

        # Temperature normalization
        sim_pos_k = sim_pos_k[:, 1:] / temperature  # Exclude self-similarity
        sim_neg_k = sim_neg_k / temperature

        # Compute positive and negative contributions
        pos_pairs = torch.exp(sim_pos_k)
        neg_sum = torch.sum(torch.exp(sim_neg_k), dim=1, keepdim=True)

        # Compute contrastive loss
        cts_loss = -(torch.sum(torch.log(pos_pairs / neg_sum), dim=1) / sigma).mean()

        # Ensure non-negativity
        cts_loss = torch.max(cts_loss, torch.tensor(0.0, device=device))

        return cts_loss
    

class KnnCtsLoss3_FNC(nn.Module):
    """
    Knn contrastive loss with stability and False Negative Cancel
    """
    def __init__(self):
        super(KnnCtsLoss3_FNC, self).__init__()

    def forward(self, features, labels, sigma=5, temperature=0.1):
        device = features.device

        # Normalize features
        features = features.contiguous().view(features.shape[0], -1)
        features = F.normalize(features, p=2, dim=1)
        sim_matrix = torch.matmul(features, features.T)
        sim_pos_k, pos_index_k = torch.topk(sim_matrix, sigma + 1, dim=1)

        # Construct masks
        batch_size = sim_matrix.size(0)
        pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
        pos_mask.scatter_(1, pos_index_k, True)  # Mark positive pairs (including self-similarity)

        # Filter negatives: Remove samples with the same labels
        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)  # Compare all pairs of labels
        neg_mask = ~pos_mask & ~label_matrix  # Exclude positive pairs and same-label pairs

        # Extract negative similarities (only valid negatives remain)
        neg_sim_matrix = sim_matrix.masked_fill(~neg_mask, float('-inf'))  # 非负样本用 -inf 填充
        sim_neg_k = neg_sim_matrix.view(batch_size, -1)

        # Temperature normalization
        sim_pos_k = sim_pos_k[:, 1:] / temperature  # Exclude self-similarity
        sim_neg_k = sim_neg_k / temperature

        # Compute positive and negative contributions
        pos_pairs = torch.exp(sim_pos_k)
        neg_sum = torch.sum(torch.exp(sim_neg_k), dim=1, keepdim=True)

        # Compute contrastive loss
        eps = 1e-8  # Stability constant
        cts_loss = -(torch.sum(torch.log(torch.clamp(pos_pairs / neg_sum, min=eps)), dim=1) / sigma).mean()

        # Ensure non-negativity
        cts_loss = torch.clamp(cts_loss, min=0.0)

        return cts_loss

if __name__ == "__main__":
    pass
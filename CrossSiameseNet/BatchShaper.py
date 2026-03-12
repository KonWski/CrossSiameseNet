import torch
import random

class BatchShaper:

    def __init__(self, device, training_type: str, alpha: float = None):
        self.device = device
        self.training_type = training_type
        self.alpha = alpha

    def shape_batch(self, anchors_mf, positive_mfs, negative_mfs, anchor_labels, model, state):

        indices_1 = (anchor_labels == 1).nonzero()[:,0].tolist()
        indices_0 = (anchor_labels == 0).nonzero()[:,0].tolist()

        anchors_mf = anchors_mf.to(self.device)
        anchors_transformed = model(anchors_mf)
        anchors_transformed_1 = anchors_transformed[indices_1,:]
        anchors_transformed_0 = anchors_transformed[indices_0,:]

        distances = torch.cdist(anchors_transformed, anchors_transformed)

        if state == "train":

            if self.training_type == "hard_batch_learning":

                positive_mfs_transformed = []
                negative_mfs_transformed = []

                for anchor_iter in range(len(anchors_transformed)):

                    anchor_label = anchor_labels[anchor_iter]

                    if anchor_label == 0:
                        distances_pos = distances[anchor_iter, indices_0]
                        distances_neg = distances[anchor_iter, indices_1]

                        id_distance_pos_max = distances_pos.argmax().item()
                        id_distance_neg_min = distances_neg.argmin().item()
                        positive_mfs_transformed.append(anchors_transformed_0[id_distance_pos_max, :])
                        negative_mfs_transformed.append(anchors_transformed_1[id_distance_neg_min, :])

                    elif anchor_label == 1:
                        distances_pos = distances[anchor_iter, indices_1]
                        distances_neg = distances[anchor_iter, indices_0]
                        id_distance_pos_max = distances_pos.argmax().item()
                        id_distance_neg_min = distances_neg.argmin().item()
                        positive_mfs_transformed.append(anchors_transformed_1[id_distance_pos_max, :])
                        negative_mfs_transformed.append(anchors_transformed_0[id_distance_neg_min, :])

                positive_mfs_transformed = torch.stack(positive_mfs_transformed, dim=0)
                negative_mfs_transformed = torch.stack(negative_mfs_transformed, dim=0)

            elif self.training_type == "hard_batch_learning_only_positives":

                positive_mfs_transformed = []
                negative_mfs_transformed = []

                for anchor_iter in range(len(anchors_transformed)):

                    anchor_label = anchor_labels[anchor_iter]

                    if anchor_label == 0:

                        distances_pos = distances[anchor_iter, indices_0]
                        distances_neg = distances[anchor_iter, indices_1]
                        id_distance_pos = random.randrange(len(distances_pos))
                        id_distance_neg = random.randrange(len(distances_neg))
                        positive_mfs_transformed.append(anchors_transformed_0[id_distance_pos, :])
                        negative_mfs_transformed.append(anchors_transformed_1[id_distance_neg, :])

                    elif anchor_label == 1:
                        distances_pos = distances[anchor_iter, indices_1]
                        distances_neg = distances[anchor_iter, indices_0]
                        id_distance_pos_max = distances_pos.argmax().item()
                        id_distance_neg_min = distances_neg.argmin().item()
                        positive_mfs_transformed.append(anchors_transformed_1[id_distance_pos_max, :])
                        negative_mfs_transformed.append(anchors_transformed_0[id_distance_neg_min, :])

                positive_mfs_transformed = torch.stack(positive_mfs_transformed, dim=0)
                negative_mfs_transformed = torch.stack(negative_mfs_transformed, dim=0)

            elif self.training_type == "semi_hard_negative_mining":

                positive_mfs_transformed = []
                negative_mfs_transformed = []

                for anchor_iter in range(len(anchors_transformed)):

                    anchor_label = anchor_labels[anchor_iter]

                    if anchor_label == 1:
                        pos_pool = [id for id in indices_1 if id != anchor_iter]
                        anchors_transformed_pos = anchors_transformed_1
                        neg_pool = indices_0
                        anchors_transformed_neg = anchors_transformed_0
                        
                    else:
                        pos_pool = [id for id in indices_0 if id != anchor_iter]
                        anchors_transformed_pos = anchors_transformed_0
                        neg_pool = indices_1
                        anchors_transformed_neg = anchors_transformed_1

                    pos_idx = pos_pool[torch.randint(len(pos_pool), (1,)).item()]

                    distances_pos = distances[anchor_iter, pos_idx]
                    distances_neg = distances[anchor_iter, neg_pool]

                    semi_hard_mask = (distances_neg > distances_pos) & (distances_neg < distances_pos + self.alpha)
                    semi_hard_idx = semi_hard_mask.nonzero(as_tuple=True)[0].tolist()

                    if semi_hard_mask.any():
                        # pick first semi-hard (or random among them)
                        valid_ids = [neg_pool[i] for i in semi_hard_idx]
                        neg_idx = valid_ids[torch.randint(len(valid_ids), (1,)).item()]
                    else:
                        # fallback to closest negative
                        neg_idx = neg_pool[distances_neg.argmin().item()]

                    positive_mfs_transformed.append(anchors_transformed_pos[pos_idx, :])
                    negative_mfs_transformed.append(anchors_transformed_neg[neg_idx, :])

                positive_mfs_transformed = torch.stack(positive_mfs_transformed, dim=0)
                negative_mfs_transformed = torch.stack(negative_mfs_transformed, dim=0)

            else:
                positive_mfs = positive_mfs.to(self.device)
                negative_mfs = negative_mfs.to(self.device)

                positive_mfs_transformed = model(positive_mfs)
                negative_mfs_transformed = model(negative_mfs)

        elif state == "test":
                positive_mfs = positive_mfs.to(self.device)
                negative_mfs = negative_mfs.to(self.device)

                positive_mfs_transformed = model(positive_mfs)
                negative_mfs_transformed = model(negative_mfs)

        return anchors_transformed, positive_mfs_transformed, negative_mfs_transformed, anchor_labels
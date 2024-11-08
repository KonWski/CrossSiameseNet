import torch
import random
import logging

class BatchShaper:

    def __init__(self, device, training_type: str):
        self.device = device
        self.training_type = training_type

    def shape_batch(self, anchors_mf, positive_mfs, negative_mfs, anchor_labels, model, state):

        indices_1 = (anchor_labels == 1).nonzero()[:,0].tolist()
        indices_0 = (anchor_labels == 0).nonzero()[:,0].tolist()

        anchors_mf = anchors_mf.to(self.device)
        anchors_transformed = model(anchors_mf)
        anchors_transformed_1 = anchors_transformed[indices_1,:]
        anchors_transformed_0 = anchors_transformed[indices_0,:]

        if state == "train":

            if self.training_type == "hard_batch_learning":

                distances = torch.cdist(anchors_transformed, anchors_transformed)
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

                distances = torch.cdist(anchors_transformed)
                positive_mfs_transformed = []
                negative_mfs_transformed = []

                for anchor_iter in range(len(anchors_transformed)):

                    anchor_label = anchor_labels[anchor_iter]

                    if anchor_label == 0:

                        distances_pos = distances[anchor_iter, indices_0]
                        distances_neg = distances[anchor_iter, indices_1]
                        id_distance_pos = random.choice(indices_0)
                        id_distance_neg = random.choice(indices_1)
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

            elif self.training_type == "hard_batch_learning_only_positives":
                raise Exception("Training type {self.training_type} under construction")

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
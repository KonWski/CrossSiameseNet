import torch
import random
import logging

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

                distances = torch.cdist(anchors_transformed, anchors_transformed)
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

                distances = torch.cdist(anchors_transformed, anchors_transformed)
                positive_mfs_transformed = []
                negative_mfs_transformed = []
                n_anchors_switched_to_hard_batch_1 = 0
                n_anchors_switched_to_hard_batch_0 = 0

                for anchor_iter in range(len(anchors_transformed)):

                    anchor_label = anchor_labels[anchor_iter]

                    if anchor_label == 0:

                        distances_pos = distances[anchor_iter, indices_0]
                        distances_neg = distances[anchor_iter, indices_1]

                        id_distance_pos = random.randrange(len(distances_pos))
                        distance_pos = distances_pos[id_distance_pos]
                        positive_mfs_transformed.append(anchors_transformed_0[id_distance_pos, :])

                        # negative distances fulfilling the alpha condition
                        ids_neg_alpha_cond = ((distances_neg > distance_pos) & (distances_neg < self.alpha)).nonzero()[:,0].tolist()

                        # found negative observations which fulfill the condition
                        if len(ids_neg_alpha_cond) > 0:
                            
                            distances_neg_alpha_cond = distances_neg[ids_neg_alpha_cond]
                            anchors_transformed_neg_alpha_cond = anchors_transformed_1[ids_neg_alpha_cond, :]
                            id_distance_neg_min = distances_neg_alpha_cond.argmin().item()
                            negative_mfs_transformed.append(anchors_transformed_neg_alpha_cond[id_distance_neg_min, :])
                        
                        else:
                                
                            id_distance_neg_min = distances_neg.argmin().item()
                            negative_mfs_transformed.append(anchors_transformed_1[id_distance_neg_min, :])
                            n_anchors_switched_to_hard_batch_0 += 1

                    elif anchor_label == 1:

                        distances_pos = distances[anchor_iter, indices_1]
                        distances_neg = distances[anchor_iter, indices_0]

                        id_distance_pos = random.choice(indices_1)
                        distance_pos = distances_pos[id_distance_pos]
                        positive_mfs_transformed.append(anchors_transformed_1[id_distance_pos, :])

                        # negative distances fulfilling the alpha condition
                        ids_neg_alpha_cond = ((distances_neg > distance_pos) & (distances_neg < self.alpha)).nonzero()[:,0].tolist()

                        # found negative observations which fulfill the condition
                        if len(ids_neg_alpha_cond) > 0:
                            
                            distances_neg_alpha_cond = distances_neg[ids_neg_alpha_cond]
                            anchors_transformed_neg_alpha_cond = anchors_transformed_0[ids_neg_alpha_cond, :]
                            id_distance_neg_min = distances_neg_alpha_cond.argmin().item()
                            negative_mfs_transformed.append(anchors_transformed_neg_alpha_cond[id_distance_neg_min, :])
                        
                        else:

                            id_distance_neg_min = distances_neg.argmin().item()
                            negative_mfs_transformed.append(anchors_transformed_0[id_distance_neg_min, :])
                            n_anchors_switched_to_hard_batch_1 += 1

                positive_mfs_transformed = torch.stack(positive_mfs_transformed, dim=0)
                negative_mfs_transformed = torch.stack(negative_mfs_transformed, dim=0)

                if len(n_anchors_switched_to_hard_batch_1) > 0 or len(n_anchors_switched_to_hard_batch_0) > 0:
                    logging.info(f"BatchShaper | n_anchors_switched_to_hard_batch_1: {n_anchors_switched_to_hard_batch_1}")
                    logging.info(f"BatchShaper | n_anchors_switched_to_hard_batch_0: {n_anchors_switched_to_hard_batch_0}")

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
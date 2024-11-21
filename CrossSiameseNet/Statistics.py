import torch
from torch.utils.data import DataLoader

class Statistics:

    def __init__(self, device):
        self.device = device

    def distance_stats(self, model, loader):

        anchors_mf, anchor_labels = self._generate_embeddings(model, loader.dataset, 1000)

        indices_1 = (anchor_labels == 1).nonzero()[:,0].tolist()
        indices_0 = (anchor_labels == 0).nonzero()[:,0].tolist()

        anchors_mf = anchors_mf.to(self.device)
        anchors_transformed = model(anchors_mf)

        distances = torch.cdist(anchors_transformed, anchors_transformed)

        distances_0_0 = distances[indices_0, indices_0]
        distances_0_0 = distances_0_0[distances_0_0 != 0]
        distances_0_0_mean = round(torch.mean(distances_0_0).item(), 5)

        distances_1_1 = distances[indices_1, indices_1]
        distances_1_1 = distances_1_1[distances_1_1 != 0]
        distances_1_1_mean = round(torch.mean(distances_1_1).item(), 5)

        distances_0_1 = []
        for i0 in indices_0:
            for i1 in indices_1:
                distances_0_1.append(distances[i0, i1].item())
        distances_0_1_mean = round(sum(distances_0_1) / len(distances_0_1), 5)

        return distances_0_0_mean, distances_1_1_mean, distances_0_1_mean
    
    def _generate_embeddings(self, model, dataset, batch_size):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        embeddings = []
        anchor_labels = []
        for batch_id, (anchor_mf, positive_mf, negative_mf, anchor_label) in enumerate(loader):
            anchor_mf = anchor_mf.to(self.device)
            embeddings.append(model(anchor_mf).cpu())
            anchor_labels.append(anchor_label.cpu())
        return torch.cat(embeddings, dim=0), torch.cat(anchor_labels, dim=0)
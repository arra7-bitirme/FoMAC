import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionSpottingTransformerNet(nn.Module):
    def __init__(self, input_dim=8576, num_classes=17):
        super().__init__()

        # Feature projection
        self.proj = nn.Linear(input_dim, 512)

        # Temporal reasoning
        self.transformer = TemporalTransformer(
            dim=512,
            heads=8,
            layers=2
        )

        # Context aggregation
        self.vlad_past = NetVLAD(32, 512)
        self.vlad_future = NetVLAD(32, 512)

        vlad_dim = 32 * 512 * 2

        # Heads
        self.cls_head = nn.Linear(vlad_dim, num_classes + 1)
        self.reg_head = nn.Sequential(
            nn.Linear(vlad_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """

        # Step 1: project features
        x = self.proj(x)        # (B, T, 512)

        # Step 2: temporal modeling
        x = self.transformer(x)

        # Step 3: split context
        mid = x.shape[1] // 2
        past = x[:, :mid]
        future = x[:, mid:]

        # Step 4: aggregate
        v_p = self.vlad_past(past)
        v_f = self.vlad_future(future)

        # Step 5: concat
        feat = torch.cat([v_p, v_f], dim=-1)

        # Step 6: predict
        return self.cls_head(feat), self.reg_head(feat).squeeze(-1)
    
class TemporalTransformer(nn.Module):
    """
    Learns temporal relationships between frames.
    """

    def __init__(self, dim=512, heads=8, layers=2):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        return self.encoder(x)


class NetVLAD(nn.Module):
    """
    Aggregates temporal features into a fixed-length descriptor.
    """

    def __init__(self, cluster_size, feature_size):
        super().__init__()
        self.cluster_size = cluster_size
        self.feature_size = feature_size

        # Learnable cluster centers
        self.centroids = nn.Parameter(
            torch.randn(cluster_size, feature_size)
        )

        # Soft assignment of frames to clusters
        self.assignment = nn.Linear(feature_size, cluster_size)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        B, T, D = x.shape

        # (B, T, K)
        soft_assign = F.softmax(self.assignment(x), dim=-1)

        # Expand for residual computation
        x_exp = x.unsqueeze(2)           # (B, T, 1, D)
        c_exp = self.centroids.unsqueeze(0).unsqueeze(0)  # (1, 1, K, D)

        # Residuals to each cluster
        residual = x_exp - c_exp         # (B, T, K, D)

        # Aggregate
        vlad = (soft_assign.unsqueeze(-1) * residual).sum(dim=1)

        # Normalize
        vlad = F.normalize(vlad, p=2, dim=-1)

        # Flatten
        return vlad.view(B, -1)


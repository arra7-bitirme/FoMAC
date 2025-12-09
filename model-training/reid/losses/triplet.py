"""
Triplet Loss with Batch-Hard Mining

Implements triplet loss for metric learning in ReID.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet loss with batch-hard mining.
    
    For each anchor, selects:
        - Hardest positive (same identity, maximum distance)
        - Hardest negative (different identity, minimum distance)
    
    This is more effective than random sampling.
    """
    
    def __init__(self, margin: float = 0.3):
        """
        Args:
            margin: Margin for triplet loss
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            embeddings: L2-normalized embeddings (B, emb_dim)
            labels: Identity labels (B,)
            
        Returns:
            loss: Scalar triplet loss
            stats: Dictionary with mining statistics
        """
        # Compute pairwise distances (using L2-normalized embeddings)
        # Distance = 2 - 2 * cosine_similarity
        dist_mat = self._compute_distance_matrix(embeddings)
        
        # Mine hard positives and negatives
        dist_ap, dist_an, valid_triplets = self._batch_hard_mining(
            dist_mat, labels
        )
        
        if valid_triplets == 0:
            # No valid triplets found
            return torch.tensor(0.0, device=embeddings.device), {
                'triplet_loss': 0.0,
                'valid_triplets': 0,
                'mean_dist_ap': 0.0,
                'mean_dist_an': 0.0
            }
        
        # Compute triplet loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # Statistics
        stats = {
            'triplet_loss': loss.item(),
            'valid_triplets': valid_triplets,
            'mean_dist_ap': dist_ap.mean().item(),
            'mean_dist_an': dist_an.mean().item(),
            'margin_violations': (dist_ap > dist_an - self.margin).sum().item()
        }
        
        return loss, stats
    
    def _compute_distance_matrix(self, embeddings: torch.Tensor):
        """
        Compute pairwise Euclidean distance matrix.
        
        Args:
            embeddings: (B, D) L2-normalized embeddings
            
        Returns:
            dist_mat: (B, B) distance matrix
        """
        # Using L2-normalized embeddings:
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*<a,b>
        #             = 1 + 1 - 2*<a,b>  (since normalized)
        #             = 2 * (1 - <a,b>)
        
        m, n = embeddings.size(0), embeddings.size(0)
        
        # Compute dot product
        mat = torch.matmul(embeddings, embeddings.t())
        
        # Convert to distances
        dist_mat = 2 - 2 * mat
        
        # Clamp to avoid numerical issues
        dist_mat = torch.clamp(dist_mat, min=0.0)
        
        # Take square root
        dist_mat = torch.sqrt(dist_mat + 1e-12)
        
        return dist_mat
    
    def _batch_hard_mining(self, dist_mat, labels):
        """
        Batch-hard mining strategy.
        
        Args:
            dist_mat: (B, B) distance matrix
            labels: (B,) identity labels
            
        Returns:
            dist_ap: Distances to hardest positives
            dist_an: Distances to hardest negatives
            valid_count: Number of valid triplets
        """
        batch_size = dist_mat.size(0)
        
        # For each anchor, find:
        # - hardest positive: max distance to same identity
        # - hardest negative: min distance to different identity
        
        # Create masks
        labels = labels.unsqueeze(1)  # (B, 1)
        pos_mask = labels == labels.t()  # (B, B)
        neg_mask = labels != labels.t()  # (B, B)
        
        # Set diagonal to False (exclude self-comparisons)
        pos_mask.fill_diagonal_(False)
        
        # Find hardest positive for each anchor
        dist_ap = []
        dist_an = []
        
        for i in range(batch_size):
            # Hardest positive (maximum distance among same identity)
            pos_dists = dist_mat[i][pos_mask[i]]
            
            if len(pos_dists) > 0:
                hardest_pos_dist = pos_dists.max()
                
                # Hardest negative (minimum distance among different identities)
                neg_dists = dist_mat[i][neg_mask[i]]
                
                if len(neg_dists) > 0:
                    hardest_neg_dist = neg_dists.min()
                    
                    dist_ap.append(hardest_pos_dist)
                    dist_an.append(hardest_neg_dist)
        
        if len(dist_ap) == 0:
            # No valid triplets
            return None, None, 0
        
        # Stack distances
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        
        return dist_ap, dist_an, len(dist_ap)


class CombinedLoss(nn.Module):
    """
    Combined CrossEntropy + Triplet Loss for ReID training.
    """
    
    def __init__(
        self,
        num_classes: int,
        triplet_margin: float = 0.3,
        ce_weight: float = 1.0,
        triplet_weight: float = 1.0,
        label_smooth: float = 0.0
    ):
        """
        Args:
            num_classes: Number of person identities
            triplet_margin: Margin for triplet loss
            ce_weight: Weight for cross-entropy loss
            triplet_weight: Weight for triplet loss
            label_smooth: Label smoothing epsilon
        """
        super(CombinedLoss, self).__init__()
        
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.triplet_weight = triplet_weight
        
        # Cross-entropy loss
        if label_smooth > 0:
            self.ce_loss = CrossEntropyLabelSmooth(
                num_classes=num_classes,
                epsilon=label_smooth
            )
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        # Triplet loss
        self.triplet_loss = TripletLoss(margin=triplet_margin)
    
    def forward(self, embeddings, logits, labels):
        """
        Args:
            embeddings: L2-normalized embeddings (B, emb_dim)
            logits: Classification logits (B, num_classes)
            labels: Identity labels (B,)
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        # Cross-entropy loss
        ce_loss = self.ce_loss(logits, labels)
        
        # Triplet loss
        triplet_loss, triplet_stats = self.triplet_loss(embeddings, labels)
        
        # Combined loss
        total_loss = (
            self.ce_weight * ce_loss +
            self.triplet_weight * triplet_loss
        )
        
        # Loss dictionary
        loss_dict = {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'triplet_loss': triplet_stats['triplet_loss'],
            'valid_triplets': triplet_stats['valid_triplets'],
            'mean_dist_ap': triplet_stats['mean_dist_ap'],
            'mean_dist_an': triplet_stats['mean_dist_an'],
        }
        
        if 'margin_violations' in triplet_stats:
            loss_dict['margin_violations'] = triplet_stats['margin_violations']
        
        return total_loss, loss_dict


class CrossEntropyLabelSmooth(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    """
    
    def __init__(self, num_classes: int, epsilon: float = 0.1):
        """
        Args:
            num_classes: Number of classes
            epsilon: Label smoothing factor
        """
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits (B, num_classes)
            targets: Labels (B,)
            
        Returns:
            loss: Scalar loss
        """
        log_probs = self.logsoftmax(inputs)
        
        # Create smoothed labels
        targets_one_hot = torch.zeros_like(log_probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply label smoothing
        targets_smooth = (
            (1 - self.epsilon) * targets_one_hot +
            self.epsilon / self.num_classes
        )
        
        # Compute loss
        loss = (-targets_smooth * log_probs).sum(dim=1).mean()
        
        return loss


def test_triplet_loss():
    """Test triplet loss."""
    print("Testing Triplet Loss...")
    
    # Create sample embeddings
    batch_size = 16
    emb_dim = 256
    
    embeddings = torch.randn(batch_size, emb_dim)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Create labels (4 identities, 4 samples each)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    
    # Create loss
    triplet_loss = TripletLoss(margin=0.3)
    
    # Compute loss
    loss, stats = triplet_loss(embeddings, labels)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Stats: {stats}")
    
    assert loss.item() >= 0
    assert stats['valid_triplets'] > 0
    
    print("✓ Triplet loss test passed!")


if __name__ == "__main__":
    test_triplet_loss()

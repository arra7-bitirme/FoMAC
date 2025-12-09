"""
Cost Matrix for ReID-based Tracking

Combines IoU and appearance similarity for robust tracking.
"""

import numpy as np
from scipy.spatial.distance import cdist


def compute_iou_matrix(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between two sets of bounding boxes.
    
    Args:
        bboxes_a: (N, 4) array of boxes [x1, y1, x2, y2]
        bboxes_b: (M, 4) array of boxes [x1, y1, x2, y2]
        
    Returns:
        iou_matrix: (N, M) IoU matrix
    """
    N = bboxes_a.shape[0]
    M = bboxes_b.shape[0]
    
    iou_matrix = np.zeros((N, M), dtype=np.float32)
    
    for i in range(N):
        for j in range(M):
            # Compute intersection
            x1 = max(bboxes_a[i, 0], bboxes_b[j, 0])
            y1 = max(bboxes_a[i, 1], bboxes_b[j, 1])
            x2 = min(bboxes_a[i, 2], bboxes_b[j, 2])
            y2 = min(bboxes_a[i, 3], bboxes_b[j, 3])
            
            inter_w = max(0, x2 - x1)
            inter_h = max(0, y2 - y1)
            inter_area = inter_w * inter_h
            
            # Compute union
            area_a = (bboxes_a[i, 2] - bboxes_a[i, 0]) * \
                     (bboxes_a[i, 3] - bboxes_a[i, 1])
            area_b = (bboxes_b[j, 2] - bboxes_b[j, 0]) * \
                     (bboxes_b[j, 3] - bboxes_b[j, 1])
            union_area = area_a + area_b - inter_area
            
            # Compute IoU
            if union_area > 0:
                iou_matrix[i, j] = inter_area / union_area
    
    return iou_matrix


def compute_cosine_distance_matrix(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray
) -> np.ndarray:
    """
    Compute cosine distance matrix between two sets of embeddings.
    
    For L2-normalized embeddings:
        cosine_distance = 1 - cosine_similarity
    
    Args:
        embeddings_a: (N, D) array of L2-normalized embeddings
        embeddings_b: (M, D) array of L2-normalized embeddings
        
    Returns:
        dist_matrix: (N, M) cosine distance matrix
    """
    # Compute cosine similarity (dot product for normalized vectors)
    similarity = np.dot(embeddings_a, embeddings_b.T)
    
    # Clip to [-1, 1] to handle numerical errors
    similarity = np.clip(similarity, -1.0, 1.0)
    
    # Convert to distance
    distance = 1.0 - similarity
    
    return distance


def build_cost(
    iou_matrix: np.ndarray,
    emb_trk: np.ndarray,
    emb_det: np.ndarray,
    alpha: float = 0.6,
    beta: float = 0.4,
    max_distance: float = 0.7
) -> np.ndarray:
    """
    Build combined cost matrix for tracking association.
    
    Cost combines motion (IoU) and appearance (embedding) cues:
        cost = α * (1 - IoU) + β * (1 - cosine_sim)
    
    Lower cost = better match
    
    Args:
        iou_matrix: (N, M) IoU matrix between tracks and detections
        emb_trk: (N, D) embeddings for tracks
        emb_det: (M, D) embeddings for detections
        alpha: Weight for IoU term (default: 0.6)
        beta: Weight for appearance term (default: 0.4)
        max_distance: Maximum allowed distance (matches beyond this are
                     set to infinity)
        
    Returns:
        cost_matrix: (N, M) combined cost matrix
    
    Example:
        >>> iou = compute_iou_matrix(track_boxes, det_boxes)
        >>> cost = build_cost(iou, track_embs, det_embs,
        ...                   alpha=0.6, beta=0.4)
        >>> # Use cost matrix with Hungarian algorithm for assignment
    """
    # Compute IoU cost (1 - IoU)
    iou_cost = 1.0 - iou_matrix
    
    # Compute appearance cost (cosine distance)
    appearance_cost = compute_cosine_distance_matrix(emb_trk, emb_det)
    
    # Combine costs
    cost_matrix = alpha * iou_cost + beta * appearance_cost
    
    # Set high cost for matches beyond threshold
    cost_matrix[cost_matrix > max_distance] = 1e5
    
    return cost_matrix


def build_cost_iou_only(
    iou_matrix: np.ndarray,
    max_iou_distance: float = 0.7
) -> np.ndarray:
    """
    Build cost matrix using only IoU (no appearance).
    
    Args:
        iou_matrix: (N, M) IoU matrix
        max_iou_distance: Maximum IoU distance threshold
        
    Returns:
        cost_matrix: (N, M) cost matrix
    """
    cost_matrix = 1.0 - iou_matrix
    cost_matrix[cost_matrix > max_iou_distance] = 1e5
    return cost_matrix


def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        emb_a: (D,) embedding
        emb_b: (D,) embedding
        
    Returns:
        similarity: Scalar in [-1, 1]
    """
    similarity = np.dot(emb_a, emb_b)
    similarity = np.clip(similarity, -1.0, 1.0)
    return float(similarity)


def cosine_distance(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Compute cosine distance between two embeddings.
    
    Args:
        emb_a: (D,) embedding
        emb_b: (D,) embedding
        
    Returns:
        distance: Scalar in [0, 2]
    """
    return 1.0 - cosine_similarity(emb_a, emb_b)


def test_cost_matrix():
    """Test cost matrix computation."""
    print("Testing Cost Matrix...")
    
    # Create dummy data
    N_tracks = 3
    M_detections = 4
    emb_dim = 256
    
    # Random bounding boxes
    track_boxes = np.array([
        [100, 100, 150, 200],
        [200, 150, 250, 250],
        [300, 200, 350, 300]
    ], dtype=np.float32)
    
    det_boxes = np.array([
        [105, 105, 155, 205],  # Close to track 0
        [195, 145, 245, 245],  # Close to track 1
        [400, 400, 450, 500],  # Far from all tracks
        [305, 205, 355, 305]   # Close to track 2
    ], dtype=np.float32)
    
    # Random L2-normalized embeddings
    track_embs = np.random.randn(N_tracks, emb_dim).astype(np.float32)
    track_embs = track_embs / np.linalg.norm(
        track_embs, axis=1, keepdims=True
    )
    
    det_embs = np.random.randn(M_detections, emb_dim).astype(np.float32)
    det_embs = det_embs / np.linalg.norm(det_embs, axis=1, keepdims=True)
    
    # Make first detection embedding similar to first track
    det_embs[0] = track_embs[0] + np.random.randn(emb_dim) * 0.1
    det_embs[0] = det_embs[0] / np.linalg.norm(det_embs[0])
    
    # Compute IoU matrix
    iou_matrix = compute_iou_matrix(track_boxes, det_boxes)
    print(f"IoU matrix shape: {iou_matrix.shape}")
    print(f"IoU matrix:\n{iou_matrix}")
    
    # Compute cost matrix
    cost_matrix = build_cost(
        iou_matrix,
        track_embs,
        det_embs,
        alpha=0.6,
        beta=0.4,
        max_distance=0.7
    )
    
    print(f"\nCost matrix shape: {cost_matrix.shape}")
    print(f"Cost matrix:\n{cost_matrix}")
    
    # Test case: IoU=0.5, cos_sim=0.8
    # Expected cost: 0.6*(1-0.5) + 0.4*(1-0.8) = 0.3 + 0.08 = 0.38
    test_iou = np.array([[0.5]])
    test_emb_a = np.array([[1.0] + [0.0] * 255])
    test_emb_b = np.array([[0.8] + [0.6] + [0.0] * 254])
    test_emb_b = test_emb_b / np.linalg.norm(test_emb_b)
    
    test_cost = build_cost(test_iou, test_emb_a, test_emb_b,
                          alpha=0.6, beta=0.4)
    print(f"\nTest case (IoU=0.5, cos_sim≈0.8):")
    print(f"  Cost: {test_cost[0, 0]:.4f}")
    print(f"  Expected: ~0.38")
    
    # Test case: IoU=0.1, cos_sim=0.2
    # Expected cost: 0.6*(1-0.1) + 0.4*(1-0.2) = 0.54 + 0.32 = 0.86
    test_iou2 = np.array([[0.1]])
    test_emb_c = np.array([[1.0] + [0.0] * 255])
    test_emb_d = np.array([[0.2] + [0.98] + [0.0] * 254])
    test_emb_d = test_emb_d / np.linalg.norm(test_emb_d)
    
    test_cost2 = build_cost(test_iou2, test_emb_c, test_emb_d,
                           alpha=0.6, beta=0.4)
    print(f"\nTest case (IoU=0.1, cos_sim≈0.2):")
    print(f"  Cost: {test_cost2[0, 0]:.4f}")
    print(f"  Expected: ~0.86")
    
    # Verify acceptance criterion
    assert test_cost[0, 0] < test_cost2[0, 0], \
        "Cost should be lower for better matches!"
    
    print("\n✓ All tests passed!")
    print("✓ Acceptance criterion verified:")
    print("  IoU=0.5, cos_sim=0.8 has LOWER cost than")
    print("  IoU=0.1, cos_sim=0.2")


if __name__ == "__main__":
    test_cost_matrix()

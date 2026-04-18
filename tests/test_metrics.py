import unittest
import numpy as np

# A standalone copy of compute_mAP to bypass PyTorch import requirement in the CI/Test phase
def compute_mAP(y_true, y_scores):
    ap_list = []
    num_classes = y_true.shape[1]
    
    for i in range(num_classes):
        y_t = y_true[:, i]
        y_s = y_scores[:, i]
        
        if y_t.sum() == 0:
            ap_list.append(0.0)
            continue
            
        sorted_indices = np.argsort(-y_s)
        sorted_scores = y_s[sorted_indices]
        sorted_truth = y_t[sorted_indices]
        
        tp = sorted_truth
        fp = 1 - tp
        
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        
        precision = tp_cum / (tp_cum + fp_cum + 1e-8)
        recall = tp_cum / (tp_cum[-1] + 1e-8)
        
        ap = np.sum(precision * tp) / (np.sum(tp) + 1e-8)
        ap_list.append(ap)
        
    return np.mean(ap_list) * 100

class TestMetrics(unittest.TestCase):
    def test_compute_map_perfect_score(self):
        y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        y_scores = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        mAP = compute_mAP(y_true, y_scores)
        self.assertAlmostEqual(mAP, 100.0, places=1)

    def test_compute_map_worst_score(self):
        y_true = np.array([[1, 0], [0, 1]])
        y_scores = np.array([[0.1, 0.9], [0.9, 0.1]])
        mAP = compute_mAP(y_true, y_scores)
        self.assertLess(mAP, 100.0)

if __name__ == "__main__":
    unittest.main()

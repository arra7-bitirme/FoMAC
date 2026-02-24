import torch
import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm

import config as cfg
# Model Seçimi
if cfg.MODEL_TYPE == "transformer":
    from model import ActionTransformer as CurrentModel
elif cfg.MODEL_TYPE == "cnn":
    from model import CNNActionSpotter as CurrentModel
else:
    raise ValueError(f"Unknown MODEL_TYPE: {cfg.MODEL_TYPE}")


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        # raw state_dict kaydedilmiş olabilir
        return checkpoint
    raise RuntimeError("Checkpoint formatı beklenenden farklı (state_dict bulunamadı).")


def _infer_cnn_hparams_from_state_dict(state_dict):
    # DataParallel vb. prefix'leri için toleranslı arama
    def _find_key(suffix: str):
        if suffix in state_dict:
            return suffix
        for k in state_dict.keys():
            if isinstance(k, str) and k.endswith(suffix):
                return k
        return None

    k_key = _find_key("netvlad_past.cluster_weights")
    if not k_key:
        raise RuntimeError("Checkpoint CNNActionSpotter/NetVLAD anahtarlarını içermiyor.")
    k_clusters = int(state_dict[k_key].shape[0])

    proj_key = _find_key("input_proj.0.weight")
    if not proj_key:
        raise RuntimeError("Checkpoint input_proj ağırlıklarını içermiyor.")
    proj_dim = int(state_dict[proj_key].shape[0])
    feature_dim = int(state_dict[proj_key].shape[1])

    return {"k_clusters": k_clusters, "proj_dim": proj_dim, "feature_dim": feature_dim}

def load_model(checkpoint_path):
    print(f"📥 Model yükleniyor: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE)
    state_dict = _extract_state_dict(checkpoint)

    # Not: Model mimarisi config'ten okunuyor; ama config değişmişse checkpoint yüklenmez.
    # Bu yüzden CNN için gerekli hparam'ları checkpoint'ten okuyup cfg'yi override ediyoruz.
    if cfg.MODEL_TYPE == "cnn":
        inferred = _infer_cnn_hparams_from_state_dict(state_dict)

        if cfg.NETVLAD_CLUSTERS != inferred["k_clusters"]:
            print(
                f"⚠️  NETVLAD_CLUSTERS config={cfg.NETVLAD_CLUSTERS} ama checkpoint={inferred['k_clusters']}. "
                "Checkpoint değerine göre override ediyorum."
            )
        if cfg.PROJECTION_DIM != inferred["proj_dim"]:
            print(
                f"⚠️  PROJECTION_DIM config={cfg.PROJECTION_DIM} ama checkpoint={inferred['proj_dim']}. "
                "Checkpoint değerine göre override ediyorum."
            )
        if cfg.FEATURE_DIM != inferred["feature_dim"]:
            print(
                f"⚠️  FEATURE_DIM config={cfg.FEATURE_DIM} ama checkpoint={inferred['feature_dim']}. "
                "Checkpoint değerine göre override ediyorum."
            )

        cfg.NETVLAD_CLUSTERS = inferred["k_clusters"]
        cfg.PROJECTION_DIM = inferred["proj_dim"]
        cfg.FEATURE_DIM = inferred["feature_dim"]

        model = CurrentModel().to(cfg.DEVICE)
        model.load_state_dict(state_dict)
    else:
        model = CurrentModel().to(cfg.DEVICE)
        model.load_state_dict(state_dict)
        
    model.eval()
    return model

def predict(model, feature_path, threshold=0.5):
    # Features Yükle
    print(f"📂 Özellikler okunuyor: {feature_path}")
    features = np.load(feature_path, mmap_mode='r')
    total_frames = features.shape[0]
    
    predictions = []
    
    # Sliding Window
    # Adım boyutu (Stride) pencerenin yarısı olabilir veya 1 frame
    # 1 Frame en hassas olanıdır ama yavaştır.
    # Biz test için stride=FPS (1 saniye) kullanalım, production'da 1 frame.
    stride = cfg.FPS 
    window_half = cfg.WINDOW_SIZE_FRAMES // 2
    
    print("running inference...")
    for frame_idx in tqdm(range(window_half, total_frames - window_half, stride)):
        # Pencereyi kes
        start = frame_idx - window_half
        end = frame_idx + window_half
        
        # (Window, Feat) -> (1, Window, Feat)
        chunk = features[start:end].copy()
        tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(cfg.DEVICE)
        
        with torch.no_grad():
            cls_logits, reg_offset = model(tensor)
            
        # Softmax yerine Sigmoid (Multi-label)
        # Focal Loss ile eğittik, yani Binary Classification mantığı var.
        probs = torch.sigmoid(cls_logits).squeeze(0) # (18,)
        
        # En yüksek skorlu sınıf
        score, cls_id = torch.max(probs[:-1], dim=0) # Background'ı hariç tut (son sınıf)
        
        if score.item() > threshold:
            # Zaman Düzeltmesi (Regression)
            # Offset tahmini Normalize edilmiş [-0.5, +0.5] aralığında
            # Frame'e çevirmek için Window Size ile çarpmalıyız.
            predicted_shift_frames = reg_offset.item() * cfg.WINDOW_SIZE_FRAMES
            event_frame = frame_idx + predicted_shift_frames
            event_time = event_frame / cfg.FPS
            
            predictions.append({
                "time": event_time,
                "label": cfg.ID_TO_EVENT[cls_id.item()],
                "score": score.item(),
                "frame": frame_idx
            })
            
    return predictions

def nms(predictions, window_sec=10):
    """Basit NMS: Yakın olanlardan skoru düşük olanı sil"""
    if not predictions: return []
    
    # Skora göre sırala
    predictions.sort(key=lambda x: x['score'], reverse=True)
    
    keep = []
    while predictions:
        current = predictions.pop(0)
        keep.append(current)
        
        # Sadece AYNI SINIFTAN olan ve yakın olanları sil
        # Farklı sınıftan (örn: Gol ve Kart) ise silme.
        predictions = [
            p for p in predictions 
            if not (p['label'] == current['label'] and abs(p['time'] - current['time']) < window_sec)
        ]
        
    # Zamana göre sırala
    keep.sort(key=lambda x: x['time'])
    return keep

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    
    model = load_model(args.checkpoint)
    preds = predict(model, args.features)
    final_preds = nms(preds)
    
    print(f"\n🎯 TESPİT EDİLEN OLAYLAR ({len(final_preds)}):")
    print("-" * 50)
    for p in final_preds:
        m, s = divmod(int(p['time']), 60)
        print(f"⏰ {m:02d}:{s:02d} - {p['label']:<20} ({p['score']:.2f})")
    print("-" * 50)

import unittest
import numpy as np
try:
    from ultralytics import YOLO
    yolo_available = True
except ImportError:
    yolo_available = False

class TestYOLOModel(unittest.TestCase):
    @unittest.skipIf(not yolo_available, "Ultralytics YOLO kütüphanesi yüklü değil, atlanıyor.")
    def test_yolo_initialization_and_inference(self):
        # Varsayılan YOLOv11n ya da v8n modelinin yüklenmesi (Eğer yoksa indirir)
        model = YOLO('yolo11n.pt') 
        self.assertIsNotNone(model)

        # Boş bir görüntü oluştur (örn. 640x640 rgb)
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Modele boş görüntüde tahmin yaptır (Crash olup olmadığını test etmek için)
        results = model(dummy_image, verbose=False)
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 1)

if __name__ == "__main__":
    unittest.main()

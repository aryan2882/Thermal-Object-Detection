# 🔍 Tiny Object-Aware Multi-Stage Blockwise Framework for Thermal Object Detection Using EfficientDet

## 👥 Team Members
- **Aryan Raj** – VIT Chennai, aryan.raj2022@vitstudent.ac.in  
- **Tridib Chatterjee** – VIT Chennai, tridib.chatterjee2022@vitstudent.ac.in  
- **Parth Khairnar** – VIT Chennai, parth.khairnar2022@vitstudent.ac.in  

---

## 🧰 Major Frameworks / Tools Used 
- **EfficientDet** (Object Detection backbone)  
- **SRCNN & ESRGAN** (Super-resolution models)  
- **FLIR ADAS Thermal Dataset**  
- **COCO Format Annotation Tools**  

---

## 📂 Dataset Download
➡️ **[Download the FLIR ADAS Thermal Dataset here](PASTE-YOUR-LINK-HERE)**  
(contains 10,742 training and 1,145 testing thermal images, annotated for 15 object classes)


---

## ⚙️ Steps to Run

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/aryan2882/Thermal-Object-Detection.git
   cd Thermal-Object-Detection

2. **Navigate into the Project Directory**  
   ```bash
   cd YOUR_REPO
3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
4. **Download & Place Dataset**
   Place the FLIR dataset inside the data/ folder (create it if it doesn’t exist).

5. **Run Preprocessing & Filtering Stage**

   ```bash
   python stage1_filtering.py
6. **Train the EfficientDet-based Detector**

   ```bash
   python train_detector.py

🧪 Results

Model	Person mAP	Bike mAP	Car mAP	Overall mAP
ThermalDet	78.2	60.0	85.5	74.6
Ours (EfficientDet-D3 + Enhancements)	81.2	64.0	86.5	77.3
Tiny Object AP improved from 64.6% to 77.3%

Achieves real-time inference on NVIDIA RTX 3090 (≈100ms/frame)

🔁 Sample Input and Output
🔹 Input

🔸 Output

Bounding boxes shown for pedestrians, vehicles, and other classes with confidence scores. Tiny and partially occluded objects detected using our enhanced pipeline.

📌 Citation
If you use this work in your research, please cite our paper:

bibtex
Copy
Edit
@inproceedings{raj2025thermaldet,
  title={Tiny Object-Aware Multi-Stage Blockwise Framework for Thermal Object Detection Using EfficientDet},
  author={Aryan Raj and Tridib Chatterjee and Parth Khairnar},
  booktitle={Conference on [Name of Conference]},
  year={2025}
}

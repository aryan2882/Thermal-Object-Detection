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
➡️ **[Download the FLIR ADAS Thermal Dataset here](https://adas-dataset-v2.flirconservator.com/#downloadguide)**  
(contains 10,742 training and 1,145 testing thermal images, annotated for 15 object classes)


---

## ⚙️ Steps to Run

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/aryan2882/Thermal-Object-Detection.git

2. **Navigate into the Project Directory**  
   ```bash
   cd Thermal-Object-Detection
3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
4. **Download & Place Dataset**
   Place the FLIR dataset inside the coco/ folder (create it if it doesn’t exist).

5. **Run Preprocessing & Filtering Stage**

   ```bash
   python normalize.py

6. **Downloads pretrained weights and put it in weights folder**
   **[Download link](https://drive.google.com/drive/folders/1_KQM_ZiAGLuCRBtYWQUJVgA9wQ2gOPo_?usp=sharing)**

7. **Put some pictures from validation folder to data/sample folder**
   
8. **Train the EfficientDet-based Detector**

  
   ```bash
   python detect.py --data data/custom.data --cfg cfg/yolov3-spp-r.cfg --weights weights/best.pt

9. **The result with annotations will be in output folder.**

## 🧪 **Results**

We compared our model against ThermalDet on the FLIR test set across key object classes:

### 📊 Performance Comparison (mAP @ IoU=0.5)

| Model                                     | Person mAP   | Bike mAP   | Car mAP   | Overall mAP   |
|-------------------------------------------|--------------|------------|-----------|---------------|
| ThermalDet                                | 78.2%        | 60.0%      | 85.5%     | 74.6%         |
| **Ours (EfficientDet-D3 + Enhancements)** | **81.2%**    | **64.0%**  | **86.5%** | **77.3%**     |

### 🚀 Highlights
- 🔍 **Tiny Object AP** increased from **64.6% ➜ 77.3%**
- 🧠 Enhanced performance through:
  - Super-resolution preprocessing
  - Adaptive anchor box tuning




## 🔁 Sample Input and Output

### 🔹 Input
<img src="samples/input.jpg" alt="Input Thermal Image" width="400"/>

### 🔸 Output
<img src="samples/output.jpg" alt="Detected Output" width="400"/>

Bounding boxes are drawn for pedestrians, vehicles, and other classes with confidence scores.  
Our enhanced pipeline detects tiny and partially occluded objects effectively, even in low-visibility conditions.

---

## 📌 Citation

If you use this work in your research or projects, please consider citing:

```bibtex
@inproceedings{raj2025thermaldet,
  title={Tiny Object-Aware Multi-Stage Blockwise Framework for Thermal Object Detection Using EfficientDet},
  author={Aryan Raj and Tridib Chatterjee and Parth Khairnar},
  year={2025}
}

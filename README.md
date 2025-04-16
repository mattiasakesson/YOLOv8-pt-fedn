# YOLOv8 Re-Implementation Using PyTorch + FEDn (Fork)

This repository is a **fork** of [jahongir7174/YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt), a PyTorch re-implementation of YOLOv8.

This version integrates support for **FEDn** (a federated learning framework).  
Modifications may have been made — please refer to the commit history for details.  
All code is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**, in accordance with the original project.

---

## 🔧 Installation

Clone this repository and navigate to its root directory:

```bash
git clone <your-repo-url>
cd <repo-folder>
```

Create and activate a Python virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📁 Preprocessing

1. **Download the Wisard dataset with environment splits**  
   _TODO: Add instructions or download link here._

2. **Generate `train.txt` for each client**:

```bash
python create_wisard_text_complete.py
```

---

## 🧪 Set Up Studio Project (FEDn)

To train the model in a federated setting, create a studio project using FEDn.

Follow the official guide:  
👉 [How to create a FEDn project](https://fedn.scaleoutsystems.com/)

---

## 🚀 Start FEDn Client

Set the following environment variables in your terminal or kernel:

```bash
export PROJECT_URL=<REPLACE-WITH-YOUR-PROJECT-URL>
export FEDN_AUTH_TOKEN=<REPLACE-WITH-YOUR-AUTH-TOKEN>
export DATA_PATH=<REPLACE-WITH-YOUR-LOCAL-DATASET-PATH>
```

Then run:

```bash
python main_fedn.py
```

---

## 📚 References

- https://github.com/jahongir7174/YOLOv8-pt
- https://github.com/ultralytics/yolov5
- https://github.com/ultralytics/ultralytics
- https://github.com/scaleoutsystems/fedn
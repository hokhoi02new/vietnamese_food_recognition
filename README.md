# 🍜 Vietnamese Traditional Food Recognition

### Introduction
This project focuses on building a system to classify and recognize traditional Vietnamese dishes using deep learning and computer vision techniques. The goal is to support applications in tourism, restaurants, and enhance user experience.

---

### Description, Feature
- Recognition of popular Vietnamese dishes (Pho, Bun Bo Hue, Goi Cuon,……).  
- Using data preprocessing and data augmentation to improve performance. Fine-tuning several pretrained models such as InceptionV3, MobileNetV2, Vision Transformer (ViT), build a custom CNN models with residual block (resnet style). Achieve 82.4% accuracy with fine-tune ViT.
- API/UI: RESTful API built with FastAPI and a simple UI for image upload and prediction. 
- Deployment: ONXX, Docker, cloud platform Render 
---

### Project Structure
```
vietnamese_food_recognition/
│── app/                # API and UI
│   ├── app_UI.py       # UI
│   ├── main.py         # API backend (FastAPI)
│── config/             # config file
│── data/               # dataset 
│── logs/               # training logs
│   ├── training_log.csv
│── result/             # model evaluation results
│   ├── customCNN_eval.csv
│   ├── inception_v3.csv
│   ├── mobilenet_v2.csv
│   ├── ViT.csv
│── saved_models/       # saved models
│── src/                
│   ├── data_loader.py  # data loading & preprocessing
│   ├── evaluate.py     # evaluate model
│   ├── inference.py    # inference 
│   ├── models.py       # define model
│   ├── train.py        # training script
│   ├── utils.py        # utility functions (logging, plotting....)
│── test/               # tests API
│   ├── uni_test.py
├── convert_to_onxx.py  #script convert model to ONXX format for deployment
│── uploads/            # folder server: uploaded images for prediction
│── requirements.txt
│── requirements-train-local.txt   
│── Dockerfile          # docker file
│── render.yaml         # file config deploy for Render cloud platform
│── LICENSE
│── README.md


```

---

### Usage 

#### Clone the repository
```
git clone https://github.com/yourusername/vietnamese-food-recognition.git
cd vietnamese-food-recognition
```

#### Install dependencies
```
pip install -r requirements-train-local.txt
```

#### Training
Training model
```
python -m src.train --model mobilenet_v2 --epochs 20 --batch_size 32
```

Arguments:
- `--model`: choose from `customCNN | inception_v3 | mobilenet_v2 | vit`  
- `--epochs`: number of training epochs  
- `--batch_size`: batch size for training  

Trained models will be saved in `saved_models/` and logs in `logs/`.

---

#### Evaluation
Evaluate trained model:
```bash
python src/evaluate.py --model mobilenet_v2 --test_dir data/test --output_dir result
```

Results will be stored in the `result` directory.

---

#### Inference (Prediction)
Make predictions on new images:
```bash
python src/inference.py --image <path_to_image>.jpg --model mobilenet_v2 
```

---

#### Running API & UI
Start the API and Web UI:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000     # Start API
streamlit run app/app_UI.py    # Start web UI
```

Open your browser, API run at: http://127.0.0.1:8000 (local)

---
#### Test API
```bash
pytest test/uni_test.py
```

---
#### Docker 
Build and run with Docker:
```bash
docker build -t vietnamese-food-api .
docker run -p 8000:8000 vietnamese-food-api
```

---

## Deploy on Cloud platform (Render)
API is live at: https://vietnamese-food-recognition-12.onrender.com/docs




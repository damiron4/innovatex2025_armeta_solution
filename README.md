# Solution for innovatex2025_armeta

## Installation

Install all requirements using Python 3.10. Also, install the torch (cuda version) before the requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Datasets

### Signatures
```bash
https://huggingface.co/datasets/tech4humans/signature-detection
```
### Stamps
```bash
https://universe.roboflow.com/swp-3jks1/stamp-shape
```
### QR code

There weren't any good dataset, so we donwloaded different documents in PDF format
```bash 
https://www.kaggle.com/datasets/manisha717/dataset-of-pdf-files
```
Then we randomly took PDFs and artificially inserted 1 to N QR codes on the pages and got ~1000 samples



## Usage 

If you want to reproduce the whole data collection and training, you need to download all datasets and run notebooks in the following order

### Prepare the data 

- yolo/qr_dataset.ipynb
- yolo/stamp_data_filter.ipynb
- yolo/convert_signature_to_yolo.ipynb
- yolo/merge_datasets.ipynb

### Run the training

```bash
python yolo/train_yolo9.py 
``` 

## Inference

```bash
python yolo/inference_yolo9.py <path_to_your_test_data> 
``` 

The results will appear in ```./yolo/inference_results ``` 

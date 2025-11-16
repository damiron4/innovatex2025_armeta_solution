import os
from ultralytics import YOLO
from pathlib import Path
import cv2
from pdf2image import convert_from_path
import numpy as np
import json
import time


MODEL_PATH = "runs_yolov9/signature_stamp_qr3/weights/best.pt"

INPUT_SOURCE = "C:\\Users\\user\\Desktop\\My_Projects\\innovatex2025_armeta_solution\\merged_pdfs_test"  

OUTPUT_DIR = "inference_results"

CONF_THRESHOLD = 0.2  
IOU_THRESHOLD = 0.55   
IMG_SIZE = 640         

CLASS_NAMES = {
    0: 'signature',
    1: 'stamp',
    2: 'qr'
}


def convert_pdf_to_images(pdf_path, dpi=200):
    """
    Convert PDF pages to images
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for conversion (higher DPI = better quality for small details like signatures)
    
    Returns:
        List of numpy arrays (images)
    """
    try:
        pages = convert_from_path(pdf_path, dpi=200)
        images = [np.array(page.convert('RGB')) for page in pages]
        return images
    except Exception as e:
        print(f"️ Error converting PDF {pdf_path}: {e}")
        return []


def resize_to_640(image):
    """
    Resize image to 640x640 with padding (letterbox)
    
    Args:
        image: Input image (numpy array)
    
    Returns:
        Tuple of (resized_image, scale, padding)
        - resized_image: 640x640 image with letterbox padding
        - scale: scaling factor used
        - pad_x, pad_y: padding added (left/top)
    """
    h, w = image.shape[:2]
    target = 640
    
    scale = target / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.ones((target, target, 3), dtype=np.uint8) * 114  
    
    pad_y = (target - new_h) // 2
    pad_x = (target - new_w) // 2
    
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    
    return canvas, scale, pad_x, pad_y


def unscale_bbox(bbox, scale, pad_x, pad_y):
    """
    Convert bbox from 640x640 coordinates back to original image coordinates
    
    Args:
        bbox: [x1, y1, x2, y2] in 640x640 image
        scale: scaling factor
        pad_x, pad_y: padding added
    
    Returns:
        [x1, y1, x2, y2] in original coordinates
    """
    x1, y1, x2, y2 = bbox
    
    x1 = x1 - pad_x
    y1 = y1 - pad_y
    x2 = x2 - pad_x
    y2 = y2 - pad_y
    
    x1 = x1 / scale
    y1 = y1 / scale
    x2 = x2 / scale
    y2 = y2 / scale
    
    return [x1, y1, x2, y2]


def process_pdf(pdf_path, model, output_dir, json_data, annotation_counter):
    """
    Process a single PDF file - convert to images and run inference
    
    Args:
        pdf_path: Path to PDF file
        model: YOLO model
        output_dir: Directory to save results
        json_data: Dictionary to store JSON results
        annotation_counter: Counter for annotation IDs
    
    Returns:
        Tuple of (list of results, updated annotation counter, timing_info)
    """
    pdf_filename = Path(pdf_path).name  
    pdf_stem = Path(pdf_path).stem
    print(f"\n Processing PDF: {pdf_stem}")
    
    pdf_start_time = time.time()
    
    images = convert_pdf_to_images(pdf_path)
    
    if not images:
        print(f"️ No pages extracted from {pdf_stem}")
        return [], annotation_counter, {"total_time": 0, "page_times": []}
    
    print(f"   Extracted {len(images)} pages")
    
    pdf_output_dir = Path(output_dir) / pdf_stem
    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    
    json_data[pdf_filename] = {}
    
    all_results = []
    page_times = []
    
    for page_idx, img in enumerate(images):
        page_num = page_idx + 1
        page_key = f"page_{page_num}"
        
        page_start_time = time.time()
        
        orig_height, orig_width = img.shape[:2]
        
        img_resized, scale, pad_x, pad_y = resize_to_640(img)
        
        results = model.predict(
            source=img_resized,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )[0]
        
        annotated_img = results.plot()
        output_path = pdf_output_dir / f"page_{page_num}.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        
        boxes = results.boxes
        annotations = []
        
        if len(boxes) > 0:
            print(f"   Page {page_num}: Found {len(boxes)} objects")
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                class_name = CLASS_NAMES.get(cls, f"class_{cls}")
                print(f"      - {class_name}: {conf:.2%}")
                
                orig_xyxy = unscale_bbox(xyxy, scale, pad_x, pad_y)
                x1, y1, x2, y2 = orig_xyxy
                
                # Use absolute pixel coordinates
                x = float(x1)
                y = float(y1)
                width = float(x2 - x1)
                height = float(y2 - y1)
                area = width * height
                
                annotation_id = f"annotation_{annotation_counter}"
                annotation_counter += 1
                
                annotation_entry = {
                    annotation_id: {
                        "category": class_name,
                        "bbox": {
                            "x": round(x, 2),
                            "y": round(y, 2),
                            "width": round(width, 2),
                            "height": round(height, 2)
                        },
                        "area": round(area, 2)
                    }
                }
                annotations.append(annotation_entry)
        else:
            print(f"   Page {page_num}: No objects detected")
        
        if annotations:
            json_data[pdf_filename][page_key] = {
                "annotations": annotations,
                "page_size": {
                    "width": orig_width,
                    "height": orig_height
                }
            }
        
        page_end_time = time.time()
        page_time = page_end_time - page_start_time
        page_times.append(page_time)
        
        all_results.append(results)
    
    pdf_end_time = time.time()
    pdf_total_time = pdf_end_time - pdf_start_time
    
    timing_info = {
        "total_time": pdf_total_time,
        "page_times": page_times,
        "num_pages": len(images)
    }
    
    print(f"    Results saved to: {pdf_output_dir}")
    print(f"   ️  Total time: {pdf_total_time:.2f}s | Avg per page: {pdf_total_time/len(images):.2f}s")
    return all_results, annotation_counter, timing_info



def run_inference(source, save=True, show=False):
    """
    Run inference on images/video/folder
    
    Args:
        source: Path to image, folder, or video
        save: Save results with bounding boxes
        show: Display results in window
    """
    print(f" Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    print(f" Running inference on: {source}")
    
    results = model.predict(
        source=source,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        save=save,
        show=show,
        project=OUTPUT_DIR,
        name="detect",
        exist_ok=True,
        line_width=2,
        show_labels=True,
        show_conf=True,
    )
    
    for idx, result in enumerate(results):
        boxes = result.boxes
        
        if len(boxes) > 0:
            print(f"\n Image {idx + 1}: {result.path}")
            print(f"   Found {len(boxes)} objects:")
            
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                class_name = CLASS_NAMES.get(cls, f"class_{cls}")
                print(f"   - {class_name}: {conf:.2%} at [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")
        else:
            print(f"\n Image {idx + 1}: No objects detected")
    
    print(f"\n Inference complete! Results saved to: {OUTPUT_DIR}/detect")
    return results


def run_inference_single_image(image_path):
    """
    Run inference on a single image and return results
    """
    print(f" Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    results = model.predict(
        source=image_path,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False
    )[0]
    
    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()
        
        detections.append({
            'class': cls,
            'class_name': CLASS_NAMES.get(cls, f"class_{cls}"),
            'confidence': conf,
            'bbox': [float(x) for x in xyxy],  
        })
    
    return detections


def run_inference_batch(folder_path):
    """
    Run inference on all images in a folder
    """
    print(f" Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    folder = Path(folder_path)
    images = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]
    
    print(f" Found {len(images)} images in {folder_path}")
    
    results = model.predict(
        source=folder_path,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        save=True,
        project=OUTPUT_DIR,
        name="batch_detect",
        exist_ok=True,
    )
    
    print(f"\n Batch inference complete! Results saved to: {OUTPUT_DIR}/batch_detect")
    return results


def run_inference_pdfs(folder_path):
    """
    Run inference on all PDFs in a folder
    
    Args:
        folder_path: Path to folder containing PDF files
    
    Returns:
        Dictionary with PDF names as keys and results as values
    """
    overall_start_time = time.time()
    
    print(f" Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    folder = Path(folder_path)
    pdf_files = list(folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"️ No PDF files found in {folder_path}")
        return {}
    
    print(f" Found {len(pdf_files)} PDF files")
    
    output_dir = Path(OUTPUT_DIR) / "pdf_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    json_data = {}
    annotation_counter = 1
    timing_data = {}
    all_page_times = []
    total_pages = 0
    
    for pdf_file in pdf_files:
        results, annotation_counter, timing_info = process_pdf(pdf_file, model, output_dir, json_data, annotation_counter)
        all_results[pdf_file.name] = results
        timing_data[pdf_file.name] = timing_info
        all_page_times.extend(timing_info["page_times"])
        total_pages += timing_info["num_pages"]
    
    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time
    
    # Calculate statistics
    avg_time_per_pdf = overall_time / len(pdf_files) if pdf_files else 0
    avg_time_per_page = sum(all_page_times) / len(all_page_times) if all_page_times else 0
    
    
    json_output_path = output_dir / "inference_results.json"
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f" All PDFs processed!")
    print(f" Images saved to: {output_dir}")
    print(f" JSON saved to: {json_output_path}")
    print(f"\n️  TIMING STATISTICS:")
    print(f"   Total time: {overall_time:.2f}s")
    print(f"   Total PDFs: {len(pdf_files)}")
    print(f"   Total pages: {total_pages}")
    print(f"   Average per PDF: {avg_time_per_pdf:.2f}s")
    print(f"   Average per page: {avg_time_per_page:.2f}s")
    print(f"{'='*60}")
    
    return all_results



def main():
    """
    Main function - modify INPUT_SOURCE to run inference
    """
    import sys
    
    if len(sys.argv) > 1:
        input_source = sys.argv[1]
    else:
        input_source = INPUT_SOURCE
    
    if not os.path.exists(input_source):
        print(f" Error: {input_source} not found!")
        print(f"\nUsage:")
        print(f"  python inference_yolo9.py <path_to_image_or_folder_or_pdf>")
        print(f"\nOr modify INPUT_SOURCE in the script")
        return
    
    input_path = Path(input_source)
    
    if input_path.is_file():
        if input_path.suffix.lower() == '.pdf':
            print(f" Loading model: {MODEL_PATH}")
            model = YOLO(MODEL_PATH)
            output_dir = Path(OUTPUT_DIR) / "pdf_results"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            json_data = {}
            annotation_counter = 1
            results, annotation_counter, timing_info = process_pdf(input_path, model, output_dir, json_data, annotation_counter)
            
            # Add timing statistics for single PDF
            
            
            json_output_path = output_dir / "inference_results.json"
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"\n JSON saved to: {json_output_path}")
            print(f"\n TIMING: {timing_info['total_time']:.2f}s total, {timing_info['total_time']/timing_info['num_pages']:.2f}s per page")
        else:
            run_inference(input_source, save=True, show=False)
    
    elif input_path.is_dir():
        pdf_files = list(input_path.glob("*.pdf"))
        if pdf_files:
            run_inference_pdfs(input_source)
        else:
            run_inference(input_source, save=True, show=False)
    
    else:
        print(f" Invalid input: {input_source}")


if __name__ == "__main__":
    main()

# import ultralytics
# import cv2
# from ultralytics import YOLO
# import json
# import numpy as np
# from paddleocr import PaddleOCR


# class object:
#     def __init__(self,cls,x1,y1,x2,y2):
#         self.cls=cls
#         self.x1=x1
#         self.y1=y1
#         self.x2=x2
#         self.y2=y2
        

# def CROP(original_img, final_detection_list):
#     cropped_B_BBox = []
    
#     for object_element in final_detection_list:
#         conf = object_element[0]
#         cls = object_element[1]
#         x1, y1 = object_element[2][0]
#         x2, y2 = object_element[2][1]
        
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
#         h, w = original_img.shape[:2]
#         x1 = max(0, min(x1, w))
#         y1 = max(0, min(y1, h))
#         x2 = max(0, min(x2, w))
#         y2 = max(0, min(y2, h))
        
#         cropped_img = original_img[y1:y2, x1:x2]
        
#         if cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
#             cropped_B_BBox.append(cropped_img)
    
#     return cropped_B_BBox
        

# def B_Box_extracion(img):
#     output = model.predict(img)
#     print(output)
#     objects = []
#     print("**" * 10)
#     for results in output:
#         print("__ITRATION__")
#         boxes = results.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0].tolist()
#             conf = float(box.conf[0])
#             cls = int(box.cls[0])
#             print(f"Class: {cls}, Confidence: {conf:.2f}, BBox: ({x1}, {y1}, {x2}, {y2})")
#             bounding_box = [conf, cls, ((x1, y1), (x2, y2))] 
#             objects.append(bounding_box)
    
#     filtered_objects = overlap_removal(objects)
#     print(f"After overlap removal: {len(filtered_objects)}")
    
#     return filtered_objects  # ADD THIS LINE
            
            
# def overlap_removal(objects_list):
#     if len(objects_list) == 0:
#         return []
    
#     sorted_objects = sorted(objects_list, key=lambda x: x[0], reverse=True)
#     filtered_objects = []
    
#     for current_obj in sorted_objects:
#         keep = True
#         current_conf, current_cls, current_coords = current_obj
#         x1_curr, y1_curr = current_coords[0]
#         x2_curr, y2_curr = current_coords[1]
        
#         for kept_obj in filtered_objects:
#             kept_conf, kept_cls, kept_coords = kept_obj
            
#             if current_cls != kept_cls:
#                 continue
            
#             x1_kept, y1_kept = kept_coords[0]
#             x2_kept, y2_kept = kept_coords[1]
            
#             x1_inter = max(x1_curr, x1_kept)
#             y1_inter = max(y1_curr, y1_kept)
#             x2_inter = min(x2_curr, x2_kept)
#             y2_inter = min(y2_curr, y2_kept)
            
#             inter_width = max(0, x2_inter - x1_inter)
#             inter_height = max(0, y2_inter - y1_inter)
#             inter_area = inter_width * inter_height
            
#             curr_area = (x2_curr - x1_curr) * (y2_curr - y1_curr)
#             kept_area = (x2_kept - x1_kept) * (y2_kept - y1_kept)
#             union_area = curr_area + kept_area - inter_area
            
#             if union_area > 0:
#                 iou = inter_area / union_area
                
#                 if iou > 0.5:
#                     keep = False
#                     break
        
#         if keep:
#             filtered_objects.append(current_obj)
    
#     return filtered_objects
    
    
# def text_Extraction(Filtered_list):
#     output = []
#     for i, image_np in enumerate(Filtered_list):
#         try:
#             # Call the predict method directly (no extra parameters)
#             results = ocr.predict(image_np)
            
#             if results and results[0]:
#                 for line in results[0]:
#                     text = line[1][0]       # extracted text
#                     conf = line[1][1]       # confidence
#                     output.append({"text": text, "confidence": conf})
#                     print(f"Crop {i}: {text} (conf: {conf:.2f})")
#             else:
#                 print(f"Crop {i}: No text detected")
#         except Exception as e:
#             print(f"Error processing crop {i}: {e}")
#             continue

#     return output
 

# if __name__=="__main__":
#     model = YOLO("D:\\Degendorf Institute of Technology\\Semster 1\\Machine vision\\Project\\Egyptian National ID Information Extraction Using YOLO and OCR\\YOLO-ID-Extraction-main\\model_Parameters\\best.pt")
#     ocr = PaddleOCR(lang='ar')
#     img = cv2.imread("D:\\Degendorf Institute of Technology\\Semster 1\\Machine vision\\Project\\Egyptian National ID Information Extraction Using YOLO and OCR\\YOLO-ID-Extraction-main\\TEST_ID.jpg")
#     filtered_bbox_list = B_Box_extracion(img)
#     filtered_cropped = CROP(img, filtered_bbox_list)
    
#     # Extract text from cropped images
#     extracted_text = text_Extraction(filtered_cropped)
    
#     # Print results
#     print("\nExtracted Text:")
#     for item in extracted_text:
#         print(f"Text: {item['text']}, Confidence: {item['confidence']:.2f}")
#/////////////////////////

import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_pir_apply_shape_optimization_pass"] = "0"

import cv2
from ultralytics import YOLO
import numpy as np
from paddleocr import PaddleOCR

CLASS_NAMES = {
    0: "Code",
    1: "Image",
    2: "City",
    3: "Family Name",
    4: "Name",
    5: "Neighborhood",
    6: "Number",
    7: "State"
}

# ── Bounding-box detection ────────────────────────────────────────────────────

def B_Box_extraction(img):
    output = model.predict(img)
    objects = []
    for results in output:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            print(f"  Class: {cls}, Conf: {conf:.2f}, BBox: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
            objects.append([conf, cls, ((x1, y1), (x2, y2))])
    filtered = overlap_removal(objects)
    print(f"  After overlap removal: {len(filtered)} objects kept")
    return filtered


# ── NMS ───────────────────────────────────────────────────────────────────────

def overlap_removal(objects_list, iou_threshold=0.9):
    if not objects_list:
        return []
    sorted_objects = sorted(objects_list, key=lambda x: x[0], reverse=True)
    filtered = []
    for current in sorted_objects:
        curr_conf, curr_cls, curr_coords = current
        cx1, cy1 = curr_coords[0]
        cx2, cy2 = curr_coords[1]
        keep = True
        for kept in filtered:
            _, kept_cls, kept_coords = kept
            if curr_cls != kept_cls:
                continue
            kx1, ky1 = kept_coords[0]
            kx2, ky2 = kept_coords[1]
            ix1, iy1 = max(cx1, kx1), max(cy1, ky1)
            ix2, iy2 = min(cx2, kx2), min(cy2, ky2)
            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            curr_area  = (cx2 - cx1) * (cy2 - cy1)
            kept_area  = (kx2 - kx1) * (ky2 - ky1)
            union_area = curr_area + kept_area - inter_area
            if union_area > 0 and (inter_area / union_area) > iou_threshold:
                keep = False
                break
        if keep:
            filtered.append(current)
    return filtered


# ── Crop ──────────────────────────────────────────────────────────────────────

def CROP(original_img, final_detection_list):
    h, w = original_img.shape[:2]
    cropped = []
    for obj in final_detection_list:
        _conf, _cls, coords = obj
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x1 = int(max(0, min(x1, w)))
        y1 = int(max(0, min(y1, h)))
        x2 = int(max(0, min(x2, w)))
        y2 = int(max(0, min(y2, h)))
        crop = original_img[y1:y2, x1:x2]
        if crop.shape[0] > 0 and crop.shape[1] > 0:
            cropped.append((_cls, crop))
    return cropped


# ── OCR ───────────────────────────────────────────────────────────────────────

def text_Extraction(cropped_list):
    output = []

    for i, (cls, image_np) in enumerate(cropped_list):
        try:
            results = ocr.predict(image_np)
            found_any = False

            if results:
                for res in results:
                    # ── Method 1: dict with 'rec_texts' key (most common in v3) ──
                    if isinstance(res, dict):
                        texts  = res.get('rec_texts', [])
                        scores = res.get('rec_scores', [1.0] * len(texts))
                        for text, score in zip(texts, scores):
                            output.append({
                                "class": cls,
                                "field": CLASS_NAMES.get(cls, f"Class_{cls}"),
                                "text": text,
                                "confidence": float(score)
                            })
                            print(f"  Crop {i}: '{text}'  (conf: {float(score):.2f})")
                            found_any = True

                    # ── Method 2: result object with .json attribute ──────────
                    elif hasattr(res, 'json'):
                        j = res.json  # plain dict
                        # Pipeline result: j = {'res': {'rec_texts': [...], 'rec_scores': [...]}}
                        inner = j.get('res', j)
                        texts  = inner.get('rec_texts', [])
                        scores = inner.get('rec_scores', [1.0] * len(texts))
                        # Single-line result: j = {'res': {'rec_text': '...', 'rec_score': 0.99}}
                        if not texts and 'rec_text' in inner:
                            texts  = [inner['rec_text']]
                            scores = [inner.get('rec_score', 1.0)]
                        for text, score in zip(texts, scores):
                            output.append({
                                "class": cls,
                                "field": CLASS_NAMES.get(cls, f"Class_{cls}"),
                                "text": text,
                                "confidence": float(score)
                            })
                            print(f"  Crop {i}: '{text}'  (conf: {float(score):.2f})")
                            found_any = True

                    # ── Method 3: object with direct attributes ───────────────
                    elif hasattr(res, 'rec_texts') and res.rec_texts:
                        for text, score in zip(res.rec_texts, res.rec_scores):
                            output.append({
                                "class": cls,
                                "field": CLASS_NAMES.get(cls, f"Class_{cls}"),
                                "text": text,
                                "confidence": float(score)
                            })
                            print(f"  Crop {i}: '{text}'  (conf: {float(score):.2f})")
                            found_any = True

                    # ── Debug: print unknown structure so we can see it ───────
                    else:
                        print(f"  Crop {i}: unknown result type {type(res)} — {dir(res)}")

            if not found_any:
                print(f"  Crop {i}: No text detected")

        except Exception as e:
            print(f"  Crop {i}: Error — {e}")

    return output


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    MODEL_PATH = (
        "D:\\Degendorf Institute of Technology\\Semster 1\\Machine vision\\"
        "Project\\Egyptian National ID Information Extraction Using YOLO and OCR\\"
        "YOLO-ID-Extraction-main\\model_Parameters\\best.pt"
    )
    IMAGE_PATH = (
        "D:\\Degendorf Institute of Technology\\Semster 1\\Machine vision\\"
        "Project\\Egyptian National ID Information Extraction Using YOLO and OCR\\"
        "YOLO-ID-Extraction-main\\TEST_ID2.jpg"
    )

    model = YOLO(MODEL_PATH)

    ocr = PaddleOCR(
        lang='ar',
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device='cpu',
        engine='onnxruntime',
    )

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

    print("=" * 55)
    print("Step 1 — YOLO detection")
    print("=" * 55)
    filtered_bbox_list = B_Box_extraction(img)

    print("\n" + "=" * 55)
    print("Step 2 — Cropping")
    print("=" * 55)
    filtered_cropped = CROP(img, filtered_bbox_list)
    print(f"  {len(filtered_cropped)} valid crop(s) produced")

    print("\n" + "=" * 55)
    print("Step 3 — Arabic OCR")
    print("=" * 55)
    extracted_text = text_Extraction(filtered_cropped)

    print("\n" + "=" * 55)
    print("Final Extracted Text")
    print("=" * 55)
    if extracted_text:
        print("\n" + "=" * 60)
        print("EGYPTIAN ID INFORMATION")
        print("=" * 60)

        # Keep only best confidence for each field
        final_results = {}

        for item in extracted_text:

            field = item["field"]

            if field not in final_results:
                final_results[field] = item

            elif item["confidence"] > final_results[field]["confidence"]:
                final_results[field] = item

        # for field, item in final_results.items():

        #     print(
        #         f"{field:<15}: "
        #         f"{item['text']}"
        #     )

        with open("output.txt", "w", encoding="utf-8") as f:

            f.write("EGYPTIAN ID INFORMATION\n")
            f.write("=" * 40 + "\n")

            for field, item in final_results.items():

                line = f"{field:<15}: {item['text']}\n"

                print(line, end="")
                f.write(line)

        print("\nResults saved to output.txt")

    else:
        print("No text was extracted.")
        
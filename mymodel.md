---

## 📦 Download Pre-trained Model

You can download the pre-trained YOLOv8 model used in this project from the link below:

🔗 [Download mymodel.pt from Google Drive](https://drive.google.com/file/d/1Ho7zi_UJFrgC0zxBLQyoOygy--_zpUzX/view?usp=sharing)

📁 **Save it as:** `mymodel.pt`  
📂 **Location:** Put this file in the **root project folder**, like this:

```
project/
├── imgid_folder.py
├── imgid_final.py
├── mymodel.pt           👈 [Place it here after download]
├── requirements.txt
├── README.md
├── final_outputs/
├── idcard_outputs/
```

Once downloaded and placed correctly, it will work **out of the box** with both:
- `imgid_folder.py` – batch processing of images
- `imgid_final.py` – one-image processing with form auto-fill

This model is trained specifically for the OCR ID card recognition task described in this project.


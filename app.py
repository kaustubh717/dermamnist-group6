"""Group 6 – DermaMNIST skin lesion classifier (Gradio app).

Run:  python app.py      -> opens http://127.0.0.1:7860 in the browser
Needs model_group6.pt in the same folder.
"""
import os

import gradio as gr
import numpy as np
import torch
from PIL import Image

MODEL_PATH = "model_group6.pt"
MEAN = [0.7631, 0.5381, 0.5614]    # same normalisation as in training (starter notebook)
STD = [0.1366, 0.1543, 0.1692]
CLASSES = ["actinic keratoses", "basal cell carcinoma", "benign keratosis", "dermatofibroma",
           "melanoma", "melanocytic nevi", "vascular lesions"]
INFO = {
    "actinic keratoses": ("pre-cancerous", "Rough, scaly patch from sun damage; can develop into skin cancer."),
    "basal cell carcinoma": ("malignant", "The most common skin cancer; grows slowly, rarely spreads."),
    "benign keratosis": ("benign", "Harmless growth, e.g. seborrhoeic keratosis or solar lentigo."),
    "dermatofibroma": ("benign", "Small, firm, harmless nodule in the skin."),
    "melanoma": ("malignant", "The most dangerous skin cancer; early detection is critical."),
    "melanocytic nevi": ("benign", "A common mole; usually harmless."),
    "vascular lesions": ("benign", "Blood-vessel lesions such as angiomas; usually harmless."),
}

model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()


def preprocess(img):
    """Any uploaded image -> normalised tensor (1, 3, 28, 28), exactly like the training data."""
    small = img.convert("RGB").resize((28, 28), Image.BILINEAR)
    x = torch.tensor(np.array(small)).permute(2, 0, 1).float() / 255
    x = (x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)
    return x.unsqueeze(0), small


def classify(img):
    if img is None:
        return None, None, "Upload an image or click an example."
    x, small = preprocess(img)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top3 = probs.topk(3)
    result = {CLASSES[i]: float(p) for p, i in zip(top3.values, top3.indices)}

    best = CLASSES[top3.indices[0]]
    kind, text = INFO[best]
    risky = [CLASSES[i] for i in top3.indices.tolist() if INFO[CLASSES[i]][0] != "benign"]
    lines = [f"### Most likely: {best} ({probs[top3.indices[0]]:.0%}) – *{kind}*", text, ""]
    if risky:
        lines.append(f"⚠️ **Please have this checked by a dermatologist:** the top-3 include "
                     f"{', '.join(risky)}.")
    else:
        lines.append("✅ All top-3 classes are benign lesion types.")
    preview = small.resize((224, 224), Image.NEAREST)        # what the model actually sees
    return result, preview, "\n\n".join(lines)


def example_images():
    """A few validation images (one per class) saved as PNG, for one-click demos."""
    os.makedirs("examples", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    paths = []
    try:
        from medmnist import DermaMNIST
        d = DermaMNIST(split="val", download=True, size=28, root="data")
        for c in [4, 1, 5, 6, 3]:                      # melanoma, BCC, nevus, vascular, dermatofibroma
            i = int(np.where(d.labels[:, 0] == c)[0][0])
            p = f"examples/{CLASSES[c].replace(' ', '_')}.png"
            Image.fromarray(d.imgs[i]).resize((112, 112), Image.NEAREST).save(p)
            paths.append(p)
    except Exception as e:                            # no internet: app still works without examples
        print("examples skipped:", e)
    return paths


with gr.Blocks(title="Group 6 – Skin lesion classifier") as demo:
    gr.Markdown("# Skin lesion classifier – Group 6\n"
                "Upload a dermatoscopy image. Our CNN (class-weighted loss + LR schedule + BatchNorm, "
                "trained on DermaMNIST) returns the **top-3 classes with probabilities**.  \n"
                "Test balanced accuracy: **0.603** (baseline 0.43–0.48).")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Your image", sources=["upload"])
            btn = gr.Button("Classify", variant="primary")
            examples = example_images()
            if examples:
                gr.Examples(examples, inputs=inp, label="Examples (validation images; click one)")
        with gr.Column():
            out_label = gr.Label(num_top_classes=3, label="Top-3 prediction")
            out_text = gr.Markdown()
            out_small = gr.Image(label="What the model sees (28 × 28 pixels)", height=224)
    btn.click(classify, inputs=inp, outputs=[out_label, out_small, out_text])
    inp.change(classify, inputs=inp, outputs=[out_label, out_small, out_text])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
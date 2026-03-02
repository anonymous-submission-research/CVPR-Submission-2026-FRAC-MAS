# analyze_results.py
import os, csv, argparse, numpy as np, matplotlib.pyplot as plt
from PIL import Image
import torch, torch.nn as nn, torchvision.transforms as T
import timm, torchvision.models as tvmodels
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import cv2

def get_model(name, num_classes):
    name = name.lower()
    if name.startswith('swin'):
        m = timm.create_model('swin_small_patch4_window7_224', pretrained=False)
        if hasattr(m, 'reset_classifier'):
            m.reset_classifier(num_classes=num_classes)
        else:
            m.head = nn.Linear(m.head.in_features, num_classes)
        return m
    if name.startswith('convnext'):
        m = timm.create_model('convnext_tiny', pretrained=False)
        if hasattr(m, 'reset_classifier'):
            m.reset_classifier(num_classes=num_classes)
        else:
            m.head.fc = nn.Linear(m.head.fc.in_features, num_classes)
        return m
    if name.startswith('densenet'):
        m = tvmodels.densenet169(pretrained=False)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m
    raise ValueError('model')

def load_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return [r for r in reader]

def transform_for_eval(img_size=224):
    return T.Compose([T.Resize((img_size,img_size)), T.ToTensor(),
                      T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

def save_confusion(cm, labels, out_path):
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right'); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j,i, str(cm[i,j]), ha='center', va='center', color='black')
    plt.colorbar(im)
    plt.tight_layout(); plt.savefig(out_path); plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument('--test-csv')
    parser.add_argument('--img-root', default='.')
    parser.add_argument('--model', default='swin')
    parser.add_argument('--img-size', default=224)
    parser.add_argument('--class-names')
    parser.add_argument('--out-dir', default='outputs/analysis')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    class_names = [s.strip() for s in args.class_names.split(',')]
    num_classes = len(class_names)
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    model = get_model(args.model, num_classes)
    ck = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ck['model_state_dict'])
    model.to(device); model.eval()

    rows = load_csv(args.test_csv)
    tf = transform_for_eval(args.img_size)
    preds, trues, paths, probs = [], [], [], []
    os.makedirs(os.path.join(args.out_dir,'examples'), exist_ok=True)

    for r in rows:
        img_path = r['image_path'] if os.path.isabs(r['image_path']) else os.path.join(args.img_root, r['image_path'])
        img = Image.open(img_path).convert('RGB')
        t = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(t)
            p = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred = int(p.argmax())
        preds.append(pred); trues.append(int(r['label'])); paths.append(img_path); probs.append(p)

    cm = confusion_matrix(trues, preds)
    p, r, f1, _ = precision_recall_fscore_support(trues, preds, average=None, labels=list(range(num_classes)), zero_division=0)

    # print per-class metrics
    for i,name in enumerate(class_names):
        print(f'{i} {name}: support={(cm[i].sum())}, prec={p[i]:.3f}, rec={r[i]:.3f}, f1={f1[i]:.3f}')
    print('macro-f1:', np.mean(f1))

    # save confusion matrix image
    save_confusion(cm, class_names, os.path.join(args.out_dir,'confusion_matrix.png'))

    # write misclassified csv
    miscsv = os.path.join(args.out_dir,'misclassified.csv')
    with open(miscsv,'w') as f:
        writer = csv.writer(f); writer.writerow(['image_path','true','pred','top1','top2'])
        for path, t, pr, prob in zip(paths,trues,preds,probs):
            if t!=pr:
                top2 = np.argsort(prob)[-2:][::-1].tolist()
                writer.writerow([path, t, pr, int(np.argmax(prob)), int(top2[0])])

    # Save example images for top confused pairs
    # find the biggest off-diagonal cells
    cm_off = cm.copy(); np.fill_diagonal(cm_off, 0)
    flat = [(cm_off[i,j],i,j) for i in range(num_classes) for j in range(num_classes)]
    flat = sorted(flat, reverse=True)
    for count,i,j in flat[:6]:  # top 6 confusion pairs
        if count==0: continue
        pair_dir = os.path.join(args.out_dir, 'examples', f'{i}_to_{j}')
        os.makedirs(pair_dir, exist_ok=True)
        saved=0
        for path,t,pred,prob in zip(paths,trues,preds,probs):
            if t==i and pred==j and saved<10:
                img = Image.open(path).convert('RGB')
                img.save(os.path.join(pair_dir, os.path.basename(path)))
                saved+=1

    print('Saved misclassified list and example images in', args.out_dir)

if __name__=='__main__':
    main()

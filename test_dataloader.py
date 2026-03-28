import yaml
from doclayout_yolo.utils import LOGGER
from doclayout_yolo.data.dataset import YOLODataset
from doclayout_yolo.data.build import build_dataloader

data_file = 'viet-pcc.yaml'
with open(data_file) as f:
    data = yaml.safe_load(f)

# Try to load a batch from the training set
try:
    dataset = YOLODataset(
        img_path=data['train'],
        imgsz=1024,
        batch_size=2,
        augment=False,
        hyp={},
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix='train: ',
        use_segments=False,
        use_keypoints=False,
        classes=None,
        data=data
    )
    print(f"Loaded {len(dataset)} images.")
    if len(dataset) > 0:
        labels = dataset.get_labels()
        print(f"Sample labels from first image: {labels[0]['labels']}")
except Exception as e:
    print(f"Error loading dataset: {e}")

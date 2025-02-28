import glob

from .lane_dataset_loader import LaneDatasetLoader


class NoLabelDataset(LaneDatasetLoader):
    def __init__(self, img_h=720, img_w=1280, max_lanes=None, root=None, img_ext='.jpg', **_):
        """Use this loader if you want to test a model on an image without annotations or implemented loader."""
        # DGG added
        self.split = 'test'
        self.list = []

        self.root = root
        if root is None:
            raise Exception('Please specify the root directory')

        self.img_w, self.img_h = img_w, img_h
        self.img_ext = img_ext
        self.annotations = []
        self.load_annotations()

        # Force max_lanes, used when evaluating testing with models trained on other datasets
        # On NoLabelDataset, always force it
        #self.max_lanes = max_lanes
        self.max_lanes = 4 if max_lanes is None else max_lanes



    def get_img_heigth(self, _):
        return self.img_h

    def get_img_width(self, _):
        return self.img_w

    def get_metrics(self, lanes, _):
        return 0, 0, [1] * len(lanes), [1] * len(lanes)

    def load_annotations(self):
        self.annotations = []
        pattern = '{}/**/*{}'.format(self.root, self.img_ext)
        print('Looking for image files with the pattern', pattern)
        for file in glob.glob(pattern, recursive=True):
            self.annotations.append({'lanes': [], 'path': file})
            self.list.append(file)

    def eval(self, _, __, ___, ____, _____):
        return "", None

    def eval_predictions(self, predictions, output_basedir):
        """
        DGG added...
        Args:
            predictions:
            output_basedir:

        Returns: a dict: {'TP': total_tp, 'FP': total_fp, 'FN': total_fn, 'Precision': precision, 'Recall': recall, 'F1': f1}

        """
        print('Generating prediction output...')

        # TODO: run the model here

        return None

    def __getitem__(self, idx):
        return self.annotations[idx]

    def __len__(self):
        return len(self.annotations)

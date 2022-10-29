import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from tqdm import tqdm
from torchvision.datasets import CocoDetection
from torchvision import transforms
import cv2
from model.fcos import FCOSDetector
import torch
import random
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
random.seed(0)

class COCOGenerator(CocoDetection):
    CLASSES_NAME = (
        '__back_ground__', 'drone')

    def __init__(self, imgs_path, anno_path, resize_size=[512, 640]):
        super().__init__(imgs_path, anno_path)

        print("INFO====>check annos, filtering invalid data......")
        self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}

        self.resize_size = resize_size
        self.mean = [0.40789654, 0.44719302, 0.47026115]  # [0.5,0.5,0.5]#[0.40789654, 0.44719302, 0.47026115]
        self.std = [0.28863828, 0.27408164, 0.27809835]  # [0.5,0.5,0.5]#[0.28863828, 0.27408164, 0.27809835]
        self.imgz = dict()
        self.boxz = dict()

        test_seq =  ["02_102609_4"]

        self.imgz[test_seq[0]], self.boxz[test_seq[0]], _, _,_ = self.get_item(0)

        self.names = []
        for i in self.coco.imgs:
            self.names.append(self.coco.imgs[i]["file_name"])

    def __getitem__(self, index):

        imgx, boxx, classes, scale,bb = self.get_item(index)

        crop_reapt_feat = self.imgz["02_102609_4"]
        boxz = self.boxz["02_102609_4"]

        return crop_reapt_feat, boxz, imgx, boxx, classes, scale,bb

    def preprocess_img_boxes(self, image, boxes, input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side, max_side = input_ksize
        h, w, _ = image.shape
        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))
        image_paded = np.zeros(shape=[384, 384, 3], dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized
        bb = boxes.copy()
        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes, scale,bb

    def get_item(self, index):
        img, ann = super().__getitem__(index)

        ann = [o for o in ann if o['iscrowd'] == 0]
        boxes = [o['bbox'] for o in ann]
        boxes = np.array(boxes, dtype=np.float32)
        # xywh-->xyxy
        boxes[..., 2:] = boxes[..., 2:] + boxes[..., :2]
        img = np.array(img)

        img, boxes, scale,bb = self.preprocess_img_boxes(img, boxes, self.resize_size)

        classes = [o['category_id'] for o in ann]
        classes = [self.category2id[c] for c in classes]

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(self.mean, self.std, inplace=True)(img)
        classes = np.array(classes, dtype=np.int64)
        return img, boxes, classes, scale,bb



def evaluate_coco(generator, model, l):
    data = l[0]
    for index in tqdm(range(2,7)):
        img_z, boxz, imgx, boxx, gt_labels, scale, bb = generator[index-1]
        scores, labels, boxes = model([img_z.unsqueeze(dim=0).cuda(), torch.from_numpy(boxz).unsqueeze(dim=0).cuda(),
                                       imgx.unsqueeze(dim=0).cuda(), data])

        boxes = boxes.detach().cpu().numpy()
        boxes /= scale

        boxes[:, :, 2] -= boxes[:, :, 0]
        boxes[:, :, 3] -= boxes[:, :, 1]

        bbox = boxes[0].tolist()
        bbox = bbox[0]
        path = os.path.join('/data1/wcx/githubdata/',str(index)+'.bmp')
        img = cv2.imread(path)
        cv2.rectangle(img, (round(bbox[0]), round(bbox[1])), (round(bbox[0])+round(bbox[2]), round(bbox[1])+round(bbox[3])),
                      (0, 0, 255), 1)
        cv2.imwrite(str(index)+'.bmp', img)


if __name__ == "__main__":
    with torch.no_grad():
        generator = COCOGenerator("./data",
                                  "./test.json",
                                  resize_size=[384, 384])  # resize_size=[256,256]
        tmp = [generator.imgz["02_102609_4"]]
        photos = ['2.bmp','3.bmp','4.bmp','5.bmp','6.bmp']
        model = FCOSDetector()
        temp_stic = torch.load("./model.pth", map_location=torch.device('cpu'))
        model.load_state_dict(temp_stic,strict=False)
        model = model.cuda().eval()
        need = model.fcos_body
        t = []
        for img_z in tmp:
            t.append(img_z.unsqueeze(dim=0).cuda())
        l = []
        for i in t:
            tmp = need(1, 1, 1, 1, i)
            l.append(tmp)
        model.fcos_body.backbone.structural_reparam()
        evaluate_coco(generator, model, l)


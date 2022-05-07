import numpy as np
import cv2
import torch

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ConvertUcharToFloat(object):
    """
    Convert img form uchar to float32
    """

    def __call__(self, data):
        data = [x.astype(np.float32) for x in data]
        return data


class RandomContrast(object):
    """
    Get random contrast img
    """
    def __init__(self, phase, lower=0.8, upper=1.2, prob=0.5):
        self.phase = phase
        self.lower = lower
        self.upper = upper
        self.prob = prob
        assert self.upper >= self.lower, "contrast upper must be >= lower!"
        assert self.lower > 0, "contrast lower must be non-negative!"

    def __call__(self, data):
        if self.phase in ['od', 'seg']:
            img, _ = data
            if torch.rand(1) < self.prob:
                alpha = torch.FloatTensor(1).uniform_(self.lower, self.upper)
                img *= alpha.numpy()
            return_data = img, _
        elif self.phase == 'cd':
            img1, label1, img2, label2 = data
            if torch.rand(1) < self.prob:
                alpha = torch.FloatTensor(1).uniform_(self.lower, self.upper)
                img1 *= alpha.numpy()
            if torch.rand(1) < self.prob:
                alpha = torch.FloatTensor(1).uniform_(self.lower, self.upper)
                img2 *= alpha.numpy()
            return_data = img1, label1, img2, label2
        return return_data


class RandomBrightness(object):
    """
    Get random brightness img
    """
    def __init__(self, phase, delta=10, prob=0.5):
        self.phase = phase
        self.delta = delta
        self.prob = prob
        assert 0. <= self.delta < 255., "brightness delta must between 0 to 255"

    def __call__(self, data):
        if self.phase in ['od', 'seg']:
            img, _ = data
            if torch.rand(1) < self.prob:
                delta = torch.FloatTensor(1).uniform_(- self.delta, self.delta)
                img += delta.numpy()
            return_data = img, _

        elif self.phase == 'cd':
            img1, label1, img2, label2 = data
            if torch.rand(1) < self.prob:
                delta = torch.FloatTensor(1).uniform_(- self.delta, self.delta)
                img1 += delta.numpy()
            if torch.rand(1) < self.prob:
                delta = torch.FloatTensor(1).uniform_(- self.delta, self.delta)
                img2 += delta.numpy()
            return_data = img1, label1, img2, label2

        return return_data


class ConvertColor(object):
    """
    Convert img color BGR to HSV or HSV to BGR for later img distortion.
    """
    def __init__(self, phase, current='RGB', target='HSV'):
        self.phase = phase
        self.current = current
        self.target = target

    def __call__(self, data):

        if self.phase in ['od', 'seg']:
            img, _ = data
            if self.current == 'RGB' and self.target == 'HSV':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif self.current == 'HSV' and self.target == 'RGB':
                img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            else:
                raise NotImplementedError("Convert color fail!")
            return_data = img, _

        elif self.phase == 'cd':
            img1, label1, img2, label2 = data
            if self.current == 'RGB' and self.target == 'HSV':
                img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
                img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
            elif self.current == 'HSV' and self.target == 'RGB':
                img1 = cv2.cvtColor(img1, cv2.COLOR_HSV2RGB)
                img2 = cv2.cvtColor(img2, cv2.COLOR_HSV2RGB)
            else:
                raise NotImplementedError("Convert color fail!")
            return_data = img1, label1, img2, label2

        return return_data


class RandomSaturation(object):
    """
    get random saturation img
    apply the restriction on saturation S
    """
    def __init__(self, phase, lower=0.8, upper=1.2, prob=0.5):
        self.phase = phase
        self.lower = lower
        self.upper = upper
        self.prob = prob
        assert self.upper >= self.lower, "saturation upper must be >= lower!"
        assert self.lower > 0, "saturation lower must be non-negative!"

    def __call__(self, data):
        if self.phase in ['od', 'seg']:
            img, _ = data
            if torch.rand(1) < self.prob:
                alpha = torch.FloatTensor(1).uniform_(self.lower, self.upper)
                img[:, :, 1] *= alpha.numpy()
            return_data = img, _
        elif self.phase == 'cd':
            img1, label1, img2, label2 = data
            if torch.rand(1) < self.prob:
                alpha = torch.FloatTensor(1).uniform_(self.lower, self.upper)
                img1[:, :, 1] *= alpha.numpy()
            if torch.rand(1) < self.prob:
                alpha = torch.FloatTensor(1).uniform_(self.lower, self.upper)
                img2[:, :, 1] *= alpha.numpy()
            return_data = img1, label1, img2, label2
        return return_data


class RandomHue(object):
    """
    get random Hue img
    apply the restriction on Hue H
    """
    def __init__(self, phase, delta=10., prob=0.5):
        self.phase = phase
        self.delta = delta
        self.prob = prob
        assert 0 <= self.delta < 360, "Hue delta must between 0 to 360!"

    def __call__(self, data):
        if self.phase in ['od', 'seg']:
            img, _ = data
            if torch.rand(1) < self.prob:
                alpha = torch.FloatTensor(1).uniform_(-self.delta, self.delta)
                img[:, :, 0] += alpha.numpy()
                img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
                img[:, :, 0][img[:, :, 0] < 0.0] += 360.0
            return_data = img, _

        elif self.phase == 'cd':
            img1, label1, img2, label2 = data
            if torch.rand(1) < self.prob:
                alpha = torch.FloatTensor(1).uniform_(-self.delta, self.delta)
                img1[:, :, 0] += alpha.numpy()
                img1[:, :, 0][img1[:, :, 0] > 360.0] -= 360.0
                img1[:, :, 0][img1[:, :, 0] < 0.0] += 360.0
            if torch.rand(1) < self.prob:
                alpha = torch.FloatTensor(1).uniform_(-self.delta, self.delta)
                img2[:, :, 0] += alpha.numpy()
                img2[:, :, 0][img2[:, :, 0] > 360.0] -= 360.0
                img2[:, :, 0][img2[:, :, 0] < 0.0] += 360.0

            return_data = img1, label1, img2, label2

        return return_data


class RandomChannelNoise(object):
    """
    Get random shuffle channels
    """
    def __init__(self, phase, prob=0.4):
        self.phase = phase
        self.prob = prob
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, data):
        if self.phase in ['od', 'seg']:
            img, _ = data
            if torch.rand(1) < self.prob:
                shuffle_factor = self.perms[torch.randint(0, len(self.perms), size=[])]
                img = img[:, :, shuffle_factor]
            return_data = img, _

        elif self.phase == 'cd':
            img1, label1, img2, label2 = data
            if torch.rand(1) < self.prob:
                shuffle_factor = self.perms[torch.randint(0, len(self.perms), size=[])]
                img1 = img1[:, :, shuffle_factor]
            if torch.rand(1) < self.prob:
                shuffle_factor = self.perms[torch.randint(0, len(self.perms), size=[])]
                img2 = img2[:, :, shuffle_factor]
            return_data = img1, label1, img2, label2

        return return_data


class ImgDistortion(object):
    """
    Change img by distortion
    """
    def __init__(self, phase, prob=0.5):
        self.phase = phase
        self.prob = prob
        self.operation = [
            RandomContrast(phase),
            ConvertColor(phase, current='RGB', target='HSV'),
            RandomSaturation(phase),
            RandomHue(phase),
            ConvertColor(phase, current='HSV', target='RGB'),
            RandomContrast(phase)
        ]
        self.random_brightness = RandomBrightness(phase)
        self.random_light_noise = RandomChannelNoise(phase)

    def __call__(self, data):
        if torch.rand(1) < self.prob:
            data = self.random_brightness(data)
            if torch.rand(1) < self.prob:
                distort = Compose(self.operation[:-1])
            else:
                distort = Compose(self.operation[1:])
            data = distort(data)
            data = self.random_light_noise(data)
        return data


class ExpandImg(object):
    """
    Get expand img
    """
    def __init__(self, phase, prior_mean, prob=0.5, expand_ratio=0.2):
        self.phase = phase
        self.prior_mean = np.array(prior_mean) * 255
        self.prob = prob
        self.expand_ratio = expand_ratio

    def __call__(self, data):
        if self.phase == 'seg':
            img, label = data
            if torch.rand(1) < self.prob:
                return data
            height, width, channels = img.shape
            ratio_width = self.expand_ratio * torch.rand([])
            ratio_height = self.expand_ratio * torch.rand([])
            left, right = torch.randint(high=int(max(1, width * ratio_width)), size=[2])
            top, bottom = torch.randint(high=int(max(1, width * ratio_height)), size=[2])
            img = cv2.copyMakeBorder(
                img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=self.prior_mean)
            label = cv2.copyMakeBorder(
                label, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=0)
            return img, label
        elif self.phase == 'cd':
            img1, label1, img2, label2 = data
            if torch.rand(1) < self.prob:
                return data
            height, width, channels = img1.shape
            ratio_width = self.expand_ratio * torch.rand([])
            ratio_height = self.expand_ratio * torch.rand([])
            left, right = torch.randint(high=int(max(1, width * ratio_width)), size=[2])
            top, bottom = torch.randint(high=int(max(1, width * ratio_height)), size=[2])
            img1 = cv2.copyMakeBorder(
                img1, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=self.prior_mean)
            label1 = cv2.copyMakeBorder(
                label1, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=0)
            img2 = cv2.copyMakeBorder(
                img2, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=self.prior_mean)
            label2 = cv2.copyMakeBorder(
                label2, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=0)
            return img1, label1, img2, label2

        elif self.phase == 'od':
            if torch.rand(1) < self.prob:
                return data
            img, label = data
            height, width, channels = img.shape
            ratio_width = self.expand_ratio * torch.rand([])
            ratio_height = self.expand_ratio * torch.rand([])
            left, right = torch.randint(high=int(max(1, width * ratio_width)), size=[2])
            top, bottom = torch.randint(high=int(max(1, width * ratio_height)), size=[2])
            left = int(left)
            right = int(right)
            top = int(top)
            bottom = int(bottom)
            img = cv2.copyMakeBorder(
                img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.prior_mean)

            label[:, 1::2] += left
            label[:, 2::2] += top
            return img, label


class RandomSampleCrop(object):
    """
    Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        label (Tensor): the class label for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
        img (Image): the cropped image
        boxes (Tensor): the adjusted bounding boxes in pt form
        label (Tensor): the class label for each bbox
    """
    def __init__(self,
                 phase,
                 original_size=[512, 512],
                 prob=0.5,
                 crop_scale_ratios_range=[0.8, 1.2],
                 aspect_ratio_range=[4./5, 5./4]):
        self.phase = phase
        self.prob = prob
        self.scale_range = crop_scale_ratios_range
        self.original_size = original_size
        self.aspect_ratio_range = aspect_ratio_range  # h/w
        self.max_try_times = 500

    def __call__(self, data):
        if self.phase == 'seg':
            img, label = data
            w, h, c = img.shape
            if torch.rand(1) < self.prob:
                return data
            else:
                try_times = 0
                while try_times < self.max_try_times:
                    crop_w = torch.randint(
                        min(w, int(self.scale_range[0] * self.original_size[0])),
                        min(w + 1, int(self.scale_range[1] * self.original_size[0])),
                        size=[]
                    )
                    crop_h = torch.randint(
                        min(h, int(self.scale_range[0] * self.original_size[1])),
                        min(h + 1, int(self.scale_range[1] * self.original_size[1])),
                        size=[]
                    )
                    # aspect ratio constraint
                    if self.aspect_ratio_range[0] < crop_h / crop_w < self.aspect_ratio_range[1]:
                        break
                    else:
                        try_times += 1
                if try_times >= self.max_try_times:
                    print("try times over max threshold!", flush=True)
                    return img, label

                left = torch.randint(0, w - crop_w + 1, size=[])
                top = torch.randint(0, h - crop_h + 1, size=[])
                img = img[top:(top + crop_h), left:(left + crop_w), :]
                label = label[top:(top + crop_h), left:(left + crop_w)]
                return img, label

        elif self.phase == 'od':
            if torch.rand(1) < self.prob:
                return data
            img, label = data
            w, h, c = img.shape

            while True:
                crop_w = torch.randint(
                    min(w, int(self.scale_range[0] * self.original_size[0])),
                    min(w + 1, int(self.scale_range[1] * self.original_size[0])),
                    size=[]
                )
                crop_h = torch.randint(
                    min(h, int(self.scale_range[0] * self.original_size[1])),
                    min(h + 1, int(self.scale_range[1] * self.original_size[1])),
                    size=[]
                )

                # aspect ratio constraint
                if self.aspect_ratio_range[0] < crop_h / crop_w < self.aspect_ratio_range[1]:
                    break

            left = torch.randint(0, w - crop_w + 1, size=[])
            top = torch.randint(0, h - crop_h + 1, size=[])
            left = left.numpy()
            top = top.numpy()
            crop_h = crop_h.numpy()
            crop_w = crop_w.numpy()
            img = img[top:(top + crop_h), left:(left + crop_w), :]
            if len(label):
                # keep overlap with gt box IF center in sampled patch
                centers = (label[:, 1:3] + label[:, 3:]) / 2.0
                # mask in all gt boxes that above and to the left of centers
                m1 = (left <= centers[:, 0]) * (top <= centers[:, 1])
                # mask in all gt boxes that under and to the right of centers
                m2 = ((left + crop_w) >= centers[:, 0]) * ((top + crop_h) > centers[:, 1])
                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # take only matching gt boxes
                current_label = label[mask, :]

                # adjust to crop (by substracting crop's left,top)
                current_label[:, 1::2] -= left
                current_label[:, 2::2] -= top
                label = current_label
            return img, label


class RandomMirror(object):
    def __init__(self, phase, prob=0.5):
        self.phase = phase
        self.prob = prob

    def __call__(self, data):
        if self.phase == 'seg':
            img, label = data
            if torch.rand(1) < self.prob:
                img = img[:, ::-1]
                label = label[:, ::-1]
            return img, label
        elif self.phase == 'cd':
            img1, label1, img2, label2 = data
            if torch.rand(1) < self.prob:
                img1 = img1[:, ::-1]
                label1 = label1[:, ::-1]
                img2 = img2[:, ::-1]
                label2 = label2[:, ::-1]
            return img1, label1, img2, label2
        elif self.phase == 'od':
            img, label = data
            if torch.rand(1) < self.prob:
                _, width, _ = img.shape
                img = img[:, ::-1]
                label[:, 1::2] = width - label[:, 3::-2]
            return img, label


class RandomFlipV(object):
    def __init__(self, phase, prob=0.5):
        self.phase = phase
        self.prob = prob

    def __call__(self, data):
        if self.phase == 'seg':
            img, label = data
            if torch.rand(1) < self.prob:
                img = img[::-1, :]
                label = label[::-1, :]
            return img, label
        elif self.phase == 'cd':
            img1, label1, img2, label2 = data
            if torch.rand(1) < self.prob:
                img1 = img1[::-1, :]
                label1 = label1[::-1, :]
                img2 = img2[::-1, :]
                label2 = label2[::-1, :]
            return img1, label1, img2, label2
        elif self.phase == 'od':
            img, label = data
            if torch.rand(1) < self.prob:
                height, _, _ = img.shape
                img = img[::-1, :]
                label[:, 2::2] = height - label[:, 4:1:-2]
            return img, label


class Resize(object):
    def __init__(self, phase, size):
        self.phase = phase
        self.size = size

    def __call__(self, data):
        if self.phase == 'seg':
            img, label = data
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            # for label
            label = cv2.resize(label, self.size, interpolation=cv2.INTER_NEAREST)
            return img, label
        elif self.phase == 'cd':
            img1, label1, img2, label2 = data
            img1 = cv2.resize(img1, self.size, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, self.size, interpolation=cv2.INTER_LINEAR)
            # for label
            label1 = cv2.resize(label1, self.size, interpolation=cv2.INTER_NEAREST)
            label2 = cv2.resize(label2, self.size, interpolation=cv2.INTER_NEAREST)
            return img1, label1, img2, label2
        elif self.phase == 'od':
            img, label = data
            height, width, _ = img.shape
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            label[:, 1::2] = label[:, 1::2] / width * self.size[0]
            label[:, 2::2] = label[:, 2::2] / height * self.size[1]
            return img, label


class Normalize(object):
    def __init__(self, phase, prior_mean, prior_std):
        self.phase = phase
        self.prior_mean = np.array([[prior_mean]], dtype=np.float32)
        self.prior_std = np.array([[prior_std]], dtype=np.float32)

    def __call__(self, data):
        if self.phase in ['od', 'seg']:
            img, _ = data
            img = img / 255.
            img = (img - self.prior_mean) / (self.prior_std + 1e-10)

            return img, _
        elif self.phase == 'cd':
            img1, label1, img2, label2 = data
            img1 = img1 / 255.
            img1 = (img1 - self.prior_mean) / (self.prior_std + 1e-10)
            img2 = img2 / 255.
            img2 = (img2 - self.prior_mean) / (self.prior_std + 1e-10)

            return img1, label1, img2, label2


class InvNormalize(object):
    def __init__(self, prior_mean, prior_std):
        self.prior_mean = np.array([[prior_mean]], dtype=np.float32)
        self.prior_std = np.array([[prior_std]], dtype=np.float32)

    def __call__(self, img):
        img = img * self.prior_std + self.prior_mean
        img = img * 255.
        img = np.clip(img, a_min=0, a_max=255)
        return img


class Augmentations(object):
    def __init__(self, size, prior_mean=0, prior_std=1, pattern='train', phase='seg', *args, **kwargs):
        self.size = size
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.phase = phase

        augments = {
            'train': Compose([
                ConvertUcharToFloat(),
                ImgDistortion(self.phase),
                ExpandImg(self.phase, self.prior_mean),
                RandomSampleCrop(self.phase, original_size=self.size),
                RandomMirror(self.phase),
                RandomFlipV(self.phase),
                Resize(self.phase, self.size),
                Normalize(self.phase, self.prior_mean, self.prior_std),
            ]),
            'val': Compose([
                ConvertUcharToFloat(),
                Resize(self.phase, self.size),
                Normalize(self.phase, self.prior_mean, self.prior_std),
            ]),
            'test': Compose([
                ConvertUcharToFloat(),
                Resize(self.phase, self.size),
                Normalize(self.phase, self.prior_mean, self.prior_std),
            ])
        }
        self.augment = augments[pattern]

    def __call__(self, data):
        return self.augment(data)


import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import numpy as np

class Compose:
    """Composes several transforms together for both image and targets."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert image to tensor and keep target as is."""
    
    def __call__(self, image, target):
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        return image, target


class Normalize_:
    """Normalize image with mean and standard deviation."""
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, target):
        # Normalize each channel
        image = (image - self.mean) / self.std
        return image, target


class Normalize:
    """Normalize image with mean and standard deviation."""
    
    def __init__(self, mean, std):
        # Convert to tensor if list is provided
        if isinstance(mean, list):
            self.mean = torch.tensor(mean).view(-1, 1, 1)
        elif isinstance(mean, torch.Tensor):
            self.mean = mean.view(-1, 1, 1)
        else:
            self.mean = mean
            
        if isinstance(std, list):
            self.std = torch.tensor(std).view(-1, 1, 1)
        elif isinstance(std, torch.Tensor):
            self.std = std.view(-1, 1, 1)
        else:
            self.std = std
    
    def __call__(self, image, target):
        # Move mean/std to same device as image if they're tensors
        if isinstance(self.mean, torch.Tensor):
            self.mean = self.mean.to(image.device)
        if isinstance(self.std, torch.Tensor):
            self.std = self.std.to(image.device)
            
        # Normalize each channel
        image = (image - self.mean) / self.std
        return image, target


class RandomHorizontalFlip:
    """Horizontally flip the image and bounding boxes with given probability."""
    
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            # Flip image
            image = F.hflip(image)
            
            # Flip bounding boxes
            if target['boxes'] is not None:
                _, width = image.shape[-2:]
                boxes = target['boxes']
                # Flip x coordinates: x_new = width - x_old
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target['boxes'] = boxes
        
        return image, target


class RandomVerticalFlip:
    """Vertically flip the image and bounding boxes with given probability."""
    
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            # Flip image
            image = F.vflip(image)
            
            # Flip bounding boxes
            if target['boxes'] is not None:
                height, _ = image.shape[-2:]
                boxes = target['boxes']
                # Flip y coordinates: y_new = height - y_old
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
                target['boxes'] = boxes
        
        return image, target


class RandomRotation:
    """Rotate image and bounding boxes by random angle."""
    
    def __init__(self, degrees=15, prob=0.5):
        self.degrees = degrees
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            angle = random.uniform(-self.degrees, self.degrees)
            
            # Rotate image
            image = F.rotate(image, angle)
            
            # For bounding boxes, we need to rotate each corner
            if target['boxes'] is not None:
                h, w = image.shape[-2:]
                boxes = target['boxes']
                
                # Get rotation matrix
                import math
                theta = math.radians(angle)
                cos, sin = math.cos(theta), math.sin(theta)
                
                # Center of rotation
                cx, cy = w / 2, h / 2
                
                # Rotate each box corner
                rotated_boxes = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    corners = torch.tensor([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                    
                    # Translate to origin
                    corners[:, 0] -= cx
                    corners[:, 1] -= cy
                    
                    # Rotate
                    new_x = corners[:, 0] * cos - corners[:, 1] * sin
                    new_y = corners[:, 0] * sin + corners[:, 1] * cos
                    
                    # Translate back
                    new_x += cx
                    new_y += cy
                    
                    # Get new bounding box
                    new_x1 = torch.min(new_x)
                    new_y1 = torch.min(new_y)
                    new_x2 = torch.max(new_x)
                    new_y2 = torch.max(new_y)
                    
                    # Clamp to image boundaries
                    new_x1 = torch.clamp(new_x1, 0, w)
                    new_x2 = torch.clamp(new_x2, 0, w)
                    new_y1 = torch.clamp(new_y1, 0, h)
                    new_y2 = torch.clamp(new_y2, 0, h)
                    
                    rotated_boxes.append([new_x1, new_y1, new_x2, new_y2])
                
                target['boxes'] = torch.stack(rotated_boxes) if rotated_boxes else boxes
        
        return image, target


class RandomRotation__:
    """Rotate image and bounding boxes by random angle."""
    
    def __init__(self, degrees=15, prob=0.5):
        self.degrees = degrees
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            angle = random.uniform(-self.degrees, self.degrees)
            
            # Rotate image
            image = F.rotate(image, angle)
            
            # For bounding boxes, we need to rotate each corner
            if target['boxes'] is not None and len(target['boxes']) > 0:
                h, w = image.shape[-2:]
                boxes = target['boxes']
                
                # Get rotation matrix
                import math
                theta = math.radians(angle)
                cos, sin = math.cos(theta), math.sin(theta)
                
                # Center of rotation
                cx, cy = w / 2, h / 2
                
                # Rotate each box corner
                rotated_boxes = []
                valid_indices = []
                MIN_SIZE = 1.0
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    
                    # Get all four corners
                    corners = torch.tensor([
                        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                    ])
                    
                    # Rotate each corner
                    rotated_corners = []
                    for corner in corners:
                        x, y = corner
                        # Translate to origin
                        x -= cx
                        y -= cy
                        # Rotate
                        new_x = x * cos - y * sin
                        new_y = x * sin + y * cos
                        # Translate back
                        new_x += cx
                        new_y += cy
                        rotated_corners.append([new_x, new_y])
                    
                    # Find min and max of rotated corners
                    rotated_corners = torch.tensor(rotated_corners)
                    new_x1 = torch.min(rotated_corners[:, 0])
                    new_y1 = torch.min(rotated_corners[:, 1])
                    new_x2 = torch.max(rotated_corners[:, 0])
                    new_y2 = torch.max(rotated_corners[:, 1])
                    
                    # Clamp to image boundaries
                    new_x1 = torch.clamp(new_x1, 0, w)
                    new_x2 = torch.clamp(new_x2, 0, w)
                    new_y1 = torch.clamp(new_y1, 0, h)
                    new_y2 = torch.clamp(new_y2, 0, h)
                    
                    # If box becomes degenerate, keep original
                    if new_x2 - new_x1 < MIN_SIZE or new_y2 - new_y1 < MIN_SIZE:
                        # Keep original box but adjust for rotation? No, skip this box
                        # For now, just keep the original box
                        rotated_boxes.append(box)
                        valid_indices.append(i)
                    else:
                        rotated_boxes.append(torch.tensor([new_x1, new_y1, new_x2, new_y2]))
                        valid_indices.append(i)
                
                if rotated_boxes:
                    target['boxes'] = torch.stack(rotated_boxes)
                    if 'labels' in target and target['labels'] is not None:
                        target['labels'] = target['labels'][valid_indices]
        
        return image, target


class ColorJitter:
    """Randomly change brightness, contrast, saturation and hue."""
    
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, prob=0.8):
        self.color_jitter = T.ColorJitter(brightness, contrast, saturation, hue)
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            # Convert to PIL for ColorJitter
            if isinstance(image, torch.Tensor):
                # Convert tensor (C, H, W) to PIL
                from torchvision.transforms.functional import to_pil_image
                image_pil = to_pil_image(image)
                image_pil = self.color_jitter(image_pil)
                image = F.to_tensor(image_pil)
        
        return image, target


class RandomGaussianBlur:
    """Apply Gaussian blur with given probability."""
    
    def __init__(self, kernel_size=5, sigma=(0.1, 2.0), prob=0.5):
        self.gaussian_blur = T.GaussianBlur(kernel_size, sigma)
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = self.gaussian_blur(image)
        return image, target


class RandomAffine:
    """Apply random affine transformation."""
    
    def __init__(self, degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5, prob=0.5):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            h, w = image.shape[-2:]
            
            # Generate random parameters
            angle = random.uniform(-self.degrees, self.degrees)
            translate = (random.uniform(-self.translate[0] * w, self.translate[0] * w),
                        random.uniform(-self.translate[1] * h, self.translate[1] * h))
            scale = random.uniform(self.scale[0], self.scale[1])
            shear = random.uniform(-self.shear, self.shear)
            
            # Apply affine to image
            image = F.affine(image, angle, translate, scale, shear)
            
            # Apply same transformation to bounding boxes
            if target['boxes'] is not None:
                boxes = target['boxes']
                
                # Simplified: we'll use a similar approach as rotation
                # For production, consider using torchvision.ops.box_convert with affine transforms
                
        return image, target


class RandomGaussianNoise:
    """Add random Gaussian noise to the image."""
    
    def __init__(self, mean=0.0, std=0.05, prob=0.3):
        self.mean = mean
        self.std = std
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            noise = torch.randn_like(image) * self.std + self.mean
            image = image + noise
            # Clip to valid range [0, 1]
            image = torch.clamp(image, 0, 1)
        return image, target


class RandomSaltPepperNoise:
    """Add salt and pepper noise to the image."""
    
    def __init__(self, salt_prob=0.01, pepper_prob=0.01, prob=0.3):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            # Create noise mask
            salt_mask = torch.rand_like(image) < self.salt_prob
            pepper_mask = torch.rand_like(image) < self.pepper_prob
            
            # Apply noise
            image = torch.where(salt_mask, torch.ones_like(image), image)
            image = torch.where(pepper_mask, torch.zeros_like(image), image)
        
        return image, target


class RandomErasing:
    """Randomly erase a rectangular patch from the image."""
    
    def __init__(self, prob=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), fill='random'):
        self.prob = prob
        self.scale = scale
        self.ratio = ratio
        self.fill = fill
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            _, h, w = image.shape
            
            # Randomly choose area and aspect ratio
            area = h * w
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            # Calculate patch dimensions
            ph = int(round(np.sqrt(target_area * aspect_ratio)))
            pw = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if ph < h and pw < w:
                # Random top-left corner
                top = random.randint(0, h - ph)
                left = random.randint(0, w - pw)
                
                # Fill the patch
                if self.fill == 'random':
                    # Random noise
                    image[:, top:top+ph, left:left+pw] = torch.rand_like(image[:, top:top+ph, left:left+pw])
                elif self.fill == 'mean':
                    # Mean pixel value
                    image[:, top:top+ph, left:left+pw] = image.mean(dim=(1, 2), keepdim=True)
                elif self.fill == 'zeros':
                    # Black
                    image[:, top:top+ph, left:left+pw] = 0
                elif self.fill == 'ones':
                    # White
                    image[:, top:top+ph, left:left+pw] = 1
        
        return image, target


class RandomCutout:
    """Cutout: Randomly mask out square patches (similar to Erasing but with fixed size)."""
    
    def __init__(self, size=0.1, prob=0.5):
        self.size = size
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            _, h, w = image.shape
            patch_h = int(h * self.size)
            patch_w = int(w * self.size)
            
            top = random.randint(0, h - patch_h)
            left = random.randint(0, w - patch_w)
            
            # Apply cutout
            image[:, top:top+patch_h, left:left+patch_w] = 0
        
        return image, target


class RandomMixUp:
    """MixUp augmentation: blend with another random image."""
    
    def __init__(self, alpha=0.2, prob=0.3):
        self.alpha = alpha
        self.prob = prob
        self.cached_dataset = None
        self.cached_indices = None
    
    def __call__(self, image, target):
        if random.random() < self.prob and self.cached_dataset is not None:
            # Get random image from dataset
            idx = random.choice(self.cached_indices)
            mix_image, mix_target = self.cached_dataset[idx]
            
            # Mixing coefficient
            lam = np.random.beta(self.alpha, self.alpha)
            
            # Mix images
            image = lam * image + (1 - lam) * mix_image
            
            # For boxes, we keep original (mixing boxes is complex)
            # In a full implementation, you'd need to combine boxes from both images
        
        return image, target
    
    def set_dataset(self, dataset, indices):
        """Cache dataset for MixUp."""
        self.cached_dataset = dataset
        self.cached_indices = indices


class RandomElasticTransform_:
    """Elastic deformation (good for medical/tokamak images)."""
    
    def __init__(self, alpha=50.0, sigma=5.0, prob=0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            # Convert to numpy for elastic transform
            c, h, w = image.shape
            image_np = image.numpy()
            
            # Create random displacement fields
            dx = np.random.randn(h, w) * self.sigma
            dy = np.random.randn(h, w) * self.sigma
            
            # Scale by alpha
            dx *= self.alpha / (h - 1)
            dy *= self.alpha / (w - 1)
            
            # Create mesh grid
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            indices_x = (x + dx).reshape(-1)
            indices_y = (y + dy).reshape(-1)
            
            # Apply transform
            from scipy.ndimage import map_coordinates
            transformed = np.zeros_like(image_np)
            for i in range(c):
                transformed[i] = map_coordinates(image_np[i], [indices_y, indices_x], 
                                                order=1, mode='reflect').reshape(h, w)
            
            image = torch.from_numpy(transformed).float()
            
            # Note: Elastic transform of bounding boxes is complex
            # For simplicity, we keep original boxes (approximation)
        
        return image, target


class RandomResizedCrop:
    """Crop random portion of image and resize to target size."""
    
    def __init__(self, size, scale=(0.5, 1.0), ratio=(0.75, 1.33)):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, image, target):
        _, h, w = image.shape
        
        # Find random crop that respects scale and ratio
        for _ in range(10):  # Try 10 times
            target_area = random.uniform(*self.scale) * (h * w)
            aspect_ratio = random.uniform(*self.ratio)
            
            crop_w = int(round(np.sqrt(target_area * aspect_ratio)))
            crop_h = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if 0 < crop_h <= h and 0 < crop_w <= w:
                top = random.randint(0, h - crop_h)
                left = random.randint(0, w - crop_w)
                
                # Crop image
                image = image[:, top:top+crop_h, left:left+crop_w]
                
                # Resize image
                image = F.resize(image, self.size)
                
                # Adjust bounding boxes
                if target['boxes'] is not None:
                    boxes = target['boxes'].clone()
                    
                    # Translate and scale boxes
                    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) * (self.size[1] / crop_w)
                    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) * (self.size[0] / crop_h)
                    
                    # Clamp to image boundaries
                    boxes[:, 0] = torch.clamp(boxes[:, 0], 0, self.size[1])
                    boxes[:, 1] = torch.clamp(boxes[:, 1], 0, self.size[0])
                    boxes[:, 2] = torch.clamp(boxes[:, 2], 0, self.size[1])
                    boxes[:, 3] = torch.clamp(boxes[:, 3], 0, self.size[0])
                    
                    target['boxes'] = boxes
                
                return image, target
        
        # Fallback: center crop
        crop_h = min(h, int(h * self.scale[0]))
        crop_w = min(w, int(w * self.scale[0]))
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        
        image = image[:, top:top+crop_h, left:left+crop_w]
        image = F.resize(image, self.size)
        
        if target['boxes'] is not None:
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) * (self.size[1] / crop_w)
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) * (self.size[0] / crop_h)
            
            boxes[:, 0] = torch.clamp(boxes[:, 0], 0, self.size[1])
            boxes[:, 1] = torch.clamp(boxes[:, 1], 0, self.size[0])
            boxes[:, 2] = torch.clamp(boxes[:, 2], 0, self.size[1])
            boxes[:, 3] = torch.clamp(boxes[:, 3], 0, self.size[0])
            
            target['boxes'] = boxes
        
        return image, target


class RandomGridShuffle:
    """Shuffle patches of the image (makes model learn spatial relationships)."""
    
    def __init__(self, grid_size=(2, 2), prob=0.3):
        self.grid_size = grid_size
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            c, h, w = image.shape
            gh, gw = self.grid_size
            
            patch_h = h // gh
            patch_w = w // gw
            
            # Split image into patches
            patches = []
            for i in range(gh):
                for j in range(gw):
                    patch = image[:, i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
                    patches.append(patch)
            
            # Shuffle patches
            random.shuffle(patches)
            
            # Reassemble
            new_image = torch.zeros_like(image)
            idx = 0
            for i in range(gh):
                for j in range(gw):
                    new_image[:, i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] = patches[idx]
                    idx += 1
            
            image = new_image
            
            # Note: Bounding boxes are not transformed (too complex)
            # This transform is better for unsupervised/self-supervised learning
        
        return image, target


class RandomSolarize:
    """Solarize: invert pixels above threshold."""
    
    def __init__(self, threshold=0.5, prob=0.2):
        self.threshold = threshold
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = torch.where(image > self.threshold, 1 - image, image)
        return image, target


class RandomPosterize:
    """Reduce number of bits per channel."""
    
    def __init__(self, bits=4, prob=0.2):
        self.bits = bits
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = torch.round(image * (2**self.bits - 1)) / (2**self.bits - 1)
        return image, target


class RandomSharpness:
    """Adjust sharpness."""
    
    def __init__(self, factor=2.0, prob=0.3):
        self.factor = factor
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.adjust_sharpness(image, self.factor)
        return image, target


class RandomAutocontrast:
    """Maximize image contrast."""
    
    def __init__(self, prob=0.2):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.autocontrast(image)
        return image, target


class RandomEqualize:
    """Equalize histogram."""
    
    def __init__(self, prob=0.2):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.equalize(image)
        return image, target


class MixUp:
    """MixUp: blend with another random image - GREAT for small datasets."""
    
    def __init__(self, alpha=0.4, prob=0.7):
        self.alpha = alpha
        self.prob = prob
        self.dataset_cache = None
        self.indices_cache = None
    
    def __call__(self, image, target):
        if random.random() < self.prob and self.dataset_cache is not None:
            # Get random image from dataset
            idx = random.choice(self.indices_cache)
            mix_image, mix_target = self.dataset_cache[idx]
            
            # Mixing coefficient from Beta distribution
            lam = np.random.beta(self.alpha, self.alpha)
            lam = max(lam, 1 - lam)  # Keep more of original
            
            # Mix images
            image = lam * image + (1 - lam) * mix_image
            
            # For boxes: combine boxes from both images
            if target['boxes'] is not None and mix_target['boxes'] is not None:
                # Combine boxes and labels
                combined_boxes = torch.cat([target['boxes'], mix_target['boxes']], dim=0)
                combined_labels = torch.cat([
                    torch.ones(len(target['boxes'])), 
                    torch.ones(len(mix_target['boxes']))
                ], dim=0)
                
                target['boxes'] = combined_boxes
                target['labels'] = combined_labels
        
        return image, target
    
    def set_dataset(self, dataset, indices):
        """Cache dataset for MixUp."""
        self.dataset_cache = dataset
        self.indices_cache = indices


class CutMix:
    """CutMix: replace rectangular region with patch from another image."""
    
    def __init__(self, alpha=0.4, prob=0.5):
        self.alpha = alpha
        self.prob = prob
        self.dataset_cache = None
        self.indices_cache = None
    
    def __call__(self, image, target):
        if random.random() < self.prob and self.dataset_cache is not None:
            # Get random image from dataset
            idx = random.choice(self.indices_cache)
            mix_image, mix_target = self.dataset_cache[idx]
            
            _, h, w = image.shape
            
            # Sample lambda from Beta
            lam = np.random.beta(self.alpha, self.alpha)
            
            # Get bounding box
            cut_ratio = np.sqrt(1. - lam)
            cut_w = int(w * cut_ratio)
            cut_h = int(h * cut_ratio)
            
            # Center coordinates
            cx = np.random.randint(w)
            cy = np.random.randint(h)
            
            # Bounding box coordinates
            x1 = np.clip(cx - cut_w // 2, 0, w)
            y1 = np.clip(cy - cut_h // 2, 0, h)
            x2 = np.clip(cx + cut_w // 2, 0, w)
            y2 = np.clip(cy + cut_h // 2, 0, h)
            
            # Replace region with patch from mix_image
            image[:, y1:y2, x1:x2] = mix_image[:, y1:y2, x1:x2]
            
            # Adjust lambda based on actual area
            lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
            
            # Combine boxes (only keep boxes from original that are not in cut region)
            if target['boxes'] is not None:
                boxes = target['boxes']
                keep = []
                for i, box in enumerate(boxes):
                    box_x1, box_y1, box_x2, box_y2 = box
                    # Check if box is mostly outside cut region
                    if not (box_x1 > x1 and box_x2 < x2 and box_y1 > y1 and box_y2 < y2):
                        keep.append(i)
                
                if keep:
                    target['boxes'] = boxes[keep]
                    target['labels'] = torch.ones(len(keep), dtype=torch.int64)
        
        return image, target
    
    def set_dataset(self, dataset, indices):
        """Cache dataset for CutMix."""
        self.dataset_cache = dataset
        self.indices_cache = indices


class RandomElasticTransform:
    """Elastic deformation - GREAT for plasma/tokamak images."""
    
    def __init__(self, alpha=80.0, sigma=8.0, prob=0.7):
        self.alpha = alpha
        self.sigma = sigma
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            c, h, w = image.shape
            image_np = image.numpy()
            
            # Create random displacement fields
            dx = np.random.randn(h, w) * self.sigma
            dy = np.random.randn(h, w) * self.sigma
            
            # Scale by alpha
            dx *= self.alpha / (h - 1)
            dy *= self.alpha / (w - 1)
            
            # Create mesh grid
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            indices_x = (x + dx).reshape(-1)
            indices_y = (y + dy).reshape(-1)
            
            # Apply transform
            from scipy.ndimage import map_coordinates
            transformed = np.zeros_like(image_np)
            for i in range(c):
                transformed[i] = map_coordinates(image_np[i], [indices_y, indices_x], 
                                                order=1, mode='reflect').reshape(h, w)
            
            image = torch.from_numpy(transformed).float()
            
            # Also deform bounding boxes (approximation)
            if target['boxes'] is not None:
                boxes = target['boxes'].clone()
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    # Deform corners
                    corners_x = torch.tensor([x1, x2, x2, x1])
                    corners_y = torch.tensor([y1, y1, y2, y2])
                    
                    # Apply same deformation field
                    # Simplified: use average displacement in each region
                    x1_new = x1 + dx[int(y1), int(x1)]
                    y1_new = y1 + dy[int(y1), int(x1)]
                    x2_new = x2 + dx[int(y2), int(x2)]
                    y2_new = y2 + dy[int(y2), int(x2)]
                    
                    boxes[i] = torch.tensor([x1_new, y1_new, x2_new, y2_new])
                
                target['boxes'] = boxes
        
        return image, target


class RandomResizedCropWithBoxes:
    """Crop and resize - ESSENTIAL for scale invariance."""
    
    def __init__(self, size, scale=(0.3, 1.0), ratio=(0.8, 1.2)):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, image, target):
        _, h, w = image.shape
        
        # Try multiple times to get a crop with boxes
        for _ in range(20):
            target_area = random.uniform(*self.scale) * (h * w)
            aspect_ratio = random.uniform(*self.ratio)
            
            crop_w = int(round(np.sqrt(target_area * aspect_ratio)))
            crop_h = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if 0 < crop_h <= h and 0 < crop_w <= w:
                top = random.randint(0, h - crop_h)
                left = random.randint(0, w - crop_w)
                
                # Check if crop contains at least one box
                if target['boxes'] is not None:
                    boxes = target['boxes']
                    boxes_in_crop = []
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        if (x2 > left and x1 < left + crop_w and 
                            y2 > top and y1 < top + crop_h):
                            boxes_in_crop.append(box)
                    
                    # If no boxes in crop, try again
                    if len(boxes_in_crop) == 0 and random.random() > 0.3:
                        continue
                
                # Crop image
                image_cropped = image[:, top:top+crop_h, left:left+crop_w]
                
                # Resize image
                image_resized = F.resize(image_cropped, self.size)
                
                # Adjust bounding boxes
                if target['boxes'] is not None:
                    boxes = target['boxes'].clone()
                    
                    # Translate and scale
                    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) * (self.size[1] / crop_w)
                    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) * (self.size[0] / crop_h)
                    
                    # Clamp and filter
                    keep = []
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box
                        # Filter out boxes that are too small or outside
                        if (x2 > 0 and x1 < self.size[1] and 
                            y2 > 0 and y1 < self.size[0] and
                            (x2 - x1) * (y2 - y1) > 100):  # Minimum area
                            boxes[i] = torch.tensor([
                                max(0, x1), max(0, y1),
                                min(self.size[1], x2), min(self.size[0], y2)
                            ])
                            keep.append(i)
                    
                    if keep:
                        target['boxes'] = boxes[keep]
                        target['labels'] = torch.ones(len(keep), dtype=torch.int64)
                    else:
                        target['boxes'] = None
                        target['labels'] = None
                
                return image_resized, target
        
        # Fallback: center crop
        crop_h = int(h * 0.8)
        crop_w = int(w * 0.8)
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        
        image = image[:, top:top+crop_h, left:left+crop_w]
        image = F.resize(image, self.size)
        
        return image, target


class Mosaic:
    """Mosaic augmentation: combine 4 images into one - HUGE for small datasets."""
    
    def __init__(self, size, prob=0.5):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.prob = prob
        self.dataset_cache = None
        self.indices_cache = None
    
    def __call__(self, image, target):
        if random.random() < self.prob and self.dataset_cache is not None:
            # Get 3 additional random images
            other_indices = random.sample(self.indices_cache, 3)
            other_images = []
            other_targets = []
            
            for idx in other_indices:
                img, tgt = self.dataset_cache[idx]
                # Resize to half size
                img = F.resize(img, (self.size[0] // 2, self.size[1] // 2))
                # Adjust boxes
                if tgt['boxes'] is not None:
                    boxes = tgt['boxes'].clone()
                    boxes[:, [0, 2]] *= (self.size[1] // 2) / img.shape[2]
                    boxes[:, [1, 3]] *= (self.size[0] // 2) / img.shape[1]
                    tgt['boxes'] = boxes
                other_images.append(img)
                other_targets.append(tgt)
            
            # Resize main image to half
            main_img = F.resize(image, (self.size[0] // 2, self.size[1] // 2))
            if target['boxes'] is not None:
                boxes = target['boxes'].clone()
                boxes[:, [0, 2]] *= (self.size[1] // 2) / image.shape[2]
                boxes[:, [1, 3]] *= (self.size[0] // 2) / image.shape[1]
                target['boxes'] = boxes
            
            # Create mosaic
            mosaic_img = torch.zeros(3, self.size[0], self.size[1])
            
            # Top-left
            mosaic_img[:, :self.size[0]//2, :self.size[1]//2] = main_img
            # Top-right
            mosaic_img[:, :self.size[0]//2, self.size[1]//2:] = other_images[0]
            # Bottom-left
            mosaic_img[:, self.size[0]//2:, :self.size[1]//2] = other_images[1]
            # Bottom-right
            mosaic_img[:, self.size[0]//2:, self.size[1]//2:] = other_images[2]
            
            # Combine boxes and adjust offsets
            all_boxes = []
            if target['boxes'] is not None:
                for box in target['boxes']:
                    all_boxes.append(box)
            
            # Adjust offsets for other images
            offsets = [(0, 0), (0, self.size[1]//2), 
                      (self.size[0]//2, 0), (self.size[0]//2, self.size[1]//2)]
            
            for i, tgt in enumerate(other_targets):
                if tgt['boxes'] is not None:
                    for box in tgt['boxes']:
                        box[0] += offsets[i+1][1]
                        box[1] += offsets[i+1][0]
                        box[2] += offsets[i+1][1]
                        box[3] += offsets[i+1][0]
                        all_boxes.append(box)
            
            if all_boxes:
                target['boxes'] = torch.stack(all_boxes)
                target['labels'] = torch.ones(len(all_boxes), dtype=torch.int64)
            
            image = mosaic_img
        
        return image, target
    
    def set_dataset(self, dataset, indices):
        """Cache dataset for Mosaic."""
        self.dataset_cache = dataset
        self.indices_cache = indices


def get_train_transform_for_tiny_dataset(mean=None, std=None, img_size=(512, 512)):
    """
    EXTREME augmentation strategy for datasets with <50 images.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    # This will be set later
    dataset_cache = None
    indices_cache = None
    
    transforms_list = []
    
    # === STAGE 1: HEAVY COLOR & NOISE (applied early) ===
    transforms_list.extend([
        ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.3, prob=0.9),
        RandomSolarize(threshold=0.5, prob=0.3),
        RandomPosterize(bits=3, prob=0.3),
        RandomAutocontrast(prob=0.3),
        RandomEqualize(prob=0.3),
        RandomSharpness(factor=3.0, prob=0.4),
    ])
    
    # === STAGE 2: HEAVY NOISE & BLUR ===
    transforms_list.extend([
        RandomGaussianNoise(std=0.1, prob=0.5),
        RandomSaltPepperNoise(salt_prob=0.03, pepper_prob=0.03, prob=0.3),
        RandomGaussianBlur(kernel_size=7, sigma=(0.1, 3.0), prob=0.6),
    ])
    
    # === STAGE 3: AGGRESSIVE GEOMETRIC ===
    transforms_list.append(
        RandomResizedCropWithBoxes(size=img_size, scale=(0.3, 1.0), ratio=(0.7, 1.4))
    )
    
    transforms_list.extend([
        RandomHorizontalFlip(prob=0.5),
        RandomVerticalFlip(prob=0.3),
        RandomRotation(degrees=30, prob=0.7),
        RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.7, 1.3), 
                    shear=20, prob=0.6),
    ])
    
    # === STAGE 4: OCCLUSION ===
    transforms_list.extend([
        RandomErasing(prob=0.4, scale=(0.02, 0.25), ratio=(0.3, 3.3), fill='random'),
        RandomCutout(size=0.15, prob=0.4),
    ])
    
    # === STAGE 5: DEFORMATION ===
    transforms_list.append(
        RandomElasticTransform(alpha=80.0, sigma=8.0, prob=0.5)
    )
    
    # === STAGE 6: FINAL PROCESSING ===
    transforms_list.extend([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    
    transform = Compose(transforms_list)
    
    # Add MixUp, CutMix, Mosaic as post-compose transforms
    class AdvancedAugmentation:
        def __init__(self, transform, dataset, indices):
            self.transform = transform
            self.mixup = MixUp(alpha=0.4, prob=0.5)
            self.cutmix = CutMix(alpha=0.4, prob=0.4)
            self.mosaic = Mosaic(size=img_size, prob=0.3)
            
            # Set datasets
            self.mixup.set_dataset(dataset, indices)
            self.cutmix.set_dataset(dataset, indices)
            self.mosaic.set_dataset(dataset, indices)
        
        def __call__(self, image, target):
            # Apply base transforms
            image, target = self.transform(image, target)
            
            # Apply advanced augmentations randomly
            aug_type = random.random()
            if aug_type < 0.2:  # 20% MixUp
                image, target = self.mixup(image, target)
            elif aug_type < 0.4:  # 20% CutMix
                image, target = self.cutmix(image, target)
            elif aug_type < 0.5:  # 10% Mosaic
                image, target = self.mosaic(image, target)
            
            return image, target
    
    return transform, AdvancedAugmentation


def compute_dataset_stats_with_augmentation(dataset, num_samples=500):
    """
    Compute stats with augmentation to better represent transformed data.
    """
    print("Computing dataset statistics with augmentation...")
    
    # Create a temporary transform with normalization disabled
    temp_transform = Compose([
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, prob=0.5),
        RandomGaussianBlur(kernel_size=5, sigma=(0.1, 2.0), prob=0.3),
        RandomHorizontalFlip(prob=0.3),
        ToTensor(),
    ])
    
    means = []
    stds = []
    
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for idx in indices:
        image, _ = dataset[idx]
        
        # Apply augmentation
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        
        image_aug, _ = temp_transform(image, {'boxes': None, 'labels': None, 'frame_index': ''})
        
        means.append(image_aug.mean(dim=(1, 2)))
        stds.append(image_aug.std(dim=(1, 2)))
    
    mean = torch.stack(means).mean(dim=0)
    std = torch.stack(stds).mean(dim=0)
    
    print(f"Dataset mean (with augmentation): {mean.tolist()}")
    print(f"Dataset std (with augmentation): {std.tolist()}")
    
    return mean, std


def get_val_transform(mean=None, std=None):
    """
    Get validation transforms (no augmentation, just normalization).
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]  # ImageNet default
    if std is None:
        std = [0.229, 0.224, 0.225]   # ImageNet default
    
    return Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])


# Old functions :
def get_train_transform(mean=None, std=None, img_size=None):
    """
    Get training transforms with EXTREME data augmentation.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    transforms_list = []
    
    # 1. COLOR AUGMENTATIONS (applied early, before geometric)
    transforms_list.extend([
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, prob=0.9),
        RandomSolarize(threshold=0.5, prob=0.2),
        RandomPosterize(bits=4, prob=0.2),
        RandomAutocontrast(prob=0.2),
        RandomEqualize(prob=0.2),
        RandomSharpness(factor=2.0, prob=0.3),
    ])
    
    # 2. NOISE & BLUR
    transforms_list.extend([
        RandomGaussianNoise(std=0.05, prob=0.3),
        RandomSaltPepperNoise(salt_prob=0.01, pepper_prob=0.01, prob=0.2),
        RandomGaussianBlur(kernel_size=5, sigma=(0.1, 2.0), prob=0.4),
    ])
    
    # 3. GEOMETRIC AUGMENTATIONS
    if img_size is not None:
        transforms_list.append(
            RandomResizedCrop(size=img_size, scale=(0.5, 1.0), ratio=(0.8, 1.2))
        )
    
    transforms_list.extend([
        RandomHorizontalFlip(prob=0.5),
        RandomVerticalFlip(prob=0.2),
        RandomRotation(degrees=20, prob=0.5),  # Increased rotation
        RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10, prob=0.4),
    ])
    
    # 4. OCCLUSION AUGMENTATIONS
    transforms_list.extend([
        RandomErasing(prob=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), fill='random'),
        RandomCutout(size=0.1, prob=0.3),
    ])
    
    # 5. ELASTIC DEFORMATIONS (good for tokamak plasma)
    transforms_list.append(
        RandomElasticTransform(alpha=30.0, sigma=5.0, prob=0.2)
    )
    
    # 6. FINAL PROCESSING
    transforms_list.extend([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    
    return Compose(transforms_list)


def get_extreme_train_transform(mean=None, std=None, img_size=None):
    """
    EXTREME version - maximum augmentation for very small datasets.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    return Compose([
        # Color (extreme)
        ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.3, prob=0.95),
        RandomSolarize(threshold=0.5, prob=0.3),
        RandomPosterize(bits=3, prob=0.3),
        RandomAutocontrast(prob=0.3),
        RandomEqualize(prob=0.3),
        RandomSharpness(factor=3.0, prob=0.4),
        
        # Noise (extreme)
        RandomGaussianNoise(std=0.1, prob=0.4),
        RandomSaltPepperNoise(salt_prob=0.02, pepper_prob=0.02, prob=0.3),
        RandomGaussianBlur(kernel_size=7, sigma=(0.1, 3.0), prob=0.5),
        
        # Geometric (extreme)
        RandomResizedCrop(size=img_size, scale=(0.3, 1.0), ratio=(0.5, 2.0)) if img_size else None,
        RandomHorizontalFlip(prob=0.5),
        RandomVerticalFlip(prob=0.3),
        RandomRotation(degrees=30, prob=0.6),
        RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.7, 1.3), shear=15, prob=0.5),
        
        # Occlusion (extreme)
        RandomErasing(prob=0.5, scale=(0.02, 0.3), ratio=(0.3, 3.3), fill='random'),
        RandomCutout(size=0.15, prob=0.4),
        
        # Deformation
        RandomElasticTransform(alpha=50.0, sigma=6.0, prob=0.3),
        
        # Spatial shuffling (risky for detection)
        # RandomGridShuffle(grid_size=(2, 2), prob=0.1),
        
        # Final processing
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])


def compute_dataset_stats(dataset, num_samples=1000):
    """
    Compute mean and std of the dataset for normalization.
    """
    print("Computing dataset statistics for normalization...")
    
    means = []
    stds = []
    
    # Sample random indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for idx in indices:
        image, _ = dataset[idx]
        
        # Ensure image is tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        
        # Compute mean and std per channel
        means.append(image.mean(dim=(1, 2)))
        stds.append(image.std(dim=(1, 2)))
    
    # Calculate overall mean and std
    mean = torch.stack(means).mean(dim=0)
    std = torch.stack(stds).mean(dim=0)
    
    print(f"Dataset mean: {mean.tolist()}")
    print(f"Dataset std: {std.tolist()}")
    
    return mean, std


def get_train_transform_old(mean=None, std=None):
    """
    Get training transforms with data augmentation.
    If mean and std are not provided, will use ImageNet defaults.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]  # ImageNet default
    if std is None:
        std = [0.229, 0.224, 0.225]   # ImageNet default
    
    return Compose([
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, prob=0.8),
        RandomGaussianBlur(kernel_size=5, sigma=(0.1, 2.0), prob=0.3),
        RandomHorizontalFlip(prob=0.5),
        RandomVerticalFlip(prob=0.1),  # Less common in natural images but could be useful for tokamak
        RandomRotation(degrees=10, prob=0.3),
        # RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=3, prob=0.3),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
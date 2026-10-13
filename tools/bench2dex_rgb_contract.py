"""Three-view composition copied from the task21 baseline contract; no robot assumptions."""
import numpy as np

CAMERAS = ("cam_overhead", "cam_wrist_left", "cam_wrist_right")

def compose_rgb(images, width=640):
    """Return uint8 CHW: overhead above two wrists, aspect-preserving letterbox."""
    import cv2

    if width < 8 or width % 4:
        raise ValueError("view_width must be >=8 and divisible by 4")

    def fit(image, w, h):
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
            raise ValueError("Expected nonempty uint8 HWC RGB")
        scale = min(w / image.shape[1], h / image.shape[0])
        rw, rh = (
            max(1, round(image.shape[1] * scale)),
            max(1, round(image.shape[0] * scale)),
        )
        result = np.zeros((h, w, 3), np.uint8)
        result[(h - rh) // 2 : (h - rh) // 2 + rh, (w - rw) // 2 : (w - rw) // 2 + rw] = cv2.resize(
            image,
            (rw, rh),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        return result

    top = fit(images[CAMERAS[0]], width, width * 3 // 4)
    bottom = np.concatenate([fit(images[k], width // 2, width * 3 // 8) for k in CAMERAS[1:]], axis=1)
    return np.ascontiguousarray(np.concatenate([top, bottom], axis=0).transpose(2, 0, 1))

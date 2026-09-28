import os
import queue
import threading

import numpy as np
import torch


def mode():
    v = (os.environ.get("GFPGAN_FAST") or "1").strip().lower()
    if v in ("0", "off", "false", "no"):
        return "off"
    if v in ("lossless", "exact"):
        return "lossless"
    return "full"


def enabled(sw):
    m = mode()
    if m == "off":
        return False
    if m == "lossless":
        return sw in ("sw1", "sw2", "sw3")
    return True


def _pre_hook_1(_module, args):
    if args and torch.is_tensor(args[0]):
        return (args[0].contiguous(memory_format=torch.channels_last),) + tuple(args[1:])
    return None


def tune_bg_upsampler(bg):
    import types
    import cv2

    if not enabled('sw1'):
        return bg

    if enabled('sw7'):
        bg.model.to(memory_format=torch.channels_last)
        head = getattr(bg.model, 'conv_first', None) or bg.model
        head.register_forward_pre_hook(_pre_hook_1)

    if enabled('sw4'):
        try:
            from basicsr.archs.rrdbnet_arch import RRDB
            if not getattr(RRDB, '_ao_opt_done', False):
                RRDB.forward = torch.compile(RRDB.forward)
                RRDB._ao_opt_done = True
        except Exception:                                         # noqa: BLE001
            pass

    stock_enhance = type(bg).enhance

    def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
        if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
            return stock_enhance(self, img, outscale=outscale, alpha_upsampler=alpha_upsampler)
        h_input, w_input = img.shape[0:2]
        self.pre_process(cv2.cvtColor(img.astype(np.float32) / 255., cv2.COLOR_BGR2RGB))
        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()
        out = self.post_process().detach().squeeze(0).float().clamp_(0, 1)
        out = (out[[2, 1, 0], :, :] * 255.0).round_().to(torch.uint8)
        output = out.permute(1, 2, 0).contiguous().cpu().numpy()
        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(output, (int(w_input * outscale), int(h_input * outscale)),
                                interpolation=cv2.INTER_LANCZOS4)
        return output, 'RGB'

    bg.enhance = types.MethodType(enhance, bg)
    return bg


class paste_context:

    def __init__(self, helper):
        self.helper = helper
        self.on = enabled('sw2') or enabled('sw5')

    def __enter__(self):
        if not self.on:
            return self
        import cv2
        self._cv2 = cv2
        self._resize, self._warp, self._blur = cv2.resize, cv2.warpAffine, cv2.GaussianBlur
        exact = enabled('sw2')
        f32 = enabled('sw5')
        resize, warp, blur = self._resize, self._warp, self._blur

        def _f32(src):
            if f32 and getattr(src, 'dtype', None) == np.float64:
                return src.astype(np.float32)
            return src

        def resize_(src, dsize, *a, **kw):
            if (exact and dsize is not None and not kw.get('fx') and not kw.get('fy')
                    and getattr(src, 'ndim', 0) >= 2
                    and src.shape[1] == dsize[0] and src.shape[0] == dsize[1]):
                return src.copy()
            return resize(_f32(src), dsize, *a, **kw)

        def warp_(src, M, dsize, *a, **kw):
            return warp(_f32(src), M, dsize, *a, **kw)

        def blur_(src, ksize, sigmaX, *a, **kw):
            return blur(_f32(src), ksize, sigmaX, *a, **kw)

        cv2.resize, cv2.warpAffine, cv2.GaussianBlur = resize_, warp_, blur_
        return self

    def __exit__(self, *exc):
        if self.on:
            self._cv2.resize = self._resize
            self._cv2.warpAffine = self._warp
            self._cv2.GaussianBlur = self._blur
        return False


class OutputWriter:

    def __init__(self, enable=True):
        self.on = enable and enabled('sw3')
        if not self.on:
            return
        self.q = queue.Queue(maxsize=1)
        self.err = None
        self.t = threading.Thread(target=self._loop, name='gfpgan-writer', daemon=True)
        self.t.start()

    def _loop(self):
        from basicsr.utils import imwrite
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                return
            img, path = item
            try:
                imwrite(img, path)
            except Exception as exc:                              # noqa: BLE001
                self.err = exc
            finally:
                self.q.task_done()

    def write(self, img, path):
        from basicsr.utils import imwrite
        if not self.on:
            return imwrite(img, path)
        if self.err is not None:
            raise self.err
        self.q.put((img, path))

    def close(self):
        if not self.on:
            return
        self.q.join()
        if self.err is not None:
            raise self.err

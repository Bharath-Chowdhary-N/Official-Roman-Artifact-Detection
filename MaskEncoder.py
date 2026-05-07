import base64
import gzip
import numpy as np
from typing import Union


class MaskEncoder:
    """
    Encoder for binary segmentation masks → custom compact format.
    Exact inverse of MaskDecoder:
      1. Flatten 2D mask to 1D.
      2. Bit-pack: 8 pixels per byte, MSB first (pixel 0 → bit 7).
      3. RLE encode the packed bytes: [count, value, count, value, ...].
      4. Optionally encode as 'base64' or 'gzip-base64' (default).

    Supports three output formats:
      - 'array'      : plain numpy uint8 RLE array (largest)
      - 'base64'     : RLE bytes as ASCII base64 string (medium)
      - 'gzip-base64': RLE bytes gzip-compressed then base64-encoded (smallest)
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.total_pixels = width * height

    # ---------------------------------------------------------
    # Step 1 — Bit-pack flat binary mask into bytes
    # ---------------------------------------------------------
    def pack_bits(self, flat_mask: np.ndarray) -> np.ndarray:
        """
        Pack flat binary mask (values 0/1) into bytes, 8 pixels per byte,
        MSB first: pixel[i*8 + j] → bit (7-j) of byte i.
        Mirrors MaskDecoder.unpack_bits exactly.
        """
        flat = np.asarray(flat_mask, dtype=np.uint8).ravel()
        pad = (-len(flat)) % 8
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
        n_bytes = len(flat) // 8
        packed = np.zeros(n_bytes, dtype=np.uint8)
        for bit_pos in range(8):
            packed |= (flat[bit_pos::8] & 1).astype(np.uint8) << (7 - bit_pos)
        return packed

    # ---------------------------------------------------------
    # Step 2 — RLE encode packed byte array
    # ---------------------------------------------------------
    def rle_encode(self, packed: np.ndarray) -> np.ndarray:
        """
        RLE encode byte array as [count, value, count, value, ...].
        Count is capped at 255 so all values fit in uint8.
        Mirrors MaskDecoder.rle_decode exactly.
        """
        out = []
        i = 0
        while i < len(packed):
            val = int(packed[i])
            count = 1
            while i + count < len(packed) and int(packed[i + count]) == val and count < 255:
                count += 1
            out.extend([count, val])
            i += count
        return np.array(out, dtype=np.uint8)

    # ---------------------------------------------------------
    # Step 3 — Encode RLE array to chosen format
    # ---------------------------------------------------------
    def encode_rle(
        self, rle_arr: np.ndarray, encoding: str = 'gzip-base64'
    ) -> Union[str, np.ndarray]:
        """
        Encode RLE array to the specified format.
        Mirrors MaskDecoder.decode_encoded_rle exactly.
        """
        if encoding == 'array':
            return rle_arr
        elif encoding == 'base64':
            return base64.b64encode(rle_arr.tobytes()).decode('ascii')
        elif encoding == 'gzip-base64':
            compressed = gzip.compress(rle_arr.tobytes())
            return base64.b64encode(compressed).decode('ascii')
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

    # ---------------------------------------------------------
    # Full encode pipeline
    # ---------------------------------------------------------
    def encode(
        self,
        mask_2d: np.ndarray,
        encoding: str = 'gzip-base64',
    ) -> Union[str, np.ndarray]:
        """
        Full pipeline: 2D binary mask → encoded RLE string (or array).

        Args:
            mask_2d:  2D numpy array of shape (height, width) with values {0, 1}.
            encoding: 'array', 'base64', or 'gzip-base64' (default).

        Returns:
            Encoded RLE — ASCII string for base64/gzip-base64, numpy array for 'array'.
        """
        flat = mask_2d.ravel().astype(np.uint8)
        packed = self.pack_bits(flat)
        rle = self.rle_encode(packed)
        return self.encode_rle(rle, encoding)

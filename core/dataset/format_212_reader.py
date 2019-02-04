from io import BufferedReader
from core.util.middle_byte_decoder_212 import middle_byte_decoder
import numpy as np


class Format212reader:
    """
        Each sample is represented by a 12-bit two's complement amplitude. The first sample is obtained from the 12
        least significant bits of the first byte pair (stored least significant byte first).  The second sample is
        formed from the 4 remaining bits of the first byte pair (which are the 4 high bits of the 12-bit sample) and the
        next byte (which contains the remaining 8 bits of the second sample). The process is repeated for each
        successive pair of samples. Most of the signal files in PhysioBank are written in format 212.
        """

    @classmethod
    def from_bytestream(cls, byte_stream: BufferedReader, read_size=None):
        signal = np.array([])
        chunk = byte_stream.read(read_size)
        while chunk:
            signal = np.concatenate((signal, cls.from_bytes(chunk)))
            chunk = byte_stream.read(read_size)
        return signal

    @classmethod
    def from_bytes(cls, bytes_stream):
        if len(bytes_stream) % 3 != 0:
            raise ValueError(f"byte stream length is not a multiple of 3: {len(bytes_stream)}")
        blocks = len(bytes_stream) // 3
        s_left = np.ndarray((blocks,), "B", bytes_stream[::3])
        middle = np.ndarray((blocks,), "B", bytes_stream[1::3])
        s_right = np.ndarray((blocks,), "B", bytes_stream[2::3])
        middle_byte = middle_byte_decoder[middle]
        signal = np.empty((s_left.size + s_right.size,), dtype=np.int16)
        signal[0::2] = s_left + middle_byte[:, 0]
        signal[1::2] = s_right + middle_byte[:, 1]
        return signal

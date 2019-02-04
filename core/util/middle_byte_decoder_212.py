import numpy as np

middle_byte_decoder = np.array(
    [[0, 0], [256, 0], [512, 0], [768, 0], [1024, 0], [1280, 0], [1536, 0], [1792, 0], [-2048, 0], [-1792, 0],
     [-1536, 0], [-1280, 0], [-1024, 0], [-768, 0], [-512, 0], [-256, 0], [0, 256], [256, 256], [512, 256], [768, 256],
     [1024, 256], [1280, 256], [1536, 256], [1792, 256], [-2048, 256], [-1792, 256], [-1536, 256], [-1280, 256],
     [-1024, 256], [-768, 256], [-512, 256], [-256, 256], [0, 512], [256, 512], [512, 512], [768, 512], [1024, 512],
     [1280, 512], [1536, 512], [1792, 512], [-2048, 512], [-1792, 512], [-1536, 512], [-1280, 512], [-1024, 512],
     [-768, 512], [-512, 512], [-256, 512], [0, 768], [256, 768], [512, 768], [768, 768], [1024, 768], [1280, 768],
     [1536, 768], [1792, 768], [-2048, 768], [-1792, 768], [-1536, 768], [-1280, 768], [-1024, 768], [-768, 768],
     [-512, 768], [-256, 768], [0, 1024], [256, 1024], [512, 1024], [768, 1024], [1024, 1024], [1280, 1024],
     [1536, 1024], [1792, 1024], [-2048, 1024], [-1792, 1024], [-1536, 1024], [-1280, 1024], [-1024, 1024],
     [-768, 1024], [-512, 1024], [-256, 1024], [0, 1280], [256, 1280], [512, 1280], [768, 1280], [1024, 1280],
     [1280, 1280], [1536, 1280], [1792, 1280], [-2048, 1280], [-1792, 1280], [-1536, 1280], [-1280, 1280],
     [-1024, 1280], [-768, 1280], [-512, 1280], [-256, 1280], [0, 1536], [256, 1536], [512, 1536], [768, 1536],
     [1024, 1536], [1280, 1536], [1536, 1536], [1792, 1536], [-2048, 1536], [-1792, 1536], [-1536, 1536], [-1280, 1536],
     [-1024, 1536], [-768, 1536], [-512, 1536], [-256, 1536], [0, 1792], [256, 1792], [512, 1792], [768, 1792],
     [1024, 1792], [1280, 1792], [1536, 1792], [1792, 1792], [-2048, 1792], [-1792, 1792], [-1536, 1792], [-1280, 1792],
     [-1024, 1792], [-768, 1792], [-512, 1792], [-256, 1792], [0, -2048], [256, -2048], [512, -2048], [768, -2048],
     [1024, -2048], [1280, -2048], [1536, -2048], [1792, -2048], [-2048, -2048], [-1792, -2048], [-1536, -2048],
     [-1280, -2048], [-1024, -2048], [-768, -2048], [-512, -2048], [-256, -2048], [0, -1792], [256, -1792],
     [512, -1792], [768, -1792], [1024, -1792], [1280, -1792], [1536, -1792], [1792, -1792], [-2048, -1792],
     [-1792, -1792], [-1536, -1792], [-1280, -1792], [-1024, -1792], [-768, -1792], [-512, -1792], [-256, -1792],
     [0, -1536], [256, -1536], [512, -1536], [768, -1536], [1024, -1536], [1280, -1536], [1536, -1536], [1792, -1536],
     [-2048, -1536], [-1792, -1536], [-1536, -1536], [-1280, -1536], [-1024, -1536], [-768, -1536], [-512, -1536],
     [-256, -1536], [0, -1280], [256, -1280], [512, -1280], [768, -1280], [1024, -1280], [1280, -1280], [1536, -1280],
     [1792, -1280], [-2048, -1280], [-1792, -1280], [-1536, -1280], [-1280, -1280], [-1024, -1280], [-768, -1280],
     [-512, -1280], [-256, -1280], [0, -1024], [256, -1024], [512, -1024], [768, -1024], [1024, -1024], [1280, -1024],
     [1536, -1024], [1792, -1024], [-2048, -1024], [-1792, -1024], [-1536, -1024], [-1280, -1024], [-1024, -1024],
     [-768, -1024], [-512, -1024], [-256, -1024], [0, -768], [256, -768], [512, -768], [768, -768], [1024, -768],
     [1280, -768], [1536, -768], [1792, -768], [-2048, -768], [-1792, -768], [-1536, -768], [-1280, -768],
     [-1024, -768], [-768, -768], [-512, -768], [-256, -768], [0, -512], [256, -512], [512, -512], [768, -512],
     [1024, -512], [1280, -512], [1536, -512], [1792, -512], [-2048, -512], [-1792, -512], [-1536, -512], [-1280, -512],
     [-1024, -512], [-768, -512], [-512, -512], [-256, -512], [0, -256], [256, -256], [512, -256], [768, -256],
     [1024, -256], [1280, -256], [1536, -256], [1792, -256], [-2048, -256], [-1792, -256], [-1536, -256], [-1280, -256],
     [-1024, -256], [-768, -256], [-512, -256], [-256, -256]])

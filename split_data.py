from ultralytics.data.utils import autosplit

autosplit('datasets/val', (0.8, 0.2, 0.0))

"""
Label utilities for pneumonia detection
"""

def normalize_label(label):
    """Normalize label to standard format"""
    if isinstance(label, str):
        label = label.lower().strip()
        if label in ['normal', 'not pneumonia']:
            return 'Normal'
        elif label in ['pneumonia', 'lung opacity']:
            return 'Pneumonia'
    return label
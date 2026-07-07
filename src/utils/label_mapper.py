from config.constants import LABEL_INDEX_TO_NAME

def get_class_name(label_idx):
    return LABEL_INDEX_TO_NAME.get(label_idx, "Unknown")

def get_all_labels():
    return list(LABEL_INDEX_TO_NAME.values())
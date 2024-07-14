ELEMENTS_CONFIG = []
frame = None
should_play_sound = False

PERSON = 'person'
# coco labels
COCO_SAFE_LABELS = ['dog', 'cat', 'backpack', 'sports ball', 'donut', 'tv', 'laptop', 'book', 'teddy bear']
COCO_DANGEROUS_LABELS = ['wine glass', 'cup', 'fork', 'knife', 'remote', 'oven', 'toaster', 'vase',
                         'potted plant', 'scissors', 'hair drier', 'skateboard', 'bottle', "baseball bat",  "bowl",  "sports ball", "banana", "apple", 'broccoli', 'carrot']

SHARP_LABELS = ['wine glass', 'fork', 'knife', 'vase', 'potted plant', 'scissors', 'bottle', "baseball bat"]

HOT_LABELS = ['cup', 'oven', 'toaster', 'hair drier', "bowl"]

CHOKE_LABELS = ['bottle', "banana", "apple", 'broccoli', 'carrot']

FALL_LABELS = ['skateboard', "sports ball"]


# imagenet 100

dangerous_labels = SHARP_LABELS + HOT_LABELS + CHOKE_LABELS + FALL_LABELS
# dangerous_labels = ['bottle', 'banana']

MIN_DIST_THRESHOLD = 100
BOT_STR_TOKEN = "6837047207:AAEgO2dR4sRu-vyqFhdgcTlOb0blIEdtN5M"
BOT_ID = '@ToddlerGuardbot'

# colors for bounding boxes
class Colors:
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    PURPLE = (128, 0, 128)
    PINK = (160, 64, 240)
    ORANGE = (255, 165, 0)
    YELLOW = (255, 255, 0)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)



# Dictionary to map label categories to their colors
label_category_colors = {
    "COCO_SAFE_LABELS": Colors.GREEN,
    "HOT_LABELS": Colors.RED,
    "SHARP_LABELS": Colors.ORANGE,
    "CHOKE_LABELS": Colors.BLUE,
    "PERSON": Colors.PURPLE
}

# Mapping of categories to their respective label sets
label_categories = {
    "COCO_SAFE_LABELS": COCO_SAFE_LABELS,
    "HOT_LABELS": HOT_LABELS,
    "SHARP_LABELS": SHARP_LABELS,
    "CHOKE_LABELS": CHOKE_LABELS,
    "PERSON": {PERSON}
}

# Generate the label_color_mapping dictionary
label_color_mapping = {
    label: color
    for category, color in label_category_colors.items()
    for label in label_categories[category]
}



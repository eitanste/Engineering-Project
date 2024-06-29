PERSON = 'person'
# coco labels
cocc_safe_labels = ['dog', 'cat', 'backpack', 'sports ball', 'bottle', 'banana', 'apple', 'orange', 'donut',
                    'tv', 'laptop', 'cell phone', 'book', 'teddy bear']
coco_dangerous_labels = ['wine glass', 'cup', 'fork', 'knife', 'remote', 'oven', 'toaster', 'vase',
                         'potted plant', 'scissors', 'hair drier', 'skateboard', 'bottle']

# imagenet 100

dangerous_labels = coco_dangerous_labels
# dangerous_labels = ['bottle', 'banana']

MIN_DIST_THRESHOLD = 100
BOT_STR_TOKEN = "6837047207:AAEgO2dR4sRu-vyqFhdgcTlOb0blIEdtN5M"
BOT_ID = '@ToddlerGuardbot'
FRAME_SAMPLE_PARAMETER = 5

# colors for bounding boxes
GREEN_COLOR = (0, 255, 0)
RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)

import os

REPO_ID="tejacherukuri/guided-context-gating"
WEIGHT_FILE="gcg.weights.keras"
LABELENCODER_FILE="labelencoder.pkl"
# Files should be loaded from Huggingface
FROM_HF=True

ROOT_DIR = os.getcwd()

# Dataset details
DATA_PATH = "/Users/tejacherukuri/TReNDS/MyResearch/Datasets/Zenodo-DR7"
LABELS = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Very Severe NPDR', 'PDR', 'Advanced PDR']
IMAGE_SIZE = (512,512,3)
NUM_CLASSES = 7

# Training Parameters
EPOCHS = 100
BATCH_SIZE = 32

# Saving to local when running
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'saves', 'gcg.weights.keras')  
LABELENCODER_SAVE_PATH = os.path.join(ROOT_DIR, 'saves', 'labelencoder.pkl')
HEATMAPS_SAVE_PATH = os.path.join(ROOT_DIR, 'heatmaps')      

# Attention Layer to get features
GCG_LAYER_OUTPUT = 'attention_gate'

#Inference
TEST_IMAGES = [
    os.path.join(ROOT_DIR, 'test_images', '184_No_DR.jpg'),
    os.path.join(ROOT_DIR, 'test_images', '198_Moderate_NPDR.jpg'),
    os.path.join(ROOT_DIR, 'test_images', '440_Severe_NPDR.jpg'),
    os.path.join(ROOT_DIR, 'test_images', '500_Very_Severe_NPDR.jpg'),
    os.path.join(ROOT_DIR, 'test_images', '635_PDR.jpg'),
    os.path.join(ROOT_DIR, 'test_images', '705_Advanced_PDR.jpg')
]
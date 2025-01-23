import os

# Get the absolute path to the root directory (where gcg, test_images, etc. are located)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset details
data_path = "/Users/tejacherukuri/TReNDS/MyResearch/Datasets/Zenodo-DR7"
labels = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Very Severe NPDR', 'PDR', 'Advanced PDR']
image_size = (512,512,3)
num_classes = 7
labelencoder_save_path = os.path.join(ROOT_DIR, 'saves', 'labelencoder.pkl')
heatmaps_save_path = os.path.join(ROOT_DIR, 'heatmaps')

# Training Parameters
EPOCHS = 100
batch_size = 32
model_path = os.path.join(ROOT_DIR, 'saves', 'gcg.weights.keras')

# Attention Layer to get features
gcg_layer_name = 'attention_gate'

#Inference
test_images = [
    os.path.join(ROOT_DIR, 'test_images', '184_No_DR.jpg'),
    os.path.join(ROOT_DIR, 'test_images', '198_Moderate_NPDR.jpg'),
    os.path.join(ROOT_DIR, 'test_images', '440_Severe_NPDR.jpg'),
    os.path.join(ROOT_DIR, 'test_images', '500_Very_Severe_NPDR.jpg'),
    os.path.join(ROOT_DIR, 'test_images', '635_PDR.jpg'),
    os.path.join(ROOT_DIR, 'test_images', '705_Advanced_PDR.jpg')
]
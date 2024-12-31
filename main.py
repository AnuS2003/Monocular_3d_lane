from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from sklearn.preprocessing import PolynomialFeatures # type: ignore
from PIL import Image
import io
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchvision import models, transforms # type: ignore
import matplotlib.pyplot as plt # type: ignore
from scipy.ndimage import gaussian_filter
import base64
import os
from fastapi.middleware.cors import CORSMiddleware
import cv2

# Initialize the FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],  # Replace with your frontend origin
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the 3D Lane Detection API!"}

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define U-Net with Attention class
class UNetWithAttention(nn.Module):
    def __init__(self):
        super(UNetWithAttention, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, x):
        x1 = self.encoder[0:2](x)
        x2 = self.encoder[2:4](x1)
        x3 = self.encoder[4:](x2)
        middle = self.middle(x3)
        x4 = self.decoder[0:2](middle)
        x5 = self.decoder[2:](x4)
        return x5
    
model_2d = UNetWithAttention().to(DEVICE)

# Ensure model weights are loaded correctly
model_path_2d = Path(r'C:\Users\Anupama S\Desktop\3d\backend\app\models\lane_detection_weights.pth')
if model_path_2d.exists():
    model_2d.load_state_dict(torch.load(model_path_2d, map_location=DEVICE))
else:
    raise FileNotFoundError("Model weights file not found.")
model_2d.eval()

# Define model components
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove final layers

    def forward(self, x):
        return self.backbone(x)

class MaskDecoder(nn.Module):
    def __init__(self, input_channels):
        super(MaskDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_channels, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(256, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.decoder(x)

class CameraPoseRegressor(nn.Module):
    def __init__(self, input_channels, hidden_dim=256):
        super(CameraPoseRegressor, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels * 8 * 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x):
        return self.fc(x)

class LaneRegressor(nn.Module):
    def __init__(self, input_channels, hidden_dim=128):
        super(LaneRegressor, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels * 8 * 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x):
        return self.fc(x)

class LaneDetectionModel(nn.Module):
    def __init__(self):
        super(LaneDetectionModel, self).__init__()
        self.encoder = ImageEncoder()
        self.mask_decoder = MaskDecoder(input_channels=2048)
        self.camera_pose_regressor = CameraPoseRegressor(input_channels=2048)
        self.lane_regressor = LaneRegressor(input_channels=2048)

    def forward(self, x):
        features = self.encoder(x)
        lane_mask = self.mask_decoder(features)
        theta = self.camera_pose_regressor(features)
        lane_anchors = self.lane_regressor(features)
        return lane_mask, lane_anchors, theta

# Load the model
model = LaneDetectionModel().to(DEVICE)
model_path = Path(r'C:\Users\Anupama S\Desktop\3d\backend\app\models\reconstruction_of_3d_lane_model_weights.pth')
try:
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        raise FileNotFoundError(f"Model weights file not found at {model_path}.")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")
model.eval()


# Utility functions
# Define image preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to model's input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])
def process_lane_image(frame):
    input_tensor = preprocess(frame).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model_2d(input_tensor)
        preds = torch.sigmoid(output)
        pred_mask = (preds > 0.5).float()
    pred_mask = pred_mask.squeeze().cpu().numpy()
    kernel = np.ones((5, 5), np.uint8)
    mask_dilated = cv2.dilate(pred_mask.astype(np.uint8), kernel, iterations=2)
    mask_eroded = cv2.erode(mask_dilated, kernel, iterations=2)
    mask_blurred = cv2.GaussianBlur(mask_eroded, (5, 5), 0) * 255
    return mask_blurred.astype(np.uint8)

def preprocess_predictions(pred_mask, min_area=1000, poly_degree=2):
    """
    Preprocess the predicted lane mask to refine and map lane points.
    """
    if len(pred_mask.shape) > 2:
        pred_mask = pred_mask[:, :, 0]  # Handle 3D mask

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(pred_mask, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=2)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lane_points = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            contour_points = contour.reshape(-1, 2)
            x_points, y_points = contour_points[:, 0], contour_points[:, 1]

            poly = PolynomialFeatures(degree=poly_degree)
            x_poly = poly.fit_transform(x_points.reshape(-1, 1))

            coeffs = np.linalg.lstsq(x_poly, y_points, rcond=None)[0]
            y_fitted = np.polyval(coeffs[::-1], x_points)

            lane_points.append(list(zip(x_points, y_fitted)))
    return lane_points


def reconstruct_3d_lane(anchors, theta):
    x = np.linspace(-1, 1, num=500)  # X-coordinates across the lane

    # Ensure anchors is iterable for np.polyval
    anchors = np.atleast_1d(anchors)

    degree = 2 if len(anchors) == 3 else 3  # Quadratic or cubic fit
    y = np.polyval(np.flip(anchors[:degree+1]), x)  # Higher-degree curve

    # Compute depth (Z) considering pitch and yaw
    pitch, yaw = theta[0], theta[1]
    z = np.tan(pitch) * x + np.tan(yaw) * y

    points_3d = np.stack([x, y, z], axis=1)
    return points_3d

def plot_3d_lane(ax, points_3d, color='b'):
    ax.plot(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=color, linewidth=2)
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=color, s=5)
    ax.grid(True)

# API endpoint
@app.post("/visualize-3d-lane/")  # Fixed endpoint definition
async def visualize_3d_lane(file: UploadFile = File(...)):
    try:
        # Load and preprocess the image
        content = await file.read()
        frame = Image.open(io.BytesIO(content)).convert("RGB")

        # Transformation to tensor
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        lane_image = process_lane_image(frame)

        frame_tensor = transform(frame).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            lane_mask, lane_anchors, theta = model(frame_tensor)

        # Move outputs to CPU if needed
        lane_mask_np = lane_mask.cpu().squeeze(1).numpy()
        lane_mask_np[0] = gaussian_filter(lane_mask_np[0], sigma=2)
        lane_anchors_np = lane_anchors.cpu().numpy()
        theta_np = theta.cpu().numpy()

        # Process image tensor back to numpy
        frame_np = frame_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
        frame_np = (frame_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        frame_np = np.clip(frame_np, 0, 1)

        # Preprocess predictions
        refined_lanes = preprocess_predictions(lane_image)

        # Reconstruct 3D lanes
        reconstructed_lanes = []
        for anchors, theta in zip(lane_anchors_np, theta_np):
            for lane_anchor in anchors:
                reconstructed_lanes.append(reconstruct_3d_lane(lane_anchor, theta))

        # Plot results
        fig = plt.figure(figsize=(20, 8))
        ax1 = fig.add_subplot(1, 4, 1)
        ax1.imshow(frame)
        ax1.set_title("Input Image")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 4, 2)
        ax2.imshow(frame, alpha=0.5)
        heatmap_overlay = ax2.imshow(lane_mask_np[0], cmap="jet", alpha=0.6)
        plt.colorbar(heatmap_overlay, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title("2D Lane Detection")
        ax2.axis("off")

        ax3 = fig.add_subplot(1, 4, 3)
        ax3.imshow(lane_image, cmap='gray')
        ax3.set_title("Refined Lane Mask")
        ax3.axis("off")

        ax4 = fig.add_subplot(1, 4, 4, projection='3d')
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        for idx, lane_points in enumerate(reconstructed_lanes):
            color = colors[idx % len(colors)]
            plot_3d_lane(ax4, lane_points, color=color)

        ax4.set_title("3D Lane Reconstruction")
        ax4.set_xlabel("X (meters)")
        ax4.set_ylabel("Y (meters)")
        ax4.set_zlabel("Z (meters)")
        ax4.view_init(elev=15, azim=-60)

        plt.tight_layout()



        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        # Convert to base64
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return JSONResponse(content={"result": img_base64})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
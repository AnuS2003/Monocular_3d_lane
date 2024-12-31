# Monocular 3D Lane Detection

## Project Description

This project implements a **Monocular 3D Lane Detection** system. It uses a **FastAPI** backend for processing and a **React** frontend for the user interface. The system detects 3D lane markings from monocular image input and visualizes them in real-time.

---

## How to Use
### Models
To use the models, download them from the provided Jupyter notebook:

- **Lane Detection Model**
- **3D Lane Reconstruction Model**
  
Once downloaded, place the model files in the appropriate `backend/app/models/` directory to ensure the backend can load and use them during inference.
### Frontend (React)

1. **Install Dependencies**  
   Navigate to the `frontend/` directory and install the required packages:
   ```bash
   cd frontend
   npm install
2. **Start the Development Server**
     ```bash
     npm run dev
### Backend (FastAPI)

1. **Install Dependencies**  
   Navigate to the `frontend/` directory and install the required packages:
   ```bash
   cd backend
2. **Create a virtual environment (Windows)**
   ```bash
   python -m venv venv
   venv\Scripts\activate 
   pip install -r requirements.txt
2. **Start the Development Server**
     ```bash
     uvicorn main:app --reload
### Folder Structure
```bash
/backend
    /app
        /models          # Model weights for lane detection
        /main.py         # FastAPI backend entry point
    /test
    /venv
/frontend
    /src
        /components     # React components for the frontend UI
          /FileUpload.jsx
          /Preview.jsx
        /App.jsx         # Main React component
        /index.css
        /main.jsx
    /public
    /node_modules
    package.json        # Frontend dependencies
    vite.config.js      # Vite configuration for React build
    eslint.config.js
    index.html
    package-lock.json
    postcss.config.js
    tailwind.cinfig.js

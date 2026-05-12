## 🚀 Quick Start (Docker)

### 1. Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
- An **NVIDIA API Key** (get one from [NVIDIA Build](https://build.nvidia.com/)).

### 2. Configuration
Create a `.env` file in the root directory:
```env
NVIDIA_API_KEY=your_nvidia_api_key_here
```

### 3. Start the Server
Run the following command in the project root:
```bash
docker compose up --build
```

### 4. Access the Application
- **Frontend UI**: [http://localhost:8000/home](http://localhost:8000/home)
---

## 🛠 Manual Setup (Local)

If you prefer to run without Docker:

### 1. Requirements
- Python 3.10+
- Tesseract OCR installed on your system.

### 2. Install Dependencies
```bash
cd backend
pip install -r requirements-po-extraction.txt
```

### 3. Run the API
```bash
python api.py
```
*Note: Ensure your `.env` file is present in the `backend/` directory if running manually, or exported to your environment.*

---
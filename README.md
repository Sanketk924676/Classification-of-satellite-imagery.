# Classification of Satellite Imagery

This is a machine learning–based project designed to classify satellite images. It uses a trained ML model to process images and predict their categories.

## Features
- Upload and process satellite images.
- Classify images using a trained machine learning model.
- Simple Flask web interface.

## Installation

1. **Clone the repository**
```bash
git clone <your-repo-link>
cd Classification-of-satellite-imagery
```

2. **Install dependencies**
Make sure you have Python 3.8+ installed. Then run:
```bash
pip install -r requirements.txt
```

## Requirements
From `requirements.txt`:
- Flask
- Pillow
- NumPy
- Matplotlib

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser at `http://127.0.0.1:5000` to use the interface.

3. Upload a satellite image to classify it.

## Project Structure
```
Classification-of-satellite-imagery/
│
├── app.py                 # Main Flask application
├── model/                 # Pretrained ML model (if applicable)
├── static/                # Static files (CSS, JS, Images)
├── templates/             # HTML templates
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

## Contributing
Pull requests are welcome. For major changes, open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License.

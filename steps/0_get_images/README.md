# On the Raspbery Pi 5

### Install Requirements
```bash
git clone https://github.com/BetaUtopia/Hailo8l.git
```

```bash
sudo apt-get update
sudo apt-get install -y python3-picamera2 
pip install opencv-python Pillow picamera2 --break-system-packages
```

### Get Images
```bash
cd Hailo8l
python steps/0_get_images/from_raspberypi.py --classname Green_Cube --total_pictures 100 --onkeypress
```
<!-- # Get Images
python from_raspberypi.py --classname Green_Cube --total_pictures 100 -->

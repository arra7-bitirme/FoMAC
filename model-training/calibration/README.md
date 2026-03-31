# Standalone SoccerNet Calibration

This is a standalone version of the **No Bells Just Whistles (NBJW)** calibration module from the [SoccerNet Game State Reconstruction](https://github.com/SoccerNet/sn-gamestate) project.

## Installation

1.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Run the demo script:
    ```bash
    python demo.py
    ```

    The script will:
    - Download the pre-trained HRNet models (`SV_kp.pth` and `SV_lines.pth`) if they are not present.
    - Create a dummy `input_image.jpg` if one does not exist (you should replace this with a real soccer broadcast image).
    - Detect keypoints and lines on the pitch.
    - Compute the camera calibration parameters (pan, tilt, roll, focal length, position).

## Directory Structure

-   `nbjw_calib/`: Contains the HRNet model definitions and utility functions for keypoint extraction.
-   `sn_calibration_baseline/`: Contains the camera model and geometry utilities.
-   `demo.py`: Main script to demonstrate usage.

## License

This code is derived from the [sn-gamestate](https://github.com/SoccerNet/sn-gamestate) repository. Please refer to their license for usage terms.

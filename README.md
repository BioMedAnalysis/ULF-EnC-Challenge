# ULF-EnC-Challenge
ULF-EnC: Ultra-Low-Field MRI Image Enhancement Challenge

📌 Project Overview

The ULF-EnC Challenge is designed to advance deep learning techniques for enhancing ultra-low-field (64mT) MRI to match high-field (3T) MRI quality. This challenge provides a paired dataset of 3T and 64mT MRI scans and aims to push the boundaries of medical image enhancement by encouraging participants to develop robust algorithms that improve diagnostic capabilities.

🏆 Challenge Goals

Develop deep learning models to enhance 64mT MRI images.

Maintain anatomical integrity and clinical relevance.

Benchmark solutions using SSIM and PSNR metrics.

Promote accessibility to high-quality MRI in low-resource settings.

## 📦 ULF-EnC-Challenge

```
ULF-EnC-Challenge/
├── data/
│   ├── test/
│   │   ├── 3T/         # Ground truth high-field MRI
│   │   ├── 64mT/       # Ultra-low-field MRI
├── submissions/
│   ├── team1/          # Example team submission
│   ├── team2/          # Additional submissions
├── evaluation.py       # Evaluation script for ranking submissions
├── generate_dummy_data.py  # Script for testing with synthetic data
├── leaderboard.csv     # Generated results after evaluation
├── evaluation.log      # Log file with warnings/errors
├── README.md           # Project documentation
```

🖥️ Installation & Setup

1️⃣ Install Required Dependencies

Ensure you have Python 3.x installed, then run:
pip install numpy nibabel pandas scikit-image

git clone https://github.com/your-username/ULF-EnC-Challenge.git

cd ULF-EnC-Challenge

python generate_dummy_data.py

python evaluation.py

cat leaderboard.csv

📊 Evaluation Process

Submissions are evaluated per subject and per modality.

Two metrics are used:

Structural Similarity Index (SSIM)

Peak Signal-to-Noise Ratio (PSNR)

Mean Absolute Error (MAE)

Normalised Mean Square Error (NMSE)

Leaderboard ranks teams based on a weighted combination of SSIM (70%), PSNR (10%), MAE (10%), and NMSE (10%) is used to evaluate reconstruction quality, emphasizing structural fidelity while incorporating intensity-based and error-based metrics.

Results are stored in leaderboard.csv.

🚀 Submission Guidelines

🔹 File Naming

Each team must submit enhanced MRI images in the following format:

subject_{ID}_enhanced_{MODALITY}.nii.gz

subject_1_enhanced_T1.nii.gz
subject_1_enhanced_T2.nii.gz
subject_1_enhanced_FLAIR.nii.gz

📂 submissions/
    ├── 📂 team_name/
    │   ├── subject_1_enhanced_T1.nii.gz
    │   ├── subject_1_enhanced_T2.nii.gz
    │   ├── subject_1_enhanced_FLAIR.nii.gz
    │   ├── ...

🔹 Submission Format

All submissions must be in .nii.gz format.

Each team must upload a Docker image with their trained model.

The submission must produce 3D reconstructions for each test subject.

📜 License

This project is licensed under MIT License.

📬 Contact & Support

For inquiries, please reach out via GitHub Issues or email the organizers.

🎯 Future Work

Extend the dataset with more paired 3T-64mT scans.

Introduce additional deep learning-based evaluation metrics.

Collaborate with medical experts to refine clinical impact.

🚀 Join us in pushing the boundaries of medical imaging!


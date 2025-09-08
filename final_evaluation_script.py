import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

# === CONFIGS ===
REFERENCE_DIR = "/Testing_data/"
SUBMISSION_DIR = "/Sumbissions/"
MASKS_DIR = "/Masks/"

MODALITIES = ["T1", "T2", "FLAIR"]
NUM_TEST_CASES = [9, 34, 76, 36, 55, 50, 103, 52, 90, 63, 91, 42, 96, 48, 85]
MAX_PSNR = 32.0


def load_inputs(pred_path, ref_path, mask_path):
    """
    Load input images

    :param pred_path: enhanced image path
    :type pred_path: str
    :param ref_path: GT image path
    :type ref_path: str
    :param mask_path: mask image path
    :type mask_path: str
    :return: loaded images
    :rtype: tuple
    """
    pred_nii = nib.load(pred_path)
    ref_nii = nib.load(ref_path)
    mask_nii = nib.load(mask_path)

    pred_img = pred_nii.get_fdata()
    ref_img = ref_nii.get_fdata()
    mask_img = mask_nii.get_fdata()

    if pred_img is None or ref_img is None or mask_img is None:
        return None, None, None, None, None, None

    pred_img = (pred_img - np.min(pred_img)) / (np.max(pred_img) - np.min(pred_img))
    ref_img = (ref_img - np.min(ref_img)) / (np.max(ref_img) - np.min(ref_img))
    mask = (mask_img > 0.5).astype(np.uint8)

    pred_masked = pred_img * mask
    ref_masked = ref_img * mask

    return pred_masked, ref_masked, pred_img, ref_img, mask


# === Metrics Calculation===

def calculate_mae_masked(enhanced_img, reference_img):
    """
    Calculate MAE masked

    :param enhanced_img: enhanced image
    :type enhanced_img: numpy array
    :param reference_img: GT image
    :type reference_img: numpy array
    :return: mae value
    :rtype: float
    """
    return np.abs(enhanced_img - reference_img)


def calculate_nmse_masked(enhanced_img, reference_img, mask):
    """
    Calculate NMSE masked

    :param enhanced_img:enhanced image
    :type enhanced_img: numpy array
    :param reference_img: GT image
    :type reference_img: numpy array
    :param mask: binary mask
    :type mask: numpy array
    :return: nmse value
    :rtype: float
    """
    mse = ((enhanced_img - reference_img) ** 2)
    norm_factor = np.mean(reference_img[mask == 1] ** 2)
    nmse_masked = mse / norm_factor if norm_factor > 0 else np.inf

    return nmse_masked


def calculate_psnr_masked(pred, ref, data_range=1.0):
    """
    Calculate PSNR masked

    :param pred: enhanced image
    :type pred: numpy array
    :param ref: GT image
    :type ref: numpy array
    :param data_range: data range
    :type data_range: float
    :return: psnr value
    :rtype: float
    """
    pred = pred.astype(np.float32)
    ref = ref.astype(np.float32)
    mse = (pred - ref) ** 2
    psnr_map = 10 * np.log10(data_range ** 2 / (mse + 1e-8))

    return psnr_map


def calculate_metrics(pred_path, ref_path, mask_path):
    """
    Calculate all metrices

    :param pred_path: enhanced images saved folder path
    :type pred_path: str
    :param ref_path: ground truth reference saved folder path
    :type ref_path: str
    :param mask_path: binary masks saved folder path
    :type mask_path: str
    :return: sim_score_masked, psnr_score_masked, mae_score_masked, nmse_score_masked
    :rtype: tuple
    """
    pred_masked, ref_masked, pred_img, ref_img, mask = load_inputs(pred_path, ref_path, mask_path)

    if pred_img is None or ref_img is None:
        return None, None, None, None

    ssim_tensor = ssim(pred_masked, ref_masked, data_range=1.0, full=True)
    ssim_score_masked = np.sum(ssim_tensor[1] * mask) / np.count_nonzero(mask)

    psnr_tensor = calculate_psnr_masked(ref_masked, pred_masked, data_range=1.0)
    psnr_score_masked = np.sum(psnr_tensor * mask) / np.count_nonzero(mask)

    mae_tensor = calculate_mae_masked(pred_masked, ref_masked)
    mae_score_masked = np.sum(mae_tensor * mask) / np.count_nonzero(mask)

    nmse_tensor = calculate_nmse_masked(pred_masked, ref_masked, mask)
    nmse_score_masked = np.sum(nmse_tensor * mask) / np.count_nonzero(mask)

    if nmse_score_masked > 1:
        nmse_score_masked = 1

    return ssim_score_masked, psnr_score_masked, mae_score_masked, nmse_score_masked


def evaluate_output_dir(output_dir):
    """
    Evaluate predictions of each participant

    :param output_dir: put dir with saved data for Enhanced, Reference and Mask images
    :type output_dir: str
    :return: final score for each participant
    :rtype: float
    """
    scores = {"SSIM": [], "PSNR": [], "MAE": [], "NMSE": [], "SSIMMasked": [], "PSNRMasked": [], "MAEMasked": [],
              "NMSEMasked": []}

    for subject_id in NUM_TEST_CASES:

        subj_folder = f"POCEMR{subject_id:03}"
        enhanced_dir = os.path.join(output_dir, subj_folder, "Enhanced")
        gt_dir = os.path.join(REFERENCE_DIR, subj_folder, "3T")
        mask_dir = os.path.join(MASKS_DIR, subj_folder)

        for modality in MODALITIES:
            pred_file = os.path.join(enhanced_dir, f"{subj_folder}_{modality}.nii.gz")
            ref_file = os.path.join(gt_dir, f"{subj_folder}_{modality}.nii.gz")
            mask_file = os.path.join(mask_dir, f"{subj_folder}_T1.nii.gz")

            ssim_masked_val, psnr_masked_val, mae_masked_val, nmse_masked_val = calculate_metrics(
                pred_file, ref_file, mask_file)

            scores["SSIMMasked"].append(ssim_masked_val)
            scores["PSNRMasked"].append(psnr_masked_val)
            scores["MAEMasked"].append(mae_masked_val)
            scores["NMSEMasked"].append(nmse_masked_val)

    # Aggregate metrics
    mean_ssim_masked = np.mean(scores["SSIMMasked"])
    mean_psnr_masked = np.mean(np.array(scores["PSNRMasked"]))
    mean_mae_masked = np.mean(scores["MAEMasked"])
    mean_nmse_masked = np.mean(scores["NMSEMasked"])

    # Final score calculation
    final_score = (
            0.7 * mean_ssim_masked +
            0.1 * (mean_psnr_masked / MAX_PSNR) +
            0.1 * (1 - mean_mae_masked) +
            0.1 * (1 - mean_nmse_masked)
    )
    return final_score

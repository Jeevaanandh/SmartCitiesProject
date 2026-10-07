# ============================================================
# PDM-NATIVE TERNARY CNN
# OPTIMIZED MAC VERSION
#
# Pipeline:
#
# WAV
#   ↓
# 16 kHz mono PCM
#   ↓
# PCM → PDM  (ONE TIME)
#   ↓
# Packed PDM cache
#   ↓
# Ternary PDM CNN
#   ↓
# 4 classes
#
# Classes:
#   gunshot
#   scream
#   carcrash
#   normal
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import random
from pathlib import Path
from math import gcd

import numpy as np

from scipy.io import wavfile
from scipy.signal import resample_poly

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader
)

from tqdm.auto import tqdm

from numba import njit


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

DATASET_DIR = (
    "/Users/JeevaanandhIlayaraja/"
    "Downloads/Dataset"
)

# Cached PDM will be stored here.
#
# This avoids converting WAV → PDM repeatedly.
CACHE_DIR = (
    "/Users/JeevaanandhIlayaraja/"
    "Downloads/pdm_cache"
)


# ------------------------------------------------------------
# CLASSES
# ------------------------------------------------------------

CLASSES = [
    "gunshot",
    "scream",
    "carcrash",
    "normal"
]

NUM_CLASSES = len(CLASSES)


# ------------------------------------------------------------
# DATASET SPLIT
# ------------------------------------------------------------

TRAIN_PER_CLASS = 400
VAL_PER_CLASS = 50
TEST_PER_CLASS = 50


# ------------------------------------------------------------
# AUDIO
# ------------------------------------------------------------

TARGET_SAMPLE_RATE = 16000

# PDM clock
#
# 3.072 MHz / 16 kHz = 192
#

PDM_RATE = 3_072_000

OVERSAMPLING_RATIO = (
    PDM_RATE // TARGET_SAMPLE_RATE
)

assert (
    PDM_RATE % TARGET_SAMPLE_RATE == 0
)


# ------------------------------------------------------------
# WINDOW
# ------------------------------------------------------------

WINDOW_SECONDS = 0.5

PCM_SAMPLES_PER_WINDOW = int(
    TARGET_SAMPLE_RATE *
    WINDOW_SECONDS
)

PDM_SAMPLES_PER_WINDOW = int(
    PDM_RATE *
    WINDOW_SECONDS
)


# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------

BATCH_SIZE = 8

# Start with 5 to verify everything.
# Change to 30 after the pipeline works.
EPOCHS = 15

LEARNING_RATE = 1e-3

SEED = 42


# ------------------------------------------------------------
# MODEL CHECKPOINT
# ------------------------------------------------------------

MODEL_PATH = (
    "best_pdm_ternary_cnn.pth"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


print("=" * 60)
print("PDM-NATIVE TERNARY CNN")
print("=" * 60)

print()

print("Device:", DEVICE)

print(
    "Dataset:",
    DATASET_DIR
)

print(
    "Cache:",
    CACHE_DIR
)

print(
    "Classes:",
    CLASSES
)

print(
    "PCM sample rate:",
    TARGET_SAMPLE_RATE
)

print(
    "PDM rate:",
    PDM_RATE
)

print(
    "Oversampling ratio:",
    OVERSAMPLING_RATIO
)

print(
    "Window:",
    WINDOW_SECONDS,
    "seconds"
)

print(
    "PDM samples/window:",
    f"{PDM_SAMPLES_PER_WINDOW:,}"
)

print()


# ============================================================
# CREATE CACHE DIRECTORY
# ============================================================

os.makedirs(
    CACHE_DIR,
    exist_ok=True
)


# ============================================================
# WAV LOADING
# ============================================================

def load_wav(path):

    """
    Load WAV and convert to:

        mono
        16 kHz
        float32
        approximately [-1, +1]
    """

    sample_rate, audio = (
        wavfile.read(path)
    )

    # IMPORTANT:
    # Preserve original dtype BEFORE
    # converting to float32.

    original_dtype = audio.dtype


    # --------------------------------------------------------
    # Stereo → mono
    # --------------------------------------------------------

    if audio.ndim == 2:

        audio = audio.astype(
            np.float32
        )

        audio = np.mean(
            audio,
            axis=1
        )


    # --------------------------------------------------------
    # Integer PCM
    # --------------------------------------------------------

    elif np.issubdtype(
        original_dtype,
        np.integer
    ):

        info = np.iinfo(
            original_dtype
        )

        audio = audio.astype(
            np.float32
        )

        scale = max(
            abs(info.min),
            info.max
        )

        if scale > 0:

            audio /= scale


    # --------------------------------------------------------
    # Floating-point WAV
    # --------------------------------------------------------

    else:

        audio = audio.astype(
            np.float32
        )

        max_value = np.max(
            np.abs(audio)
        )

        if max_value > 1.0:

            audio /= max_value


    # --------------------------------------------------------
    # Resample → 16 kHz
    # --------------------------------------------------------

    if sample_rate != TARGET_SAMPLE_RATE:

        g = gcd(
            sample_rate,
            TARGET_SAMPLE_RATE
        )

        up = (
            TARGET_SAMPLE_RATE // g
        )

        down = (
            sample_rate // g
        )

        audio = resample_poly(
            audio,
            up,
            down
        ).astype(
            np.float32
        )


    # --------------------------------------------------------
    # Remove DC offset
    # --------------------------------------------------------

    audio -= np.mean(audio)


    # --------------------------------------------------------
    # Keep within valid range
    # --------------------------------------------------------

    audio = np.clip(
        audio,
        -1.0,
        1.0
    )

    return audio.astype(
        np.float32
    )


# ============================================================
# FAST SIGMA-DELTA MODULATOR
# ============================================================

@njit(cache=True)
def sigma_delta_modulator(
    signal
):

    """
    First-order 1-bit sigma-delta modulator.

    Input:
        oversampled PCM

    Output:
        PDM represented as uint8

        0 = low
        1 = high
    """

    pdm = np.empty(
        len(signal),
        dtype=np.uint8
    )

    integrator = 0.0


    for i in range(
        len(signal)
    ):

        integrator += (
            signal[i]
        )

        if integrator >= 0.0:

            pdm[i] = 1

            integrator -= 1.0

        else:

            pdm[i] = 0

            integrator += 1.0


    return pdm


# ============================================================
# PCM → PDM
# ============================================================

def pcm_to_pdm(pcm):

    """
    Convert 16-kHz PCM into
    simulated 3.072-MHz PDM.
    """

    # --------------------------------------------------------
    # Interpolate PCM → PDM rate
    # --------------------------------------------------------

    oversampled = resample_poly(
        pcm,
        PDM_RATE,
        TARGET_SAMPLE_RATE
    ).astype(
        np.float32
    )


    # --------------------------------------------------------
    # Sigma-delta
    # --------------------------------------------------------

    pdm = sigma_delta_modulator(
        oversampled
    )


    return pdm


# ============================================================
# PACK PDM
# ============================================================

def pack_pdm(pdm):

    """
    Store 8 PDM bits in one byte.

    This reduces storage by ~32x
    compared with float32.
    """

    return np.packbits(
        pdm,
        bitorder="little"
    )


# ============================================================
# FIND DATASET FILES
# ============================================================

def find_class_files():

    class_files = {}


    for class_name in CLASSES:

        class_dir = (
            Path(DATASET_DIR)
            / class_name
        )


        if not class_dir.exists():

            raise FileNotFoundError(
                f"\nClass directory "
                f"not found:\n"
                f"{class_dir}"
            )


        files = sorted(
            list(
                class_dir.glob("*.wav")
            )
            +
            list(
                class_dir.glob("*.WAV")
            )
        )


        if len(files) == 0:

            raise RuntimeError(
                f"\nNo WAV files found "
                f"in:\n{class_dir}"
            )


        class_files[
            class_name
        ] = files


        print(
            f"{class_name:10s}: "
            f"{len(files)} files"
        )


    return class_files


# ============================================================
# CREATE TRAIN / VAL / TEST SPLITS
# ============================================================

def create_splits():

    class_files = (
        find_class_files()
    )


    rng = random.Random(
        SEED
    )


    train_files = []
    val_files = []
    test_files = []


    for class_index, class_name in enumerate(
        CLASSES
    ):

        files = class_files[
            class_name
        ].copy()


        rng.shuffle(files)


        required = (
            TRAIN_PER_CLASS
            +
            VAL_PER_CLASS
            +
            TEST_PER_CLASS
        )


        if len(files) < required:

            raise RuntimeError(
                f"{class_name} has "
                f"{len(files)} files, "
                f"but {required} "
                f"are required."
            )


        train = files[
            :TRAIN_PER_CLASS
        ]


        val = files[
            TRAIN_PER_CLASS:
            TRAIN_PER_CLASS
            +
            VAL_PER_CLASS
        ]


        test = files[
            TRAIN_PER_CLASS
            +
            VAL_PER_CLASS:
            required
        ]


        train_files.extend(
            [
                (str(f), class_index)
                for f in train
            ]
        )


        val_files.extend(
            [
                (str(f), class_index)
                for f in val
            ]
        )


        test_files.extend(
            [
                (str(f), class_index)
                for f in test
            ]
        )


    rng.shuffle(
        train_files
    )

    rng.shuffle(
        val_files
    )

    rng.shuffle(
        test_files
    )


    return (
        train_files,
        val_files,
        test_files
    )


# ============================================================
# PREPROCESS ONE AUDIO FILE
# ============================================================

def create_pdm_window(
    wav_path,
    split_name
):

    # --------------------------------------------------------
    # WAV → PCM
    # --------------------------------------------------------

    pcm = load_wav(
        wav_path
    )


    # --------------------------------------------------------
    # Select window
    # --------------------------------------------------------

    if len(pcm) < PCM_SAMPLES_PER_WINDOW:

        padded = np.zeros(
            PCM_SAMPLES_PER_WINDOW,
            dtype=np.float32
        )

        padded[
            :len(pcm)
        ] = pcm

        pcm_window = padded


    else:

        max_start = (
            len(pcm)
            -
            PCM_SAMPLES_PER_WINDOW
        )


        # Training:
        # random window
        #
        # Validation/test:
        # center window

        if split_name == "train":

            start = random.randint(
                0,
                max_start
            )

        else:

            start = (
                max_start // 2
            )


        pcm_window = pcm[
            start:
            start +
            PCM_SAMPLES_PER_WINDOW
        ]


    # --------------------------------------------------------
    # PCM → PDM
    # --------------------------------------------------------

    pdm = pcm_to_pdm(
        pcm_window
    )


    # --------------------------------------------------------
    # Make exact length
    # --------------------------------------------------------

    if len(pdm) >= PDM_SAMPLES_PER_WINDOW:

        pdm = pdm[
            :PDM_SAMPLES_PER_WINDOW
        ]

    else:

        padded = np.zeros(
            PDM_SAMPLES_PER_WINDOW,
            dtype=np.uint8
        )

        padded[
            :len(pdm)
        ] = pdm

        pdm = padded


    return pdm


# ============================================================
# PRECOMPUTE PDM CACHE
# ============================================================

def preprocess_split(
    files,
    split_name
):

    split_dir = (
        Path(CACHE_DIR)
        / split_name
    )


    split_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    print()
    print(
        "=" * 60
    )

    print(
        f"PREPROCESSING: "
        f"{split_name.upper()}"
    )

    print(
        "=" * 60
    )


    for index, (
        wav_path,
        label
    ) in enumerate(
        tqdm(files)
    ):

        output_path = (
            split_dir
            /
            f"{index:05d}_class{label}.bin"
        )


        # Skip if already cached

        if output_path.exists():

            continue


        # Create PDM

        pdm = create_pdm_window(
            wav_path,
            split_name
        )


        # Pack bits

        packed = pack_pdm(
            pdm
        )


        # Save

        packed.tofile(
            output_path
        )


    print(
        f"\nFinished {split_name}"
    )


# ============================================================
# CACHED DATASET
# ============================================================

class CachedPDMDataset(
    Dataset
):


    def __init__(
        self,
        cache_dir
    ):

        self.files = sorted(
            Path(cache_dir).glob(
                "*.bin"
            )
        )


        if len(self.files) == 0:

            raise RuntimeError(
                f"No cached PDM files "
                f"found in {cache_dir}"
            )


    def __len__(self):

        return len(
            self.files
        )


    def __getitem__(
        self,
        index
    ):

        path = self.files[
            index
        ]


        # ----------------------------------------------------
        # Read packed PDM
        # ----------------------------------------------------

        packed = np.fromfile(
            path,
            dtype=np.uint8
        )


        # ----------------------------------------------------
        # Unpack PDM
        # ----------------------------------------------------

        pdm = np.unpackbits(
            packed,
            bitorder="little"
        )


        pdm = pdm[
            :PDM_SAMPLES_PER_WINDOW
        ]


        # ----------------------------------------------------
        # Convert to float
        # ----------------------------------------------------

        pdm = pdm.astype(
            np.float32
        )


        # [time]
        #
        # →
        #
        # [1, time]

        pdm = torch.from_numpy(
            pdm
        ).unsqueeze(0)


        # ----------------------------------------------------
        # Extract label
        # ----------------------------------------------------

        filename = path.stem


        label = int(
            filename.split(
                "class"
            )[1]
        )


        label = torch.tensor(
            label,
            dtype=torch.long
        )


        return pdm, label


# ============================================================
# TERNARY WEIGHT FUNCTION
# ============================================================

class TernaryFunction(
    torch.autograd.Function
):


    @staticmethod
    def forward(
        ctx,
        weights
    ):

        threshold = (
            0.7 *
            weights.abs().mean()
        )


        ternary = torch.zeros_like(
            weights
        )


        ternary[
            weights > threshold
        ] = 1.0


        ternary[
            weights < -threshold
        ] = -1.0


        ctx.save_for_backward(
            weights
        )


        return ternary


    @staticmethod
    def backward(
        ctx,
        grad_output
    ):

        weights, = (
            ctx.saved_tensors
        )


        grad = (
            grad_output.clone()
        )


        mask = (
            weights.abs() <= 1.0
        ).float()


        grad *= mask


        return grad


def ternary_weights(
    weights
):

    return TernaryFunction.apply(
        weights
    )


# ============================================================
# TERNARY CONVOLUTION
# ============================================================

class TernaryConv1d(
    nn.Module
):


    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0
    ):

        super().__init__()


        self.stride = stride

        self.padding = padding


        self.weight = nn.Parameter(
            torch.randn(
                out_channels,
                in_channels,
                kernel_size
            ) * 0.05
        )


        self.bias = nn.Parameter(
            torch.zeros(
                out_channels
            )
        )


    def forward(self, x):

        ternary = (
            ternary_weights(
                self.weight
            )
        )


        return F.conv1d(
            x,
            ternary,
            self.bias,
            stride=self.stride,
            padding=self.padding
        )


# ============================================================
# MODEL
# ============================================================

class PDMTernaryCNN(
    nn.Module
):


    def __init__(
        self,
        num_classes=4
    ):

        super().__init__()


        # ----------------------------------------------------
        # FIRST PDM-NATIVE LAYER
        #
        # 3.072 MHz / 192 = 16 kHz
        #
        # The model directly consumes PDM.
        # ----------------------------------------------------

        self.conv1 = TernaryConv1d(
            1,
            16,
            kernel_size=384,
            stride=192,
            padding=96
        )


        self.bn1 = (
            nn.BatchNorm1d(16)
        )


        self.conv2 = TernaryConv1d(
            16,
            32,
            kernel_size=9,
            stride=2,
            padding=4
        )


        self.bn2 = (
            nn.BatchNorm1d(32)
        )


        self.conv3 = TernaryConv1d(
            32,
            64,
            kernel_size=9,
            stride=2,
            padding=4
        )


        self.bn3 = (
            nn.BatchNorm1d(64)
        )


        self.conv4 = TernaryConv1d(
            64,
            64,
            kernel_size=7,
            stride=2,
            padding=3
        )


        self.bn4 = (
            nn.BatchNorm1d(64)
        )


        self.pool = (
            nn.AdaptiveAvgPool1d(1)
        )


        self.classifier = (
            nn.Linear(
                64,
                num_classes
            )
        )


    def forward(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = F.relu(x)


        x = self.conv2(x)

        x = self.bn2(x)

        x = F.relu(x)


        x = self.conv3(x)

        x = self.bn3(x)

        x = F.relu(x)


        x = self.conv4(x)

        x = self.bn4(x)

        x = F.relu(x)


        x = self.pool(x)


        x = x.squeeze(-1)


        x = self.classifier(x)


        return x


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Dataset split
    # --------------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        "DATASET"
    )

    print(
        "=" * 60
    )


    (
        train_files,
        val_files,
        test_files
    ) = create_splits()


    print()

    print(
        "Training files:",
        len(train_files)
    )

    print(
        "Validation files:",
        len(val_files)
    )

    print(
        "Test files:",
        len(test_files)
    )


    # --------------------------------------------------------
    # Preprocess PDM
    # --------------------------------------------------------

    preprocess_split(
        train_files,
        "train"
    )

    preprocess_split(
        val_files,
        "val"
    )

    preprocess_split(
        test_files,
        "test"
    )


    # --------------------------------------------------------
    # Cached datasets
    # --------------------------------------------------------

    train_dataset = (
        CachedPDMDataset(
            f"{CACHE_DIR}/train"
        )
    )


    val_dataset = (
        CachedPDMDataset(
            f"{CACHE_DIR}/val"
        )
    )


    test_dataset = (
        CachedPDMDataset(
            f"{CACHE_DIR}/test"
        )
    )


    print()

    print(
        "Cached training:",
        len(train_dataset)
    )

    print(
        "Cached validation:",
        len(val_dataset)
    )

    print(
        "Cached test:",
        len(test_dataset)
    )


    # --------------------------------------------------------
    # DataLoaders
    #
    # num_workers=0 is intentional for Mac/MPS.
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = (
        PDMTernaryCNN(
            NUM_CLASSES
        )
        .to(DEVICE)
    )


    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )


    print()

    print(
        "Total parameters:",
        f"{total_parameters:,}"
    )


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = (
        nn.CrossEntropyLoss()
    )


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = (
        torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE
        )
    )


    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_val_accuracy = 0.0


    print()
    print(
        "=" * 60
    )

    print(
        "TRAINING"
    )

    print(
        "=" * 60
    )


    for epoch in range(
        EPOCHS
    ):


        # ====================================================
        # TRAIN
        # ====================================================

        model.train()


        train_loss = 0.0

        train_correct = 0

        train_total = 0


        progress = tqdm(
            train_loader,
            desc=(
                f"Epoch "
                f"{epoch + 1}/{EPOCHS}"
            )
        )


        for inputs, labels in progress:

            inputs = inputs.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            outputs = model(
                inputs
            )


            loss = criterion(
                outputs,
                labels
            )


            loss.backward()


            optimizer.step()


            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            train_loss += (
                loss.item()
                *
                labels.size(0)
            )


            predictions = (
                outputs.argmax(
                    dim=1
                )
            )


            train_correct += (
                (
                    predictions ==
                    labels
                )
                .sum()
                .item()
            )


            train_total += (
                labels.size(0)
            )


            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )


        train_loss /= (
            train_total
        )


        train_accuracy = (
            train_correct /
            train_total
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()


        val_correct = 0

        val_total = 0


        with torch.no_grad():

            for inputs, labels in (
                val_loader
            ):

                inputs = inputs.to(
                    DEVICE
                )

                labels = labels.to(
                    DEVICE
                )


                outputs = model(
                    inputs
                )


                predictions = (
                    outputs.argmax(
                        dim=1
                    )
                )


                val_correct += (
                    (
                        predictions ==
                        labels
                    )
                    .sum()
                    .item()
                )


                val_total += (
                    labels.size(0)
                )


        val_accuracy = (
            val_correct /
            val_total
        )


        print()

        print(
            f"Epoch "
            f"{epoch + 1}/{EPOCHS} | "
            f"Train Loss: "
            f"{train_loss:.4f} | "
            f"Train Acc: "
            f"{train_accuracy:.4f} | "
            f"Val Acc: "
            f"{val_accuracy:.4f}"
        )


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if (
            val_accuracy
            >
            best_val_accuracy
        ):

            best_val_accuracy = (
                val_accuracy
            )


            torch.save(
                model.state_dict(),
                MODEL_PATH
            )


            print(
                "✓ Saved best model "
                f"(Val Acc = "
                f"{val_accuracy:.4f})"
            )


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print()

    print(
        "=" * 60
    )

    print(
        "TESTING"
    )

    print(
        "=" * 60
    )


    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )


    model.eval()


    # ========================================================
    # TEST
    # ========================================================

    all_predictions = []

    all_labels = []


    with torch.no_grad():

        for inputs, labels in tqdm(
            test_loader,
            desc="Testing"
        ):

            inputs = inputs.to(
                DEVICE
            )


            outputs = model(
                inputs
            )


            predictions = (
                outputs.argmax(
                    dim=1
                )
            )


            all_predictions.extend(
                predictions
                .cpu()
                .numpy()
            )


            all_labels.extend(
                labels.numpy()
            )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print()

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=CLASSES,
            digits=4
        )
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print(
        "Confusion Matrix:"
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print(cm)


    # ========================================================
    # OVERALL ACCURACY
    # ========================================================

    accuracy = (
        np.array(
            all_predictions
        )
        ==
        np.array(
            all_labels
        )
    ).mean()


    print()

    print(
        f"Overall Test Accuracy: "
        f"{accuracy:.4f}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()

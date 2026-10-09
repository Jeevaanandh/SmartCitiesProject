import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
from scipy.signal import resample_poly
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import numba


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    "/Users/JeevaanandhIlayaraja/Downloads/Dataset"
)

CACHE_DIR = Path(
    "/Users/JeevaanandhIlayaraja/Downloads/pdm_cache"
)

CACHE_DIR.mkdir(parents=True, exist_ok=True)


CLASSES = [
    "gunshot",
    "scream",
    "carcrash",
    "normal"
]

CLASS_TO_IDX = {
    name: i for i, name in enumerate(CLASSES)
}


# Dataset split
TRAIN_PER_CLASS = 250
VAL_PER_CLASS = 75
TEST_PER_CLASS = 75


# Audio
TARGET_SAMPLE_RATE = 16000

# PDM clock
PDM_RATE = 3_072_000

# 0.5 second window
WINDOW_SECONDS = 0.5

PCM_SAMPLES_PER_WINDOW = int(
    TARGET_SAMPLE_RATE * WINDOW_SECONDS
)

PDM_SAMPLES_PER_WINDOW = int(
    PDM_RATE * WINDOW_SECONDS
)

PDM_SAMPLES_PER_PCM_SAMPLE = (
    PDM_RATE // TARGET_SAMPLE_RATE
)


# Multiple windows
TRAIN_WINDOWS_PER_FILE = 5
VAL_WINDOWS_PER_FILE = 5
TEST_WINDOWS_PER_FILE = 5


# Training
BATCH_SIZE = 8
EPOCHS = 30

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 4

SEED = 42


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# WAV LOADING
# ============================================================

def load_wav(path):
    """
    Load WAV file and convert to:

        mono
        16 kHz
        float32
        range approximately [-1, 1]
    """

    sample_rate, audio = wavfile.read(str(path))

    # Preserve original dtype before converting to float
    original_dtype = audio.dtype

    # Convert stereo -> mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Integer PCM normalization
    if np.issubdtype(original_dtype, np.integer):

        info = np.iinfo(original_dtype)

        max_value = max(
            abs(info.min),
            info.max
        )

        audio = audio.astype(np.float32) / max_value

    else:
        audio = audio.astype(np.float32)

    # Remove DC offset
    audio = audio - np.mean(audio)

    # Resample to 16 kHz
    if sample_rate != TARGET_SAMPLE_RATE:

        audio = resample_poly(
            audio,
            TARGET_SAMPLE_RATE,
            sample_rate
        )

    # Prevent numerical problems
    audio = np.clip(audio, -1.0, 1.0)

    return audio.astype(np.float32)


# ============================================================
# SIGMA-DELTA PDM MODULATOR
# ============================================================

@numba.njit(cache=True)
def sigma_delta_modulator(signal):

    output = np.empty(
        len(signal),
        dtype=np.uint8
    )

    integrator = 0.0

    for i in range(len(signal)):

        integrator += signal[i]

        if integrator >= 0.0:
            output[i] = 1
            integrator -= 1.0
        else:
            output[i] = 0
            integrator += 1.0

    return output


# ============================================================
# PCM -> PDM
# ============================================================

def pcm_to_pdm(pcm):

    # Convert 16 kHz PCM to 3.072 MHz
    high_rate_pcm = resample_poly(
        pcm,
        PDM_RATE,
        TARGET_SAMPLE_RATE
    )

    # Sigma-delta modulation
    pdm = sigma_delta_modulator(
        high_rate_pcm
    )

    return pdm


# ============================================================
# PACK PDM BITS
# ============================================================

def pack_pdm(pdm):

    return np.packbits(
        pdm,
        bitorder="little"
    )


# ============================================================
# UNPACK PDM
# ============================================================

def unpack_pdm(packed, num_bits):

    bits = np.unpackbits(
        packed,
        bitorder="little"
    )

    bits = bits[:num_bits]

    return bits


# ============================================================
# FIND DATASET FILES
# ============================================================

def get_audio_files():

    files_by_class = {}

    for class_name in CLASSES:

        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing directory: {class_dir}"
            )

        files = sorted(
            list(class_dir.glob("*.wav")) +
            list(class_dir.glob("*.WAV"))
        )

        if len(files) < (
            TRAIN_PER_CLASS +
            VAL_PER_CLASS +
            TEST_PER_CLASS
        ):
            raise RuntimeError(
                f"{class_name}: only {len(files)} WAV files found"
            )

        files_by_class[class_name] = files

        print(
            f"{class_name}: {len(files)} recordings"
        )

    return files_by_class


# ============================================================
# SPLIT DATA
# ============================================================

def split_dataset(files_by_class):

    train_files = []
    val_files = []
    test_files = []

    rng = random.Random(SEED)

    for class_name in CLASSES:

        files = files_by_class[class_name].copy()

        rng.shuffle(files)

        train_files.extend(
            [
                (path, CLASS_TO_IDX[class_name])
                for path in files[
                    :TRAIN_PER_CLASS
                ]
            ]
        )

        val_start = TRAIN_PER_CLASS

        val_end = (
            TRAIN_PER_CLASS +
            VAL_PER_CLASS
        )

        val_files.extend(
            [
                (path, CLASS_TO_IDX[class_name])
                for path in files[
                    val_start:val_end
                ]
            ]
        )

        test_start = val_end

        test_end = (
            val_end +
            TEST_PER_CLASS
        )

        test_files.extend(
            [
                (path, CLASS_TO_IDX[class_name])
                for path in files[
                    test_start:test_end
                ]
            ]
        )

    rng.shuffle(train_files)
    rng.shuffle(val_files)
    rng.shuffle(test_files)

    print()
    print(
        f"Train recordings: {len(train_files)}"
    )

    print(
        f"Validation recordings: {len(val_files)}"
    )

    print(
        f"Test recordings: {len(test_files)}"
    )

    return (
        train_files,
        val_files,
        test_files
    )


# ============================================================
# CREATE PDM CACHE
# ============================================================

def get_cache_path(wav_path):

    # Keep unique filename while avoiding collisions
    relative = wav_path.relative_to(DATASET_DIR)

    safe_name = str(relative).replace(
        os.sep,
        "_"
    )

    safe_name = safe_name.replace(
        ".wav",
        ""
    ).replace(
        ".WAV",
        ""
    )

    return CACHE_DIR / f"{safe_name}.bin"


def cache_pdm_file(wav_path):

    cache_path = get_cache_path(wav_path)

    if cache_path.exists():

        return cache_path

    print(
        f"Generating PDM: {wav_path.name}"
    )

    pcm = load_wav(wav_path)

    pdm = pcm_to_pdm(pcm)

    packed = pack_pdm(pdm)

    packed.tofile(cache_path)

    return cache_path


def prepare_pdm_cache(file_list):

    print()
    print("Preparing PDM cache...")

    for wav_path, _ in tqdm(
        file_list,
        desc="PDM conversion"
    ):

        cache_pdm_file(wav_path)

    print("PDM cache ready.")


# ============================================================
# WINDOW SELECTION
# ============================================================

def get_window_starts(
    total_samples,
    num_windows,
    mode
):

    if total_samples <= PDM_SAMPLES_PER_WINDOW:

        return [0] * num_windows

    max_start = (
        total_samples -
        PDM_SAMPLES_PER_WINDOW
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    if mode == "train":

        return [
            random.randint(
                0,
                max_start
            )
            for _ in range(num_windows)
        ]

    # --------------------------------------------------------
    # VALIDATION / TEST
    # --------------------------------------------------------
    # Deterministic evenly-spaced windows
    # --------------------------------------------------------

    if num_windows == 1:

        return [max_start // 2]

    starts = np.linspace(
        0,
        max_start,
        num_windows
    )

    return [
        int(x)
        for x in starts
    ]


# ============================================================
# DATASET
# ============================================================

class PDMDataset(Dataset):

    def __init__(
        self,
        file_list,
        mode,
        windows_per_file
    ):

        self.mode = mode

        self.windows_per_file = (
            windows_per_file
        )

        self.samples = []

        print()
        print(
            f"Building {mode} dataset..."
        )

        for wav_path, label in file_list:

            cache_path = get_cache_path(
                wav_path
            )

            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Missing cache: {cache_path}"
                )

            # Determine number of PDM bits
            packed_size = (
                cache_path.stat().st_size
            )

            total_bits = packed_size * 8

            starts = get_window_starts(
                total_bits,
                windows_per_file,
                mode
            )

            for start in starts:

                self.samples.append(
                    (
                        cache_path,
                        label,
                        start
                    )
                )

        print(
            f"{mode} windows: {len(self.samples)}"
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        cache_path, label, start = (
            self.samples[index]
        )

        # Number of bytes needed
        # for the 0.5 sec PDM window
        num_bytes = (
            PDM_SAMPLES_PER_WINDOW // 8
        )

        byte_start = start // 8

        # Read only required bytes
        with open(
            cache_path,
            "rb"
        ) as f:

            f.seek(byte_start)

            packed = np.fromfile(
                f,
                dtype=np.uint8,
                count=num_bytes
            )

        pdm = unpack_pdm(
            packed,
            PDM_SAMPLES_PER_WINDOW
        )

        # Convert to float for PyTorch
        pdm = torch.from_numpy(
            pdm.astype(np.float32)
        )

        # [time] -> [1, time]
        pdm = pdm.unsqueeze(0)

        return (
            pdm,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ============================================================
# MODEL
# ============================================================

class FloatPDMCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # First layer
        #
        # 3.072 MHz PDM
        # kernel = 384
        # stride = 192
        #
        # 192 PDM clocks correspond to
        # one 16 kHz sample period.
        # ----------------------------------------------------

        self.conv1 = nn.Conv1d(
            1,
            16,
            kernel_size=384,
            stride=192,
            padding=96
        )

        self.bn1 = nn.BatchNorm1d(16)


        # ----------------------------------------------------
        # Second layer
        # ----------------------------------------------------

        self.conv2 = nn.Conv1d(
            16,
            32,
            kernel_size=9,
            stride=2,
            padding=4
        )

        self.bn2 = nn.BatchNorm1d(32)


        # ----------------------------------------------------
        # Third layer
        # ----------------------------------------------------

        self.conv3 = nn.Conv1d(
            32,
            64,
            kernel_size=9,
            stride=2,
            padding=4
        )

        self.bn3 = nn.BatchNorm1d(64)


        # ----------------------------------------------------
        # Fourth layer
        # ----------------------------------------------------

        self.conv4 = nn.Conv1d(
            64,
            64,
            kernel_size=7,
            stride=2,
            padding=3
        )

        self.bn4 = nn.BatchNorm1d(64)


        # ----------------------------------------------------
        # Global pooling
        # ----------------------------------------------------

        self.pool = nn.AdaptiveAvgPool1d(1)


        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            64,
            len(CLASSES)
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
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for x, y in progress:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        output = model(x)

        loss = criterion(
            output,
            y
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() *
            x.size(0)
        )

        predictions = output.argmax(
            dim=1
        )

        correct += (
            predictions == y
        ).sum().item()

        total += x.size(0)

    return (
        total_loss / total,
        correct / total
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            output = model(x)

            loss = criterion(
                output,
                y
            )

            total_loss += (
                loss.item() *
                x.size(0)
            )

            predictions = output.argmax(
                dim=1
            )

            correct += (
                predictions == y
            ).sum().item()

            total += x.size(0)

    return (
        total_loss / total,
        correct / total
    )


# ============================================================
# TRAINING LOOP
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader
):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_acc = 0.0

    epochs_without_improvement = 0

    checkpoint_path = (
        "best_pdm_int8_cnn_fp32.pth"
    )

    print()
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion
            )
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print(
                f"  -> Best model saved "
                f"(Val Acc: {val_acc:.4f})"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"  -> No improvement "
                f"({epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE})"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    print()
    print(
        f"Best validation accuracy: "
        f"{best_val_acc:.4f}"
    )

    return checkpoint_path


# ============================================================
# TESTING
# ============================================================

def test_model(
    model,
    test_files
):

    print()
    print("=" * 60)
    print("TESTING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    checkpoint_path = (
        "best_pdm_int8_cnn_fp32.pth"
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE
        )
    )

    model.eval()

    # --------------------------------------------------------
    # Evaluate multiple windows per recording
    # --------------------------------------------------------

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for wav_path, label in tqdm(
            test_files,
            desc="Testing"
        ):

            cache_path = get_cache_path(
                wav_path
            )

            packed_size = (
                cache_path.stat().st_size
            )

            total_bits = packed_size * 8

            starts = get_window_starts(
                total_bits,
                TEST_WINDOWS_PER_FILE,
                "test"
            )

            window_predictions = []

            for start in starts:

                num_bytes = (
                    PDM_SAMPLES_PER_WINDOW // 8
                )

                byte_start = start // 8

                with open(
                    cache_path,
                    "rb"
                ) as f:

                    f.seek(byte_start)

                    packed = np.fromfile(
                        f,
                        dtype=np.uint8,
                        count=num_bytes
                    )

                pdm = unpack_pdm(
                    packed,
                    PDM_SAMPLES_PER_WINDOW
                )

                pdm = torch.from_numpy(
                    pdm.astype(np.float32)
                )

                pdm = pdm.unsqueeze(0)
                pdm = pdm.unsqueeze(0)

                pdm = pdm.to(DEVICE)

                output = model(pdm)

                # Store logits rather than only class
                window_predictions.append(
                    output.squeeze(0).cpu()
                )

            # ------------------------------------------------
            # Average logits across windows
            # ------------------------------------------------

            mean_logits = torch.stack(
                window_predictions
            ).mean(dim=0)

            prediction = (
                mean_logits.argmax().item()
            )

            all_predictions.append(
                prediction
            )

            all_labels.append(
                label
            )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    print()

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=CLASSES,
            digits=4
        )
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print("Confusion Matrix:")
    print(cm)

    accuracy = (
        np.array(all_predictions)
        ==
        np.array(all_labels)
    ).mean()

    print()
    print(
        f"Overall Test Accuracy: "
        f"{accuracy:.4f}"
    )




# ============================================================
# POST-TRAINING STATIC INT8 QUANTIZATION
# ============================================================

def create_int8_model(float_model, calibration_loader):
    """
    Convert the trained FP32 PDM CNN to a statically quantized INT8 model.
    FX graph-mode quantization inserts activation observers and quantizes
    supported Conv1d/Linear operations. Quantized inference runs on CPU.
    """
    try:
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    except ImportError as exc:
        raise RuntimeError(
            "This PyTorch installation does not expose FX graph-mode quantization. "
            "Upgrade PyTorch and retry."
        ) from exc

    # Quantized CPU kernels use FBGEMM on x86 and QNNPACK on ARM.
    supported = torch.backends.quantized.supported_engines
    if "qnnpack" in supported:
        backend = "qnnpack"
    elif "fbgemm" in supported:
        backend = "fbgemm"
    else:
        raise RuntimeError(
            f"No supported quantized CPU backend found. Available: {supported}"
        )

    torch.backends.quantized.engine = backend
    qconfig_mapping = get_default_qconfig_mapping(backend)

    float_model = float_model.to("cpu").eval()
    example_inputs = (torch.zeros(1, 1, PDM_SAMPLES_PER_WINDOW),)

    print()
    print("=" * 60)
    print(f"INT8 POST-TRAINING QUANTIZATION ({backend})")
    print("=" * 60)

    prepared_model = prepare_fx(
        float_model,
        qconfig_mapping,
        example_inputs
    )

    # Calibrate activation ranges using validation windows only.
    with torch.no_grad():
        for batch_x, _ in tqdm(calibration_loader, desc="INT8 calibration"):
            prepared_model(batch_x.to("cpu"))

    int8_model = convert_fx(prepared_model)
    int8_model.eval()

    # Save the converted model as a serialized PyTorch model.
    # Use the same PyTorch version when loading this file.
    torch.save(int8_model, "best_pdm_int8_cnn.pt")
    print("Saved quantized model: best_pdm_int8_cnn.pt")
    print(f"Quantized backend: {backend}")

    return int8_model


def test_quantized_model(model, test_files):
    """Evaluate the converted INT8 model using the same recording-level
    five-window logit averaging used by Model 1.
    """
    print()
    print("=" * 60)
    print("INT8 MODEL TESTING")
    print("=" * 60)

    model = model.to("cpu").eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for wav_path, label in tqdm(test_files, desc="INT8 testing"):
            cache_path = get_cache_path(wav_path)
            total_bits = cache_path.stat().st_size * 8
            starts = get_window_starts(
                total_bits, TEST_WINDOWS_PER_FILE, "test"
            )
            window_predictions = []

            for start in starts:
                num_bytes = PDM_SAMPLES_PER_WINDOW // 8
                byte_start = start // 8

                with open(cache_path, "rb") as f:
                    f.seek(byte_start)
                    packed = np.fromfile(
                        f, dtype=np.uint8, count=num_bytes
                    )

                pdm = unpack_pdm(packed, PDM_SAMPLES_PER_WINDOW)
                x = torch.from_numpy(
                    pdm.astype(np.float32)
                ).unsqueeze(0).unsqueeze(0)

                logits = model(x)
                window_predictions.append(logits.squeeze(0).cpu())

            mean_logits = torch.stack(window_predictions).mean(dim=0)
            all_predictions.append(mean_logits.argmax().item())
            all_labels.append(label)

    print()
    print(classification_report(
        all_labels, all_predictions, target_names=CLASSES, digits=4
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions))
    accuracy = (
        np.asarray(all_predictions) == np.asarray(all_labels)
    ).mean()
    print(f"\nINT8 Overall Test Accuracy: {accuracy:.4f}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PDM INT8 ABLATION CNN (FLOAT TRAINING)")
    print("=" * 60)

    print()
    print(
        f"Target sample rate: "
        f"{TARGET_SAMPLE_RATE} Hz"
    )

    print(
        f"PDM rate: "
        f"{PDM_RATE:,} Hz"
    )

    print(
        f"PDM clocks / PCM sample: "
        f"{PDM_SAMPLES_PER_PCM_SAMPLE}"
    )

    print(
        f"Window length: "
        f"{WINDOW_SECONDS} seconds"
    )

    print(
        f"PDM samples / window: "
        f"{PDM_SAMPLES_PER_WINDOW:,}"
    )

    print(
        f"Training windows / recording: "
        f"{TRAIN_WINDOWS_PER_FILE}"
    )

    print(
        f"Validation windows / recording: "
        f"{VAL_WINDOWS_PER_FILE}"
    )

    print(
        f"Test windows / recording: "
        f"{TEST_WINDOWS_PER_FILE}"
    )

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    files_by_class = get_audio_files()

    # --------------------------------------------------------
    # Split recordings
    # --------------------------------------------------------

    (
        train_files,
        val_files,
        test_files
    ) = split_dataset(
        files_by_class
    )

    # --------------------------------------------------------
    # Prepare PDM cache
    #
    # Important:
    # Split happens BEFORE window generation.
    #
    # Therefore windows from the same recording can never
    # appear in both train and validation/test.
    # --------------------------------------------------------

    prepare_pdm_cache(
        train_files +
        val_files +
        test_files
    )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_dataset = PDMDataset(
        train_files,
        mode="train",
        windows_per_file=
            TRAIN_WINDOWS_PER_FILE
    )

    val_dataset = PDMDataset(
        val_files,
        mode="val",
        windows_per_file=
            VAL_WINDOWS_PER_FILE
    )

    # --------------------------------------------------------
    # DataLoaders
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

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = FloatPDMCNN()

    model = model.to(DEVICE)

    print()
    print(model)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    checkpoint_path = train_model(
        model,
        train_loader,
        val_loader
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_model(
        model,
        test_files
    )

    # Reload the best FP32 checkpoint before quantization.
    model.load_state_dict(
        torch.load(
            "best_pdm_int8_cnn_fp32.pth",
            map_location="cpu"
        )
    )

    # Calibrate on validation data, then evaluate the INT8 model on test data.
    int8_model = create_int8_model(model, val_loader)
    test_quantized_model(int8_model, test_files)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()


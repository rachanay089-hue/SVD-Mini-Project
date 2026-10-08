import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import os


# =========================================================
# 1. CONVERT IMAGE TO GRAYSCALE MATRIX
# =========================================================

def image_to_matrix(image):
    """
    Convert RGB image into grayscale matrix.

    Each pixel becomes a numerical value from 0 to 255.
    """

    gray_image = image.convert("L")

    matrix = np.array(gray_image, dtype=float)

    return gray_image, matrix


# =========================================================
# 2. APPLY SVD
# =========================================================

def apply_svd(matrix):
    """
    Perform Singular Value Decomposition.

    A = U * Sigma * V^T
    """

    U, S, Vt = np.linalg.svd(
        matrix,
        full_matrices=False
    )

    return U, S, Vt


# =========================================================
# 3. RECONSTRUCT IMAGE USING RANK K
# =========================================================

def reconstruct_image(U, S, Vt, k):
    """
    Reconstruct image using only first k singular values.

    A_k = U_k * Sigma_k * V_k^T
    """

    # Select first k columns of U
    U_k = U[:, :k]

    # Select first k singular values
    S_k = S[:k]

    # Select first k rows of V transpose
    Vt_k = Vt[:k, :]

    # Create Sigma_k
    Sigma_k = np.diag(S_k)

    # Matrix multiplication
    reconstructed = U_k @ Sigma_k @ Vt_k

    # Pixel values must be between 0 and 255
    reconstructed = np.clip(
        reconstructed,
        0,
        255
    )

    return reconstructed


# =========================================================
# 4. CALCULATE MSE
# =========================================================

def calculate_mse(original, reconstructed):
    """
    Mean Squared Error.

    MSE = average((original - reconstructed)^2)
    """

    error = original - reconstructed

    mse = np.mean(error ** 2)

    return mse


# =========================================================
# 5. CALCULATE PSNR
# =========================================================

def calculate_psnr(mse):
    """
    PSNR = 10 log10(MAX^2 / MSE)

    MAX pixel value = 255
    """

    if mse == 0:
        return float("inf")

    psnr = 10 * np.log10(
        (255 ** 2) / mse
    )

    return psnr


# =========================================================
# 6. CALCULATE COMPRESSION
# =========================================================

def calculate_compression(height, width, k):
    """
    Original matrix storage:

        m * n

    Rank-k SVD storage:

        m*k + k + k*n

    """

    original_values = height * width

    compressed_values = (
        height * k +
        k +
        k * width
    )

    compression_ratio = (
        original_values /
        compressed_values
    )

    storage_saved = (
        1 -
        compressed_values /
        original_values
    ) * 100

    return compression_ratio, storage_saved


# =========================================================
# 7. MAIN APPLICATION
# =========================================================

class SVDCompressor:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "SVD-Based Intelligent Image Compression"
        )

        self.root.geometry(
            "1200x800"
        )

        self.root.configure(
            bg="white"
        )

        # Store image information
        self.original_image = None
        self.gray_image = None
        self.original_matrix = None

        # Store SVD
        self.U = None
        self.S = None
        self.Vt = None

        # Store compressed image
        self.compressed_matrix = None

        # Store Tkinter images
        self.original_photo = None
        self.compressed_photo = None

        # =================================================
        # TITLE
        # =================================================

        title = tk.Label(
            root,
            text="SVD-BASED INTELLIGENT IMAGE COMPRESSION",
            font=("Arial", 22, "bold"),
            bg="white"
        )

        title.pack(pady=15)

        subtitle = tk.Label(
            root,
            text=(
                "Singular Value Decomposition and "
                "Rank-k Approximation"
            ),
            font=("Arial", 12),
            bg="white"
        )

        subtitle.pack()

        # =================================================
        # CONTROL FRAME
        # =================================================

        control_frame = tk.Frame(
            root,
            bg="white"
        )

        control_frame.pack(
            pady=15
        )

        # Select image button

        select_button = tk.Button(
            control_frame,
            text="Select Image",
            font=("Arial", 11, "bold"),
            command=self.select_image,
            width=15
        )

        select_button.grid(
            row=0,
            column=0,
            padx=5
        )

        # K label

        k_label = tk.Label(
            control_frame,
            text="Rank (k):",
            font=("Arial", 11),
            bg="white"
        )

        k_label.grid(
            row=0,
            column=1,
            padx=5
        )

        # K entry

        self.k_entry = tk.Entry(
            control_frame,
            width=8,
            font=("Arial", 11)
        )

        self.k_entry.insert(
            0,
            "20"
        )

        self.k_entry.grid(
            row=0,
            column=2,
            padx=5
        )

        # Apply SVD button

        svd_button = tk.Button(
            control_frame,
            text="Apply SVD",
            font=("Arial", 11, "bold"),
            command=self.apply_svd_button,
            width=15
        )

        svd_button.grid(
            row=0,
            column=3,
            padx=5
        )

        # Reconstruct button

        reconstruct_button = tk.Button(
            control_frame,
            text="Reconstruct",
            font=("Arial", 11, "bold"),
            command=self.reconstruct_button,
            width=15
        )

        reconstruct_button.grid(
            row=0,
            column=4,
            padx=5
        )

        # Graph button

        graph_button = tk.Button(
            control_frame,
            text="Show Singular Values",
            font=("Arial", 11, "bold"),
            command=self.show_graph,
            width=20
        )

        graph_button.grid(
            row=0,
            column=5,
            padx=5
        )

        # Save button

        save_button = tk.Button(
            control_frame,
            text="Save Image",
            font=("Arial", 11, "bold"),
            command=self.save_image,
            width=15
        )

        save_button.grid(
            row=0,
            column=6,
            padx=5
        )

        # =================================================
        # IMAGE DISPLAY
        # =================================================

        image_frame = tk.Frame(
            root,
            bg="white"
        )

        image_frame.pack(
            pady=10
        )

        # ---------------- ORIGINAL ----------------

        original_frame = tk.Frame(
            image_frame,
            bg="white"
        )

        original_frame.grid(
            row=0,
            column=0,
            padx=30
        )

        original_title = tk.Label(
            original_frame,
            text="Original Image",
            font=("Arial", 14, "bold"),
            bg="white"
        )

        original_title.pack(
            pady=5
        )

        self.original_display = tk.Label(
            original_frame,
            text="No image selected",
            width=50,
            height=20,
            bg="#eeeeee"
        )

        self.original_display.pack()

        # ---------------- COMPRESSED ----------------

        compressed_frame = tk.Frame(
            image_frame,
            bg="white"
        )

        compressed_frame.grid(
            row=0,
            column=1,
            padx=30
        )

        compressed_title = tk.Label(
            compressed_frame,
            text="Rank-k Reconstructed Image",
            font=("Arial", 14, "bold"),
            bg="white"
        )

        compressed_title.pack(
            pady=5
        )

        self.compressed_display = tk.Label(
            compressed_frame,
            text="No compressed image",
            width=50,
            height=20,
            bg="#eeeeee"
        )

        self.compressed_display.pack()

        # =================================================
        # INFORMATION FRAME
        # =================================================

        info_frame = tk.Frame(
            root,
            bg="#eeeeee",
            bd=2,
            relief="groove"
        )

        info_frame.pack(
            fill="x",
            padx=30,
            pady=10
        )

        self.info_label = tk.Label(
            info_frame,
            text=(
                "Select an image to begin."
            ),
            font=("Consolas", 11),
            bg="#eeeeee",
            justify="left"
        )

        self.info_label.pack(
            pady=10
        )

        # =================================================
        # MATHEMATICAL FORMULA
        # =================================================

        formula = tk.Label(
            root,
            text=(
                "A = UΣVᵀ        →        "
                "Aₖ = UₖΣₖVₖᵀ"
            ),
            font=("Arial", 16, "bold"),
            bg="white"
        )

        formula.pack(
            pady=5
        )

    # =====================================================
    # SELECT IMAGE
    # =====================================================

    def select_image(self):

        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp"
                )
            ]
        )

        if not file_path:
            return

        try:

            # Open image
            self.original_image = Image.open(
                file_path
            ).convert("RGB")

            # Convert to grayscale
            self.gray_image, self.original_matrix = (
                image_to_matrix(
                    self.original_image
                )
            )

            # Reset previous SVD
            self.U = None
            self.S = None
            self.Vt = None

            self.compressed_matrix = None

            # Display original
            self.display_image(
                self.gray_image,
                self.original_display,
                "original"
            )

            width, height = self.gray_image.size

            self.info_label.config(
                text=(
                    f"Image: {os.path.basename(file_path)}\n"
                    f"Image dimensions: {height} × {width}\n"
                    f"Matrix A dimensions: {height} × {width}\n\n"
                    f"Next step: Click 'Apply SVD'"
                )
            )

        except Exception as e:

            messagebox.showerror(
                "Error",
                str(e)
            )

    # =====================================================
    # APPLY SVD
    # =====================================================

    def apply_svd_button(self):

        if self.original_matrix is None:

            messagebox.showwarning(
                "Warning",
                "Please select an image first."
            )

            return

        try:

            self.U, self.S, self.Vt = apply_svd(
                self.original_matrix
            )

            m, n = self.original_matrix.shape

            rank = len(self.S)

            self.info_label.config(
                text=(
                    "SVD successfully applied!\n\n"

                    f"A dimensions       : {m} × {n}\n"
                    f"U dimensions       : {self.U.shape[0]} × "
                    f"{self.U.shape[1]}\n"
                    f"Σ dimensions       : {rank} × {rank}\n"
                    f"Vᵀ dimensions      : {self.Vt.shape[0]} × "
                    f"{self.Vt.shape[1]}\n\n"

                    "Mathematical decomposition:\n"
                    "A = UΣVᵀ\n\n"

                    "Now choose Rank k and click "
                    "'Reconstruct'."
                )
            )

        except Exception as e:

            messagebox.showerror(
                "SVD Error",
                str(e)
            )

    # =====================================================
    # RECONSTRUCT
    # =====================================================

    def reconstruct_button(self):

        if self.U is None:

            messagebox.showwarning(
                "Warning",
                "Please apply SVD first."
            )

            return

        try:

            k = int(
                self.k_entry.get()
            )

        except ValueError:

            messagebox.showerror(
                "Error",
                "Rank k must be an integer."
            )

            return

        max_k = len(self.S)

        if k <= 0 or k > max_k:

            messagebox.showerror(
                "Invalid Rank",
                f"k must be between 1 and {max_k}."
            )

            return

        # Reconstruct

        self.compressed_matrix = reconstruct_image(
            self.U,
            self.S,
            self.Vt,
            k
        )

        # Convert matrix to image

        compressed_image = Image.fromarray(
            self.compressed_matrix.astype(
                np.uint8
            )
        )

        # Display

        self.display_image(
            compressed_image,
            self.compressed_display,
            "compressed"
        )

        # Calculate MSE

        mse = calculate_mse(
            self.original_matrix,
            self.compressed_matrix
        )

        # Calculate PSNR

        psnr = calculate_psnr(
            mse
        )

        # Image dimensions

        height, width = self.original_matrix.shape

        # Compression

        ratio, storage_saved = calculate_compression(
            height,
            width,
            k
        )

        # =================================================
        # DISPLAY RESULTS
        # =================================================

        self.info_label.config(
            text=(
                "RESULTS\n\n"

                f"Rank k              : {k}\n"

                f"Original matrix     : "
                f"{height} × {width}\n"

                f"Uₖ dimensions       : "
                f"{height} × {k}\n"

                f"Σₖ dimensions       : "
                f"{k} × {k}\n"

                f"Vₖᵀ dimensions      : "
                f"{k} × {width}\n\n"

                f"MSE                 : "
                f"{mse:.4f}\n"

                f"PSNR                : "
                f"{psnr:.2f} dB\n"

                f"Compression Ratio   : "
                f"{ratio:.2f}:1\n"

                f"Estimated Storage Saved : "
                f"{storage_saved:.2f}%"
            )
        )

    # =====================================================
    # DISPLAY IMAGE
    # =====================================================

    def display_image(
        self,
        image,
        label,
        image_type
    ):

        display_image = image.copy()

        display_image.thumbnail(
            (500, 350)
        )

        photo = ImageTk.PhotoImage(
            display_image
        )

        label.config(
            image=photo,
            text=""
        )

        if image_type == "original":

            self.original_photo = photo

        else:

            self.compressed_photo = photo

    # =====================================================
    # SHOW SINGULAR VALUE GRAPH
    # =====================================================

    def show_graph(self):

        if self.S is None:

            messagebox.showwarning(
                "Warning",
                "Please apply SVD first."
            )

            return

        plt.figure(
            figsize=(9, 5)
        )

        plt.plot(
            range(
                1,
                len(self.S) + 1
            ),
            self.S
        )

        plt.xlabel(
            "Singular Value Index"
        )

        plt.ylabel(
            "Singular Value"
        )

        plt.title(
            "Singular Value Distribution"
        )

        plt.grid(
            True
        )

        plt.tight_layout()

        plt.show()

    # =====================================================
    # SAVE IMAGE
    # =====================================================

    def save_image(self):

        if self.compressed_matrix is None:

            messagebox.showwarning(
                "Warning",
                "Please reconstruct an image first."
            )

            return

        file_path = filedialog.asksaveasfilename(
            title="Save Compressed Image",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg")
            ]
        )

        if not file_path:
            return

        image = Image.fromarray(
            self.compressed_matrix.astype(
                np.uint8
            )
        )

        image.save(
            file_path
        )

        messagebox.showinfo(
            "Success",
            "Compressed image saved successfully."
        )


# =========================================================
# PROGRAM START
# =========================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = SVDCompressor(
        root
    )

    root.mainloop()
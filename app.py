import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(app_dir))

from algorithms import lut, mapping, slicing
from ui.toolbar import ToolbarFrame
from ui.controls import ControlPanel, ALGORITHM_CATEGORIES, ALGORITHMS
from ui.previews import PreviewPanel


class PseudoColorStudioApp:
    """Thin orchestrator that wires the UI components together."""

    def __init__(self, root):
        self.root = root
        self.root.title("Pseudo-Color Studio - Algorithms + CNN")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root.geometry("1280x800")
        self.root.minsize(1060, 700)

        # ── Application state ────────────────────────────────────────────
        self.original_color = None
        self.original_gray = None
        self.result_bgr = None

        # CNN async state
        self._cnn_executor = ThreadPoolExecutor(max_workers=1)
        self._cnn_request_id = 0
        self._cnn_pending_request = None
        self._cnn_busy = False

        # ── UI components ────────────────────────────────────────────────
        self.toolbar = ToolbarFrame(
            root,
            on_open=self._open_image,
            on_save=self._save_result,
            on_save_all=self._save_all,
        )

        main = ctk.CTkFrame(root, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=12, pady=12)

        self.controls = ControlPanel(
            main,
            on_apply=self._apply,
            on_compare_all=self._compare_all,
        )
        self.previews = PreviewPanel(main)

        # ── Keyboard shortcuts ───────────────────────────────────────────
        self.root.bind("<Control-o>", lambda e: self._open_image())
        self.root.bind("<Control-O>", lambda e: self._open_image())
        self.root.bind("<Control-s>", lambda e: self._save_result())
        self.root.bind("<Control-S>", lambda e: self._save_all())
        self.root.bind("<Return>", lambda e: self._apply())

    # ------------------------------------------------------------------ #
    #  Image I/O                                                           #
    # ------------------------------------------------------------------ #

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("All", "*.*"),
            ],
        )
        if not path:
            return

        img_color = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_color is None:
            messagebox.showerror("Error", f"Could not load image:\n{path}")
            return

        self.original_color = img_color
        self.original_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        name = os.path.basename(path)
        self.toolbar.status_var.set(f"{name}  |  {img_color.shape[1]}×{img_color.shape[0]} px")
        self.previews.info_var.set(f"Loaded: {name}")

        self.previews.show_image(self.previews.canvas_source, self.original_color, is_gray=False)
        self.previews.show_image(self.previews.canvas_orig, self.original_gray, is_gray=True)
        self.previews.draw_placeholder(self.previews.canvas_result, "Applying…")
        self._apply()

    def _save_result(self):
        if self.result_bgr is None:
            messagebox.showinfo("Nothing to save", "Apply an algorithm first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
        )
        if path:
            cv2.imwrite(path, self.result_bgr)
            self.toolbar.status_var.set(f"Saved → {os.path.basename(path)}")

    def _save_all(self):
        if self.original_gray is None:
            messagebox.showinfo("No image", "Open an image first.")
            return
        folder = filedialog.askdirectory(title="Choose output folder")
        if not folder:
            return

        saved = 0
        for algo in ALGORITHMS:
            result = self._run_algorithm(algo, self.original_gray)
            if result is not None:
                safe = algo.replace(":", "").replace(" ", "_").replace("/", "-")
                cv2.imwrite(os.path.join(folder, f"{safe}.png"), result)
                saved += 1

        self.toolbar.status_var.set(f"All results saved → {folder}")
        messagebox.showinfo("Done", f"Saved {saved} images to:\n{folder}")

    # ------------------------------------------------------------------ #
    #  Algorithm dispatch                                                  #
    # ------------------------------------------------------------------ #

    def _apply(self):
        if self.original_gray is None:
            return
        algo = self.controls.algo_var.get()

        # Clear stale CNN request when switching away
        if algo != "CNN: Best model":
            self._cnn_pending_request = None

        if algo == "CNN: Best model":
            self._apply_cnn_async()
            return

        result = self._run_algorithm(algo, self.original_gray)
        if result is not None:
            self.result_bgr = result
            self.previews.show_image(self.previews.canvas_result, result, is_gray=False)
            self.previews.info_var.set(f"Applied: {algo}")

    def _run_algorithm(self, algo: str, gray: np.ndarray) -> np.ndarray | None:
        try:
            if algo == "LUT: Jet":
                return lut.apply_lut(gray, "jet")
            if algo == "LUT: Plasma":
                return lut.apply_lut(gray, "plasma")
            if algo == "LUT: Viridis":
                return lut.apply_lut(gray, "viridis")
            if algo == "LUT: Hot":
                return lut.apply_lut(gray, "hot")
            if algo == "LUT: Turbo":
                return lut.apply_lut(gray, "turbo")
            if algo == "LUT: Custom (B-G-R)":
                return lut.apply_custom_lut(gray, lut.build_custom_lut())
            if algo == "Slicing: 4 levels":
                return slicing.apply_level_slicing(gray, 4)
            if algo == "Slicing: 6 levels":
                return slicing.apply_level_slicing(gray, 6)
            if algo == "Slicing: 8 levels":
                return slicing.apply_level_slicing(gray, 8)
            if algo == "Slicing: Background preserved":
                return slicing.apply_level_slicing(gray, self.controls.slices_var.get(), background=True)
            if algo == "Slicing: Custom thresholds":
                thresholds = self._parse_thresholds(self.controls.thresholds_entry.get())
                return slicing.apply_custom_slicing(gray, thresholds, slicing.DEFAULT_COLORS)
            if algo == "Mapping: Sine":
                return mapping.apply_sin_mapping(gray, self.controls.freq_var.get() / 10)
            if algo == "Mapping: Density":
                return mapping.apply_density_mapping(gray)
            if algo == "Mapping: Gamma + Plasma":
                return mapping.apply_gamma_mapped(gray, self.controls.gamma_var.get() / 10)
            if algo == "Mapping: Histogram EQ":
                return mapping.apply_histogram_equalized_lut(gray)
            if algo == "CNN: Best model":
                from algorithms import cnn
                return cnn.apply_trained_model(
                    gray, 
                    output_size=(gray.shape[1], gray.shape[0]),
                    color_boost=self.controls.saturation_var.get() / 10
                )
        except Exception as exc:
            messagebox.showerror("Algorithm error", str(exc))
        return None

    @staticmethod
    def _parse_thresholds(text: str) -> list[int]:
        thresholds = []
        for part in text.split(","):
            value = part.strip()
            if not value:
                continue
            try:
                threshold = int(value)
            except ValueError as exc:
                raise ValueError(f"Invalid threshold value: {value}") from exc
            if not 0 < threshold < 255:
                raise ValueError("Thresholds must be between 1 and 254.")
            thresholds.append(threshold)
        thresholds = sorted(dict.fromkeys(thresholds))
        if len(thresholds) > 7:
            raise ValueError("Use at most 7 thresholds.")
        return thresholds

    # ------------------------------------------------------------------ #
    #  CNN async pipeline                                                  #
    # ------------------------------------------------------------------ #

    def _apply_cnn_async(self):
        if self.original_gray is None:
            return

        request_id = self._cnn_request_id + 1
        self._cnn_request_id = request_id
        gray_copy = self.original_gray.copy()
        output_size = (gray_copy.shape[1], gray_copy.shape[0])
        color_boost = self.controls.saturation_var.get() / 10
        self._cnn_pending_request = (request_id, gray_copy, output_size, color_boost)

        if self._cnn_busy:
            self.previews.info_var.set("CNN queued — previous run finishing…")
            return

        self._start_next_cnn_request()

    def _start_next_cnn_request(self):
        if self._cnn_pending_request is None:
            self._cnn_busy = False
            self.controls.set_apply_btn_state(enabled=True)
            return

        request_id, gray_copy, output_size, color_boost = self._cnn_pending_request
        self._cnn_pending_request = None
        self._cnn_busy = True

        # Clear stale result and show loading state
        self.result_bgr = None
        self.previews.draw_loading(self.previews.canvas_result)
        self.previews.info_var.set("Running CNN model…  please wait")
        self.controls.set_apply_btn_state(enabled=False)

        future = self._cnn_executor.submit(self._run_cnn_model, gray_copy, output_size, color_boost)
        future.add_done_callback(
            lambda f: self.root.after(0, lambda: self._on_cnn_finished(request_id, f))
        )

    @staticmethod
    def _run_cnn_model(gray: np.ndarray, output_size: tuple[int, int], color_boost: float) -> np.ndarray:
        from algorithms import cnn
        return cnn.apply_trained_model(gray, output_size=output_size, color_boost=color_boost)

    def _on_cnn_finished(self, request_id: int, future):
        self._cnn_busy = False
        self.controls.set_apply_btn_state(enabled=True)

        # Discard if algo switched or a newer request superseded this one
        if self.controls.algo_var.get() != "CNN: Best model" or request_id != self._cnn_request_id:
            self._start_next_cnn_request()
            return

        try:
            result = future.result()
        except Exception as exc:
            messagebox.showerror("CNN error", str(exc))
            self.previews.info_var.set("CNN failed")
            self.previews.draw_placeholder(self.previews.canvas_result, "CNN failed\nSee error dialog")
            self._start_next_cnn_request()
            return

        self.result_bgr = result
        self.previews.show_image(self.previews.canvas_result, result, is_gray=False)
        self.previews.info_var.set("Applied: CNN: Best model")
        self._start_next_cnn_request()

    # ------------------------------------------------------------------ #
    #  Compare all                                                         #
    # ------------------------------------------------------------------ #

    def _compare_all(self):
        if self.original_gray is None:
            messagebox.showinfo("No image", "Open an image first.")
            return

        compare_images: list[ImageTk.PhotoImage] = []

        win = ctk.CTkToplevel(self.root)
        win.title("All algorithms — comparison")
        win.geometry("1150x820")
        win.grab_set()

        scroll = ctk.CTkScrollableFrame(win, label_text="All Results")
        scroll.pack(fill="both", expand=True, padx=10, pady=10)

        cols = 5
        thumb = 200
        valid_idx = 0

        all_entries = [("Original", None)] + [(a, a) for a in ALGORITHMS if a != "CNN: Best model"]

        for label, algo in all_entries:
            r, c = divmod(valid_idx, cols)
            frame = ctk.CTkFrame(scroll, corner_radius=8)
            frame.grid(row=r, column=c, padx=6, pady=6)
            valid_idx += 1

            if algo is None:
                img_np = cv2.cvtColor(self.original_gray, cv2.COLOR_GRAY2RGB)
            else:
                bgr = self._run_algorithm(algo, self.original_gray)
                if bgr is None:
                    continue
                img_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            pil = Image.fromarray(img_np).resize((thumb, thumb), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil)

            canvas = tk.Canvas(frame, width=thumb, height=thumb, bg="#1e1e1e", highlightthickness=0)
            canvas.create_image(0, 0, anchor="nw", image=photo)
            compare_images.append(photo)
            canvas.pack(padx=4, pady=4)

            ctk.CTkLabel(
                frame, text=label,
                font=ctk.CTkFont(size=11), wraplength=thumb,
            ).pack(pady=(0, 6))

        def _on_compare_cnn_done(future):
            try:
                bgr = future.result()
                if bgr is None:
                    return
                img_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(img_np).resize((thumb, thumb), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(pil)

                nonlocal valid_idx
                r, c = divmod(valid_idx, cols)
                frame = ctk.CTkFrame(scroll, corner_radius=8)
                frame.grid(row=r, column=c, padx=6, pady=6)
                valid_idx += 1

                canvas = tk.Canvas(frame, width=thumb, height=thumb, bg="#1e1e1e", highlightthickness=0)
                canvas.create_image(0, 0, anchor="nw", image=photo)
                compare_images.append(photo)
                canvas.pack(padx=4, pady=4)

                ctk.CTkLabel(
                    frame, text="CNN: Best model",
                    font=ctk.CTkFont(size=11), wraplength=thumb,
                ).pack(pady=(0, 6))
            except Exception as e:
                print(f"Compare CNN failed: {e}")

        # Start CNN asynchronously for compare
        color_boost = self.controls.saturation_var.get() / 10
        out_size = (self.original_gray.shape[1], self.original_gray.shape[0])
        cnn_future = self._cnn_executor.submit(self._run_cnn_model, self.original_gray.copy(), out_size, color_boost)
        cnn_future.add_done_callback(
            lambda f: self.root.after(0, lambda: _on_compare_cnn_done(f))
        )

        ctk.CTkLabel(
            win,
            text="ℹ️  CNN: Best model is running asynchronously and will appear when finished.",
            font=ctk.CTkFont(size=11), text_color="gray60",
        ).pack(pady=(0, 8))

        # prevent GC of thumbnails while window is open
        win._compare_images = compare_images  # noqa: SLF001


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def launch():
    root = ctk.CTk()
    PseudoColorStudioApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
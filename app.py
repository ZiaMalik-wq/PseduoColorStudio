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

PREVIEW_SIZE = 320

# Algorithms grouped by category for the tabbed selector
ALGORITHM_CATEGORIES = {
    "LUT": [
        "LUT: Jet",
        "LUT: Plasma",
        "LUT: Viridis",
        "LUT: Hot",
        "LUT: Turbo",
        "LUT: Custom (B-G-R)",
    ],
    "Slicing": [
        "Slicing: 4 levels",
        "Slicing: 6 levels",
        "Slicing: 8 levels",
        "Slicing: Background preserved",
        "Slicing: Custom thresholds",
    ],
    "Mapping": [
        "Mapping: Sine",
        "Mapping: Density",
        "Mapping: Gamma + Plasma",
        "Mapping: Histogram EQ",
    ],
    "CNN": [
        "CNN: Best model",
    ],
}

# Flat list for save-all / compare-all
ALGORITHMS = [algo for algos in ALGORITHM_CATEGORIES.values() for algo in algos]


class PseudoColorAlgorithmsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pseudo-Color Studio - Algorithms + CNN")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root.geometry("1280x800")
        self.root.minsize(1060, 700)

        self.original_color = None
        self.original_gray = None
        self.result_bgr = None

        # Keep strong references to PhotoImage objects so they aren't GC'd
        self._canvas_images: dict[tk.Canvas, ImageTk.PhotoImage] = {}
        self._compare_images: list[ImageTk.PhotoImage] = []

        # CNN async state
        self._cnn_executor = ThreadPoolExecutor(max_workers=1)
        self._cnn_request_id = 0
        self._cnn_pending_request = None
        self._cnn_busy = False

        self._build_ui()
        self._draw_placeholder(self.canvas_source, "Open an image\nto begin")
        self._draw_placeholder(self.canvas_orig, "Grayscale\npreview")
        self._draw_placeholder(self.canvas_result, "Output\nwill appear here")

    # ------------------------------------------------------------------ #
    #  UI construction                                                     #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        # ── Toolbar ──────────────────────────────────────────────────────
        toolbar = ctk.CTkFrame(self.root, corner_radius=0, height=52)
        toolbar.pack(fill="x")
        toolbar.pack_propagate(False)

        ctk.CTkLabel(
            toolbar,
            text="Pseudo-Color Studio",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(side="left", padx=16, pady=10)

        btn_frame = ctk.CTkFrame(toolbar, fg_color="transparent")
        btn_frame.pack(side="left", padx=4, pady=8)

        ctk.CTkButton(btn_frame, text="Open image", width=110, command=self._open_image).pack(
            side="left", padx=4
        )
        ctk.CTkButton(btn_frame, text="Save result", width=110, command=self._save_result).pack(
            side="left", padx=4
        )
        ctk.CTkButton(btn_frame, text="Save all results", width=130, command=self._save_all).pack(
            side="left", padx=4
        )

        self.status_var = ctk.StringVar(value="No image loaded")
        ctk.CTkLabel(
            toolbar,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=12),
            text_color="gray70",
        ).pack(side="right", padx=16)

        # ── Main area ────────────────────────────────────────────────────
        main = ctk.CTkFrame(self.root, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=12, pady=12)

        self._build_controls(main)
        self._build_previews(main)

    def _build_controls(self, parent):
        # Single scrollable sidebar — no nested scrollable frames
        ctrl = ctk.CTkScrollableFrame(parent, width=290, label_text="Controls")
        ctrl.pack(side="left", fill="y", padx=(0, 10), pady=0)

        # ── Algorithm category tabs ───────────────────────────────────────
        ctk.CTkLabel(
            ctrl, text="Category", font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=12, pady=(10, 4))

        self.category_var = ctk.StringVar(value="LUT")
        tab_frame = ctk.CTkFrame(ctrl, fg_color="transparent")
        tab_frame.pack(fill="x", padx=10, pady=(0, 8))

        for cat in ALGORITHM_CATEGORIES:
            ctk.CTkButton(
                tab_frame,
                text=cat,
                width=58,
                height=28,
                corner_radius=6,
                fg_color="transparent",
                border_width=1,
                font=ctk.CTkFont(size=12),
                command=lambda c=cat: self._on_category_change(c),
            ).pack(side="left", padx=2, pady=2)

        # ── Algorithm radio buttons ──
        ctk.CTkLabel(
            ctrl, text="Algorithm", font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=12, pady=(4, 4))

        self.algo_radio_frame = ctk.CTkFrame(ctrl, fg_color="gray17", corner_radius=8)
        self.algo_radio_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.algo_var = ctk.StringVar(value=ALGORITHMS[0])
        self._radio_buttons: list[ctk.CTkRadioButton] = []
        self._rebuild_radio_buttons("LUT")
        
        ctk.CTkLabel(
            ctrl, text="Parameters", font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=12, pady=(4, 4))
        
        self.params_container = ctk.CTkFrame(ctrl, fg_color="gray17", corner_radius=8, height=160)
        self.params_container.pack(fill="x", padx=10, pady=(0, 10))
        self.params_container.pack_propagate(False)

        # Sine frequency
        self.freq_frame = ctk.CTkFrame(self.params_container, fg_color="transparent")
        self.freq_display = self._make_slider_row(
            self.freq_frame, "Sine frequency", hint="Mapping: Sine"
        )
        self.freq_var = ctk.IntVar(value=10)
        self.freq_slider = ctk.CTkSlider(
            self.freq_frame,
            variable=self.freq_var,
            from_=5,
            to=50,
            command=self._apply_slider,
        )
        self.freq_slider.pack(fill="x", padx=12, pady=(0, 6))
        self.freq_var.trace_add(
            "write",
            lambda *_: self.freq_display.configure(text=f"{self.freq_var.get() / 10:.1f}"),
        )

        # Gamma
        self.gamma_frame = ctk.CTkFrame(self.params_container, fg_color="transparent")
        self.gamma_display = self._make_slider_row(
            self.gamma_frame, "Gamma", hint="Mapping: Gamma + Plasma"
        )
        self.gamma_var = ctk.IntVar(value=15)
        self.gamma_slider = ctk.CTkSlider(
            self.gamma_frame,
            variable=self.gamma_var,
            from_=3,
            to=40,
            command=self._apply_slider,
        )
        self.gamma_slider.pack(fill="x", padx=12, pady=(0, 6))
        self.gamma_var.trace_add(
            "write",
            lambda *_: self.gamma_display.configure(text=f"{self.gamma_var.get() / 10:.1f}"),
        )

        # Slice count
        self.slices_frame = ctk.CTkFrame(self.params_container, fg_color="transparent")
        self.slices_display = self._make_slider_row(
            self.slices_frame, "Slice count", hint="Slicing: Background preserved"
        )
        self.slices_var = ctk.IntVar(value=6)
        self.slices_slider = ctk.CTkSlider(
            self.slices_frame,
            variable=self.slices_var,
            from_=2,
            to=8,
            number_of_steps=6,
            command=self._apply_slider,
        )
        self.slices_slider.pack(fill="x", padx=12, pady=(0, 6))
        self.slices_var.trace_add(
            "write",
            lambda *_: self.slices_display.configure(text=str(self.slices_var.get())),
        )

        # Custom thresholds
        self.thresholds_frame = ctk.CTkFrame(self.params_container, fg_color="transparent")
        ctk.CTkLabel(
            self.thresholds_frame,
            text="Custom thresholds",
            font=ctk.CTkFont(size=12),
        ).pack(anchor="w", padx=12, pady=(8, 2))
        self.thresholds_entry = ctk.CTkEntry(
            self.thresholds_frame, placeholder_text="64, 128, 192"
        )
        self.thresholds_entry.insert(0, "64, 128, 192")
        self.thresholds_entry.pack(fill="x", padx=12, pady=(0, 4))
        ctk.CTkLabel(
            self.thresholds_frame,
            text="Used by: Slicing: Custom thresholds",
            text_color="gray60",
            font=ctk.CTkFont(size=11),
        ).pack(anchor="w", padx=12, pady=(0, 6))

        # No-params placeholder
        self.no_params_label = ctk.CTkLabel(
            self.params_container,
            text="No parameters\nfor this algorithm",
            text_color="gray55",
            font=ctk.CTkFont(size=12),
        )

        self._update_parameter_visibility()

        # ── Divider ───────────────────────────────────────────────────────
        ctk.CTkFrame(ctrl, height=1, fg_color="gray30").pack(fill="x", padx=12, pady=8)

        # ── Action buttons ────────────────────────────────────────────────
        self.apply_btn = ctk.CTkButton(
            ctrl,
            text="Apply",
            command=self._apply,
            fg_color="#2b6ef2",
            hover_color="#245ed0",
            text_color="white",
        )
        self.apply_btn.pack(fill="x", padx=12, pady=(0, 6))

        ctk.CTkButton(
            ctrl,
            text="Apply all & compare",
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "#DCE4EE"),
            command=self._compare_all,
        ).pack(fill="x", padx=12, pady=(0, 12))

    def _build_previews(self, parent):
        prev = ctk.CTkFrame(parent, fg_color="transparent")
        prev.pack(side="left", fill="both", expand=True)

        # Column headers
        header = ctk.CTkFrame(prev, fg_color="transparent")
        header.pack(fill="x", pady=(0, 4))
        for title in ("Original Image", "Grayscale Input", "Colorized Output"):
            ctk.CTkLabel(
                header, text=title, font=ctk.CTkFont(size=13, weight="bold")
            ).pack(side="left", expand=True)

        # Canvas row
        canvas_row = ctk.CTkFrame(prev, fg_color="transparent")
        canvas_row.pack(fill="both", expand=True, pady=(0, 4))

        self.frame_source = ctk.CTkFrame(canvas_row, corner_radius=8)
        self.frame_source.pack(side="left", expand=True, fill="both", padx=(0, 6))
        self.canvas_source = self._make_canvas(self.frame_source)

        self.frame_orig = ctk.CTkFrame(canvas_row, corner_radius=8)
        self.frame_orig.pack(side="left", expand=True, fill="both", padx=(0, 6))
        self.canvas_orig = self._make_canvas(self.frame_orig)

        self.frame_result = ctk.CTkFrame(canvas_row, corner_radius=8)
        self.frame_result.pack(side="left", expand=True, fill="both")
        self.canvas_result = self._make_canvas(self.frame_result)

        # Info bar
        self.info_var = ctk.StringVar(value="Open an image to begin")
        ctk.CTkLabel(
            prev,
            textvariable=self.info_var,
            font=ctk.CTkFont(size=12),
            text_color="gray70",
        ).pack(pady=2)

    # ------------------------------------------------------------------ #
    #  Helper widgets                                                      #
    # ------------------------------------------------------------------ #

    def _make_canvas(self, parent) -> tk.Canvas:
        canvas = tk.Canvas(parent, bg="#1e1e1e", highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=2, pady=2)
        return canvas

    def _make_slider_row(self, parent, label: str, hint: str) -> ctk.CTkLabel:
        """Creates a label row with name on left and current value on right; returns value label."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(8, 0))
        ctk.CTkLabel(row, text=label, font=ctk.CTkFont(size=12)).pack(side="left")
        value_lbl = ctk.CTkLabel(row, text="", font=ctk.CTkFont(size=12))
        value_lbl.pack(side="right")
        ctk.CTkLabel(
            parent,
            text=f"Used by: {hint}",
            text_color="gray60",
            font=ctk.CTkFont(size=11),
        ).pack(anchor="w", padx=12, pady=(0, 2))
        return value_lbl

    def _draw_placeholder(self, canvas: tk.Canvas, text: str):
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(), PREVIEW_SIZE)
        ch = max(canvas.winfo_height(), PREVIEW_SIZE)
        canvas.delete("all")
        canvas.create_text(
            cw // 2,
            ch // 2,
            text=text,
            fill="#555555",
            font=("Courier", 13),
            justify="center",
        )

    def _draw_loading(self, canvas: tk.Canvas, text: str = "⏳  Running CNN…"):
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(), PREVIEW_SIZE)
        ch = max(canvas.winfo_height(), PREVIEW_SIZE)
        canvas.delete("all")
        # Subtle pulsing background rectangle
        canvas.create_rectangle(
            cw // 2 - 110, ch // 2 - 24, cw // 2 + 110, ch // 2 + 24,
            fill="#2a2a2a", outline="#444444", width=1,
        )
        canvas.create_text(
            cw // 2, ch // 2,
            text=text,
            fill="#aaaaaa",
            font=("Courier", 13),
            justify="center",
        )

    # ------------------------------------------------------------------ #
    #  Category / algorithm selection                                      #
    # ------------------------------------------------------------------ #

    def _on_category_change(self, category: str):
        self.category_var.set(category)
        # Pick first algo in the new category as default
        first_algo = ALGORITHM_CATEGORIES[category][0]
        self.algo_var.set(first_algo)
        self._rebuild_radio_buttons(category)
        self._update_parameter_visibility()
        self._apply()

    def _rebuild_radio_buttons(self, category: str):
        for rb in self._radio_buttons:
            rb.destroy()
        self._radio_buttons.clear()

        for algo in ALGORITHM_CATEGORIES[category]:
            rb = ctk.CTkRadioButton(
                self.algo_radio_frame,
                text=algo,
                variable=self.algo_var,
                value=algo,
                command=self._on_algorithm_change,
            )
            rb.pack(anchor="w", padx=12, pady=5)
            self._radio_buttons.append(rb)

    def _on_algorithm_change(self):
        self._update_parameter_visibility()
        if self.algo_var.get() != "CNN: Best model":
            self._cnn_pending_request = None
        self._apply()

    # ------------------------------------------------------------------ #
    #  Parameter visibility (fixed-height container → no layout jumps)    #
    # ------------------------------------------------------------------ #

    def _hide_all_param_children(self):
        for child in (
            self.freq_frame,
            self.gamma_frame,
            self.slices_frame,
            self.thresholds_frame,
            self.no_params_label,
        ):
            child.place_forget()

    def _update_parameter_visibility(self):
        algo = self.algo_var.get()
        self._hide_all_param_children()

        if algo == "Mapping: Sine":
            self.freq_frame.place(relx=0, rely=0, relwidth=1)
        elif algo == "Mapping: Gamma + Plasma":
            self.gamma_frame.place(relx=0, rely=0, relwidth=1)
        elif algo == "Slicing: Background preserved":
            self.slices_frame.place(relx=0, rely=0, relwidth=1)
        elif algo == "Slicing: Custom thresholds":
            self.thresholds_frame.place(relx=0, rely=0, relwidth=1)
        else:
            self.no_params_label.place(relx=0.5, rely=0.5, anchor="center")

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
        self.status_var.set(f"{name}  |  {img_color.shape[1]}×{img_color.shape[0]} px")
        self.info_var.set(f"Loaded: {name}")

        self._show_image(self.canvas_source, self.original_color, is_gray=False)
        self._show_image(self.canvas_orig, self.original_gray, is_gray=True)
        self._draw_placeholder(self.canvas_result, "Applying…")
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
            self.status_var.set(f"Saved → {os.path.basename(path)}")

    def _save_all(self):
        if self.original_gray is None:
            messagebox.showinfo("No image", "Open an image first.")
            return
        folder = filedialog.askdirectory(title="Choose output folder")
        if not folder:
            return

        saved = 0
        for algo in ALGORITHMS:
            if algo == "CNN: Best model":
                continue  # skip async CNN in batch save
            result = self._run_algorithm(algo, self.original_gray)
            if result is not None:
                safe = algo.replace(":", "").replace(" ", "_").replace("/", "-")
                cv2.imwrite(os.path.join(folder, f"{safe}.png"), result)
                saved += 1

        self.status_var.set(f"All results saved → {folder}")
        messagebox.showinfo("Done", f"Saved {saved} images to:\n{folder}")

    # ------------------------------------------------------------------ #
    #  Algorithm application                                              #
    # ------------------------------------------------------------------ #

    def _apply_slider(self, _=None):
        self._apply()

    def _apply(self):
        if self.original_gray is None:
            return
        algo = self.algo_var.get()
        if algo == "CNN: Best model":
            self._apply_cnn_async()
            return

        result = self._run_algorithm(algo, self.original_gray)
        if result is not None:
            self.result_bgr = result
            self._show_image(self.canvas_result, result, is_gray=False)
            self.info_var.set(f"Applied: {algo}")

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
                return slicing.apply_level_slicing(gray, self.slices_var.get(), background=True)
            if algo == "Slicing: Custom thresholds":
                thresholds = self._parse_thresholds(self.thresholds_entry.get())
                return slicing.apply_custom_slicing(gray, thresholds, slicing.DEFAULT_COLORS)
            if algo == "Mapping: Sine":
                return mapping.apply_sin_mapping(gray, self.freq_var.get() / 10)
            if algo == "Mapping: Density":
                return mapping.apply_density_mapping(gray)
            if algo == "Mapping: Gamma + Plasma":
                return mapping.apply_gamma_mapped(gray, self.gamma_var.get() / 10)
            if algo == "Mapping: Histogram EQ":
                return mapping.apply_histogram_equalized_lut(gray)
            if algo == "CNN: Best model":
                from algorithms import cnn
                return cnn.apply_trained_model(gray, output_size=(gray.shape[1], gray.shape[0]))
        except Exception as exc:
            messagebox.showerror("Algorithm error", str(exc))
        return None

    def _parse_thresholds(self, text: str) -> list[int]:
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
        self._cnn_pending_request = (request_id, gray_copy, output_size)

        if self._cnn_busy:
            self.info_var.set("CNN queued — previous run finishing…")
            return

        self._start_next_cnn_request()

    def _start_next_cnn_request(self):
        if self._cnn_pending_request is None:
            self._cnn_busy = False
            self._set_apply_btn_state(enabled=True)
            return

        request_id, gray_copy, output_size = self._cnn_pending_request
        self._cnn_pending_request = None
        self._cnn_busy = True

        # ── Clear stale result and show explicit loading state ────────────
        self.result_bgr = None
        self._draw_loading(self.canvas_result)
        self.info_var.set("Running CNN model…  please wait")
        self._set_apply_btn_state(enabled=False)

        future = self._cnn_executor.submit(self._run_cnn_model, gray_copy, output_size)
        future.add_done_callback(
            lambda f: self.root.after(0, lambda: self._on_cnn_finished(request_id, f))
        )

    def _run_cnn_model(self, gray: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
        from algorithms import cnn
        return cnn.apply_trained_model(gray, output_size=output_size)

    def _on_cnn_finished(self, request_id: int, future):
        self._cnn_busy = False
        self._set_apply_btn_state(enabled=True)

        # Discard if algo switched or a newer request superseded this one
        if self.algo_var.get() != "CNN: Best model" or request_id != self._cnn_request_id:
            self._start_next_cnn_request()
            return

        try:
            result = future.result()
        except Exception as exc:
            messagebox.showerror("CNN error", str(exc))
            self.info_var.set("CNN failed")
            self._draw_placeholder(self.canvas_result, "CNN failed\nSee error dialog")
            self._start_next_cnn_request()
            return

        self.result_bgr = result
        self._show_image(self.canvas_result, result, is_gray=False)
        self.info_var.set("Applied: CNN: Best model")
        self._start_next_cnn_request()

    def _set_apply_btn_state(self, enabled: bool):
        if enabled:
            self.apply_btn.configure(state="normal", text="Apply")
        else:
            self.apply_btn.configure(state="disabled", text="Running…")

    # ------------------------------------------------------------------ #
    #  Compare all                                                         #
    # ------------------------------------------------------------------ #

    def _compare_all(self):
        if self.original_gray is None:
            messagebox.showinfo("No image", "Open an image first.")
            return

        self._compare_images.clear()

        win = ctk.CTkToplevel(self.root)
        win.title("All algorithms — comparison")
        win.geometry("1150x820")
        win.grab_set()  # make it modal so it can't fall behind

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
            self._compare_images.append(photo)
            canvas.pack(padx=4, pady=4)

            ctk.CTkLabel(
                frame,
                text=label,
                font=ctk.CTkFont(size=11),
                wraplength=thumb,
            ).pack(pady=(0, 6))

        # CNN note
        ctk.CTkLabel(
            win,
            text="ℹ️  CNN: Best model is excluded from batch compare (runs asynchronously).",
            font=ctk.CTkFont(size=11),
            text_color="gray60",
        ).pack(pady=(0, 8))

    # ------------------------------------------------------------------ #
    #  Image display                                                       #
    # ------------------------------------------------------------------ #

    def _show_image(self, canvas: tk.Canvas, img: np.ndarray, is_gray: bool):
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(), PREVIEW_SIZE)
        ch = max(canvas.winfo_height(), PREVIEW_SIZE)

        pil = (
            Image.fromarray(img).convert("RGB")
            if is_gray
            else Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        )

        iw, ih = pil.size
        scale = min(cw / iw, ch / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        pil = pil.resize((nw, nh), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(pil)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, anchor="center", image=photo)
        self._canvas_images[canvas] = photo  # keep strong ref


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def launch():
    root = ctk.CTk()
    PseudoColorAlgorithmsApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
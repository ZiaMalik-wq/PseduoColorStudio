import tkinter as tk

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

PREVIEW_SIZE = 320


class PreviewPanel:
    """Three-panel image preview area (Original → Grayscale → Colorized)."""

    def __init__(self, parent):
        self.info_var = ctk.StringVar(value="Open an image to begin")

        # Keep strong references so PhotoImages aren't garbage-collected
        self._canvas_images: dict[tk.Canvas, ImageTk.PhotoImage] = {}

        prev = ctk.CTkFrame(parent, fg_color="transparent")
        prev.pack(side="left", fill="both", expand=True)

        # ── Column headers ───────────────────────────────────────────────
        header = ctk.CTkFrame(prev, fg_color="transparent")
        header.pack(fill="x", pady=(0, 4))
        for title in ("Original Image", "Grayscale Input", "Colorized Output"):
            ctk.CTkLabel(
                header, text=title, font=ctk.CTkFont(size=13, weight="bold")
            ).pack(side="left", expand=True)

        # ── Canvas row ───────────────────────────────────────────────────
        canvas_row = ctk.CTkFrame(prev, fg_color="transparent")
        canvas_row.pack(fill="both", expand=True, pady=(0, 4))

        frame_source = ctk.CTkFrame(canvas_row, corner_radius=8)
        frame_source.pack(side="left", expand=True, fill="both", padx=(0, 6))
        self.canvas_source = self._make_canvas(frame_source)

        frame_orig = ctk.CTkFrame(canvas_row, corner_radius=8)
        frame_orig.pack(side="left", expand=True, fill="both", padx=(0, 6))
        self.canvas_orig = self._make_canvas(frame_orig)

        frame_result = ctk.CTkFrame(canvas_row, corner_radius=8)
        frame_result.pack(side="left", expand=True, fill="both")
        self.canvas_result = self._make_canvas(frame_result)

        # ── Info bar ─────────────────────────────────────────────────────
        ctk.CTkLabel(
            prev, textvariable=self.info_var,
            font=ctk.CTkFont(size=12), text_color="gray70",
        ).pack(pady=2)

        # ── Initial placeholders ─────────────────────────────────────────
        self.draw_placeholder(self.canvas_source, "Open an image\nto begin")
        self.draw_placeholder(self.canvas_orig, "Grayscale\npreview")
        self.draw_placeholder(self.canvas_result, "Output\nwill appear here")

    # ------------------------------------------------------------------ #
    #  Canvas helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_canvas(parent) -> tk.Canvas:
        canvas = tk.Canvas(parent, bg="#1e1e1e", highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=2, pady=2)
        return canvas

    def draw_placeholder(self, canvas: tk.Canvas, text: str):
        """Draw centred placeholder text on a canvas."""
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(), PREVIEW_SIZE)
        ch = max(canvas.winfo_height(), PREVIEW_SIZE)
        canvas.delete("all")
        canvas.create_text(
            cw // 2, ch // 2, text=text,
            fill="#555555", font=("Courier", 13), justify="center",
        )

    def draw_loading(self, canvas: tk.Canvas, text: str = "⏳  Running CNN…"):
        """Draw a loading indicator on a canvas."""
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(), PREVIEW_SIZE)
        ch = max(canvas.winfo_height(), PREVIEW_SIZE)
        canvas.delete("all")
        canvas.create_rectangle(
            cw // 2 - 110, ch // 2 - 24, cw // 2 + 110, ch // 2 + 24,
            fill="#2a2a2a", outline="#444444", width=1,
        )
        canvas.create_text(
            cw // 2, ch // 2, text=text,
            fill="#aaaaaa", font=("Courier", 13), justify="center",
        )

    def show_image(self, canvas: tk.Canvas, img, *, is_gray: bool):
        """Render a NumPy image (BGR or grayscale) onto a canvas, scaled to fit."""
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
        self._canvas_images[canvas] = photo  # prevent GC

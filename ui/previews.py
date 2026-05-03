import tkinter as tk

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

PREVIEW_SIZE = 320
_RESIZE_DELAY_MS = 150  # debounce interval for window-resize redraws


class PreviewPanel:
    """Three-panel image preview area (Original → Grayscale → Colorized)."""

    def __init__(self, parent):
        self.info_var = ctk.StringVar(value="Open an image to begin")

        # Keep strong references so PhotoImages aren't garbage-collected
        self._canvas_images: dict[tk.Canvas, ImageTk.PhotoImage] = {}

        # Store raw image data per canvas so we can re-render on resize
        self._canvas_data: dict[tk.Canvas, tuple] = {}  # canvas -> (img, is_gray)

        # Debounce timer id per canvas
        self._resize_after_ids: dict[tk.Canvas, str] = {}

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

    def _make_canvas(self, parent) -> tk.Canvas:
        canvas = tk.Canvas(parent, bg="#1e1e1e", highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=2, pady=2)
        canvas.bind("<Configure>", lambda e, c=canvas: self._on_canvas_resize(c))
        return canvas

    def _on_canvas_resize(self, canvas: tk.Canvas):
        """Debounced handler — re-renders the stored image after resizing stops."""
        # Cancel any pending redraw for this canvas
        if canvas in self._resize_after_ids:
            canvas.after_cancel(self._resize_after_ids[canvas])

        self._resize_after_ids[canvas] = canvas.after(
            _RESIZE_DELAY_MS,
            lambda: self._redraw(canvas),
        )

    def _redraw(self, canvas: tk.Canvas):
        """Re-render the stored image data at the canvas's current size."""
        self._resize_after_ids.pop(canvas, None)
        if canvas in self._canvas_data:
            img, is_gray = self._canvas_data[canvas]
            self._render_image(canvas, img, is_gray=is_gray)

    def draw_placeholder(self, canvas: tk.Canvas, text: str):
        """Draw centred placeholder text on a canvas."""
        # Clear stored image data so resize doesn't overwrite the placeholder
        self._canvas_data.pop(canvas, None)

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
        self._canvas_data.pop(canvas, None)

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
        """Store the image data and render it onto the canvas, scaled to fit."""
        self._canvas_data[canvas] = (img, is_gray)
        self._render_image(canvas, img, is_gray=is_gray)

    def _render_image(self, canvas: tk.Canvas, img, *, is_gray: bool):
        """Internal: scale and draw a NumPy image onto a canvas at its current size."""
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

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

        # Sync pan and zoom state
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._drag_start_x = 0
        self._drag_start_y = 0

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

        # Zoom and pan bindings
        canvas.bind("<MouseWheel>", self._on_mousewheel)
        canvas.bind("<Button-4>", self._on_mousewheel)  # Linux scroll up
        canvas.bind("<Button-5>", self._on_mousewheel)  # Linux scroll down
        canvas.bind("<ButtonPress-1>", self._on_pan_start)
        canvas.bind("<B1-Motion>", self._on_pan_drag)
        canvas.bind("<Double-Button-1>", self._reset_view)
        
        return canvas

    def _on_mousewheel(self, event):
        if event.num == 5 or event.delta < 0:
            self.zoom_factor /= 1.15
        elif event.num == 4 or event.delta > 0:
            self.zoom_factor *= 1.15
            
        self.zoom_factor = max(1.0, min(self.zoom_factor, 50.0))
        
        if self.zoom_factor == 1.0:
            self.pan_x = 0
            self.pan_y = 0
            
        self._redraw_all()

    def _on_pan_start(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _on_pan_drag(self, event):
        if self.zoom_factor <= 1.0:
            return
            
        dx = event.x - self._drag_start_x
        dy = event.y - self._drag_start_y
        self.pan_x += dx
        self.pan_y += dy
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        
        self._redraw_all()

    def _reset_view(self, event):
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._redraw_all()

    def _redraw_all(self):
        for canvas in (self.canvas_source, self.canvas_orig, self.canvas_result):
            if canvas in self._canvas_data:
                img, is_gray = self._canvas_data[canvas]
                self._render_image(canvas, img, is_gray=is_gray)

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
        bs = min(cw / iw, ch / ih)
        s = bs * self.zoom_factor

        center_x = cw / 2 + self.pan_x
        center_y = ch / 2 + self.pan_y

        cx_min = center_x - (iw * s) / 2
        cy_min = center_y - (ih * s) / 2
        cx_max = center_x + (iw * s) / 2
        cy_max = center_y + (ih * s) / 2

        crop_cx_min = max(cx_min, 0)
        crop_cy_min = max(cy_min, 0)
        crop_cx_max = min(cx_max, cw)
        crop_cy_max = min(cy_max, ch)

        if crop_cx_min >= crop_cx_max or crop_cy_min >= crop_cy_max:
            canvas.delete("all")
            return

        raw_x_min = (crop_cx_min - cx_min) / s
        raw_y_min = (crop_cy_min - cy_min) / s
        raw_x_max = (crop_cx_max - cx_min) / s
        raw_y_max = (crop_cy_max - cy_min) / s

        cropped_pil = pil.crop((raw_x_min, raw_y_min, raw_x_max, raw_y_max))
        target_w = int(crop_cx_max - crop_cx_min)
        target_h = int(crop_cy_max - crop_cy_min)

        if target_w <= 0 or target_h <= 0:
            canvas.delete("all")
            return

        resized_pil = cropped_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized_pil)

        canvas.delete("all")
        canvas.create_image((crop_cx_min + crop_cx_max) / 2, (crop_cy_min + crop_cy_max) / 2, anchor="center", image=photo)
        self._canvas_images[canvas] = photo  # prevent GC

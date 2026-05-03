import customtkinter as ctk


class ToolbarFrame:
    """Top toolbar with title, action buttons, and status label."""

    def __init__(self, parent, *, on_open, on_save, on_save_all):
        self.status_var = ctk.StringVar(value="No image loaded")

        toolbar = ctk.CTkFrame(parent, corner_radius=0, height=52)
        toolbar.pack(fill="x")
        toolbar.pack_propagate(False)

        ctk.CTkLabel(
            toolbar,
            text="Pseudo-Color Studio",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(side="left", padx=16, pady=10)

        btn_frame = ctk.CTkFrame(toolbar, fg_color="transparent")
        btn_frame.pack(side="left", padx=4, pady=8)

        ctk.CTkButton(btn_frame, text="Open image", width=110, command=on_open).pack(
            side="left", padx=4
        )
        ctk.CTkButton(btn_frame, text="Save result", width=110, command=on_save).pack(
            side="left", padx=4
        )
        ctk.CTkButton(btn_frame, text="Save all results", width=130, command=on_save_all).pack(
            side="left", padx=4
        )

        ctk.CTkLabel(
            toolbar,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=12),
            text_color="gray70",
        ).pack(side="right", padx=16)

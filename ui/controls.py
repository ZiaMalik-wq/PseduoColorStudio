import customtkinter as ctk

# ------------------------------------------------------------------ #
#  Algorithm registry                                                  #
# ------------------------------------------------------------------ #

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

# Flat list for batch operations
ALGORITHMS = [algo for algos in ALGORITHM_CATEGORIES.values() for algo in algos]


# ------------------------------------------------------------------ #
#  Control panel                                                       #
# ------------------------------------------------------------------ #

class ControlPanel:
    """Left sidebar with algorithm selection, parameter sliders, and action buttons."""

    def __init__(self, parent, *, on_apply, on_compare_all):
        self._on_apply = on_apply

        ctrl = ctk.CTkScrollableFrame(parent, width=290, label_text="Controls")
        ctrl.pack(side="left", fill="y", padx=(0, 10), pady=0)

        # ── Category tabs ────────────────────────────────────────────────
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

        # ── Algorithm radio buttons ──────────────────────────────────────
        ctk.CTkLabel(
            ctrl, text="Algorithm", font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=12, pady=(4, 4))

        self._algo_radio_frame = ctk.CTkFrame(ctrl, fg_color="gray17", corner_radius=8)
        self._algo_radio_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.algo_var = ctk.StringVar(value=ALGORITHMS[0])
        self._radio_buttons: list[ctk.CTkRadioButton] = []
        self._rebuild_radio_buttons("LUT")

        # ── Parameters ───────────────────────────────────────────────────
        ctk.CTkLabel(
            ctrl, text="Parameters", font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=12, pady=(4, 4))

        self._params_container = ctk.CTkFrame(ctrl, fg_color="gray17", corner_radius=8, height=160)
        self._params_container.pack(fill="x", padx=10, pady=(0, 10))
        self._params_container.pack_propagate(False)

        # Sine frequency
        self._freq_frame = ctk.CTkFrame(self._params_container, fg_color="transparent")
        self._freq_display = self._make_slider_row(
            self._freq_frame, "Sine frequency", hint="Mapping: Sine"
        )
        self.freq_var = ctk.IntVar(value=10)
        ctk.CTkSlider(
            self._freq_frame, variable=self.freq_var, from_=5, to=50,
            command=self._apply_slider,
        ).pack(fill="x", padx=12, pady=(0, 6))
        self.freq_var.trace_add(
            "write",
            lambda *_: self._freq_display.configure(text=f"{self.freq_var.get() / 10:.1f}"),
        )

        # Gamma
        self._gamma_frame = ctk.CTkFrame(self._params_container, fg_color="transparent")
        self._gamma_display = self._make_slider_row(
            self._gamma_frame, "Gamma", hint="Mapping: Gamma + Plasma"
        )
        self.gamma_var = ctk.IntVar(value=15)
        ctk.CTkSlider(
            self._gamma_frame, variable=self.gamma_var, from_=3, to=40,
            command=self._apply_slider,
        ).pack(fill="x", padx=12, pady=(0, 6))
        self.gamma_var.trace_add(
            "write",
            lambda *_: self._gamma_display.configure(text=f"{self.gamma_var.get() / 10:.1f}"),
        )

        # Slice count
        self._slices_frame = ctk.CTkFrame(self._params_container, fg_color="transparent")
        self._slices_display = self._make_slider_row(
            self._slices_frame, "Slice count", hint="Slicing: Background preserved"
        )
        self.slices_var = ctk.IntVar(value=6)
        ctk.CTkSlider(
            self._slices_frame, variable=self.slices_var, from_=2, to=8,
            number_of_steps=6, command=self._apply_slider,
        ).pack(fill="x", padx=12, pady=(0, 6))
        self.slices_var.trace_add(
            "write",
            lambda *_: self._slices_display.configure(text=str(self.slices_var.get())),
        )

        # Custom thresholds
        self._thresholds_frame = ctk.CTkFrame(self._params_container, fg_color="transparent")
        ctk.CTkLabel(
            self._thresholds_frame, text="Custom thresholds",
            font=ctk.CTkFont(size=12),
        ).pack(anchor="w", padx=12, pady=(8, 2))
        self.thresholds_entry = ctk.CTkEntry(
            self._thresholds_frame, placeholder_text="64, 128, 192"
        )
        self.thresholds_entry.insert(0, "64, 128, 192")
        self.thresholds_entry.pack(fill="x", padx=12, pady=(0, 4))
        self.thresholds_entry.bind("<KeyRelease>", self._validate_thresholds)
        ctk.CTkLabel(
            self._thresholds_frame,
            text="Used by: Slicing: Custom thresholds",
            text_color="gray60",
            font=ctk.CTkFont(size=11),
        ).pack(anchor="w", padx=12, pady=(0, 6))

        # Color Saturation (CNN)
        self._saturation_frame = ctk.CTkFrame(self._params_container, fg_color="transparent")
        self._saturation_display = self._make_slider_row(
            self._saturation_frame, "Color Saturation", hint="CNN: Best model"
        )
        self.saturation_var = ctk.IntVar(value=12)
        ctk.CTkSlider(
            self._saturation_frame, variable=self.saturation_var, from_=5, to=20,
            command=self._apply_slider,
        ).pack(fill="x", padx=12, pady=(0, 6))
        self.saturation_var.trace_add(
            "write",
            lambda *_: self._saturation_display.configure(text=f"{self.saturation_var.get() / 10:.1f}"),
        )

        # No-params placeholder
        self._no_params_label = ctk.CTkLabel(
            self._params_container,
            text="No parameters\nfor this algorithm",
            text_color="gray55",
            font=ctk.CTkFont(size=12),
        )

        self._update_parameter_visibility()

        # ── Divider ──────────────────────────────────────────────────────
        ctk.CTkFrame(ctrl, height=1, fg_color="gray30").pack(fill="x", padx=12, pady=8)

        # ── Action buttons ───────────────────────────────────────────────
        self.apply_btn = ctk.CTkButton(
            ctrl, text="Apply", command=on_apply,
            fg_color="#2b6ef2", hover_color="#245ed0", text_color="white",
        )
        self.apply_btn.pack(fill="x", padx=12, pady=(0, 6))

        ctk.CTkButton(
            ctrl, text="Apply all & compare",
            fg_color="transparent", border_width=2,
            text_color=("gray10", "#DCE4EE"), command=on_compare_all,
        ).pack(fill="x", padx=12, pady=(0, 12))

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_slider_row(parent, label: str, hint: str) -> ctk.CTkLabel:
        """Label row with name on left and current-value label on right."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(8, 0))
        ctk.CTkLabel(row, text=label, font=ctk.CTkFont(size=12)).pack(side="left")
        value_lbl = ctk.CTkLabel(row, text="", font=ctk.CTkFont(size=12))
        value_lbl.pack(side="right")
        ctk.CTkLabel(
            parent, text=f"Used by: {hint}",
            text_color="gray60", font=ctk.CTkFont(size=11),
        ).pack(anchor="w", padx=12, pady=(0, 2))
        return value_lbl

    def _apply_slider(self, _=None):
        self._on_apply()

    def _validate_thresholds(self, event=None):
        """Validates custom threshold entry on key release and updates border color."""
        text = self.thresholds_entry.get()
        valid = True
        
        try:
            thresholds = []
            for part in text.split(","):
                value = part.strip()
                if not value:
                    continue
                threshold = int(value)
                if not 0 < threshold < 255:
                    valid = False
                    break
                thresholds.append(threshold)
            
            # Check maximum 7 thresholds
            if len(sorted(dict.fromkeys(thresholds))) > 7:
                valid = False
                
        except ValueError:
            valid = False
            
        if valid:
            # Restore to default CTkEntry border color
            self.thresholds_entry.configure(border_color=["#979DA2", "#565B5E"])
        else:
            self.thresholds_entry.configure(border_color="red")
        
        return valid

    # ------------------------------------------------------------------ #
    #  Category / algorithm selection                                      #
    # ------------------------------------------------------------------ #

    def _on_category_change(self, category: str):
        self.category_var.set(category)
        first_algo = ALGORITHM_CATEGORIES[category][0]
        self.algo_var.set(first_algo)
        self._rebuild_radio_buttons(category)
        self._update_parameter_visibility()
        self._on_apply()

    def _rebuild_radio_buttons(self, category: str):
        for rb in self._radio_buttons:
            rb.destroy()
        self._radio_buttons.clear()

        for algo in ALGORITHM_CATEGORIES[category]:
            rb = ctk.CTkRadioButton(
                self._algo_radio_frame,
                text=algo, variable=self.algo_var, value=algo,
                command=self._on_algorithm_change,
            )
            rb.pack(anchor="w", padx=12, pady=5)
            self._radio_buttons.append(rb)

    def _on_algorithm_change(self):
        self._update_parameter_visibility()
        self._on_apply()

    # ------------------------------------------------------------------ #
    #  Parameter visibility                                                #
    # ------------------------------------------------------------------ #

    def _hide_all_param_children(self):
        for child in (
            self._freq_frame, self._gamma_frame, self._slices_frame,
            self._thresholds_frame, self._saturation_frame, self._no_params_label,
        ):
            child.place_forget()

    def _update_parameter_visibility(self):
        algo = self.algo_var.get()
        self._hide_all_param_children()

        if algo == "Mapping: Sine":
            self._freq_frame.place(relx=0, rely=0, relwidth=1)
        elif algo == "Mapping: Gamma + Plasma":
            self._gamma_frame.place(relx=0, rely=0, relwidth=1)
        elif algo == "Slicing: Background preserved":
            self._slices_frame.place(relx=0, rely=0, relwidth=1)
        elif algo == "Slicing: Custom thresholds":
            self._thresholds_frame.place(relx=0, rely=0, relwidth=1)
        elif algo == "CNN: Best model":
            self._saturation_frame.place(relx=0, rely=0, relwidth=1)
        else:
            self._no_params_label.place(relx=0.5, rely=0.5, anchor="center")

    # ------------------------------------------------------------------ #
    #  Public helpers                                                      #
    # ------------------------------------------------------------------ #

    def set_apply_btn_state(self, enabled: bool):
        """Toggle the Apply button between normal and 'Running…' states."""
        if enabled:
            self.apply_btn.configure(state="normal", text="Apply")
        else:
            self.apply_btn.configure(state="disabled", text="Running…")

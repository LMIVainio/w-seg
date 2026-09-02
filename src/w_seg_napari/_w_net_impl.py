import napari
import numpy as np
import torch
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
)

from w_seg_napari._wsegutils import process_large_image


class WNetWidget(QGroupBox):
    def __init__(
        self,
        viewer: napari.Viewer,
        generator=None,
        device=None,
        parent=None,
    ):
        super().__init__('W-Net Model Inference', parent)
        self.viewer = viewer
        self.generator = generator
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.live_update_layer = None
        self.output_layer = None

        # Flexible Qt sizing policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._init_ui()

        # Connect Napari layer listeners
        if self.viewer is not None:
            self.viewer.layers.events.inserted.connect(
                self._update_layer_choices
            )
            self.viewer.layers.events.removed.connect(
                self._update_layer_choices
            )
            self.viewer.layers.events.changed.connect(
                self._update_layer_choices
            )

        self._update_layer_choices()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # 1. Image Layer Selector
        self.combo_image_layer = QComboBox()
        self.combo_image_layer.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        form_layout.addRow('Image Layer:', self.combo_image_layer)

        # 2. Patch Size Input
        self.spin_patch_size = QSpinBox()
        self.spin_patch_size.setRange(32, 2048)
        self.spin_patch_size.setSingleStep(32)
        self.spin_patch_size.setValue(256)
        form_layout.addRow('Patch Size:', self.spin_patch_size)

        # 3. Stride Input
        self.spin_stride = QSpinBox()
        self.spin_stride.setRange(16, 1024)
        self.spin_stride.setSingleStep(16)
        self.spin_stride.setValue(128)
        form_layout.addRow('Stride:', self.spin_stride)

        # 4. Mode Options (Radio Buttons)
        mode_layout = QHBoxLayout()
        self.radio_normal = QRadioButton('Normal (Fast)')
        self.radio_live = QRadioButton('Live View')
        self.radio_live.setChecked(True)

        mode_layout.addWidget(self.radio_normal)
        mode_layout.addWidget(self.radio_live)
        form_layout.addRow('Execution Mode:', mode_layout)

        # 5. Predict Button
        self.predict_btn = QPushButton('Predict Matrix Output')
        self.predict_btn.clicked.connect(self._on_predict)

        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.predict_btn)

    def _update_layer_choices(self, event=None):
        """Refreshes Image layer options in the dropdown."""
        if self.viewer is None:
            return

        current_text = self.combo_image_layer.currentText()
        self.combo_image_layer.blockSignals(True)
        self.combo_image_layer.clear()

        image_layers = [
            lay
            for lay in self.viewer.layers
            if isinstance(lay, napari.layers.Image)
        ]

        for layer in image_layers:
            self.combo_image_layer.addItem(layer.name, userData=layer)

        # Preserve selection if layer still exists
        idx = self.combo_image_layer.findText(current_text)
        if idx != -1:
            self.combo_image_layer.setCurrentIndex(idx)

        self.combo_image_layer.blockSignals(False)

    def _on_predict(self):
        """Callback executed when clicking the predict button."""
        image_layer = self.combo_image_layer.currentData()

        if image_layer is None:
            active = self.viewer.layers.selection.active
            if isinstance(active, napari.layers.Image):
                image_layer = active

        if image_layer is None:
            print('[w-seg] Error: Please select an Image layer.')
            if self.viewer:
                self.viewer.status = '[w-seg] Error: Please select an Image layer.'
            return

        if self.generator is None:
            print('[w-seg] Error: Model generator missing.')
            if self.viewer:
                self.viewer.status = '[w-seg] Error: Model generator missing.'
            return

        patch_size = self.spin_patch_size.value()
        stride = self.spin_stride.value()
        is_live = self.radio_live.isChecked()

        print(f'[w-seg] Running prediction with model on {self.device}...')
        if self.viewer:
            self.viewer.status = f'[w-seg] Running prediction on {self.device}...'

        raw_data = image_layer.data
        H, W = raw_data.shape[0], raw_data.shape[1]

        if raw_data.dtype == np.uint8:
            raw_data = raw_data.astype(np.float32) / 255.0
        else:
            raw_data = raw_data.astype(np.float32)

        input_tensor = torch.from_numpy(raw_data).permute(2, 0, 1).float()

        canvas = np.zeros((H, W), dtype=np.float32)
        if (
            self.live_update_layer is not None
            and self.live_update_layer in self.viewer.layers
        ):
            self.live_update_layer.data = canvas
            self.live_update_layer.opacity = 0.8 if is_live else 0.0
        else:
            self.live_update_layer = self.viewer.add_image(
                canvas,
                name='live_update_layer',
                colormap='red',
                contrast_limits=[0.0, 1.0],
                blending='additive',
                opacity=0.8 if is_live else 0.0,
            )

        mode_str = 'Live View' if is_live else 'Normal (Fast)'
        print(
            f'[w-seg] Stitching processing initialized using patches of size {patch_size} [{mode_str}]...'
        )

        final_sem, final_aff = process_large_image(
            image=input_tensor,
            model=self.generator,
            patch_size=patch_size,
            stride=stride,
            device=self.device,
            live_layer=self.live_update_layer,
            live_view=is_live,
        )

        sem_numpy = np.squeeze(final_sem)
        aff_numpy = np.clip(final_aff, 0.0, 1.0)

        original_name = image_layer.name
        predicted_layer_name = f'pred_{original_name}'

        # Add affinity layers
        aff_layers = self.viewer.add_image(
            aff_numpy,
            name=[f'affH_{original_name}', f'affV_{original_name}'],
            channel_axis=0,
            colormap='gray',
            blending='additive',
        )

        for layer in aff_layers:
            current_idx = self.viewer.layers.index(layer)
            self.viewer.layers.move(current_idx, 0)

        # Cleanup live layer
        if (
            self.live_update_layer is not None
            and self.live_update_layer in self.viewer.layers
        ):
            self.viewer.layers.remove(self.live_update_layer)
            self.live_update_layer = None

        if predicted_layer_name in self.viewer.layers:
            final_layer = self.viewer.layers[predicted_layer_name]
            final_layer.data = sem_numpy
            final_layer.refresh()
        else:
            final_layer = self.viewer.add_image(
                sem_numpy,
                name=predicted_layer_name,
                colormap='red',
                contrast_limits=[0.0, 1.0],
                blending='additive',
                opacity=1.0,
            )

        self.viewer.layers.selection.clear()
        self.viewer.layers.selection.add(final_layer)

        sem_idx = self.viewer.layers.index(final_layer)
        self.viewer.layers.move(sem_idx, -1)

        print('[w-seg] Processing complete! Final canvas layers blended successfully.')
        if self.viewer:
            self.viewer.status = '[w-seg] Prediction complete!'
import threading

import napari
import numpy as np
import torch
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
)

from w_seg_napari._affinity import layer_payloads
from w_seg_napari._batch_io import to_rgb
from w_seg_napari._geometry_fields import geometry_payloads
from w_seg_napari._wsegutils import get_model_config, process_large_image


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
        self._generator = generator
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self._tissue = None  # (folder, model, config) of the mask model
        self.live_update_layer = None
        self.output_layer = None
        self.worker = None
        self.cancel_event = threading.Event()

        # Flexible Qt sizing policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._init_ui()

        # Connect Napari layer listeners
        if self.viewer is not None:
            self.viewer.layers.events.inserted.connect(
                self.refresh_layer_choices
            )
            self.viewer.layers.events.removed.connect(
                self.refresh_layer_choices
            )
            self.viewer.layers.events.changed.connect(
                self.refresh_layer_choices
            )

        self.refresh_layer_choices()

    @property
    def generator(self):
        return self._generator

    @generator.setter
    def generator(self, value):
        # a new model may or may not come with a tissue mask model
        self._generator = value
        if hasattr(self, 'chk_tissue'):
            self.refresh_tissue_option()

    def _tissue_folder(self):
        """Folder of the tissue mask model of the loaded model, or None."""
        from w_seg_napari import _tissue_mask as tm
        from w_seg_napari._wsegutils import get_active_model_info

        if self._generator is None:
            return None
        return tm.find_tissue_mask(get_active_model_info().get('folder'))

    def refresh_tissue_option(self):
        """The tissue mask checkboxes: on and usable when the loaded model
        comes with a tissue mask model, off and greyed out when not."""
        has = self._tissue_folder() is not None
        self.chk_tissue.setEnabled(has)
        self.chk_tissue_convex.setEnabled(has)
        self.chk_tissue.setChecked(has)
        self.chk_tissue.setToolTip(
            'The tissue mask model of the loaded model finds the tissue on '
            'the whole image first (512 x 512); tiles of the prediction '
            'outside it are skipped. The mask is shown as a layer.'
            if has
            else 'The loaded model has no tissue mask model '
            '(tissue_mask.keras + tissue_mask.json in its folder).'
        )

    def _tissue_lookup_for(self, rgb01):
        """TissueLookup of the image, or None when the option is off."""
        if not (self.chk_tissue.isEnabled() and self.chk_tissue.isChecked()):
            return None
        from w_seg_napari import _tissue_mask as tm

        folder = self._tissue_folder()
        if folder is None:
            return None
        if self._tissue is None or self._tissue[0] != str(folder):
            model, cfg = tm.load_tissue_mask(folder, self.device)
            self._tissue = (str(folder), model, cfg)
        _, model, cfg = self._tissue
        return tm.tissue_lookup(
            model,
            cfg,
            rgb01,
            self.device,
            convex=self.chk_tissue_convex.isChecked(),
        )

    def _show_tissue_layer(self, look, scale, translate):
        """The mask as a Labels layer on top of the image."""
        h, w = look.mask.shape
        H, W = look.shape
        data = look.mask.astype(np.uint8)
        sc = (scale[-2] * H / h, scale[-1] * W / w)
        if 'tissue_mask' in self.viewer.layers:
            layer = self.viewer.layers['tissue_mask']
            layer.data = data
            layer.scale = sc
            layer.translate = translate
        else:
            self.viewer.add_labels(
                data,
                name='tissue_mask',
                scale=sc,
                translate=translate,
                opacity=0.35,
            )

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

        # 4b. Tissue mask (only when the loaded model has one)
        tissue_layout = QHBoxLayout()
        self.chk_tissue = QCheckBox('Skip background (tissue mask)')
        self.chk_tissue_convex = QCheckBox('Convex')
        self.chk_tissue_convex.setToolTip(
            'Use the convex hull of every piece of tissue: no notches, so '
            'a thin edge of the tissue is not cut off.'
        )
        tissue_layout.addWidget(self.chk_tissue)
        tissue_layout.addWidget(self.chk_tissue_convex)
        form_layout.addRow('Tissue:', tissue_layout)
        self.refresh_tissue_option()

        # 5. Action Buttons (Predict & Cancel)
        btn_layout = QHBoxLayout()
        self.predict_btn = QPushButton('Predict Matrix Output')
        self.predict_btn.setStyleSheet(
            'font-weight: bold; background-color: #2b5b84; color: white; padding: 6px;'
        )
        self.predict_btn.clicked.connect(self._on_predict)

        self.cancel_btn = QPushButton('❌ Cancel')
        self.cancel_btn.setStyleSheet(
            'font-weight: bold; background-color: #842b2b; color: white; padding: 6px;'
        )
        self.cancel_btn.setEnabled(False)  # Disabled by default
        self.cancel_btn.clicked.connect(self._on_cancel)

        btn_layout.addWidget(self.predict_btn)
        btn_layout.addWidget(self.cancel_btn)

        main_layout.addLayout(form_layout)
        main_layout.addLayout(btn_layout)

    def refresh_layer_choices(self, event=None):
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

    def _set_ui_processing_state(self, is_processing: bool):
        """Toggles widget UI controls depending on execution status."""
        self.predict_btn.setEnabled(not is_processing)
        self.combo_image_layer.setEnabled(not is_processing)
        self.spin_patch_size.setEnabled(not is_processing)
        self.spin_stride.setEnabled(not is_processing)
        self.radio_normal.setEnabled(not is_processing)
        self.radio_live.setEnabled(not is_processing)

        self.cancel_btn.setEnabled(is_processing)

    def _on_cancel(self):
        """Triggers cancellation of background worker."""
        if self.worker is not None:
            print('[w-seg] Cancelling inference request...')
            self.cancel_event.set()
            if self.viewer:
                self.viewer.status = '[w-seg] Cancelling inference process...'
            self.cancel_btn.setEnabled(False)

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
                self.viewer.status = (
                    '[w-seg] Error: Please select an Image layer.'
                )
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
            self.viewer.status = (
                f'[w-seg] Running prediction on {self.device}...'
            )

        # Greyscale (H, W), single-channel or RGBA images become (H, W, 3)
        try:
            raw_data = to_rgb(image_layer.data)
        except ValueError as e:
            print(f'[w-seg] Error: {e}')
            if self.viewer:
                self.viewer.status = f'[w-seg] Error: {e}'
            return
        H, W = raw_data.shape[0], raw_data.shape[1]

        if raw_data.dtype == np.uint8:
            raw_data = raw_data.astype(np.float32) / 255.0
        else:
            raw_data = raw_data.astype(np.float32)

        input_tensor = torch.from_numpy(raw_data).permute(2, 0, 1).float()

        tile_filter = None
        try:
            look = self._tissue_lookup_for(raw_data)
        except Exception as e:  # a broken mask model must not stop the run
            look = None
            print(f'[w-seg] Tissue mask not used: {e}')
        if look is not None:
            self._show_tissue_layer(
                look, image_layer.scale, image_layer.translate
            )
            tile_filter = lambda y, x, size: look.keep_tile(y, x, size, 0.01)  # noqa: E731
            print(
                f'[w-seg] Tissue mask: {look.mask.mean():.0%} of the image '
                'is tissue.'
            )

        # Capture spatial metadata from cropped input layer
        layer_scale = image_layer.scale
        layer_translate = image_layer.translate

        canvas = np.zeros((H, W), dtype=np.float32)
        if (
            self.live_update_layer is not None
            and self.live_update_layer in self.viewer.layers
        ):
            self.live_update_layer.data = canvas
            self.live_update_layer.scale = layer_scale
            self.live_update_layer.translate = layer_translate
            self.live_update_layer.opacity = 0.8 if is_live else 0.0
        else:
            self.live_update_layer = self.viewer.add_image(
                canvas,
                name='live_update_layer',
                colormap='red',
                contrast_limits=[0.0, 1.0],
                blending='additive',
                opacity=0.8 if is_live else 0.0,
                scale=layer_scale,
                translate=layer_translate,
            )

        mode_str = 'Live View' if is_live else 'Normal (Fast)'
        print(
            f'[w-seg] Stitching processing initialized using patches of size {patch_size} [{mode_str}]...'
        )

        self._set_ui_processing_state(True)
        self.cancel_event.clear()

        # The model's config says how its input is preprocessed and how its
        # affinity channels are laid out (old models get a built-in config).
        cfg = get_model_config(self.generator)
        pre = cfg['preprocessing']
        output_names = (cfg['outputs']['wall'], cfg['outputs']['affinity'])
        geometry_name = cfg['outputs'].get('geometry')  # wasp_v4 models only

        # Instantiate background thread worker using napari.qt.thread_worker
        @thread_worker
        def _run_inference_worker():
            return (
                yield from process_large_image(
                    image=input_tensor,
                    model=self.generator,
                    patch_size=patch_size,
                    stride=stride,
                    device=self.device,
                    live_view=is_live,
                    is_cancelled=self.cancel_event.is_set,
                    hue_encoding=pre['hue_encoding'],
                    enable_lab=pre['enable_lab'],
                    output_names=output_names,
                    geometry_name=geometry_name,
                    tile_filter=tile_filter,
                )
            )

        self.worker = _run_inference_worker()
        self.worker.yielded.connect(self._update_live_canvas)
        self.worker.returned.connect(
            lambda res: self._on_prediction_finished(
                res, image_layer.name, layer_scale, layer_translate, cfg
            )
        )
        self.worker.errored.connect(self._on_prediction_error)
        self.worker.start()

    def _update_live_canvas(self, update_info):
        """Safely updates live display canvas on the Qt main thread."""
        if update_info is None:
            return

        if (
            self.live_update_layer is not None
            and self.live_update_layer in self.viewer.layers
        ):
            y, x, patch_size, local_norm = update_info

            # Direct slice update on main thread array
            self.live_update_layer.data[
                y : y + patch_size, x : x + patch_size
            ] = local_norm

            # Trigger VisPy redrawing
        self.live_update_layer.refresh()

    def _cleanup_live_layer(self):
        """Removes the live processing layer."""
        if (
            self.live_update_layer is not None
            and self.live_update_layer in self.viewer.layers
        ):
            self.viewer.layers.remove(self.live_update_layer)
            self.live_update_layer = None

    def _on_prediction_error(self, exc):
        """Handles worker thread exception."""
        self._cleanup_live_layer()
        self._set_ui_processing_state(False)
        print(f'[w-seg] Error during processing: {exc}')
        if self.viewer:
            self.viewer.status = f'[w-seg] Error: {exc}'

    def _on_prediction_finished(
        self, result, original_name, layer_scale, layer_translate, cfg
    ):
        """Called when thread returns output or gets cancelled."""
        final_geo = None
        if result and len(result) == 3:
            final_sem, final_aff, final_geo = result
        else:
            final_sem, final_aff = result if result else (None, None)

        self._cleanup_live_layer()
        self._set_ui_processing_state(False)

        if self.cancel_event.is_set() or final_sem is None:
            print('[w-seg] Prediction was cancelled or returned empty.')
            if self.viewer:
                self.viewer.status = '[w-seg] Prediction cancelled.'
            return

        sem_numpy = np.squeeze(final_sem)
        predicted_layer_name = f'pred_{original_name}'

        # Affinity layers. New models: one layer per range, 4 channels each
        # (E, S, SE, SW) on the first axis, named aff_k<range>_<image>; the
        # offsets travel in layer.metadata. Old models: affH_ and affV_.
        for payload in layer_payloads(final_aff, cfg, original_name):
            if payload['name'] in self.viewer.layers:
                aff_layer = self.viewer.layers[payload['name']]
                aff_layer.data = payload['data']
                aff_layer.metadata = payload['metadata']
                aff_layer.scale = layer_scale
                aff_layer.translate = layer_translate
            else:
                aff_layer = self.viewer.add_image(
                    payload['data'],
                    name=payload['name'],
                    metadata=payload['metadata'],
                    colormap='gray',
                    blending='additive',
                    scale=layer_scale,
                    translate=layer_translate,
                )
            self.viewer.layers.move(self.viewer.layers.index(aff_layer), 0)

        if final_geo is not None:
            self._add_geometry_layers(
                final_geo, original_name, layer_scale, layer_translate
            )

        if predicted_layer_name in self.viewer.layers:
            final_layer = self.viewer.layers[predicted_layer_name]
            final_layer.data = sem_numpy
            final_layer.scale = layer_scale
            final_layer.translate = layer_translate
            final_layer.refresh()
        else:
            final_layer = self.viewer.add_image(
                sem_numpy,
                name=predicted_layer_name,
                colormap='red',
                contrast_limits=[0.0, 1.0],
                blending='additive',
                opacity=1.0,
                scale=layer_scale,
                translate=layer_translate,
            )

        self.viewer.layers.selection.clear()
        self.viewer.layers.selection.add(final_layer)

        sem_idx = self.viewer.layers.index(final_layer)
        self.viewer.layers.move(sem_idx, -1)

        print('[w-seg] Processing complete!')
        if self.viewer:
            self.viewer.status = '[w-seg] Prediction complete!'

    def _add_geometry_layers(self, geo, source, scale, translate):
        """Edge distance, skeleton, direction (RGB) and vector layers."""
        for p in geometry_payloads(geo, source):
            name, data = p['name'], p['data']
            if name in self.viewer.layers:
                layer = self.viewer.layers[name]
                layer.data = data
                layer.scale = scale
                layer.translate = translate
                continue
            kwargs = dict(
                name=name, scale=scale, translate=translate, **p['kwargs']
            )
            if p['kind'] == 'vectors':
                self.viewer.add_vectors(data, **kwargs)
            elif p['kind'] == 'rgb':
                self.viewer.add_image(data, rgb=True, **kwargs)
            else:
                self.viewer.add_image(data, **kwargs)
        # under everything, so that on top of the original image only the
        # predicted cell wall is drawn
        for p in geometry_payloads(geo, source):
            if p['name'] in self.viewer.layers:
                self.viewer.layers.move(self.viewer.layers.index(p['name']), 0)

    def get_settings(self):
        return {
            'patch_size': self.spin_patch_size.value(),
            'stride': self.spin_stride.value(),
            'execution_mode': 'live'
            if self.radio_live.isChecked()
            else 'normal',
        }
